#!/usr/bin/env python
"""J0b -- does the TensorExpr-fuser fix still hold on this hardware and CUDA build?

This is on the critical path for every timing number in the paper, because it
decides which exact implementation is the R5 baseline. The workshop's headline
systems result depends on `torch.autograd.grad(..., is_grads_batched=True)` being
usable, which it only is because `configure_e3nn_for_batched_vjp()` disables the
TensorExpr fuser before the first forward.

Three outcomes, each of which the paper must report differently:

  fuser_needed and fix_sufficient
      the workshop result stands, and now spans two CUDA builds and two compute
      capabilities -- which strengthens it, since it shows the cause is the JIT
      layer rather than a hardware quirk

  not fuser_needed
      `configure_e3nn_for_batched_vjp()` is imposing a gratuitous handicap on the
      SERIAL baseline (it loses fusion), so under R5 every comparison must be
      redone with the fuser on

  fuser_needed and not fix_sufficient
      batched reverse mode is unavailable here; serial becomes the R5 baseline and
      the speedups revert toward the pre-resolution figures

Because a different number is reported in each branch, the branch taken must
itself be a recorded artifact rather than an assumption. Each probe runs in a
FRESH SUBPROCESS: TorchScript caches an optimized plan per graph, so the fuser
state cannot be changed after the first forward within one process.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"

PROBE = r'''
import json, os, sys, torch
sys.path.insert(0, {frozen_src!r})
# Import e3nn/torch FIRST and set the fuser state BEFORE importing the adapter,
# whose module-level configure_e3nn_for_batched_vjp() would otherwise disable it.
import e3nn
if {fuser_on}:
    torch._C._jit_set_texpr_fuser_enabled(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
import forcesketch.adapters.mace_mhc as mm
if {fuser_on}:                       # undo the import-time call
    torch._C._jit_set_texpr_fuser_enabled(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
else:
    mm.configure_e3nn_for_batched_vjp()

from forcesketch.adapters.mace_data import load_frames, make_loader
from forcesketch.exact.centered_basis import exact_seed_bundle

ad = mm.MaceMHCAdapter.from_checkpoint({ckpt!r}, device="cuda", dtype=torch.float32)
frames = load_frames({data!r}, limit={bsz})
batch = next(iter(make_loader(frames, ad.model, batch_size={bsz})))
b = ad.prepare(batch.to_dict())
M = ad.num_heads
bundle = exact_seed_bundle(M, {bsz}, dtype=torch.float32, device=ad.device)

out = {{"fuser_on": {fuser_on}, "n_calls": {ncalls},
       "texpr_enabled": torch._C._jit_texpr_fuser_enabled()}}
first_fail, errs, ok = None, [], 0
for i in range({ncalls}):
    try:
        g = ad.vjp_for_seeds(b, bundle.seeds, batched=True)
        torch.cuda.synchronize(); ok += 1
    except Exception as e:
        if first_fail is None:
            first_fail = i; errs.append(type(e).__name__ + ": " + str(e)[:160])
out["n_success"] = ok
out["first_failing_call_index"] = first_fail
out["errors"] = errs
# agreement with the serial path, in fp64, only if any batched call worked
if ok:
    ad64 = mm.MaceMHCAdapter.from_checkpoint({ckpt!r}, device="cuda", dtype=torch.float64)
    b64 = ad64.prepare(next(iter(make_loader(frames, ad64.model, batch_size={bsz}))).to_dict())
    bd64 = exact_seed_bundle(M, {bsz}, dtype=torch.float64, device=ad64.device)
    gs = ad64.vjp_for_seeds(b64, bd64.seeds, batched=False)
    try:
        gb = ad64.vjp_for_seeds(b64, bd64.seeds, batched=True)
        out["max_rel_err_vs_serial_fp64"] = float((gb-gs).abs().max() / gs.abs().max())
    except Exception as e:
        out["max_rel_err_vs_serial_fp64"] = None
        out["fp64_batched_error"] = str(e)[:160]
print("@@RESULT@@" + json.dumps(out))
'''


def run_probe(*, fuser_on: bool, ckpt: str, data: str, bsz: int, ncalls: int, env: dict) -> dict:
    code = PROBE.format(frozen_src=str(FROZEN / "src"), fuser_on=fuser_on,
                        ckpt=ckpt, data=data, bsz=bsz, ncalls=ncalls)
    p = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                       capture_output=True, text=True, cwd=str(FROZEN), env=env)
    for line in p.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {"fuser_on": fuser_on, "error": (p.stderr or p.stdout)[-600:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/zenodo/3BPA/trainset_100/multihead-disjoint/multihead_committee_stagetwo.model")
    ap.add_argument("--data", default="data/3bpa/test_1200K_ref.xyz")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-calls", type=int, default=12)
    args = ap.parse_args()

    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
               FORCESKETCH_MACE_LOSS="normal")
    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cap = ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else "-"

    res = {"device_name": dev, "compute_capability": cap,
           "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
           "is_mig": "MIG" in dev, "probes": {}}

    for label, fuser_on in (("fuser_ON_(upstream_default)", True), ("fuser_OFF_(shipped_fix)", False)):
        r = run_probe(fuser_on=fuser_on, ckpt=args.ckpt, data=args.data,
                      bsz=args.batch_size, ncalls=args.n_calls, env=env)
        res["probes"][label] = r
        print(f"\n=== {label} ===")
        for k in ("texpr_enabled", "n_success", "first_failing_call_index",
                  "max_rel_err_vs_serial_fp64"):
            if k in r:
                print(f"  {k:32s} {r[k]}")
        for e in r.get("errors", [])[:1]:
            print(f"  first error                      {e}")
        if "error" in r:
            print(f"  PROBE FAILED: {r['error'][-300:]}")

    on, off = res["probes"]["fuser_ON_(upstream_default)"], res["probes"]["fuser_OFF_(shipped_fix)"]
    res["fuser_needed"] = bool(on.get("first_failing_call_index") is not None)
    res["fix_sufficient"] = bool(off.get("n_success") == args.n_calls)
    verdict = ("fix still needed AND sufficient -- workshop result stands"
               if res["fuser_needed"] and res["fix_sufficient"] else
               "fix NO LONGER NEEDED -- R5 baselines must be redone with the fuser ON"
               if not res["fuser_needed"] else
               "fix NO LONGER SUFFICIENT -- batched unavailable; serial becomes the R5 baseline")
    res["verdict"] = verdict
    print(f"\nfuser_needed={res['fuser_needed']}  fix_sufficient={res['fix_sufficient']}")
    print(f"VERDICT: {verdict}")

    tagdev = dev.replace(" ", "_").replace("/", "_")
    out = ROOT / f"results/records/j0b_guard_{tagdev}_torch{torch.__version__}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
