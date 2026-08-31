#!/usr/bin/env python
"""Compiled exact implementation (plan Task 2.1; spec SS17 item 5).

Spec SS17 asks for a `torch.compile` version "where beneficial". On this stack
compiling the *gradient call* is not an option -- `torch.compile` over
`torch.autograd.grad(..., is_grads_batched=True)` fails outright -- so the viable
pattern is to compile the FORWARD and leave the reverse pass eager. That is still
worth measuring for two reasons:

  1. the forward is a real, if small, share of the step (9.4 ms of 100.6 ms exact);
  2. AOTAutograd compiles a backward from the compiled forward, so the reverse
     pass -- which dominates -- may speed up too. That is the interesting case and
     the one this script is really testing.

Compilation time is measured and reported separately, never folded into the
per-step number (spec SS44 item 12). Correctness is verified against eager before
any timing is trusted.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import torch

from forcesketch.adapters.mace_data import load_frames, make_loader
from forcesketch.adapters.mace_mhc import MaceMHCAdapter
from forcesketch.exact.centered_basis import exact_seed_bundle
from forcesketch.sketches.registry import make_sketch_seeds
from forcesketch.utils.reproducibility import git_commit, pin_numerics

CKPT = "models/zenodo/3BPA/trainset_100/multihead-disjoint/multihead_committee_stagetwo.model"
DATA = "data/3bpa/test_1200K_ref.xyz"


def timed(fn, iters: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for s, e in evs:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in evs)
    return st.median(t), t[3 * len(t) // 4] - t[len(t) // 4]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--suppress-errors", action="store_true",
                    help="fall back to eager on regions Dynamo cannot trace, instead of "
                         "aborting. Needed here: Dynamo builds a SEQUENCE_LENGTH guard on "
                         "e3nn's Irrep and calls len() on it, which e3nn raises "
                         "NotImplementedError for by design.")
    ap.add_argument("--out", default="results/raw/02c_compiled.jsonl")
    args = ap.parse_args()

    pin_numerics()
    if args.suppress_errors:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    adapter = MaceMHCAdapter.from_checkpoint(CKPT, dtype=torch.float32)
    M = adapter.num_heads
    frames = load_frames(DATA, limit=args.batch_size)
    batch = adapter.prepare(
        next(iter(make_loader(frames, adapter.model, batch_size=args.batch_size))).to_dict())

    committee = adapter._committee
    raw_model = adapter.model

    def eager_energies():
        return raw_model(batch, training=False, compute_force=False,
                         committee_heads=committee)["heads"]["energy"]

    print(f"3BPA disjoint, B={args.batch_size} "
          f"({args.batch_size * len(frames[0])} atoms), fp32, mode={args.mode}\n")

    ref = eager_energies().detach().clone()

    # --- compile the forward; time compilation separately (spec SS44 item 12) ---
    compiled = torch.compile(raw_model, mode=args.mode, dynamic=False)

    def compiled_energies():
        return compiled(batch, training=False, compute_force=False,
                        committee_heads=committee)["heads"]["energy"]

    t0 = time.perf_counter()
    compile_ok, compile_err = True, ""
    try:
        got = compiled_energies()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        compile_ok, compile_err = False, f"{type(exc).__name__}: {exc}"[:300]
        got = None
    compile_s = time.perf_counter() - t0

    if not compile_ok:
        print(f"torch.compile FAILED after {compile_s:.1f}s\n  {compile_err}")
        print("\nSpec SS17 item 5 is therefore reported as unavailable on this stack.")
        rec = {"experiment_id": "02c_compiled", "compiled": False, "error": compile_err,
               "compile_seconds": compile_s, "mode": args.mode}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rec) + "\n")
        return 0

    err = (got.detach() - ref).abs().max().item() / ref.abs().max().item()
    print(f"compiled in {compile_s:.1f}s (reported separately, never in the step time)")
    print(f"correctness vs eager: max rel err {err:.2e}"
          f"  {'OK' if err < 1e-5 else 'FAIL -- results not trusted'}\n")
    if err >= 1e-5:
        return 1

    sha, dirty = git_commit()
    records = []
    print(f"{'lanes':>6}{'eager ms':>11}{'compiled ms':>13}{'speedup':>10}")
    print("-" * 40)
    for L, bundle in [(7, exact_seed_bundle(M, args.batch_size, dtype=torch.float32,
                                            device=adapter.device))] + [
            (K, make_sketch_seeds("haar", M=M, K=K, batch_size=args.batch_size,
                                  seed=1000003, dtype=torch.float32,
                                  device=adapter.device)) for K in (2, 3, 4)]:
        def step_eager(b=bundle):
            e = eager_energies()
            return torch.autograd.grad(e, batch["positions"], grad_outputs=b.seeds[0],
                                       retain_graph=True)[0] if False else \
                torch.stack([torch.autograd.grad(e, batch["positions"],
                                                 grad_outputs=b.seeds[i],
                                                 retain_graph=(i < b.K - 1))[0]
                             for i in range(b.K)])

        def step_compiled(b=bundle):
            e = compiled_energies()
            return torch.stack([torch.autograd.grad(e, batch["positions"],
                                                    grad_outputs=b.seeds[i],
                                                    retain_graph=(i < b.K - 1))[0]
                                for i in range(b.K)])

        t_e, iqr_e = timed(step_eager, args.iters, args.warmup)
        t_c, iqr_c = timed(step_compiled, args.iters, args.warmup)
        print(f"{L:>6}{t_e:>11.2f}{t_c:>13.2f}{t_e / t_c:>9.2f}x")
        records.append({
            "experiment_id": "02c_compiled", "git_commit": sha, "git_dirty": dirty,
            "dataset": "3bpa", "lanes": L, "batch_size": args.batch_size,
            "mode": args.mode, "compiled": True, "compile_seconds": compile_s,
            "eager_ms": t_e, "eager_iqr_ms": iqr_e,
            "compiled_ms": t_c, "compiled_iqr_ms": iqr_c,
            "speedup": t_e / t_c, "correctness_rel_err": err,
            "precision": "fp32", "gpu": torch.cuda.get_device_name(0),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    best = max(r["speedup"] for r in records)
    print(f"\nbest speedup {best:.2f}x -- "
          f"{'worth keeping' if best > 1.05 else 'NOT beneficial; eager stays the baseline'}")
    print(f"wrote {args.out} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
