#!/usr/bin/env python
"""Does torch.compile COMPOSE with batched reverse mode? (revision E1)

Spec SS45 says the speedups must be quoted against the fastest correct baseline.
Two independent accelerations are available here:

  * batched reverse mode -- 3.6x at B=1, usable only once the TensorExpr fuser is
    disabled (see `configure_e3nn_for_batched_vjp`);
  * torch.compile on the forward -- 1.04-1.10x, via AOTAutograd also cheapening
    the backward (scripts/02c_compiled_forward.py).

If they composed, compiled+batched would be the primary baseline. This script
answers that by measurement rather than assumption, and records the answer either
way -- a negative systems result is still a result, and it is the justification
for reporting the eager batched path as primary.

Note the failure, if it occurs, needs several calls to appear: TorchScript emits an
optimized plan only after two warm-ups, so a single successful call proves nothing.
We therefore make five consecutive calls before declaring composition.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch

from forcesketch.adapters.mace_data import load_frames, make_loader
from forcesketch.adapters.mace_mhc import MaceMHCAdapter
from forcesketch.utils.reproducibility import git_commit, pin_numerics

CKPT = "models/zenodo/3BPA/trainset_100/multihead-disjoint/multihead_committee_stagetwo.model"
DATA = "data/3bpa/test_1200K_ref.xyz"


def timed(fn, iters: int, warm: int) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return st.median(sorted(s.elapsed_time(e) for s, e in ev))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--out", default="results/raw/02e_compile_batched.jsonl")
    args = ap.parse_args()

    pin_numerics()
    sha, dirty = git_commit()
    frames = load_frames(DATA, limit=max(args.batch_sizes))
    records = []

    for B in args.batch_sizes:
        # A fresh adapter per batch size: torch.compile mutates the module, and a
        # failed compile must not contaminate the next measurement.
        ad = MaceMHCAdapter.from_checkpoint(CKPT, dtype=torch.float32)
        M = ad.num_heads
        batch = ad.prepare(
            next(iter(make_loader(frames[:B], ad.model, batch_size=B))).to_dict())
        seeds = torch.randn(args.lanes, B, M, device=ad.device, dtype=ad.dtype)

        ref = ad.vjp_for_seeds(batch, seeds, batched=True)
        t_eager = timed(lambda: ad.vjp_for_seeds(batch, seeds, batched=True),
                        args.iters, args.warmup)

        torch._dynamo.config.suppress_errors = True
        ad.model.compile(mode="default")
        rec = {"experiment_id": "02e_compile_batched", "git_commit": sha,
               "git_dirty": dirty, "dataset": "3bpa", "batch_size": B,
               "lanes": args.lanes, "num_heads": M, "precision": "fp32",
               "eager_batched_ms": t_eager, "n_calls": args.calls,
               "gpu": torch.cuda.get_device_name(0)}
        try:
            for _ in range(args.calls):
                out = ad.vjp_for_seeds(batch, seeds, batched=True)
            t_comp = timed(lambda: ad.vjp_for_seeds(batch, seeds, batched=True),
                           args.iters, args.warmup)
            rec.update(composes=True, compiled_batched_ms=t_comp,
                       speedup=t_eager / t_comp,
                       rel_err=float((out - ref).abs().max() / ref.abs().max()))
            print(f"B={B:>2} COMPOSES  eager {t_eager:.2f} ms  compiled {t_comp:.2f} ms "
                  f"({rec['speedup']:.3f}x, rel err {rec['rel_err']:.1e})")
        except RuntimeError as ex:
            rec.update(composes=False, error=type(ex).__name__, message=str(ex)[:300])
            print(f"B={B:>2} DOES NOT COMPOSE: {str(ex)[:120]}")
        records.append(rec)
        del ad
        torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_path} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
