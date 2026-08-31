#!/usr/bin/env python
"""Lane-scaling microbenchmark on the real committee (plan Task 2.3).

The decisive early measurement: how does reverse-pass latency scale with the
number of head-space cotangent lanes L? Exact force UQ needs L = M = 8 (one mean
lane + M-1 centered directions); ForceSketch(K) needs L = K+1. The ratio
T(L=8)/T(L=K+1) is the ceiling on the incremental UQ speedup, and spec SS50 kills
the project if it stays under 1.2x.

Implementation note. This script measures the SERIAL loop only, which is what the
reference implementation's `get_outputs_committee` does. That is deliberate: the
lane-cost model T(L) = a + bL that the paper fits is a property of the serial
path, and mixing implementations into one fit would make the slope meaningless.

It is NOT the strongest available baseline. Batched reverse mode
(`is_grads_batched=True`) initially appears unusable here -- it succeeds for the
first two calls in a fresh process and then raises

    RuntimeError: Cannot access data pointer of Tensor that doesn't have storage

-- but the cause is the TensorExpr fuser fusing the reverse graph of e3nn's
scripted `_spherical_harmonics`, and disabling that fuser before the first forward
fixes it (see `forcesketch.adapters.mace_mhc.configure_e3nn_for_batched_vjp`).
Spec SS45 forbids reporting a speedup against a weaker baseline than the one
available, so the head-to-head comparison lives in `02d_batched_vs_serial.py` and
that is what the paper's speedups are measured against.

Per spec SS44: CUDA events, per-iteration timing, >=100 warmup, median + IQR.
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

VARIANTS = {
    "disjoint": ("models/zenodo/3BPA/trainset_100/multihead-disjoint",
                 "data/3bpa/test_1200K_ref.xyz"),
    "rmd17-ethanol": ("models/zenodo/rMD17/full_trainset/multihead-disjoint",
                      "data/rmd17/ethanol_test_ref.xyz"),
    "rmd17-aspirin": ("models/zenodo/rMD17/full_trainset/multihead-disjoint",
                      "data/rmd17/aspirin_test_ref.xyz"),
    "rmd17-azobenzene": ("models/zenodo/rMD17/full_trainset/multihead-disjoint",
                         "data/rmd17/azobenzene_test_ref.xyz"),
}


def timed(fn, iters: int, warmup: int) -> tuple[float, float, float]:
    """Median, IQR and min in ms, using one CUDA event pair per iteration and a
    single synchronize at the end (spec SS44: sync only at timing boundaries)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for s, e in evs:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in evs)
    q1, q3 = t[len(t) // 4], t[3 * len(t) // 4]
    return st.median(t), q3 - q1, t[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--variant", default="disjoint", choices=list(VARIANTS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pin_numerics()
    ckpt_dir, data_path = VARIANTS[args.variant]
    adapter = MaceMHCAdapter.from_checkpoint(
        f"{ckpt_dir}/multihead_committee_stagetwo.model", dtype=torch.float32)
    M = adapter.num_heads
    frames = load_frames(data_path, limit=max(args.batch_sizes))
    sha, dirty = git_commit()
    records = []

    print(f"{args.variant}: M={M}, {len(frames[0])} atoms/structure")
    print(f"{args.iters} measured iters, {args.warmup} warmup, CUDA events, serial VJP\n")
    print(f"{'B':>3} {'atoms':>6} | " + "  ".join(f"L={L}" for L in range(1, M + 1)))

    for B in args.batch_sizes:
        batch = adapter.prepare(
            next(iter(make_loader(frames[:B], adapter.model, batch_size=B))).to_dict()
        )
        n_atoms = int(batch["positions"].shape[0])
        row, med_by_L = [], {}
        for L in range(1, M + 1):
            seeds = torch.randn(L, B, M, device=adapter.device, dtype=adapter.dtype)

            def step(seeds=seeds):
                return adapter.vjp_for_seeds(batch, seeds, batched=False)

            med, iqr, _ = timed(step, args.iters, args.warmup)
            med_by_L[L] = med
            row.append(f"{med:6.2f}")
            records.append({
                "experiment_id": "lane_scaling", "git_commit": sha, "git_dirty": dirty,
                "system": args.variant, "method": "serial_vjp", "lanes": L, "batch_size": B,
                "num_atoms": n_atoms, "num_heads": M, "precision": "fp32",
                "gpu": torch.cuda.get_device_name(0),
                "median_ms": med, "iqr_ms": iqr,
            })
        print(f"{B:>3} {n_atoms:>6} | " + "  ".join(row))
        ceil3 = med_by_L[M] / med_by_L[4]
        ceil2 = med_by_L[M] / med_by_L[3]
        verdict = ("KILL (<1.2x)" if ceil3 < 1.2 else
                   "weak (<1.5x)" if ceil3 < 1.5 else
                   "H4 min" if ceil3 < 2.0 else "H4 STRONG")
        print(f"{'':>10} | ceiling K=3: {ceil3:.2f}x   K=2: {ceil2:.2f}x   -> {verdict}")

    out = Path(args.out or f"results/raw/lane_scaling_{args.variant}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {out} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
