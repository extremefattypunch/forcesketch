#!/usr/bin/env python
"""Serial vs batched reverse mode across (batch size, lane count) -- the paper's SS5.3 table.

This is the measurement that decides which exact baseline is the honest one. Spec
SS17 item 3 asks for batched reverse mode; SS45 forbids reporting a speedup against a
weaker baseline than the one available. So the two implementations are measured on
exactly the same batches, lanes, dtype and iteration protocol, and the FASTER of the
two at each (B, L) is what ForceSketch must beat.

Batched reverse mode is usable here ONLY because
`forcesketch.adapters.mace_mhc.configure_e3nn_for_batched_vjp()` disables the
TensorExpr fuser at import time. Without it this script raises

    RuntimeError: Cannot access data pointer of Tensor that doesn't have storage

from the third call onwards -- not the first, which is what makes the cause hard to
see. The appendix of the paper gives the full diagnosis; the short version is that
e3nn's `_spherical_harmonics` is a module-level `@torch.jit.script` free function,
TorchScript emits an optimized plan after two warm-ups, and the fuser then fuses the
REVERSE graph into a kernel that demands a raw `data_ptr()` -- which vmap's
BatchedTensor cannot provide.

Per spec SS44: CUDA events, one event pair per iteration, >=100 warmup, median + IQR,
and a single synchronize at the timing boundary.
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


def timed(fn, iters: int, warmup: int) -> tuple[float, float]:
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
    return st.median(t), t[3 * len(t) // 4] - t[len(t) // 4]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", default="results/raw/02d_batched_vs_serial.jsonl")
    args = ap.parse_args()

    pin_numerics()
    adapter = MaceMHCAdapter.from_checkpoint(CKPT, dtype=torch.float32)
    M = adapter.num_heads
    frames = load_frames(DATA, limit=max(args.batch_sizes))
    sha, dirty = git_commit()
    records = []

    print(f"M={M}, {args.iters} iters / {args.warmup} warmup, CUDA events, fp32")
    print(f"{'B':>3} {'L':>2} | {'serial':>9} {'batched':>9} | winner")

    for B in args.batch_sizes:
        batch = adapter.prepare(
            next(iter(make_loader(frames[:B], adapter.model, batch_size=B))).to_dict()
        )
        n_atoms = int(batch["positions"].shape[0])
        for L in range(1, M + 1):
            seeds = torch.randn(L, B, M, device=adapter.device, dtype=adapter.dtype)
            med = {}
            for impl, batched in (("serial", False), ("batched", True)):
                try:
                    med[impl], iqr = timed(
                        lambda b=batched: adapter.vjp_for_seeds(batch, seeds, batched=b),
                        args.iters, args.warmup)
                except torch.cuda.OutOfMemoryError:
                    # Recorded rather than skipped: an OOM at some (B, L) IS the
                    # result, and silently dropping it would flatter the baseline.
                    print(f"{B:>3} {L:>2} | {impl} OOM")
                    torch.cuda.empty_cache()
                    continue
                records.append({
                    "experiment_id": "02d_batched_vs_serial", "git_commit": sha,
                    "git_dirty": dirty, "dataset": "3bpa", "batch_size": B,
                    "num_atoms": n_atoms, "lanes": L, "impl": impl,
                    "median_ms": med[impl], "iqr_ms": iqr, "num_heads": M,
                    "precision": "fp32", "iters": args.iters, "warmup": args.warmup,
                    "gpu": torch.cuda.get_device_name(0),
                })
            if len(med) == 2:
                win = min(med, key=med.get)
                print(f"{B:>3} {L:>2} | {med['serial']:9.2f} {med['batched']:9.2f} | "
                      f"{win} by {max(med.values())/min(med.values()):.2f}x")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {out} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
