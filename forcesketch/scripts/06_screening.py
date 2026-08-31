#!/usr/bin/env python
"""Experiment 6: uncertainty screening gate (plan Task 5.1; spec SS33, SS34; H5).

Converts ForceSketch from an approximation into a computational policy: sketch
every structure cheaply, and fall back to exact force UQ only where the calibrated
optimistic bound cannot rule out the high-uncertainty threshold.

This is spec SS51's escalation rung 3, and it is the right response to the SS27
result on this system: replacing exact UQ outright (H2) fails at K <= 4, but
screening (H5) succeeds comfortably, which is exactly the framing SS66 prescribes.

Splits are 20/20/60 per SS33. c_alpha and tau are fitted on the CALIBRATION split
only; the test split is touched once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from forcesketch.screening.fallback_gate import calibrate, evaluate_gate, split_indices
from forcesketch.sketches.control_variate import (
    control_variate_seeds,
    control_variate_variance,
    leading_head_directions,
)
from forcesketch.sketches.registry import make_sketch_seeds
from forcesketch.utils.reproducibility import git_commit


def sketch_global_score(F, M, method, K, seed, *, r0=0, Q=None):
    """Exact evaluation of the estimator from the cached head-force matrix."""
    S = F.shape[0]
    if r0:
        bundle, _ = control_variate_seeds(Q, M=M, K=K, batch_size=S, seed=seed)
        G = torch.einsum("sadm,ksm->ksad", F, bundle.seeds)
        v = control_variate_variance(G, r0=r0, M=M)
    else:
        bundle = make_sketch_seeds(method, M=M, K=K, batch_size=S, seed=seed,
                                   dtype=torch.float64)
        G = torch.einsum("sadm,ksm->ksad", F, bundle.seeds)
        v = bundle.variance_scale * (G**2).sum(dim=0)
    return v.sum(dim=(1, 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="disjoint")
    ap.add_argument("--split", default="test_1200K")
    ap.add_argument("--target-p", type=float, default=0.05)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("configs/seeds.yaml").read_text())
    SEEDS, SPLIT_SEED = cfg["sketch_seeds"], cfg["calibration_split_seed"]

    c = torch.load(f"results/processed/head_forces_{args.variant}_{args.split}.pt",
                   weights_only=True)
    F, M = c["F"], c["M"]
    S = F.shape[0]
    S_ex = F.var(dim=-1, unbiased=True).sum(dim=(1, 2))

    cal_i, val_i, test_i = split_indices(S, seed=SPLIT_SEED)
    tau = float(torch.quantile(S_ex[cal_i], 1.0 - args.target_p))
    print(f"{args.variant}/{args.split}: n={S} "
          f"(calib {len(cal_i)} / val {len(val_i)} / test {len(test_i)})")
    print(f"tau = {tau:.4f}  (exact top-{args.target_p:.0%}, fitted on CALIBRATION only)\n")

    Q = {r0: leading_head_directions(F[cal_i], r0) for r0 in (1, 2, 3)}
    configs = [("haar", 2, 0), ("haar", 3, 0), ("haar", 4, 0),
               ("control_variate", 3, 1), ("control_variate", 4, 1),
               ("control_variate", 4, 2), ("control_variate", 5, 2)]

    sha, dirty = git_commit()
    records = []
    print(f"{'method':<24}{'K':>2}{'alpha':>7}{'recall':>9}{'skipped':>9}"
          f"{'FNR':>8}{'speedup':>9}   H5")
    print("-" * 76)
    for method, K, r0 in configs:
        for alpha in (0.10, 0.05, 0.01):
            rows = []
            for seed in SEEDS:
                s_hat = sketch_global_score(F, M, method, K, seed,
                                            r0=r0, Q=Q.get(r0))
                gc = calibrate(S_ex[cal_i], s_hat[cal_i], alpha=alpha, tau=tau)
                # K independent centered directions are already in hand, so the
                # fallback completes the basis with M-1-K more, not a fresh M.
                m = evaluate_gate(gc, S_ex[test_i], s_hat[test_i],
                                  cost_sketch_lanes=K + 1, cost_exact_lanes=M,
                                  dirs_done=K)
                m.update({
                    "experiment_id": "06_screening", "git_commit": sha, "git_dirty": dirty,
                    "dataset": "3bpa", "split": args.split, "variant": args.variant,
                    "method": method, "K": K, "r0": r0, "sketch_seed": seed,
                    "num_heads": M, "precision": "float64",
                })
                records.append(m)
                rows.append([m["high_uq_recall"], m["frac_exact_skipped"],
                             m["false_negative_rate"], m["screening_speedup"]])
            a = np.mean(rows, axis=0)
            label = f"{method}{f' r0={r0}' if r0 else ''}"
            h5 = "PASS" if (a[0] >= 0.95 and a[1] >= 0.50) else ""
            print(f"{label:<24}{K:>2}{alpha:>7.2f}{a[0]:>9.3f}{a[1]:>9.3f}"
                  f"{a[2]:>8.3f}{a[3]:>8.2f}x   {h5}")

    out = Path(f"results/raw/06_screening_{args.variant}_{args.split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"\nwrote {out} ({len(records)} records)")
    print("\nH5 target: skip >=50% of exact evaluations while retaining >=95% of "
          "exact high-uncertainty configurations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
