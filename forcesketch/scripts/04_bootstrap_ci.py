#!/usr/bin/env python
"""Paired bootstrap confidence intervals (plan Task 4.2; spec SS47).

Every headline fidelity number gets a 95% CI from resampling structures, plus the
seed-to-seed spread reported separately. Also runs the paired difference test that
actually decides spec SS49 -- ForceSketch vs head subsampling at equal budget --
because comparing two independently-computed intervals is not a test.

Scores are recomputed from the cached head-force matrix, so this costs no model
execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from analysis.bootstrap import paired_bootstrap, paired_bootstrap_difference
from forcesketch.sketches.control_variate import (
    control_variate_seeds,
    control_variate_variance,
    leading_head_directions,
)
from forcesketch.sketches.registry import make_sketch_seeds
from forcesketch.screening.fallback_gate import split_indices
from forcesketch.utils.reproducibility import git_commit

CONFIGS = [
    ("haar", 1, 0, {}), ("haar", 2, 0, {}), ("haar", 3, 0, {}), ("haar", 4, 0, {}),
    ("gaussian", 3, 0, {}), ("rademacher", 3, 0, {}), ("pairwise", 3, 0, {}),
    # The "+mean lane" arm must USE the mean force it is charged for, otherwise
    # the comparison is against a deliberately hobbled baseline. See
    # sketches/generators.py::head_subsample_exact_mean_seeds.
    ("head_subsample_exact_mean", 3, 0, {}),
    ("head_subsample_exact_mean", 4, 0, {}),
    ("head_subsample", 4, 0, {}),
    ("control_variate", 3, 1, {}), ("control_variate", 4, 2, {}),
]


def global_scores(F, M, method, K, seeds, *, r0=0, Q=None, **kw) -> torch.Tensor:
    """-> [n_seeds, S] approximate global disagreement."""
    S = F.shape[0]
    rows = []
    for seed in seeds:
        if r0:
            b, _ = control_variate_seeds(Q, M=M, K=K, batch_size=S, seed=seed)
            G = torch.einsum("sadm,ksm->ksad", F, b.seeds)
            v = control_variate_variance(G, r0=r0, M=M)
        else:
            b = make_sketch_seeds(method, M=M, K=K, batch_size=S, seed=seed,
                                  dtype=torch.float64, **kw)
            G = torch.einsum("sadm,ksm->ksad", F, b.seeds)
            v = (b.variance_scale * (G**2).sum(dim=0) if b.estimator_kind == "quadratic"
                 else G.var(dim=0, unbiased=True))
        rows.append(v.sum(dim=(1, 2)))
    return torch.stack(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="disjoint")
    ap.add_argument("--split", default="test_1200K")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="spec SS47 default 1000; use 10000 for the final freeze")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("configs/seeds.yaml").read_text())
    SEEDS, BOOT_SEED = cfg["sketch_seeds"], cfg["bootstrap_seed"]

    c = torch.load(f"results/processed/head_forces_{args.variant}_{args.split}.pt",
                   weights_only=True)
    F, M = c["F"], c["M"]
    S_ex = F.var(dim=-1, unbiased=True).sum(dim=(1, 2))
    cal_i, _, _ = split_indices(F.shape[0], seed=cfg["calibration_split_seed"])
    Q = {r0: leading_head_directions(F[cal_i], r0) for r0 in (1, 2, 3)}

    print(f"{args.variant}/{args.split}: {F.shape[0]} structures, "
          f"{len(SEEDS)} seeds, {args.n_boot} bootstrap resamples\n")
    print(f"{'method':<24}{'K':>2}  {'Spearman':>22}  {'top-5% recall':>22}  {'seed sd':>8}")
    print("-" * 84)

    sha, dirty = git_commit()
    records, cache = [], {}
    for method, K, r0, kw in CONFIGS:
        s_hat = global_scores(F, M, method, K, SEEDS, r0=r0, Q=Q.get(r0), **kw)
        cache[(method, K, r0, bool(kw.get("with_mean_lane")))] = s_hat
        ci = paired_bootstrap(S_ex, s_hat, n_boot=args.n_boot, seed=BOOT_SEED)
        label = (f"{method}{f' r0={r0}' if r0 else ''}"
                 f"{'+mean' if kw.get('with_mean_lane') else ''}")
        print(f"{label:<24}{K:>2}  {str(ci['spearman']):>22}  "
              f"{str(ci['top5_recall']):>22}  {ci['top5_recall'].seed_std:8.4f}")
        rec = {"experiment_id": "04_bootstrap_ci", "git_commit": sha, "git_dirty": dirty,
               "variant": args.variant, "split": args.split, "method": method, "K": K,
               "r0": r0, "with_mean_lane": bool(kw.get("with_mean_lane")),
               "n_seeds": len(SEEDS), "n_structures": int(F.shape[0])}
        for m, iv in ci.items():
            rec.update(iv.as_dict(m))
        records.append(rec)

    # --- spec SS49: the paired difference that actually decides the comparison ---
    print("\nSS49 paired differences (same structure resamples; CI excluding 0 = real)")
    print("-" * 84)
    comparisons = [
        ("haar K=3 vs head_subsample K=3 +exact mean (equal 4 total lanes)",
         ("haar", 3, 0, False), ("head_subsample_exact_mean", 3, 0, False)),
        ("haar K=3 vs head_subsample K=4        (equal 4 total lanes, NO exact mean)",
         ("haar", 3, 0, False), ("head_subsample", 4, 0, False)),
        ("haar K=3 vs gaussian K=3              (orthogonalization gain)",
         ("haar", 3, 0, False), ("gaussian", 3, 0, False)),
        ("control_variate r0=2 K=4 vs haar K=4  (control-variate gain)",
         ("control_variate", 4, 2, False), ("haar", 4, 0, False)),
    ]
    for label, ka, kb in comparisons:
        d = paired_bootstrap_difference(S_ex, cache[ka], cache[kb],
                                        n_boot=args.n_boot, seed=BOOT_SEED)
        verdict = "significant" if (d.ci_lo > 0 or d.ci_hi < 0) else "NOT significant"
        print(f"  {label}\n      delta top-5% recall = {d}   -> {verdict}")
        records.append({"experiment_id": "04_bootstrap_diff", "git_commit": sha,
                        "variant": args.variant, "split": args.split, "comparison": label,
                        "delta_top5_recall": d.point, "ci_lo": d.ci_lo, "ci_hi": d.ci_hi,
                        "significant": bool(d.ci_lo > 0 or d.ci_hi < 0),
                        "n_boot": args.n_boot})

    out = Path(f"results/raw/04_bootstrap_{args.variant}_{args.split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {out} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
