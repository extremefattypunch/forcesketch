#!/usr/bin/env python
"""Experiment 3: ForceSketch fidelity sweep (plan Tasks 4.0 and 4.1; spec SS27, SS28, SS38).

Evaluates Gaussian, Haar, centered Rademacher, pairwise and head subsampling at
K in {1,2,3,4,M-1} over the ten frozen seeds, against the exact committee
uncertainty, and reports every spec SS28 metric.

This runs entirely from the cached head-force matrix F: every estimator is a
linear functional of F (g_k = F w_k), so the sweep is exact without re-running the
model. Only the timing experiments need live execution.

Head subsampling is evaluated in BOTH budget framings (spec SS49): at equal
uncertainty lanes K, and at equal TOTAL lanes K+1 (where ForceSketch spends one
lane on the exact mean force and head subsampling can spend it on another head).
Reporting only one of the two would settle the paper's central comparison by
definition rather than by measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from analysis.metrics import (
    acquisition_metrics,
    atom_localization,
    head_space_spectrum,
    kendall,
    numeric_fidelity,
    spearman,
)
from forcesketch.estimators.std_correction import std_correction_for
from forcesketch.sketches.registry import make_sketch_seeds
from forcesketch.utils.reproducibility import git_commit

METHODS = ["gaussian", "haar", "rademacher", "pairwise", "head_subsample"]


def apply_sketch(F: torch.Tensor, bundle) -> tuple[torch.Tensor, torch.Tensor]:
    """F [S, A, 3, M], seeds [K, S, M] -> (v_hat [S, A, 3], sigma_hat [S, A, 3]).

    g_k = F w_k exactly, so no model execution is needed.
    """
    G = torch.einsum("sadm,ksm->ksad", F, bundle.seeds.double())  # [K, S, A, 3]
    if bundle.estimator_kind == "quadratic":
        v = bundle.variance_scale * (G**2).sum(dim=0)
    else:
        v = G.var(dim=0, unbiased=True)
    sigma = v.clamp_min(0).sqrt() / bundle.std_correction
    return v, sigma


def scores_from(v: torch.Tensor, sigma: torch.Tensor) -> dict:
    return {
        "S_global": v.sum(dim=(1, 2)),      # [S]
        "u_atom_mhc": sigma.mean(dim=-1),   # [S, A]
        "u_atom_rms": v.mean(dim=-1).sqrt(),
    }


def ranking(scores: dict, sigma: torch.Tensor, kind: str) -> torch.Tensor:
    """The per-structure scalar the acquisition rule ranks by (spec SS49, paper SS4).

    `maxcomp` and `maxatom` are the two readings of Beck et al.'s "maximum
    disagreement of the force components"; `global` is the sum over all 3N
    coordinates, which concentrates and is therefore the statistically easier
    target. All three are computed from the same v/sigma, so switching costs
    nothing -- which is why we report rather than choose.
    """
    if kind == "maxcomp":
        return sigma.flatten(start_dim=1).max(dim=1).values       # max_d sigma_d
    if kind == "maxatom":
        return scores["u_atom_mhc"].max(dim=1).values             # max_a u^MHC_a
    if kind == "global":
        return scores["S_global"]
    raise ValueError(f"unknown acquisition scalar {kind!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="disjoint")
    ap.add_argument("--split", default="test_1200K")
    ap.add_argument("--score", default="global", choices=["maxcomp", "maxatom", "global"],
                    help="acquisition scalar structures are ranked by (paper SS4)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds_cfg = yaml.safe_load(Path("configs/seeds.yaml").read_text())
    SEEDS = seeds_cfg["sketch_seeds"]

    cache = torch.load(f"results/processed/head_forces_{args.variant}_{args.split}.pt",
                       weights_only=True)
    F, M, A = cache["F"], cache["M"], cache["A"]
    S = F.shape[0]
    r = M - 1
    print(f"{args.variant}/{args.split}: {S} structures x {A} atoms, M={M}\n")

    # --- Task 4.0: spec SS38 spectrum, run BEFORE the sweep so it can predict it
    spec = head_space_spectrum(F)
    print("head-space spectrum (spec SS38)")
    print("  eigenvalue fractions: " + "  ".join(f"{f:.3f}" for f in spec["fraction"]))
    print(f"  cumulative:           " + "  ".join(f"{c:.3f}" for c in spec["cumulative"]))
    print(f"  srank(FQ)=tr(A)/lmax {spec['stable_rank_FQ']:.2f} / {r}   "
          f"effective rank {spec['effective_rank']:.2f}   "
          f"top1 {spec['top1_fraction']:.3f} (isotropic {spec['isotropic_top1_fraction']:.3f})   "
          f"dirs for 90% {spec['n_for_90pct']}")
    print("  -> a FLAT spectrum favours random sketching; a concentrated one hurts it\n")

    # exact reference
    v_ex = F.var(dim=-1, unbiased=True)
    exact = scores_from(v_ex, v_ex.sqrt())
    exact_rank = ranking(exact, v_ex.sqrt(), args.score)
    print(f"acquisition scalar: {args.score}\n")

    sha, dirty = git_commit()
    records = []
    print(f"{'method':<22}{'K':>2} {'rho_S':>7} {'tau':>7} {'rec@5%':>8} {'jac@5%':>7} "
          f"{'p90err':>8} {'bias':>8} {'gap':>4}")
    print("-" * 76)

    configs = [(m, K, {}) for m in METHODS for K in (1, 2, 3, 4, 5, 6, r)]
    configs += [("head_subsample", K, {"with_mean_lane": True}) for K in (2, 3, 4, r + 1)]

    for method, K, kw in configs:
        if method == "haar" and K > r:
            continue
        if method == "head_subsample" and (K < 2 or K > M):
            continue
        agg: dict[str, list[float]] = {}
        for seed in SEEDS:
            bundle = make_sketch_seeds(method, M=M, K=K, batch_size=S, seed=seed,
                                       dtype=torch.float64, **kw)
            v, sig = apply_sketch(F, bundle)
            approx = scores_from(v, sig)
            approx_rank = ranking(approx, sig, args.score)
            m = {}
            m["spearman"] = spearman(exact_rank, approx_rank)
            m["kendall"] = kendall(exact_rank, approx_rank)
            m.update({f"{k}_S": val for k, val in
                      numeric_fidelity(exact_rank, approx_rank).items()})
            # spec SS47 / paper SS4: an extreme-value scalar is biased upward by the
            # max of a noisy sigma, which the marginal finite-K correction does NOT fix.
            m["rank_bias_ratio"] = float(approx_rank.mean() / exact_rank.mean())
            for p, tag in ((0.01, "top1"), (0.05, "top5"), (0.10, "top10")):
                for k, val in acquisition_metrics(exact_rank, approx_rank, p).items():
                    m[f"{tag}_{k}"] = val
            m.update(atom_localization(exact["u_atom_mhc"], approx["u_atom_mhc"]))
            for k, val in m.items():
                agg.setdefault(k, []).append(val)
            records.append({
                "experiment_id": "03_sketch_fidelity", "git_commit": sha, "git_dirty": dirty,
                "dataset": "3bpa", "split": args.split, "variant": args.variant,
                "method": method, "K": K, "sketch_seed": seed, "score": args.score,
                "with_mean_lane": bool(kw.get("with_mean_lane", False)),
                "uq_lanes": bundle.lane_budget.uq_lanes,
                "total_lanes": bundle.lane_budget.total_lanes,
                "exact_mean_force": bundle.lane_budget.exact_mean_force,
                "num_heads": M, "n_structures": S, "precision": "float64", **m,
            })
        mean = {k: sum(v) / len(v) for k, v in agg.items()}
        label = f"{method}{'+mean' if kw.get('with_mean_lane') else ''}"
        print(f"{label:<22}{K:>2} {mean['spearman']:7.3f} {mean['kendall']:7.3f} "
              f"{mean['top5_recall']:8.3f} {mean['top5_jaccard']:7.3f} "
              f"{mean['p90_rel_err_S']:8.3f} {mean['rank_bias_ratio']:8.3f} "
              f"{mean['median_rank_gap_when_wrong']:4.0f}")

    suffix = "" if args.score == "global" else f"_{args.score}"
    out = Path(args.out
               or f"results/raw/03_sketch_fidelity_{args.variant}_{args.split}{suffix}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    Path(f"results/processed/spectrum_{args.variant}_{args.split}.json").write_text(
        json.dumps(spec, indent=2))
    print(f"\nwrote {out} ({len(records)} records)")

    # --- spec SS51 statistical kill gate ----------------------------------
    best = {}
    for rec in records:
        if rec["K"] <= 4:
            key = (rec["method"], rec["K"])
            best.setdefault(key, []).append((rec["top5_recall"], rec["spearman"]))
    any_pass = False
    for (method, K), vals in sorted(best.items()):
        recall = sum(v[0] for v in vals) / len(vals)
        rho = sum(v[1] for v in vals) / len(vals)
        if recall >= 0.85 or rho >= 0.85:
            any_pass = True
    print("=" * 76)
    print(f"SS51 statistical kill gate: {'PASS' if any_pass else 'TRIGGERED'} "
          f"(triggered only if EVERY K<=4 method has recall<0.85 AND rho<0.85)")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
