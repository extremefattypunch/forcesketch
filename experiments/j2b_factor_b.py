#!/usr/bin/env python
"""J2b -- Factor B: does the screening gate preserve the decisions the exact
estimator would have made, measured against REFERENCE-FORCE ERROR?

Factor A (j2a) asked whether the exact oracle is worth anything. Factor B asks
what the gate costs relative to that oracle, and is deliberately stated as a
difference so it stays meaningful whatever Factor A turned out to be: "whatever
you chose to trust, we reproduce it at a fraction of the cost."

The policy comparison is exact:

    exact policy : compute the exact score for every structure, select s_exact >= tau
    gate policy  : skip where c_alpha * s_hat < tau; compute exact for the rest;
                   select those with s_exact >= tau

The gate's selected set is a subset of the exact policy's, so its recall of any
positive class can only be lower. Factor B is that gap, evaluated on positives
defined by REFERENCE ERROR (top-5% e_max) rather than by the proxy -- which is
the whole point, and what the workshop paper could not do.

Split discipline, enforced by manifest rather than by convention:
    design -> the control-variate basis Q_r0, and tau
    cal    -> c_alpha, and nothing else
    test   -> every number reported here

Primary acquisition statistic is `global`, following j2a: it beats max-component
as a predictor of reference error in 24/24 cells (19/24 significantly) and is
also the statistically easier one to sketch. max-component is retained because it
is the rule Beck et al. actually used.

Per-seed, never seed-averaged before scoring: deployment draws ONE sketch, so the
seed-wise minimum is reported alongside the mean.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch.sketches.control_variate import (  # noqa: E402
    control_variate_seeds, control_variate_variance, leading_head_directions,
)
from forcesketch.sketches.registry import make_sketch_seeds  # noqa: E402
from forcesketch_journal.evaluation.tail_metrics import top_p_mask  # noqa: E402

EPS = 1e-30


# ---------------------------------------------------------------- estimators
def acquisition(v: torch.Tensor, sigma: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "maxcomp":
        return sigma.amax(dim=(1, 2))
    if kind == "global":
        return v.sum(dim=(1, 2))
    raise ValueError(kind)


def est_exact(F):
    v = F.var(dim=-1, unbiased=True)
    return v, v.sqrt()


def est_haar(F, M, K, seed):
    b = make_sketch_seeds("haar", M=M, K=K, batch_size=F.shape[0], seed=seed, dtype=torch.float64)
    G = torch.einsum("sadm,ksm->ksad", F, b.seeds)
    v = b.variance_scale * (G ** 2).sum(dim=0)
    return v, v.clamp_min(0).sqrt() / b.std_correction


def est_cv(F, M, K, seed, Q, r0):
    b, _ = control_variate_seeds(Q, M=M, K=K, batch_size=F.shape[0], seed=seed)
    G = torch.einsum("sadm,ksm->ksad", F, b.seeds)
    v = control_variate_variance(G, r0=r0, M=M)
    return v, v.clamp_min(0).sqrt() / b.std_correction


def est_leading_only(F, M, Q, r0):
    """J9.1 -- the r0 exact leading directions, no randomised residual at all.

    Deterministic (no probe seed). Biased low by construction, since it discards
    the residual energy; the conformal ratio absorbs the bias, so the gate stays
    valid and the question is purely how much decision quality the residual sketch
    is actually buying. `control_variate_seeds` refuses K == r0, so the estimator
    is written out directly rather than routed through it.
    """
    G = torch.einsum("sadm,mk->ksad", F, Q.double())
    return (G ** 2).sum(dim=0) / (M - 1), None


def est_head_exact_mean(F, M, K, seed):
    S = F.shape[0]
    g = torch.Generator().manual_seed((seed * 7919 + K) % (2 ** 31))
    idx = torch.stack([torch.randperm(M, generator=g)[:K] for _ in range(S)])
    Fbar = F.mean(dim=-1, keepdim=True)
    sel = torch.gather(F, 3, idx[:, None, None, :].expand(S, F.shape[1], 3, K))
    v = (M / (K * (M - 1))) * ((sel - Fbar) ** 2).sum(dim=-1)
    return v, v.clamp_min(0).sqrt()


def conformal_c(s_exact: torch.Tensor, s_hat: torch.Tensor, alpha: float) -> float:
    """ceil((n+1)(1-alpha))-th order statistic of the ratios (split conformal).

    Returns +inf when the requested alpha is infeasible for this calibration set,
    i.e. when ceil((n+1)(1-alpha)) > n and no order statistic carries the
    finite-sample guarantee. An infinite c_alpha makes the gate skip nothing,
    which is the safe failure.

    This previously clamped `k` to `n`, silently substituting the maximum observed
    ratio and emitting an anti-conservative gate labelled with the requested alpha.
    `splits.alpha_feasible` and `splits.coverage_law` document the opposite
    contract, and `fallback_gate.calibrate` in the frozen tree already implements
    it correctly -- this function was the outlier. Found by adversarial review.

    Reachable only with alpha < 1/(n_cal+1); the smallest n_cal in this project is
    92 (water, blocked split), so only alpha=0.01 on water triggers it. Every
    shipped record uses alpha=0.05, where all n_cal (92-428) are feasible, so no
    published number was affected.
    """
    r = (s_exact / (s_hat + EPS)).sort().values
    n = r.numel()
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(r[max(1, k) - 1])


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="global", choices=["global", "maxcomp"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-p", type=float, default=0.05, help="tau = (1-p) quantile on design")
    ap.add_argument("--error-p", type=float, default=0.05, help="positives = top-p by e_max")
    ap.add_argument("--r0", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--scheme", default="contiguous_block")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-dirs", nargs="+", default=None)
    args = ap.parse_args()

    seeds = yaml.safe_load((FROZEN / "configs/seeds.yaml").read_text())["sketch_seeds"]
    out = pathlib.Path(args.out or f"results/records/j2b_factor_b_{args.score}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    dirs=[pathlib.Path(d) for d in (args.cache_dirs or [FROZEN/"results/processed"])]
    for cache in sorted({p for d in dirs for p in d.glob("head_forces_*.pt")}):
        tag = cache.stem.replace("head_forces_", "")
        man = json.loads((ROOT / f"manifests/splits/{tag}__{args.scheme}.json").read_text())
        d = torch.load(cache, weights_only=True, map_location="cpu")
        F, f_ref = d["F"].double(), d["f_ref"].double()
        M, S = int(d["M"]), F.shape[0]

        design = torch.tensor(man["roles"]["design"])
        cal = torch.tensor(man["roles"]["cal"])
        test = torch.tensor(man["roles"]["test"])

        v_ex, sig_ex = est_exact(F)
        s_exact = acquisition(v_ex, sig_ex, args.score)
        tau = float(s_exact[design].quantile(1.0 - args.target_p))   # DESIGN only

        e_max = (F.mean(dim=-1) - f_ref).norm(dim=-1).max(dim=1).values
        pos_test = top_p_mask(e_max[test], args.error_p)             # physical positives

        # The exact policy is the ceiling every gate is measured against.
        #
        # tau transfers from design, so the realised test selection fraction is
        # NOT p -- it ranges ~0.02-0.07 here. Recall is therefore prevalence-
        # limited and must be read alongside `sel_frac`; `enrichment`
        # (= recall / sel_frac) is the prevalence-free version and is the number
        # to compare across systems.
        sel_exact = s_exact[test] >= tau
        sel_frac_exact = float(sel_exact.double().mean())
        exact_err_recall = float((sel_exact & pos_test).sum() / pos_test.sum().clamp(min=1))
        exact_enrichment = exact_err_recall / max(sel_frac_exact, 1e-12)

        Q = leading_head_directions(F[design], args.r0)              # DESIGN only

        gates = {
            "control_variate": lambda sd: est_cv(F, M, args.K, sd, Q, args.r0),
            "haar": lambda sd: est_haar(F, M, args.K, sd),
            "head_exact_mean": lambda sd: est_head_exact_mean(F, M, args.K, sd),
            "leading_only": lambda sd: est_leading_only(F, M, Q, args.r0),
        }
        lanes = {"control_variate": args.K + 1, "haar": args.K + 1,
                 "head_exact_mean": args.K + 1, "leading_only": args.r0 + 1}

        for gname, fn in gates.items():
            probe_seeds = [None] if gname == "leading_only" else seeds
            for sd in probe_seeds:
                v_h, sig_h = fn(sd)
                s_hat = acquisition(v_h, sig_h if sig_h is not None else v_h.clamp_min(0).sqrt(),
                                    args.score)
                c_alpha = conformal_c(s_exact[cal], s_hat[cal], args.alpha)   # CAL only

                skip = c_alpha * s_hat[test] < tau
                sel_gate = sel_exact & ~skip

                high_uq = s_exact[test] >= tau
                cleared_err = e_max[test][skip]
                records.append({
                    "experiment_id": "j2b_factor_b", "experiment_version": 1,
                    "system": tag, "scheme": args.scheme, "split_role": "test",
                    "split_sha256": man["content_sha256"][:12],
                    "gate": gname, "score": args.score, "K": args.K, "r0": args.r0,
                    "alpha": args.alpha, "probe_seed": sd, "uq_lanes": lanes[gname],
                    "n_test": int(test.numel()), "tau": tau, "c_alpha": c_alpha,
                    "n_pos_error": int(pos_test.sum()), "n_high_uq": int(high_uq.sum()),
                    # --- proxy fidelity (what the workshop measured) ---
                    "high_uq_recall": float((high_uq & ~skip).sum() / high_uq.sum().clamp(min=1)),
                    "frac_exact_skipped": float(skip.double().mean()),
                    # --- physical decision quality (Factor B) ---
                    "exact_error_recall": exact_err_recall,
                    "exact_sel_frac": sel_frac_exact,
                    "exact_enrichment": exact_enrichment,
                    "gate_error_recall": float((sel_gate & pos_test).sum() / pos_test.sum().clamp(min=1)),
                    "gate_sel_frac": float(sel_gate.double().mean()),
                    "factor_b_gap": exact_err_recall - float((sel_gate & pos_test).sum()
                                                             / pos_test.sum().clamp(min=1)),
                    # --- safety: what error survives in the skipped pile ---
                    "max_error_cleared_mev_A": float(cleared_err.max()) * 1000 if skip.any() else 0.0,
                    "mean_error_cleared_mev_A": float(cleared_err.mean()) * 1000 if skip.any() else 0.0,
                    "max_error_overall_mev_A": float(e_max[test].max()) * 1000,
                })

    with out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"wrote {len(records)} records to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
