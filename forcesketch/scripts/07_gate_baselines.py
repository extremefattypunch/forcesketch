#!/usr/bin/env python
"""Screening-gate baselines on one Pareto (revision Stage C).

A calibrated gate is a generic wrapper: any score correlated with exact force
disagreement can be calibrated conservatively. So the paper's screening claim only
stands if ForceSketch's gate beats the cheaper gates a reviewer will reach for.
This script runs all of them under one protocol:

  energy        std of the M head energies. FREE -- all M come from the single
                forward pass the model already runs, so ZERO reverse lanes.
  head-exact    head subsampling using the EXACT mean force, which ForceSketch
                already pays a lane for:
                    v_hat_d = M/(K(M-1)) * sum_{i in S} (F_di - Fbar_d)^2
                unbiased under uniform sampling, and markedly lower variance than
                the ddof=1 sample variance among the K drawn heads.
  haar          orthogonal head-space sketch.
  cv            low-rank control variate (r0 exact leading directions + residual).

Two protocol fixes over the previous version:

  * THREE disjoint splits. Previously Q_{r0} and c_alpha were both fitted on the
    calibration split (and the validation split was computed but never used), so
    the control-variate subspace was learned on the very structures used to claim
    calibration. Now: design (fit Q_{r0}) / calibration (fit c_alpha) / test.
  * SPLIT-CONFORMAL c_alpha: the ceil((n+1)(1-alpha))-th order statistic of the
    calibration ratios, which gives marginal coverage >= 1-alpha under
    exchangeability, rather than an in-sample empirical quantile.

Acquisition scalar is selectable. Beck et al. rank structures by the maximum
disagreement of their force components, so `maxcomp` is primary; `maxatom` is the
other reading of that phrase, and `global` is the statistically easier sum.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from analysis.bootstrap import Interval
from forcesketch.sketches.control_variate import (
    control_variate_seeds,
    control_variate_variance,
    leading_head_directions,
)
from forcesketch.sketches.registry import make_sketch_seeds
from forcesketch.utils.reproducibility import git_commit

SYSTEMS = {
    "3bpa": "disjoint_test_1200K",
    "ethanol": "rmd17-disjoint_ethanol",
    "aspirin": "rmd17-disjoint_aspirin",
    "azobenzene": "rmd17-disjoint_azobenzene",
}


# --------------------------------------------------------------------------
# acquisition scalars
# --------------------------------------------------------------------------
def acquisition(v: torch.Tensor, sigma: torch.Tensor, kind: str) -> torch.Tensor:
    """v, sigma are [S, A, 3] -> [S]."""
    if kind == "maxcomp":                      # Beck et al.: max over force components
        return sigma.amax(dim=(1, 2))
    if kind == "maxatom":                      # max over per-atom MHC scores
        return sigma.mean(dim=-1).amax(dim=-1)
    if kind == "global":                       # sum of coordinate variances
        return v.sum(dim=(1, 2))
    raise ValueError(kind)


# --------------------------------------------------------------------------
# estimators -> (v_hat, sigma_hat), all offline from the cached F
# --------------------------------------------------------------------------
def est_exact(F):
    v = F.var(dim=-1, unbiased=True)
    return v, v.sqrt()


def est_haar(F, M, K, seed):
    b = make_sketch_seeds("haar", M=M, K=K, batch_size=F.shape[0], seed=seed,
                          dtype=torch.float64)
    G = torch.einsum("sadm,ksm->ksad", F, b.seeds)
    v = b.variance_scale * (G**2).sum(dim=0)
    return v, v.clamp_min(0).sqrt() / b.std_correction


def est_cv(F, M, K, seed, Q, r0):
    b, _ = control_variate_seeds(Q, M=M, K=K, batch_size=F.shape[0], seed=seed)
    G = torch.einsum("sadm,ksm->ksad", F, b.seeds)
    v = control_variate_variance(G, r0=r0, M=M)
    return v, v.clamp_min(0).sqrt() / b.std_correction


def est_head_exact_mean(F, M, K, seed):
    """Head subsampling that USES the exact mean force (revision Stage C3)."""
    S = F.shape[0]
    g = torch.Generator().manual_seed((seed * 7919 + K) % (2**31))
    idx = torch.stack([torch.randperm(M, generator=g)[:K] for _ in range(S)])  # [S,K]
    Fbar = F.mean(dim=-1, keepdim=True)                                       # exact mean
    sel = torch.gather(F, 3, idx[:, None, None, :].expand(S, F.shape[1], 3, K))
    v = (M / (K * (M - 1))) * ((sel - Fbar) ** 2).sum(dim=-1)
    return v, v.clamp_min(0).sqrt()


def est_energy(E):
    """FREE gate: std of head energies. Returns a per-structure score only."""
    return E.std(dim=-1, unbiased=True)


# --------------------------------------------------------------------------
# split-conformal calibration
# --------------------------------------------------------------------------
def conformal_c(s_exact: torch.Tensor, s_hat: torch.Tensor, alpha: float,
                eps: float = 1e-30) -> float:
    """ceil((n+1)(1-alpha))-th order statistic of the ratios (split conformal)."""
    r = (s_exact / (s_hat + eps)).sort().values
    n = r.numel()
    k = min(n, max(1, math.ceil((n + 1) * (1 - alpha))))
    return float(r[k - 1])


def gate_metrics(c_alpha, s_exact, s_hat, tau, *, lanes_gate, dirs_done,
                 lanes_exact, lane_ms, M: int = 8):
    """Metrics on the test split, with fallback cost accounted UNIFORMLY.

    Every gate pays the mean-force lane regardless, because MD needs the force
    whether or not uncertainty is computed. On fallback, a gate that has already
    evaluated `dirs_done` linearly independent directions inside the centered
    subspace needs only r - dirs_done further reverse passes to complete the exact
    basis: the existing g_j are reused, and an orthonormal frame spanning them is
    a linear recombination of vectors already in hand, not new reverse passes.

    An earlier version credited this reuse to the control variate ALONE and
    charged every baseline a fresh M-lane recompute. That is precisely the kind of
    asymmetry this paper objects to elsewhere, and it inflated the control
    variate's apparent margin -- under fair accounting the baselines are no longer
    net slowdowns, and the control variate's advantage narrows to what its
    decision quality actually earns.
    """
    skip = c_alpha * s_hat < tau
    high = s_exact >= tau
    n, nh = s_exact.numel(), int(high.sum())
    tp = int((high & ~skip).sum())
    n_run = int((~skip).sum())
    lanes_complete = (M - 1) - dirs_done
    ms_gate = lane_ms(lanes_gate) * n + lane_ms(lanes_complete) * n_run
    ms_exact = lane_ms(lanes_exact) * n
    return {
        "high_uq_recall": tp / max(nh, 1),
        "false_negative_rate": int((high & skip).sum()) / max(nh, 1),
        "frac_exact_skipped": float(skip.float().mean()),
        "screening_speedup": ms_exact / max(ms_gate, 1e-9),
        "n": n, "n_high_uq": nh, "c_alpha": c_alpha,
        "test_prevalence": nh / max(n, 1), "lanes_complete": lanes_complete,
    }


def boot_ci(fn, n, n_boot, seed, alpha=0.05):
    g = torch.Generator().manual_seed(seed)
    vals = np.array([fn(torch.randint(0, n, (n,), generator=g)) for _ in range(n_boot)])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="maxcomp", choices=["maxcomp", "maxatom", "global"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-p", type=float, default=0.05)
    ap.add_argument("--r0", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("configs/seeds.yaml").read_text())
    SEEDS, SPLIT_SEED = cfg["sketch_seeds"], cfg["calibration_split_seed"]
    # measured lane cost, eager batched, B=16 (see results/raw/02d_batched_vs_serial.jsonl)
    bv = {(r["batch_size"], r["lanes"], r["impl"]): r["median_ms"]
          for r in (json.loads(l) for l in open("results/raw/02d_batched_vs_serial.jsonl"))}
    def lane_ms(L, B=16):
        return min(bv[(B, max(L, 1), "serial")], bv[(B, max(L, 1), "batched")])

    sha, dirty = git_commit()
    records = []
    print(f"acquisition scalar: {args.score}   alpha={args.alpha}   r0={args.r0}\n")
    hdr = f"{'system':<12}{'gate':<22}{'lanes':>6}{'skip':>17}{'high-UQ recall':>22}{'speedup':>9}"
    print(hdr); print("-" * len(hdr))

    for name, tag in SYSTEMS.items():
        c = torch.load(f"results/processed/head_forces_{tag}.pt", weights_only=True)
        F, E, M = c["F"], c["E"], c["M"]
        S = F.shape[0]
        v_ex, sig_ex = est_exact(F)
        u_ex = acquisition(v_ex, sig_ex, args.score)

        # THREE disjoint splits: design (fit Q) / calibration (fit c_alpha) / test
        g = torch.Generator().manual_seed(SPLIT_SEED)
        perm = torch.randperm(S, generator=g)
        n1, n2 = int(0.2 * S), int(0.4 * S)
        design, calib, test = perm[:n1], perm[n1:n2], perm[n2:]
        # All three pairwise, not just the adjacent two: the point of the split is
        # that c_alpha is never fitted on structures used to fit Q_{r0} OR to test.
        ds, cs, ts = map(lambda t: set(t.tolist()), (design, calib, test))
        assert not (ds & cs) and not (cs & ts) and not (ds & ts), "splits overlap"
        assert len(ds) + len(cs) + len(ts) == len(perm), "splits do not partition"
        tau = float(torch.quantile(u_ex[design], 1.0 - args.target_p))
        Q = leading_head_directions(F[design], args.r0)          # design split ONLY

        # (uncertainty lanes K, independent centered directions obtained, estimator).
        # The energy gate spends NO reverse pass on uncertainty -- head energies come
        # free from the forward -- but still pays the mean-force lane, as all do.
        gates = {
            "energy (free)": (0, 0, lambda sd: est_energy(E)),
            "head-exact-mean K=4": (4, 4, lambda sd: est_head_exact_mean(F, M, 4, sd)),
            "haar K=4": (4, 4, lambda sd: est_haar(F, M, 4, sd)),
            "control-variate K=4": (4, 4, lambda sd: est_cv(F, M, 4, sd, Q, args.r0)),
        }
        for gname, (K, dirs_done, fn) in gates.items():
            seeds = SEEDS if K else SEEDS[:1]      # energy gate is deterministic
            # Precompute the score for every seed, then bootstrap the SEED-AVERAGED
            # statistic over structures -- the same convention as analysis/bootstrap.py.
            # Bootstrapping a single seed while reporting a 10-seed mean would give an
            # interval that need not even contain its own point estimate.
            s_hats, c_alphas = [], []
            for sd in seeds:
                out = fn(sd)
                sh = out if torch.is_tensor(out) else acquisition(out[0], out[1], args.score)
                s_hats.append(sh)
                c_alphas.append(conformal_c(u_ex[calib], sh[calib], args.alpha))

            def seed_avg(idx_local, field):
                vals = [gate_metrics(ca, u_ex[test][idx_local], sh[test][idx_local], tau,
                                     lanes_gate=K + 1, dirs_done=dirs_done,
                                     lanes_exact=M, lane_ms=lane_ms, M=M)[field]
                        for sh, ca in zip(s_hats, c_alphas)]
                return float(np.mean(vals))

            full = torch.arange(len(test))
            mean = {f: seed_avg(full, f) for f in
                    ("high_uq_recall", "false_negative_rate", "frac_exact_skipped",
                     "screening_speedup", "test_prevalence", "lanes_complete")}
            rec_lo, rec_hi = boot_ci(lambda i: seed_avg(i, "high_uq_recall"),
                                     len(test), args.n_boot, cfg["bootstrap_seed"])
            skip_lo, skip_hi = boot_ci(lambda i: seed_avg(i, "frac_exact_skipped"),
                                       len(test), args.n_boot, cfg["bootstrap_seed"])
            print(f"{name:<12}{gname:<22}{K + 1:>6}"
                  f"  {mean['frac_exact_skipped']:.3f} [{skip_lo:.3f},{skip_hi:.3f}]"
                  f"  {mean['high_uq_recall']:.3f} [{rec_lo:.3f},{rec_hi:.3f}]"
                  f"{mean['screening_speedup']:>9.2f}x")
            records.append({"experiment_id": "07_gate_baselines", "git_commit": sha,
                            "git_dirty": dirty, "system": name, "split": tag,
                            "gate": gname, "score": args.score, "K": K,
                            "total_lanes": K + 1, "uq_lanes": K, "alpha": args.alpha,
                            "n_design": len(design), "n_cal": len(calib),
                            "n_test": len(test), "batch_size_for_cost": 16,
                            "r0": args.r0 if "control" in gname else 0,
                            "n_seeds": len(seeds), "tau": tau,
                            "recall_ci_lo": rec_lo, "recall_ci_hi": rec_hi,
                            "skip_ci_lo": skip_lo, "skip_ci_hi": skip_hi,
                            "n_boot": args.n_boot, **mean})
        print()

    out = Path(args.out or f"results/raw/07_gate_baselines_{args.score}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {out} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
