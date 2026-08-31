"""Fidelity metrics (spec SS28) and the SS38 head-space spectrum.

Tie handling in top-p% sets is the easiest thing here to get silently wrong: with
near-tied scores, which items land inside the boundary changes recall without
anything looking broken. `top_p_set` therefore takes an explicit policy and
`strict_k` is NOT the default -- "inclusive" keeps every item tied with the
threshold, which is what an acquisition run would actually select.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from scipy import stats

TiePolicy = Literal["inclusive", "strict_k"]


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------
def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(stats.spearmanr(a.numpy(), b.numpy()).statistic)


def kendall(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(stats.kendalltau(a.numpy(), b.numpy(), variant="b").statistic)


# --------------------------------------------------------------------------
# acquisition sets
# --------------------------------------------------------------------------
def top_p_set(scores: torch.Tensor, p: float, *, policy: TiePolicy = "inclusive") -> np.ndarray:
    """Boolean mask of the top p-fraction. `p` is a fraction, e.g. 0.05."""
    n = scores.numel()
    k = max(1, int(round(p * n)))
    if policy == "strict_k":
        idx = torch.topk(scores, k).indices
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx] = True
        return mask.numpy()
    thresh = torch.topk(scores, k).values[-1]
    return (scores >= thresh).numpy()


def acquisition_metrics(exact: torch.Tensor, approx: torch.Tensor, p: float,
                        *, policy: TiePolicy = "inclusive") -> dict:
    e = top_p_set(exact, p, policy=policy)
    a = top_p_set(approx, p, policy=policy)
    inter = np.logical_and(e, a).sum()
    return {
        "recall": float(inter / max(e.sum(), 1)),
        "precision": float(inter / max(a.sum(), 1)),
        "jaccard": float(inter / max(np.logical_or(e, a).sum(), 1)),
    }


# --------------------------------------------------------------------------
# numeric fidelity
# --------------------------------------------------------------------------
def numeric_fidelity(exact: torch.Tensor, approx: torch.Tensor, eps: float = 1e-30) -> dict:
    rel = (approx - exact).abs() / (exact.abs() + eps)
    return {
        "median_rel_err": float(rel.median()),
        "p90_rel_err": float(rel.quantile(0.90)),
        "nrmse": float((approx - exact).pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()),
    }


# --------------------------------------------------------------------------
# atom localization (spec SS28)
# --------------------------------------------------------------------------
def atom_localization(exact: torch.Tensor, approx: torch.Tensor) -> dict:
    """`exact`/`approx` are [S, A] per-atom scores.

    Reports the agreement fraction AND the rank distance when they disagree. The
    second number is what distinguishes "picked a different region of the molecule"
    from "picked the runner-up", and without it the metric reads far worse than the
    behaviour warrants.
    """
    e_arg = exact.argmax(dim=1)
    a_arg = approx.argmax(dim=1)
    same = (e_arg == a_arg)
    order = approx.argsort(dim=1, descending=True)
    rank_of_true = (order == e_arg.unsqueeze(1)).float().argmax(dim=1).double()
    gaps = rank_of_true[~same]
    return {
        "same_max_atom_frac": float(same.float().mean()),
        "median_rank_gap_when_wrong": float(gaps.median()) if gaps.numel() else 0.0,
        "p90_rank_gap_when_wrong": float(gaps.quantile(0.9)) if gaps.numel() else 0.0,
        "top3_contains_true_max_frac": float((rank_of_true < 3).float().mean()),
    }


# --------------------------------------------------------------------------
# spec SS38 head-space spectrum
# --------------------------------------------------------------------------
def head_space_spectrum(F: torch.Tensor) -> dict:
    r"""Eigenvalues of \bar A = (1/n) sum_i P F_i^T F_i P  (spec SS38).

    F is [S, A, 3, M]. Returns the r = M-1 nonzero eigenvalues of the averaged
    centered head-space Gram matrix, plus the stable rank.

    Why this matters, and why it is run EARLY rather than as a post-hoc curiosity:
    spec SS38 hypothesizes that a low-rank disagreement subspace explains why small
    K works. The opposite is true for RANDOM sketching -- a concentrated spectrum
    raises the variance of a random K-dimensional projection, because the sketch
    either captures the dominant directions or misses them. The spectrum therefore
    PREDICTS whether K <= 4 can clear the SS51 gate, and tells us which SS51
    escalation rung to reach for.
    """
    S, A, _, M = F.shape
    X = F.reshape(-1, M).double()                 # [S*A*3, M]
    X = X - X.mean(dim=-1, keepdim=True)          # P applied on the right
    Abar = (X.T @ X) / X.shape[0]                 # [M, M]
    eig = torch.linalg.eigvalsh(Abar).flip(0)     # descending
    eig = eig[: M - 1].clamp_min(0)               # drop the null direction
    frac = (eig / eig.sum()).tolist()
    cum = np.cumsum(frac).tolist()

    # Two distinct quantities that a previous version conflated -- it returned the
    # SAME formula under both names, and reported it as "stable rank". They differ
    # by ~1.7x here, so both are given with their definitions and the ambiguous
    # bare name is not used anywhere.
    #
    #   stable rank of FQ (the matrix we actually sketch):
    #       ||FQ||_F^2 / ||FQ||_2^2 = tr(A) / lambda_max
    #   effective / participation rank:
    #       tr(A)^2 / tr(A^2)
    stable_rank_FQ = float(eig.sum() / eig[0])
    stable_rank_A = float((eig**2).sum() / eig[0] ** 2)
    effective_rank = float(eig.sum() ** 2 / (eig**2).sum())

    return {
        "eigenvalues": eig.tolist(),
        "fraction": frac,
        "cumulative": cum,
        "stable_rank_FQ": stable_rank_FQ,
        "stable_rank_A": stable_rank_A,
        "effective_rank": effective_rank,
        "isotropic_top1_fraction": 1.0 / (M - 1),
        "top1_fraction": frac[0],
        "n_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
    }
