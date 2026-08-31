r"""Exact centered head-space basis and the mean-force seed (spec SS8, SS9).

The centering projector is P = I - (1/M) 11^T, and the centered head space has
dimension r = M - 1. Spec SS8 asks for any Q in R^{M x (M-1)} with

    Q^T Q = I,    Q^T 1 = 0,    Q Q^T = P

We use the Helmert basis, which is deterministic by construction. That matters:
a random QR would give the "exact" path a random seed, and the exact reference
must be reproducible without one.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from forcesketch.types import LaneBudget, SeedBundle


def helmert_basis(
    M: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    recenter: bool = True,
) -> Tensor:
    r"""Deterministic centered basis Q, shape [M, M-1] (spec SS8).

    Column j (1-indexed) is
        q_j = (1/sqrt(j(j+1))) * (1, ..., 1 [j times], -j, 0, ..., 0)

    Built in float64 then cast. `recenter` re-applies Q -= Q.mean(0) after the
    cast so that Q^T 1 == 0 holds to *working* precision. Without it, a float32
    basis leaks the mean force into the uncertainty lanes with weight ~eps*rho,
    where rho = |mean force| / sqrt(variance) can be 10-1000 for a real committee.
    """
    if M < 2:
        raise ValueError(f"need M >= 2 heads, got {M}")
    Q = torch.zeros(M, M - 1, dtype=torch.float64, device=device)
    for j in range(1, M):
        scale = 1.0 / math.sqrt(j * (j + 1))
        Q[:j, j - 1] = scale
        Q[j, j - 1] = -j * scale
    Q = Q.to(dtype)
    if recenter:
        Q = Q - Q.mean(dim=0, keepdim=True)
    return Q


def assert_valid_centered_basis(Q: Tensor, *, atol: float | None = None) -> None:
    """Raise if Q fails any of the three spec SS8 properties."""
    M, r = Q.shape
    if atol is None:
        atol = 1e-11 if Q.dtype == torch.float64 else 1e-5
    eye = torch.eye(r, dtype=Q.dtype, device=Q.device)
    P = torch.eye(M, dtype=Q.dtype, device=Q.device) - 1.0 / M
    checks = {
        "Q^T Q = I": (Q.T @ Q - eye).abs().max().item(),
        "Q^T 1 = 0": Q.sum(dim=0).abs().max().item(),
        "Q Q^T = P": (Q @ Q.T - P).abs().max().item(),
    }
    bad = {k: v for k, v in checks.items() if v > atol}
    if bad:
        raise ValueError(f"invalid centered basis (atol={atol:g}): {bad}")


def mean_seed(
    M: int, batch_size: int, *, dtype: torch.dtype, device: torch.device | str = "cpu"
) -> Tensor:
    """s_0 = (1/M) 1, shaped [1, B, M] (spec SS9).

    One VJP against this seed gives the exact committee mean force, so MD-like use
    costs 1 mean-force lane + K uncertainty lanes. Spec SS9 requires the mean-force
    lane be counted explicitly, which `LaneBudget` does.
    """
    return torch.full((1, batch_size, M), 1.0 / M, dtype=dtype, device=device)


def exact_seed_bundle(
    M: int, batch_size: int, *, dtype: torch.dtype, device: torch.device | str = "cpu"
) -> SeedBundle:
    """The exact reference: all r = M-1 Helmert directions (spec SS8).

    v_d = (1/(M-1)) sum_j g_{j,d}^2, so variance_scale = 1/r and no finite-K
    correction is needed. rng_seed is None because the basis is deterministic.
    """
    Q = helmert_basis(M, dtype=dtype, device=device)
    seeds = Q.T.unsqueeze(1).expand(M - 1, batch_size, M).contiguous()
    return SeedBundle(
        seeds=seeds,
        method="exact",
        K=M - 1,
        M=M,
        estimator_kind="quadratic",
        variance_scale=1.0 / (M - 1),
        std_correction=1.0,
        lane_budget=LaneBudget(uq_lanes=M - 1, mean_lanes=1, exact_mean_force=True),
        per_structure=False,
        rng_seed=None,
    )
