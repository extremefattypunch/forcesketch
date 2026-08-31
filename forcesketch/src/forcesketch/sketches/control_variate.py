r"""Low-rank head-space control variate (spec SS39; SS51 escalation rung 4).

Motivation, from the measured spectrum rather than from theory alone. Spec SS38
hypothesizes that a low-rank disagreement subspace explains why small K works. For
RANDOM sketching the opposite holds: a concentrated spectrum raises the variance of
a random K-dimensional projection, because the sketch either captures the dominant
directions or misses them. On the 3BPA disjoint committee the leading centered
direction carries 33% of the disagreement energy, which is precisely the structure
that hurts a plain Haar sketch -- and precisely the structure a control variate
removes.

The construction. Split the centered head space into

    P = Q_{r0} Q_{r0}^T  +  R

where Q_{r0} holds the r0 leading eigenvectors of the calibration-set average
\bar A = (1/n) sum_i P F_i^T F_i P, and R projects onto the residual. Then

    r * v_d = ||e_d^T F Q_{r0}||^2  +  ||e_d^T F R||^2

Evaluate the first term EXACTLY with r0 cotangent lanes, and sketch only the
second with K' = K - r0 Haar directions drawn inside the residual subspace:

    v_hat_d = (1/r) [ sum_{j<=r0} g_{j,d}^2  +  ((r - r0)/K') sum_k G_{k,d}^2 ]

which is unbiased, because a random K'-dimensional subspace of the (r - r0)-
dimensional residual captures K'/(r - r0) of its squared norm in expectation.

Eigenvectors MUST come from a calibration split that is disjoint from the
evaluation set, or the reported fidelity is contaminated.
"""

from __future__ import annotations

import torch
from torch import Tensor

from forcesketch.estimators.std_correction import haar_std_correction
from forcesketch.exact.centered_basis import helmert_basis
from forcesketch.types import LaneBudget, SeedBundle
from forcesketch.utils.reproducibility import make_generator


def leading_head_directions(F_calib: Tensor, r0: int) -> Tensor:
    """Top-r0 eigenvectors of the averaged centered head-space Gram matrix.

    `F_calib` is [S, A, 3, M] from the CALIBRATION split only. Returns [M, r0]
    with orthonormal columns spanning a subspace of the centered head space.
    """
    M = F_calib.shape[-1]
    X = F_calib.reshape(-1, M).double()
    X = X - X.mean(dim=-1, keepdim=True)
    Abar = (X.T @ X) / X.shape[0]
    evals, evecs = torch.linalg.eigh(Abar)
    Q = evecs[:, torch.argsort(evals, descending=True)[:r0]]
    # re-center defensively: the null eigenvector is 1/sqrt(M) and must not leak in
    Q = Q - Q.mean(dim=0, keepdim=True)
    Q, _ = torch.linalg.qr(Q)
    return Q.contiguous()


def control_variate_seeds(
    Q_lead: Tensor, *, M: int, K: int, batch_size: int, seed: int | None,
    dtype: torch.dtype = torch.float64, device="cpu",
) -> tuple[SeedBundle, int]:
    """Build the r0 exact lanes plus K-r0 residual Haar lanes.

    Returns (bundle, r0). The bundle's `seeds` are [K, B, M] with the first r0
    lanes exact; `variance_scale` is None because the two blocks carry different
    normalizations -- use `control_variate_variance` rather than the generic
    estimator.
    """
    r0 = Q_lead.shape[1]
    r = M - 1
    if not r0 < K <= r:
        raise ValueError(f"need r0 < K <= r; got r0={r0}, K={K}, r={r}")
    k_res = K - r0

    # orthonormal basis of the residual subspace inside the centered space
    Q_cent = helmert_basis(M, dtype=torch.float64)            # [M, r]
    coeff = Q_cent.T @ Q_lead.double()                        # [r, r0]
    Qc, _ = torch.linalg.qr(coeff, mode="complete")
    B_res = Q_cent @ Qc[:, r0:]                               # [M, r-r0]

    g = make_generator(seed, "control_variate", K, batch_size)
    A = torch.randn(batch_size, r - r0, k_res, generator=g, dtype=torch.float64)
    O, R = torch.linalg.qr(A)
    O = O * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)
    W_res = torch.einsum("mj,bjk->kbm", B_res, O)             # [k_res, B, M]

    W_exact = Q_lead.double().T.unsqueeze(1).expand(r0, batch_size, M)
    seeds = torch.cat([W_exact, W_res], dim=0).to(dtype=dtype, device=device)

    bundle = SeedBundle(
        seeds=seeds.contiguous(), method="haar", K=K, M=M,
        estimator_kind="quadratic", variance_scale=None,
        std_correction=haar_std_correction(max(k_res, 1), max(r - r0, 1))
        if k_res < r - r0 else 1.0,
        lane_budget=LaneBudget(uq_lanes=K, mean_lanes=1, exact_mean_force=True),
        per_structure=True, rng_seed=seed,
    )
    return bundle, r0


def control_variate_variance(G: Tensor, *, r0: int, M: int) -> Tensor:
    r"""G [K, ..., ] lane forces -> v_hat.

    v_hat = (1/r) [ sum_{j<r0} g_j^2 + ((r-r0)/K') sum_{k>=r0} g_k^2 ]
    """
    r = M - 1
    K = G.shape[0]
    k_res = K - r0
    exact_part = (G[:r0] ** 2).sum(dim=0)
    resid_part = (G[r0:] ** 2).sum(dim=0) * ((r - r0) / k_res) if k_res else 0.0
    return (exact_part + resid_part) / r
