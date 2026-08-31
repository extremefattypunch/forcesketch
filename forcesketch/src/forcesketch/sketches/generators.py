r"""The five seed generators (spec SS10, SS12, SS14, SS15, plus Rademacher).

Every generator has the same keyword-only signature and returns a `SeedBundle`
whose `seeds` are `[K, B, M]`. Seeds are constructed in float64 regardless of the
working dtype -- they are only K*B*M floats, and a float32 Haar QR has
orthonormality error ~1e-7 that lands directly in v_hat.

Normalization table (r = M - 1), all verified by Monte Carlo:

    method        seeds w_k              v_hat_d                       correction
    ------------  ---------------------  ----------------------------  ----------
    gaussian      P z,  z ~ N(0, I_M)    (1/(K r)) sum_k g^2           c_K
    rademacher    P eps, eps ~ +-1       (1/(K r)) sum_k g^2           c_K (approx)
    haar          Q O,  O Haar r x K     (1/K)     sum_k g^2           c_haar(K, r)
    pairwise      e_i - e_j              (1/(2K))  sum_k g^2           1
    head_subsample one-hot, SRSWOR       var(g, ddof=1)                1
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from forcesketch.estimators.std_correction import (
    gaussian_std_correction,
    haar_std_correction,
    sample_std_correction_c4,
)
from forcesketch.exact.centered_basis import helmert_basis
from forcesketch.types import LaneBudget, SeedBundle
from forcesketch.utils.reproducibility import make_generator

CONSTRUCT_DTYPE = torch.float64


def _n_draws(batch_size: int, per_structure: bool) -> int:
    return batch_size if per_structure else 1


def _finish(seeds: Tensor, batch_size: int, per_structure: bool, dtype, device) -> Tensor:
    """[K, n_draws, M] -> [K, B, M] in the working dtype."""
    if not per_structure and batch_size > 1:
        seeds = seeds.expand(seeds.shape[0], batch_size, seeds.shape[2])
    return seeds.to(dtype=dtype, device=device).contiguous()


def gaussian_seeds(
    *, M: int, K: int, batch_size: int, seed: int | None,
    device="cpu", dtype=torch.float32, per_structure: bool = True,
) -> SeedBundle:
    r"""Spec SS10. w_k = P z_k. E[g_d^2] = r*v_d, and K*v_hat_d/v_d ~ chi^2_K."""
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    g = make_generator(seed, "gaussian", K, batch_size)
    n = _n_draws(batch_size, per_structure)
    z = torch.randn(K, n, M, generator=g, dtype=CONSTRUCT_DTYPE)
    w = z - z.mean(dim=-1, keepdim=True)  # P z
    return SeedBundle(
        seeds=_finish(w, batch_size, per_structure, dtype, device),
        method="gaussian", K=K, M=M,
        estimator_kind="quadratic",
        variance_scale=1.0 / (K * (M - 1)),
        std_correction=gaussian_std_correction(K),
        lane_budget=LaneBudget(uq_lanes=K, mean_lanes=1, exact_mean_force=True),
        per_structure=per_structure, rng_seed=seed,
    )


def rademacher_seeds(
    *, M: int, K: int, batch_size: int, seed: int | None,
    device="cpu", dtype=torch.float32, per_structure: bool = True,
) -> SeedBundle:
    r"""w = P eps, eps ~ Unif{+-1}^M. Same normalization as Gaussian since
    E[eps eps^T] = I, but strictly lower variance -- which is why SS36 compares them."""
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    g = make_generator(seed, "rademacher", K, batch_size)
    n = _n_draws(batch_size, per_structure)
    eps = (torch.randint(0, 2, (K, n, M), generator=g, dtype=torch.int64) * 2 - 1).to(
        CONSTRUCT_DTYPE
    )
    w = eps - eps.mean(dim=-1, keepdim=True)
    return SeedBundle(
        seeds=_finish(w, batch_size, per_structure, dtype, device),
        method="rademacher", K=K, M=M,
        estimator_kind="quadratic",
        variance_scale=1.0 / (K * (M - 1)),
        std_correction=gaussian_std_correction(K),
        lane_budget=LaneBudget(uq_lanes=K, mean_lanes=1, exact_mean_force=True),
        per_structure=per_structure, rng_seed=seed,
    )


def haar_seeds(
    *, M: int, K: int, batch_size: int, seed: int | None,
    device="cpu", dtype=torch.float32, per_structure: bool = True,
) -> SeedBundle:
    r"""Spec SS12. W = Q O with O a Haar-distributed orthonormal r x K frame, so
    W^T W = I and W^T 1 = 0. Exact at K = r (spec SS21).

    The sign correction O <- O * sign(diag(R)) makes the columns properly
    Haar-marginal. It does not change v_hat, which depends only on span(W), but it
    costs nothing and keeps the distribution honest.
    """
    r = M - 1
    if not 1 <= K <= r:
        raise ValueError(f"Haar sketching needs 1 <= K <= r = {r}, got K={K}")
    g = make_generator(seed, "haar", K, batch_size)
    n = _n_draws(batch_size, per_structure)
    A = torch.randn(n, r, K, generator=g, dtype=CONSTRUCT_DTYPE)
    O, R = torch.linalg.qr(A)
    O = O * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)
    Q = helmert_basis(M, dtype=CONSTRUCT_DTYPE)
    W = torch.einsum("mr,nrk->knm", Q, O)  # [K, n, M]
    return SeedBundle(
        seeds=_finish(W, batch_size, per_structure, dtype, device),
        method="haar", K=K, M=M,
        estimator_kind="quadratic",
        variance_scale=1.0 / K,
        std_correction=haar_std_correction(K, r),
        lane_budget=LaneBudget(uq_lanes=K, mean_lanes=1, exact_mean_force=True),
        per_structure=per_structure, rng_seed=seed,
    )


def pairwise_seeds(
    *, M: int, K: int, batch_size: int, seed: int | None,
    device="cpu", dtype=torch.float32, per_structure: bool = True,
    replace: bool = True,
) -> SeedBundle:
    r"""Spec SS14. w = e_i - e_j for a uniformly sampled unordered pair.

    E[(1/2)(f_i - f_j)_d^2] = v_d, so variance_scale = 1/(2K).

    `replace=True` matches SS14's derivation (i.i.d. draws across lanes).
    `replace=False` samples distinct pairs without replacement -- still unbiased,
    but lower variance, so it is an ablation and not the default.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    n_pairs = M * (M - 1) // 2
    if not replace and K > n_pairs:
        raise ValueError(f"cannot draw {K} distinct pairs without replacement from {n_pairs}")
    g = make_generator(seed, "pairwise", K, batch_size)
    n = _n_draws(batch_size, per_structure)

    iu = torch.triu_indices(M, M, offset=1)  # [2, n_pairs]
    if replace:
        pick = torch.randint(0, n_pairs, (K, n), generator=g)
    else:
        pick = torch.stack([torch.randperm(n_pairs, generator=g)[:K] for _ in range(n)], dim=1)
    i, j = iu[0][pick], iu[1][pick]  # [K, n]

    w = torch.zeros(K, n, M, dtype=CONSTRUCT_DTYPE)
    w.scatter_(2, i.unsqueeze(-1), 1.0)
    w.scatter_(2, j.unsqueeze(-1), -1.0)
    return SeedBundle(
        seeds=_finish(w, batch_size, per_structure, dtype, device),
        method="pairwise", K=K, M=M,
        estimator_kind="quadratic",
        variance_scale=1.0 / (2 * K),
        std_correction=1.0,
        lane_budget=LaneBudget(uq_lanes=K, mean_lanes=1, exact_mean_force=True),
        per_structure=per_structure, rng_seed=seed,
        head_indices=torch.stack([i, j], dim=-1),
    )


def head_subsample_exact_mean_seeds(
    *, M: int, K: int, batch_size: int, seed: int | None,
    device="cpu", dtype=torch.float32, per_structure: bool = True,
) -> SeedBundle:
    r"""Head subsampling that USES the exact mean force it pays a lane for.

    `head_subsample_seeds` spends a mean lane under `with_mean_lane=True` but then
    estimates with the sample variance among the K drawn heads, throwing that mean
    away. That understates the baseline, and comparing against it flatters
    ForceSketch -- the same asymmetry this project objects to elsewhere. The fair
    estimator at the same budget is

        v_hat_d = M / (K (M-1)) * sum_{i in S} (F_di - Fbar_d)^2 ,

    which is unbiased under uniform sampling without replacement, because
    E sum_{i in S} (F_di - Fbar_d)^2 = (K/M) sum_m (F_dm - Fbar_d)^2 = (K/M)(M-1) v_d.

    Note this is just a QUADRATIC sketch with centered one-hot probes: with
    w_i = P e_i = e_i - (1/M) 1 we get F w_i = f_i - Fbar exactly, so the existing
    machinery applies unchanged and only the scale differs.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if K > M:
        raise ValueError(f"cannot draw {K} distinct heads from M={M}")
    g = make_generator(seed, "head_subsample_exact_mean", K, batch_size)
    n = _n_draws(batch_size, per_structure)

    idx = torch.stack([torch.randperm(M, generator=g)[:K] for _ in range(n)], dim=1)  # [K, n]
    w = torch.zeros(K, n, M, dtype=CONSTRUCT_DTYPE)
    w.scatter_(2, idx.unsqueeze(-1), 1.0)
    w -= 1.0 / M                      # centre each one-hot: w_i = P e_i
    return SeedBundle(
        seeds=_finish(w, batch_size, per_structure, dtype, device),
        method="head_subsample_exact_mean", K=K, M=M,
        estimator_kind="quadratic",
        variance_scale=M / (K * (M - 1)),
        std_correction=1.0,
        lane_budget=LaneBudget(uq_lanes=K, mean_lanes=1, exact_mean_force=True),
        per_structure=per_structure, rng_seed=seed,
        head_indices=idx,
    )


def head_subsample_seeds(
    *, M: int, K: int, batch_size: int, seed: int | None,
    device="cpu", dtype=torch.float32, per_structure: bool = True,
    with_mean_lane: bool = False,
    correction: Literal["none", "c4"] = "none",
) -> SeedBundle:
    r"""Spec SS15, the mandatory baseline. K one-hot head seeds drawn without
    replacement, then the ddof=1 sample variance across those K head forces.

    `with_mean_lane` is the fairness knob (spec SS49). Bare head subsampling spends
    all K lanes on heads and cannot form the exact mean force; adding a mean lane
    costs K+1 total, matching ForceSketch(K). Both framings are evaluated, because
    choosing one would settle SS49 by definition rather than by measurement.
    """
    if K < 2:
        raise ValueError(f"head subsampling needs K >= 2 to form a variance, got {K}")
    if K > M:
        raise ValueError(f"cannot draw {K} distinct heads from M={M}")
    g = make_generator(seed, "head_subsample", K, batch_size)
    n = _n_draws(batch_size, per_structure)

    idx = torch.stack([torch.randperm(M, generator=g)[:K] for _ in range(n)], dim=1)  # [K, n]
    w = torch.zeros(K, n, M, dtype=CONSTRUCT_DTYPE)
    w.scatter_(2, idx.unsqueeze(-1), 1.0)
    return SeedBundle(
        seeds=_finish(w, batch_size, per_structure, dtype, device),
        method="head_subsample", K=K, M=M,
        estimator_kind="sample_variance",
        variance_scale=None,
        std_correction=sample_std_correction_c4(K) if correction == "c4" else 1.0,
        lane_budget=LaneBudget(
            uq_lanes=K, mean_lanes=int(with_mean_lane), exact_mean_force=with_mean_lane
        ),
        per_structure=per_structure, rng_seed=seed,
        head_indices=idx,
    )
