"""Analytic estimator variances, so Monte Carlo tests use DERIVED tolerances
rather than hand-tuned constants. With a fixed seed a 5-sigma band is
deterministic and non-flaky.

All three were derived and then confirmed against Monte Carlo:

    relstd_gauss(K)      = sqrt(2/K)
    relstd_haar(K, r)    = sqrt(2(r-K) / (K(r+2)))
    Var_haar / Var_gauss = (r-K)/(r+2)     -- exactly

The last identity turns spec SS22's qualitative "compare Gaussian and Haar
variance" into an exact assertion.
"""

from __future__ import annotations

import math

import torch


def gaussian_rel_std(K: int) -> float:
    return math.sqrt(2.0 / K)


def haar_rel_std(K: int, r: int) -> float:
    return math.sqrt(2.0 * (r - K) / (K * (r + 2))) if K < r else 0.0


def rademacher_rel_std(K: int, h: torch.Tensor) -> torch.Tensor:
    """h = P F^T e_d, the centered head-force vector at coordinate d."""
    kurt = (h**4).sum(-1) / (h**2).sum(-1) ** 2
    return torch.sqrt(2.0 * (1.0 - kurt) / K)


def haar_over_gaussian_var(K: int, r: int) -> float:
    return (r - K) / (r + 2)


def mc_tolerance(rel_std: float, n_draws: int, n_sigma: float = 5.0) -> float:
    return n_sigma * rel_std / math.sqrt(n_draws)


def sqrt_rel_std(rel_std_of_v: float) -> float:
    """Delta method: relstd(sqrt(v_hat)) ~= relstd(v_hat)/2."""
    return 0.5 * rel_std_of_v
