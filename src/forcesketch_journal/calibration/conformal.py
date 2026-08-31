"""Split-conformal calibration constant, and its finite-sample coverage.

Moved here from `experiments/j2b_factor_b.py`. It carries the paper's central
safety claim -- that the gate's skip decisions retain a stated fraction of
high-uncertainty structures -- and while it lived in an experiment script it had
no test. Anything that a published guarantee rests on belongs in the package,
where it can be exercised against the law it claims to satisfy.
"""

from __future__ import annotations

import math

import numpy as np
import torch

EPS = 1e-30


def conformal_c(s_exact: torch.Tensor, s_hat: torch.Tensor, alpha: float) -> float:
    """ceil((n+1)(1-alpha))-th order statistic of the ratios (split conformal).

    Returns +inf when the requested alpha is infeasible for this calibration set,
    i.e. when ceil((n+1)(1-alpha)) > n and no order statistic carries the
    finite-sample guarantee. An infinite c_alpha makes the gate skip nothing,
    which is the safe failure.

    This previously clamped `k` to `n`, silently substituting the maximum observed
    ratio and emitting an anti-conservative gate labelled with the requested
    alpha. Found by adversarial review.
    """
    r = (s_exact / (s_hat + EPS)).sort().values
    n = r.numel()
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(r[k - 1])


def coverage_simulation(n_cal: int, alpha: float, *, n_trials: int = 4000,
                        seed: int = 0) -> dict:
    """Realised coverage of `conformal_c` on exchangeable synthetic data.

    The guarantee is distribution-free, so any exchangeable draw exercises it.
    Returns the realised coverage alongside the Beta(k, n+1-k) mean the theory
    predicts; the two must agree to Monte-Carlo error.
    """
    g = torch.Generator().manual_seed(seed)
    k = math.ceil((n_cal + 1) * (1.0 - alpha))
    hits = 0
    for _ in range(n_trials):
        # exchangeable ratios: any positive distribution works
        cal = torch.rand(n_cal, generator=g, dtype=torch.float64).exp()
        test = torch.rand(1, generator=g, dtype=torch.float64).exp()
        c = conformal_c(cal, torch.ones_like(cal), alpha)
        hits += int(float(test[0]) <= c)
    return {"n_cal": n_cal, "alpha": alpha, "k": k,
            "realised": hits / n_trials,
            "beta_mean": k / (n_cal + 1) if k <= n_cal else 1.0,
            "n_trials": n_trials}
