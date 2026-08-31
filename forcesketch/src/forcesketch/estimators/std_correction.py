r"""Finite-K standard-deviation corrections (spec SS11, SS13).

sqrt(v_hat) is a downward-biased estimator of sqrt(v). Both corrections below
satisfy E[sqrt(v_hat)] = c * sqrt(v), so sigma_hat = sqrt(v_hat) / c is unbiased.

The Haar constant deserves a note. Spec SS13 writes it with Beta functions:

    c^Haar_{K,r} = sqrt(r/K) * B((K+1)/2, (r-K)/2) / B(K/2, (r-K)/2)

which is numerically SINGULAR at K = r, where it evaluates B(K/2, 0). Using
B(a,b) = G(a)G(b)/G(a+b), the G((r-K)/2) factors cancel between numerator and
denominator, leaving a Gamma-only form that is finite everywhere and evaluates to
exactly 1.0 at K = r -- which is the convention SS13 states separately:

    c^Haar_{K,r} = sqrt(r/K) * G((K+1)/2) G(r/2) / (G(K/2) G((r+1)/2))

Spec SS13 requires this be checked against Monte Carlo before use in the paper;
see tests/test_std_correction.py, which verifies unbiasedness at 4e5 draws, the
agreement with the Beta form wherever the Beta form is defined, and the
c_haar -> c_gaussian limit as r -> infinity.
"""

from __future__ import annotations

from math import exp, lgamma, sqrt
from typing import Literal

from forcesketch.types import Method


def _check_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def gaussian_std_correction(K: int) -> float:
    r"""c_K = sqrt(2/K) * G((K+1)/2) / G(K/2)   (spec SS11).

    K = 1..8 -> 0.797885, 0.886227, 0.921318, 0.939986, 0.951533, 0.959369,
                0.965030, 0.969311.  Monotone increasing, c_K -> 1 - 1/(4K).
    """
    _check_positive_int("K", K)
    return sqrt(2.0 / K) * exp(lgamma((K + 1) / 2) - lgamma(K / 2))


def haar_std_correction(K: int, r: int) -> float:
    r"""c^Haar_{K,r}, spec SS13, in the stable Gamma-only form derived above.

    r = 7 (i.e. M = 8), K = 1..7 -> 0.826797, 0.918341, 0.954703, 0.974048,
                                    0.986013, 0.994133, 1.0
    """
    _check_positive_int("K", K)
    _check_positive_int("r", r)
    if K > r:
        raise ValueError(f"Haar sketching needs K <= r = M-1; got K={K}, r={r}")
    if K == r:
        # Analytically exactly 1.0 (the Gamma terms cancel pairwise). Kept as an
        # explicit branch because spec SS21 asserts std_correction == 1.0 by exact
        # equality, and that should not depend on lgamma round-off.
        return 1.0
    return sqrt(r / K) * exp(
        lgamma((K + 1) / 2) + lgamma(r / 2) - lgamma(K / 2) - lgamma((r + 1) / 2)
    )


def sample_std_correction_c4(K: int) -> float:
    r"""c4(K), the bias of a ddof=1 sample std of K iid Gaussians.

    Identity: c4(K) == gaussian_std_correction(K - 1). Requires K >= 2.

    NOT the default for head subsampling -- see `std_correction_for`.
    """
    _check_positive_int("K", K)
    if K < 2:
        raise ValueError(f"c4 requires K >= 2, got {K}")
    return gaussian_std_correction(K - 1)


def std_correction_for(
    method: Method,
    K: int,
    r: int,
    *,
    head_subsample_correction: Literal["none", "c4"] = "none",
) -> float:
    """Dispatch the correct constant for a method.

    Head subsampling defaults to no correction. c4 is available as a documented
    ablation, but it does NOT de-bias this estimator: head forces are neither
    Gaussian nor drawn with replacement, and measured E[s]/(c4*sigma) runs
    1.02-1.035 for K=2..7 at M=8 -- it over-corrects and does not converge to 1.
    Applying it would hand the mandatory SS15 baseline an advantage it cannot
    legitimately claim, so it is opt-in and reported separately.

    Pairwise has no closed-form correction; 1.0 is used and the residual bias is
    measured rather than assumed away.
    """
    if method in ("exact", "onehot"):
        return 1.0
    if method in ("gaussian", "rademacher"):
        # Rademacher's statistic is not exactly chi-squared, so c_K is an
        # approximation here; the residual bias is measured in the SS36 ablation.
        return gaussian_std_correction(K)
    if method == "haar":
        return haar_std_correction(K, r)
    if method == "pairwise":
        return 1.0
    if method == "head_subsample":
        return sample_std_correction_c4(K) if head_subsample_correction == "c4" else 1.0
    raise ValueError(f"unknown method {method!r}")
