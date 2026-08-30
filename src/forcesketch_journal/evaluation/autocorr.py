"""Integrated autocorrelation time along a trajectory ordering.

This sets two things the split protocol depends on: the guard band between
design/calibration/test blocks, and the block length for the moving-block
bootstrap. Both need tau_int, not a guess.

The estimator is the standard automatic-windowing one (Sokal): accumulate
rho(t) until the window W satisfies W >= c * tau_int(W), with c = 5. A fixed
window either truncates a slow tail (underestimating tau) or accumulates pure
noise from the large-lag rho estimates (inflating it); the self-consistent
window is what avoids both.

A degenerate answer is itself informative. If the sequence is not actually
time-ordered, rho(1) ~ 0, tau_int -> 1, and contiguous blocking silently
degenerates into a random split. The caller must record tau_int rather than
assume the ordering was meaningful, which is why `integrated_autocorr_time`
returns the diagnostics alongside the number.
"""

from __future__ import annotations

import numpy as np


def autocorr_function(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Normalised autocorrelation rho(t), t = 0..max_lag, via FFT.

    Uses the biased estimator (divide by n, not n-t): it has lower variance at
    large lag and is what the standard tau_int estimators assume.
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        # A NaN or inf used to propagate to an all-zero ACF, so tau_int came back
        # as exactly 1.0 and the series was silently reported as independent --
        # which would then set the guard band and block length to their minimum.
        raise ValueError("autocorr_function received a non-finite series; "
                         f"{int((~np.isfinite(x)).sum())} of {x.size} values are NaN/inf")
    n = x.size
    if max_lag is None:
        max_lag = n // 2
    y = x - x.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, nfft)
    acf = np.fft.irfft(f * np.conjugate(f), nfft)[: max_lag + 1]
    acf /= n
    return acf / acf[0] if acf[0] > 0 else np.zeros_like(acf)


def integrated_autocorr_time(x: np.ndarray, c: float = 5.0) -> dict:
    """tau_int with Sokal automatic windowing.

    Returns tau_int, the chosen window, rho(1), and a `time_ordered` flag that
    is False when the series shows essentially no lag-1 correlation -- in which
    case blocking buys nothing and the split scheme should say so.
    """
    rho = autocorr_function(x)
    taus = 1.0 + 2.0 * np.cumsum(rho[1:])
    window = len(taus)
    for w in range(1, len(taus) + 1):
        if w >= c * taus[w - 1]:
            window = w
            break
    tau = float(max(taus[window - 1], 1.0)) if len(taus) else 1.0
    return {
        "tau_int": tau,
        "window": int(window),
        "rho_1": float(rho[1]) if len(rho) > 1 else 0.0,
        "n": int(np.asarray(x).size),
        "time_ordered": bool(len(rho) > 1 and rho[1] > 0.05),
    }


def block_length(tau_int: float) -> int:
    """Moving-block bootstrap block length / guard band width: ceil(2 * tau_int)."""
    return max(1, int(np.ceil(2.0 * tau_int)))
