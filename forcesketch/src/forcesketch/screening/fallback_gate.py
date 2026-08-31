r"""Conservative exact-fallback screening gate (spec SS33, SS34; hypothesis H5).

The point of this module is that ForceSketch does not have to REPLACE exact force
UQ to be useful. It only has to decide, cheaply and conservatively, which
structures cannot possibly be above the exact high-uncertainty threshold -- those
can skip the exact computation entirely, and the rest fall back to it.

Construction (spec SS33). On a calibration split compute the ratio

    r_i = S(x_i) / (S_hat(x_i) + eps)

take a conservative upper quantile c_alpha, and define the optimistic bound

    U(x) = c_alpha * S_hat(x)

Then skip exact evaluation whenever U(x) < tau. Because c_alpha upper-bounds the
ratio for (1 - alpha) of calibration structures, U over-estimates S for those, so
`U(x) < tau` implies `S(x) < tau` at the calibrated confidence.

The calibration quantile is fitted ONLY on the calibration split, and the
threshold tau is defined from the calibration split too. The test split is
touched once, at the end.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class GateCalibration:
    c_alpha: float
    alpha: float
    tau: float
    eps: float

    def bound(self, s_hat: Tensor) -> Tensor:
        return self.c_alpha * s_hat

    def skip_mask(self, s_hat: Tensor) -> Tensor:
        """True where exact evaluation can be skipped."""
        return self.bound(s_hat) < self.tau


def calibrate(
    s_exact: Tensor, s_hat: Tensor, *, alpha: float, tau: float, eps: float = 1e-30
) -> GateCalibration:
    r"""Fit c_alpha on the CALIBRATION split only, by SPLIT CONFORMAL.

    c_alpha is the ceil((n+1)(1-alpha))-th order statistic of the ratios
    r_i = S_i / (S_hat_i + eps), NOT `torch.quantile(ratio, 1-alpha)`.

    The distinction is not cosmetic. Split conformal guarantees marginal coverage
        P[ S(x) <= c_alpha * S_hat(x) ] >= 1 - alpha
    for an exchangeable test point, and the (n+1) is exactly what pays for the
    unseen point: with n calibration ratios there are n+1 possible ranks for the
    test ratio. An interpolating quantile of n points is anti-conservative by
    O(1/n) and carries no finite-sample guarantee at all, which matters here
    because the coverage claim is the whole basis for calling the gate safe.

    If ceil((n+1)(1-alpha)) > n -- too few calibration points to certify the
    requested alpha -- there is no finite c_alpha with the guarantee, so we return
    +inf, which makes the gate skip nothing rather than silently under-cover.
    """
    ratio = torch.sort((s_exact / (s_hat + eps)).flatten()).values
    n = ratio.numel()
    k = math.ceil((n + 1) * (1.0 - alpha))
    c = float("inf") if k > n else float(ratio[k - 1])
    return GateCalibration(c_alpha=c, alpha=alpha, tau=tau, eps=eps)


@dataclass(frozen=True, slots=True)
class LaneCostModel:
    r"""Latency of an L-lane reverse pass, as T(L) = intercept + slope * L.

    Lane COUNT is not lane COST: the forward pass and graph construction are paid
    once regardless of L. Using raw lane counts would overstate the gate's speedup,
    because it would charge the shared forward once per lane.

    THE DEFAULTS BELOW ARE NOT A SECOND COST MODEL. They are a cached copy of the
    least-squares fit that `analysis.tables.fit_cost_model` computes from
    `results/raw/lane_scaling_disjoint.jsonl` (serial path, 3BPA, RTX 5070 Laptop,
    fp32, CUDA events). The paper prints that fit and nothing else, and
    `tests/test_cost_model.py` fails if these drift apart -- the manuscript
    previously carried four mutually inconsistent slopes, so this is enforced
    rather than trusted.
    """

    intercept_ms: float = 10.38
    slope_ms_per_lane: float = 11.78

    def __call__(self, lanes: int) -> float:
        return self.intercept_ms + self.slope_ms_per_lane * lanes


MEASURED_COST = LaneCostModel()


def evaluate_gate(
    cal: GateCalibration,
    s_exact: Tensor,
    s_hat: Tensor,
    *,
    cost_sketch_lanes: int,
    cost_exact_lanes: int,
    cost_model: LaneCostModel | None = None,
    dirs_done: int | None = None,
) -> dict:
    """Every spec SS34 metric on a held-out split.

    The gate always pays the sketch, and additionally pays the exact cost on the
    structures it does not skip. That is the honest accounting -- a gate that skips
    nothing is SLOWER than computing exact directly, and this formula shows it.

    Two speedups are reported and must not be conflated (spec SS45):
      * `screening_speedup_lanes` -- the idealized lane-count model.
      * `screening_speedup`       -- the MEASURED T(L) model, which is the number
                                     that reflects wall-clock and is the one to quote.
    """
    skip = cal.skip_mask(s_hat)
    high = s_exact >= cal.tau

    tp = int((high & ~skip).sum())      # high-UQ correctly sent to exact
    fn = int((high & skip).sum())       # high-UQ wrongly skipped -- the dangerous case
    n = s_exact.numel()
    frac_skipped = float(skip.float().mean())

    n_exact_run = int((~skip).sum())
    # On fallback, a gate that already evaluated `dirs_done` independent centered
    # directions completes the exact basis with (M-1) - dirs_done further reverse
    # passes rather than a fresh M. Pass it, or leave it None for the conservative
    # full-recompute accounting. Crediting this reuse to one method and not its
    # baselines would manufacture an advantage, so callers must apply it uniformly.
    lanes_complete = (cost_exact_lanes if dirs_done is None
                      else (cost_exact_lanes - 1) - dirs_done)
    lane_gate = cost_sketch_lanes * n + lanes_complete * n_exact_run
    lane_exact = cost_exact_lanes * n

    cm = cost_model or MEASURED_COST
    ms_gate = cm(cost_sketch_lanes) * n + cm(lanes_complete) * n_exact_run
    ms_exact = cm(cost_exact_lanes) * n

    return {
        "screening_speedup_lanes": float(lane_exact / max(lane_gate, 1)),
        "screening_ms_gate": ms_gate,
        "screening_ms_exact": ms_exact,
        "alpha": cal.alpha,
        "c_alpha": cal.c_alpha,
        "tau": cal.tau,
        "n": n,
        "n_high_uq": int(high.sum()),
        "high_uq_recall": float(tp / max(int(high.sum()), 1)),
        "false_negative_rate": float(fn / max(int(high.sum()), 1)),
        "n_false_negatives": fn,
        "frac_exact_skipped": frac_skipped,
        "precision": float(tp / max(int((~skip).sum()), 1)),
        "lanes_complete": lanes_complete,
        "screening_lane_cost": lane_gate,
        "exact_lane_cost": lane_exact,
        "screening_speedup": float(ms_exact / max(ms_gate, 1e-9)),
    }


def split_indices(n: int, *, seed: int, fracs=(0.2, 0.2, 0.6)) -> tuple[Tensor, Tensor, Tensor]:
    """Spec SS33's 20% calibration / 20% validation / 60% test split."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_cal = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    return perm[:n_cal], perm[n_cal:n_cal + n_val], perm[n_cal + n_val:]
