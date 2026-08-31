"""Core value types for ForceSketch (spec SS6-SS16).

The two load-bearing ideas in this module:

1. `SeedBundle` carries its OWN normalization (`variance_scale`, `std_correction`).
   The six sketch methods have six different normalizations; keeping each next to
   the seeds that require it means `estimators.variance` has a single code path
   and cannot pick the wrong constant.

2. `LaneBudget` is attached to every bundle and is never optional. Spec SS15/SS49
   make "ForceSketch(K) vs head-subsampling at equal budget" the paper's most
   important comparison, and "equal budget" is ambiguous (equal total reverse
   lanes, or equal uncertainty lanes?). Recording the accounting on the bundle
   makes a one-sided comparison impossible to produce by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

import torch
from torch import Tensor

Method: TypeAlias = Literal[
    "exact", "onehot", "gaussian", "haar", "rademacher", "pairwise", "head_subsample"
]
EstimatorKind: TypeAlias = Literal["quadratic", "sample_variance"]


@dataclass(frozen=True, slots=True)
class VarianceConvention:
    r"""Sample- vs population-variance convention.

    ddof=1 -> v_d = (1/(M-1)) sum_m (F_dm - Fbar_d)^2     [spec SS7, DEFAULT]
    ddof=0 -> v_d = (1/M)     sum_m (F_dm - Fbar_d)^2     [arXiv:2508.09907 Eq. 4]

    v^(0) = ((M-1)/M) v^(1), a single global constant. It does NOT change ranks or
    top-p% sets, but it DOES change absolute values and therefore any calibrated
    threshold tau in the SS33 screening gate.

    Note on which is correct: the reference paper's Eq. 4 is written with 1/N, but
    its released implementation computes `torch.std(..., dim=-1)`, which is torch's
    *unbiased* 1/(M-1). Spec SS7 therefore matches the code that produced the
    paper's numbers, and ddof=1 is the default here.
    """

    ddof: Literal[0, 1] = 1

    def target_scale(self, num_heads: int) -> float:
        """Multiplicative factor taking a ddof=1 variance to this convention."""
        return 1.0 if self.ddof == 1 else (num_heads - 1) / num_heads

    def std_scale(self, num_heads: int) -> float:
        return self.target_scale(num_heads) ** 0.5


SPEC_CONVENTION = VarianceConvention(ddof=1)
MHC_PAPER_CONVENTION = VarianceConvention(ddof=0)


@dataclass(frozen=True, slots=True)
class LaneBudget:
    """Reverse-mode cotangent lane accounting (spec SS16).

    `exact_mean_force` is the honest part: bare head subsampling spends all its
    lanes on heads and can only form an *approximate* mean force from the K it
    drew, which is not acceptable for MD. Recording that distinguishes a fair
    comparison from a flattering one.
    """

    uq_lanes: int
    mean_lanes: int
    exact_mean_force: bool

    @property
    def total_lanes(self) -> int:
        return self.uq_lanes + self.mean_lanes


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """K uncertainty cotangent seeds plus everything needed to turn the resulting
    lane forces into an unbiased variance estimate.

    Attributes:
        seeds: [K, B, M] cotangent seeds in head space.
        estimator_kind: 'quadratic' -> v_hat = variance_scale * sum_k g_k^2
                        'sample_variance' -> v_hat = var(g, dim=0, unbiased=True)
        variance_scale: the per-method constant; None for sample_variance methods.
        std_correction: c with E[sqrt(v_hat)] = c * sqrt(v). Divide by it (SS11/SS13).
        rng_seed: None exactly when the method is deterministic (exact/onehot).
    """

    seeds: Tensor
    method: Method
    K: int
    M: int
    estimator_kind: EstimatorKind
    variance_scale: float | None
    std_correction: float
    lane_budget: LaneBudget
    per_structure: bool
    rng_seed: int | None
    head_indices: Tensor | None = None

    @property
    def r(self) -> int:
        """Dimension of the centered head space, M - 1."""
        return self.M - 1

    def to(self, *, dtype: torch.dtype | None = None, device=None) -> SeedBundle:
        seeds = self.seeds.to(dtype=dtype or self.seeds.dtype, device=device or self.seeds.device)
        return replace(self, seeds=seeds)


@dataclass(frozen=True, slots=True)
class UncertaintyScores:
    """All spec SS7 quantities, computed once from one set of lane forces."""

    coord_var: Tensor  # [N_total, 3]  v_d
    coord_std: Tensor  # [N_total, 3]  sigma_hat_d (finite-K corrected)
    atom_sum: Tensor  # [N_total]     S_a
    atom_rms: Tensor  # [N_total]     u_a^RMS
    atom_mhc: Tensor  # [N_total]     u_a^MHC  (primary)
    global_trace: Tensor  # [B]        S, valid atoms only
    max_atom_mhc: Tensor  # [B]
    max_coord_std: Tensor  # [B]
    convention: VarianceConvention
    lane_budget: LaneBudget
