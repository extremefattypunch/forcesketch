"""Per-head forces and the direct variance reference (spec SS7, SS15, SS23)."""

from __future__ import annotations

import torch
from torch import Tensor

from forcesketch.types import SPEC_CONVENTION, VarianceConvention


def variance_from_member_forces(
    forces: Tensor,
    *,
    convention: VarianceConvention = SPEC_CONVENTION,
    reduce_dtype: torch.dtype | None = torch.float64,
) -> Tensor:
    """Ground-truth v_d straight from all M head forces. `forces [M, ...] -> [...]`.

    This is spec SS23's fourth independent path: it never touches a centered basis
    or a cotangent, so agreement with the VJP routes is a real cross-check rather
    than a restatement.
    """
    f = forces if reduce_dtype is None else forces.to(reduce_dtype)
    v = f.var(dim=0, unbiased=True)
    scale = convention.target_scale(forces.shape[0])
    return v if scale == 1.0 else v * scale


def head_subsample_variance(
    forces: Tensor,
    *,
    convention: VarianceConvention = SPEC_CONVENTION,
    reduce_dtype: torch.dtype | None = torch.float64,
) -> Tensor:
    """ddof=1 sample variance across the K subsampled head lanes (spec SS15)."""
    return variance_from_member_forces(forces, convention=convention, reduce_dtype=reduce_dtype)
