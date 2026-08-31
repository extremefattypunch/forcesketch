r"""Coordinate variance from lane forces (spec SS7, SS10-SS15).

One function, one dispatch, because the normalization lives on the SeedBundle:

    quadratic:       v_hat = variance_scale * sum_k g_k^2
    sample_variance: v_hat = var(g, dim=0, unbiased=True)

It reduces ONLY over dim 0 (the lane axis), so it works unchanged on flat
`[K, N_total, 3]` and padded `[K, B, A_max, 3]` input.
"""

from __future__ import annotations

import torch
from torch import Tensor

from forcesketch.types import SPEC_CONVENTION, SeedBundle, VarianceConvention
from forcesketch.utils.layout import segment_sum


def coordinate_variance(
    lane_forces: Tensor,
    bundle: SeedBundle,
    *,
    convention: VarianceConvention = SPEC_CONVENTION,
    reduce_dtype: torch.dtype | None = torch.float64,
) -> Tensor:
    """`lane_forces [K, *S]` -> `v_hat [*S]`.

    `reduce_dtype` accumulates the squares in float64 by default. This costs
    nothing measurable and removes accumulation error, which matters because the
    head-space combination cancels O(|mean force|) down to O(|disagreement|).
    """
    if lane_forces.shape[0] != bundle.K:
        raise ValueError(
            f"expected {bundle.K} lanes for method {bundle.method!r}, "
            f"got {lane_forces.shape[0]}"
        )
    g = lane_forces if reduce_dtype is None else lane_forces.to(reduce_dtype)

    if bundle.estimator_kind == "quadratic":
        if bundle.variance_scale is None:
            raise ValueError(f"method {bundle.method!r} is quadratic but has no variance_scale")
        v = bundle.variance_scale * (g**2).sum(dim=0)
    elif bundle.estimator_kind == "sample_variance":
        if bundle.K < 2:
            raise ValueError(f"method {bundle.method!r} needs K >= 2, got {bundle.K}")
        v = g.var(dim=0, unbiased=True)
    else:
        raise ValueError(f"unknown estimator_kind {bundle.estimator_kind!r}")

    scale = convention.target_scale(bundle.M)
    return v if scale == 1.0 else v * scale


def global_disagreement(
    coord_var: Tensor,
    *,
    batch_index: Tensor | None = None,
    n_structures: int | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    r"""S = sum_d v_d per structure (spec SS7), over VALID atoms only (spec SS24).

    Accepts flat input (`coord_var [N_total, 3]` + batch_index) or padded input
    (`coord_var [B, A_max, 3]` + mask). Exactly one of the two must be given.
    """
    if (batch_index is None) == (mask is None):
        raise ValueError("pass exactly one of batch_index (flat) or mask (padded)")
    if mask is not None:
        return (coord_var * mask.to(coord_var.dtype).unsqueeze(-1)).sum(dim=(-2, -1))
    if n_structures is None:
        n_structures = int(batch_index.max().item()) + 1
    return segment_sum(coord_var.sum(dim=-1), batch_index, n_structures)
