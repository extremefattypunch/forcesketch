r"""Atom- and structure-level uncertainty scores (spec SS7).

    S_a       = sum_alpha v_{a,alpha}
    u_a^RMS   = sqrt( (1/3) sum_alpha v_{a,alpha} )
    u_a^MHC   = (1/3) sum_alpha sqrt( v_{a,alpha} )        <- primary (spec SS7)
    S         = sum_d v_d, valid atoms only

u^MHC and u^RMS differ in where the square root sits, so their approximation
properties differ -- u^MHC averages square roots and is therefore the one that
needs the finite-K correction. Spec SS7 requires both be retained.

This module is also the single place the ddof convention is applied, so it cannot
be double-counted upstream.
"""

from __future__ import annotations

import torch
from torch import Tensor

from forcesketch.types import (
    SPEC_CONVENTION,
    SeedBundle,
    UncertaintyScores,
    VarianceConvention,
)
from forcesketch.estimators.variance import coordinate_variance, global_disagreement
from forcesketch.utils.layout import segment_max


def coordinate_std(
    coord_var: Tensor, bundle: SeedBundle, *, apply_correction: bool = True
) -> Tensor:
    """sigma_hat_d = sqrt(v_hat_d) / c   (spec SS11, SS13).

    `clamp_min(0)` guards against a tiny negative from float round-off in the
    sample-variance path; it cannot mask a real error because a genuinely negative
    variance would be O(1), not O(eps).
    """
    s = coord_var.clamp_min(0).sqrt()
    return s / bundle.std_correction if apply_correction else s


def atom_sum_score(coord_var: Tensor) -> Tensor:
    """S_a = sum over the three Cartesian components."""
    return coord_var.sum(dim=-1)


def atom_rms_score(coord_var: Tensor) -> Tensor:
    """u_a^RMS = sqrt(mean_alpha v)."""
    return coord_var.mean(dim=-1).clamp_min(0).sqrt()


def atom_mhc_score(coord_std: Tensor) -> Tensor:
    """u_a^MHC = mean_alpha sigma_hat. Takes the CORRECTED component std."""
    return coord_std.mean(dim=-1)


def uncertainty_scores(
    lane_forces: Tensor,
    bundle: SeedBundle,
    *,
    batch_index: Tensor | None = None,
    n_structures: int | None = None,
    mask: Tensor | None = None,
    convention: VarianceConvention = SPEC_CONVENTION,
    apply_correction: bool = True,
    reduce_dtype: torch.dtype | None = torch.float64,
) -> UncertaintyScores:
    """Single entry point: lane forces -> every spec SS7 quantity.

    Accepts flat (`batch_index`) or padded (`mask`) input; exactly one is required.
    Padded entries are zeroed BEFORE any sum or max, so spec SS24 holds
    structurally rather than by convention.
    """
    if (batch_index is None) == (mask is None):
        raise ValueError("pass exactly one of batch_index (flat) or mask (padded)")

    coord_var = coordinate_variance(
        lane_forces, bundle, convention=convention, reduce_dtype=reduce_dtype
    )
    coord_std = coordinate_std(coord_var, bundle, apply_correction=apply_correction)

    if mask is not None:
        m = mask.to(coord_var.dtype).unsqueeze(-1)
        coord_var = coord_var * m
        coord_std = coord_std * m

    atom_sum = atom_sum_score(coord_var)
    atom_rms = atom_rms_score(coord_var)
    atom_mhc = atom_mhc_score(coord_std)

    if mask is not None:
        neg_inf = torch.finfo(coord_var.dtype).min
        max_atom_mhc = atom_mhc.masked_fill(~mask, neg_inf).amax(dim=-1)
        max_coord_std = coord_std.masked_fill(~mask.unsqueeze(-1), neg_inf).amax(dim=(-2, -1))
        global_trace = global_disagreement(coord_var, mask=mask)
    else:
        B = n_structures if n_structures is not None else int(batch_index.max().item()) + 1
        max_atom_mhc = segment_max(atom_mhc, batch_index, B)
        max_coord_std = segment_max(coord_std.amax(dim=-1), batch_index, B)
        global_trace = global_disagreement(
            coord_var, batch_index=batch_index, n_structures=B
        )

    return UncertaintyScores(
        coord_var=coord_var,
        coord_std=coord_std,
        atom_sum=atom_sum,
        atom_rms=atom_rms,
        atom_mhc=atom_mhc,
        global_trace=global_trace,
        max_atom_mhc=max_atom_mhc,
        max_coord_std=max_coord_std,
        convention=convention,
        lane_budget=bundle.lane_budget,
    )
