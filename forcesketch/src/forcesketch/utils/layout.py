"""Flat <-> padded tensor layout (spec SS18, SS24, SS41).

Spec SS18 documents adapter outputs as `[L, B, A_max, 3]`, but MACE uses graph
batching: flat `positions [N_total, 3]` plus a `batch_index [N_total]`. Verified
empirically: `torch.autograd.grad(E[B,M], pos[N,3], grad_outputs=seeds[L,B,M],
is_grads_batched=True)` returns exactly `[L, N_total, 3]`. So flat is not a
compromise -- it is what autograd actually produces, and SS18's padded shape
silently assumed dense padding.

Canonical internal layout is therefore FLAT. Padded is a view produced here, used
for the SS24 padding tests and the SS41 Triton kernel. Because every estimator
reduces only over dim 0 (the lane axis), one code path serves both layouts.
"""

from __future__ import annotations

import torch
from torch import Tensor


def atom_counts(batch_index: Tensor, n_structures: int) -> Tensor:
    """[N_total] -> [B] number of atoms per structure."""
    return torch.bincount(batch_index, minlength=n_structures)


def to_padded(
    flat: Tensor,
    batch_index: Tensor,
    *,
    n_structures: int | None = None,
    max_atoms: int | None = None,
    fill: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """`flat [..., N_total, C]` -> `(padded [..., B, A_max, C], mask [B, A_max])`.

    The atom axis is dim=-2, so leading lane/method axes broadcast untouched.
    """
    if batch_index.ndim != 1:
        raise ValueError(f"batch_index must be 1-D, got shape {tuple(batch_index.shape)}")
    n_total = batch_index.numel()
    if flat.shape[-2] != n_total:
        raise ValueError(
            f"flat.shape[-2]={flat.shape[-2]} does not match batch_index length {n_total}"
        )

    B = int(n_structures if n_structures is not None else batch_index.max().item() + 1)
    counts = atom_counts(batch_index, B)
    A_max = int(max_atoms if max_atoms is not None else counts.max().item())

    # Position of each atom within its own structure: 0,1,2,... per segment.
    offsets = torch.cumsum(counts, 0) - counts
    within = torch.arange(n_total, device=flat.device) - offsets[batch_index]

    lead, C = flat.shape[:-2], flat.shape[-1]
    padded = flat.new_full((*lead, B, A_max, C), fill)
    padded[..., batch_index, within, :] = flat

    mask = torch.zeros(B, A_max, dtype=torch.bool, device=flat.device)
    mask[batch_index, within] = True
    return padded, mask


def from_padded(padded: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """`padded [..., B, A_max, C]` + `mask [B, A_max]` -> `(flat [..., N_total, C], batch_index)`."""
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be bool, got {mask.dtype}")
    B, A_max = mask.shape
    if padded.shape[-3:-1] != (B, A_max):
        raise ValueError(
            f"padded atom dims {tuple(padded.shape[-3:-1])} do not match mask {(B, A_max)}"
        )
    b_idx, a_idx = mask.nonzero(as_tuple=True)
    flat = padded[..., b_idx, a_idx, :]
    return flat, b_idx


def zero_padding(padded: Tensor, mask: Tensor) -> Tensor:
    """Zero every padded slot. Applied BEFORE any sum/max so spec SS24 holds
    structurally rather than by convention."""
    return padded * mask.to(padded.dtype).unsqueeze(-1)


def segment_sum(x: Tensor, batch_index: Tensor, n_structures: int) -> Tensor:
    """`x [N_total, ...]` -> `[B, ...]`, summing within each structure."""
    out = x.new_zeros((n_structures, *x.shape[1:]))
    return out.index_add_(0, batch_index, x)


def segment_max(
    x: Tensor, batch_index: Tensor, n_structures: int, *, fill: float = float("-inf")
) -> Tensor:
    """`x [N_total, ...]` -> `[B, ...]`, max within each structure.

    Structures with no atoms return `fill`. `index_reduce_` with include_self=False
    leaves empty segments at the initialization value, which is what we want.
    """
    out = x.new_full((n_structures, *x.shape[1:]), fill)
    return out.index_reduce_(0, batch_index, x, "amax", include_self=False)
