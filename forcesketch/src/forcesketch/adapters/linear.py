r"""Analytic linear committee, the spec SS20 correctness model.

    E_m(x) = a_m^T x + b_m   =>   f_m = -grad_x E_m = -a_m, exactly.

Coefficients are stored in atom space as `coeffs [M, N_total, 3]`, so

    exact_head_forces(batch) == -coeffs

and the head-force matrix in spec SS6.2 layout is F[i, alpha, m] = -coeffs[m, i, alpha].

Every quantity in SS7-SS15 therefore has a closed form, and the entire estimator
stack is testable with zero molecular dependencies -- which is what lets spec
SS20's gate ("no molecular experiments until these pass") be cleared before the
MACE environment even exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from forcesketch.adapters.base import BaseMHCAdapter


@dataclass
class LinearBatch:
    """Deliberately minimal, and deliberately NOT a MACE/PyG object: if any core
    module grows a dependency on batch internals, the contract test breaks."""

    positions: Tensor  # [N_total, 3], requires_grad
    batch_index: Tensor  # [N_total] int64, non-decreasing


class LinearMHCAdapter(BaseMHCAdapter):
    def __init__(self, coeffs: Tensor, biases: Tensor) -> None:
        """coeffs `[M, N_total, 3]`; biases `[M]` (added once per structure)."""
        if coeffs.ndim != 3 or coeffs.shape[-1] != 3:
            raise ValueError(f"coeffs must be [M, N_total, 3], got {tuple(coeffs.shape)}")
        if biases.ndim != 1 or biases.shape[0] != coeffs.shape[0]:
            raise ValueError(f"biases must be [M={coeffs.shape[0]}], got {tuple(biases.shape)}")
        self.coeffs = coeffs
        self.biases = biases

    @property
    def num_heads(self) -> int:
        return self.coeffs.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        return self.coeffs.dtype

    @property
    def device(self) -> torch.device:
        return self.coeffs.device

    def num_structures(self, batch: LinearBatch) -> int:
        return int(batch.batch_index.max().item()) + 1

    def batch_index(self, batch: LinearBatch) -> Tensor:
        return batch.batch_index

    def positions(self, batch: LinearBatch) -> Tensor:
        return batch.positions

    def prepare(self, batch: LinearBatch) -> LinearBatch:
        if not batch.positions.requires_grad:
            batch.positions.requires_grad_(True)
        return batch

    def energies(self, batch: LinearBatch, *, create_graph: bool = False) -> Tensor:
        """E[b, m] = sum_{i in b} <coeffs[m, i], x_i> + biases[m]   ->   [B, M]."""
        B = self.num_structures(batch)
        per_atom = (self.coeffs * batch.positions.unsqueeze(0)).sum(-1)  # [M, N_total]
        out = per_atom.new_zeros(self.num_heads, B)
        out.index_add_(1, batch.batch_index, per_atom)
        return out.T + self.biases  # [B, M]


def head_force_matrix(coeffs: Tensor) -> Tensor:
    """`coeffs [M, N, 3]` -> F `[N, 3, M]` with F[..., m] = f_m = -a_m (spec SS6.2)."""
    return (-coeffs).permute(1, 2, 0).contiguous()
