"""The multi-head-committee adapter interface (spec SS18).

This is the ONLY boundary between ForceSketch and a molecular potential. Swapping
MACE for a multi-head TensorNet2 must require changing nothing but this file's
implementations, so the contract is deliberately narrow:

  * `TBatch` is opaque to the core library. No module outside `adapters/` may
    attribute-access a batch. Enforced by tests/test_adapter_contract.py.
  * Canonical layout is FLAT (`[N_total, 3]` + `batch_index`); padded is a view.
  * Lane-major everywhere: axis 0 is the cotangent lane / head index. Spec SS18
    writes `exact_head_forces -> [B, M, A_max, 3]` (batch-major) while
    `vjp_for_seeds -> [L, B, A_max, 3]` (lane-major), which contradict each other
    since head m IS lane m under a one-hot seed. Both are lane-major here.
  * `vjp_for_seeds` returns FORCES (-grad), never raw gradients. Spec SS18's
    snippet negates outside the adapter and its docstring is silent; the sign
    cancels for variance but NOT for the mean-force lane, so it is pinned here
    and tested.
"""

from __future__ import annotations

import warnings
from typing import Protocol, TypeVar, runtime_checkable

import torch
from torch import Tensor

TBatch = TypeVar("TBatch")


@runtime_checkable
class MHCAdapter(Protocol):
    """Backbone-agnostic multi-head-committee interface."""

    @property
    def num_heads(self) -> int: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def device(self) -> torch.device: ...

    def prepare(self, batch): ...
    def num_structures(self, batch) -> int: ...
    def batch_index(self, batch) -> Tensor: ...
    def positions(self, batch) -> Tensor: ...
    def energies(self, batch, *, create_graph: bool = False) -> Tensor: ...
    def vjp_for_seeds(self, batch, seeds: Tensor, **kwargs) -> Tensor: ...
    def exact_head_forces(self, batch) -> Tensor: ...


class BaseMHCAdapter:
    """Generic `vjp_for_seeds` / `exact_head_forces` on top of `energies` and
    `positions`. Subclass this and the whole ForceSketch stack works -- this is
    the swap-in point for any shared-trunk multi-head backbone.
    """

    # --- subclasses must provide ------------------------------------------
    @property
    def num_heads(self) -> int:
        raise NotImplementedError

    def num_structures(self, batch) -> int:
        raise NotImplementedError

    def batch_index(self, batch) -> Tensor:
        raise NotImplementedError

    def positions(self, batch) -> Tensor:
        raise NotImplementedError

    def energies(self, batch, *, create_graph: bool = False) -> Tensor:
        raise NotImplementedError

    def prepare(self, batch):
        return batch

    # --- generic implementations ------------------------------------------
    def vjp_for_seeds(
        self,
        batch,
        seeds: Tensor,
        *,
        batched: bool = True,
        create_graph: bool = False,
        retain_graph: bool | None = None,
        record_fallbacks: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[str, ...]]:
        """`seeds [L, B, M]` -> FORCES `[L, N_total, 3]`, i.e. forces[l] = F @ seeds[l].

        With `batched=True` this is one `is_grads_batched` call; with False it is
        a serial loop over lanes. Spec SS18 requires any vmap fallback warning be
        recorded -- set `record_fallbacks=True` to get them back alongside the
        result.
        """
        M = self.num_heads
        if seeds.ndim != 3 or seeds.shape[-1] != M:
            raise ValueError(f"seeds must be [L, B, {M}], got {tuple(seeds.shape)}")

        energies = self.energies(batch, create_graph=create_graph)
        pos = self.positions(batch)
        B = energies.shape[0]
        if seeds.shape[1] not in (1, B):
            raise ValueError(f"seeds batch dim {seeds.shape[1]} incompatible with B={B}")
        if seeds.shape[1] == 1 and B > 1:
            seeds = seeds.expand(-1, B, -1)
        seeds = seeds.to(dtype=energies.dtype, device=energies.device).contiguous()
        L = seeds.shape[0]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if batched:
                grads = torch.autograd.grad(
                    outputs=energies,
                    inputs=pos,
                    grad_outputs=seeds,
                    is_grads_batched=True,
                    create_graph=create_graph,
                    retain_graph=create_graph if retain_graph is None else retain_graph,
                )[0]
            else:
                keep = True if retain_graph is None else retain_graph
                grads = torch.stack(
                    [
                        torch.autograd.grad(
                            outputs=energies,
                            inputs=pos,
                            grad_outputs=seeds[i],
                            create_graph=create_graph,
                            retain_graph=keep or (i < L - 1),
                        )[0]
                        for i in range(L)
                    ]
                )
            fallbacks = tuple(
                str(w.message)
                for w in caught
                if "fallback" in str(w.message).lower() or "vmap" in str(w.message).lower()
            )

        forces = -grads
        return (forces, fallbacks) if record_fallbacks else forces

    def exact_head_forces(self, batch) -> Tensor:
        """Debug/reference path: per-head forces `[M, N_total, 3]` via one-hot seeds."""
        M = self.num_heads
        B = self.num_structures(batch)
        pos = self.positions(batch)
        seeds = torch.eye(M, dtype=pos.dtype, device=pos.device)
        seeds = seeds.unsqueeze(1).expand(M, B, M).contiguous()
        # batched=False on purpose: this is the reference/debug path, and
        # is_grads_batched is unreliable on MACE + e3nn 0.4.4 (see
        # adapters/mace_mhc.configure_e3nn_for_batched_vjp).
        return self.vjp_for_seeds(batch, seeds, batched=False)

    def assert_heads_nondegenerate(self, batch, *, atol: float = 0.0) -> None:
        """Guard against the one failure that is fatal AND invisible: a committee
        whose M heads are identical. Disagreement is then identically zero, every
        sketch estimate is vacuously 'exact', and nothing in a loss curve looks
        wrong. Called from the adapter, not only at setup.
        """
        f = self.exact_head_forces(batch)
        spread = f.std(dim=0).abs().max().item()
        if not spread > atol:
            raise RuntimeError(
                f"committee heads are degenerate: max std across heads = {spread:g}. "
                "Every ForceSketch result would be vacuously exact. Check that the "
                "readout was built with distinct per-head instructions and that each "
                "head's training slice was non-empty."
            )
