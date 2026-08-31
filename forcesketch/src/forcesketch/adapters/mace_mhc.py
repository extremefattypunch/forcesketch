"""MACE multi-head-committee adapter (spec SS18).

The ONLY module in the project permitted to import mace/e3nn/ase. Everything
downstream sees `energies(batch) -> [B, M]` and nothing else, which is what makes
a multi-head TensorNet2 a drop-in replacement.

Note what this adapter deliberately does NOT use: MACE's own
`get_outputs_committee`, which loops M sequential reverse passes. That loop is
the spec SS26 item-1 baseline (and is exposed here as `member_forces_reference`
for exactly that purpose), but ForceSketch computes its own VJPs from head-space
cotangents, which is the entire point.
"""

from __future__ import annotations

from pathlib import Path

import e3nn
import torch
from torch import Tensor

from forcesketch.adapters.base import BaseMHCAdapter


def configure_e3nn_for_batched_vjp() -> None:
    """Make `torch.autograd.grad(..., is_grads_batched=True)` usable with MACE.

    MUST run before the first forward pass of any e3nn/MACE model in the process:
    TorchScript caches an optimized plan per graph, and once one exists these flags
    have no effect. That is why this is called at import time of this module.

    Root cause. `is_grads_batched=True` runs the backward under
    `torch._vmap_internals._vmap`, so cotangents are BatchedTensors with no storage.
    Independently, e3nn's `_spherical_harmonics` is a module-level
    `@torch.jit.script` FREE FUNCTION -- not a Module, which is why counting
    `torch.jit.ScriptModule` instances reports zero and is misleading -- sitting
    directly on the positions -> energy path. TorchScript's profiling executor emits
    an optimized plan only after two warm-up executions; that plan wraps the body in
    a `prim::DifferentiableGraph`, and the TensorExpr (NNC) fuser then fuses the
    REVERSE graph into a `prim::TensorExprGroup`. `TensorExprKernel` requires a raw
    `data_ptr()` for every input, which a BatchedTensor cannot provide. Hence the
    characteristic pattern: calls 0 and 1 succeed, call 2 raises
    "Cannot access data pointer of Tensor that doesn't have storage".

    Disabling the fuser is therefore the fix, and it works on the pickled checkpoint
    as shipped -- verified 20/20 consecutive batched calls matching the serial path
    to 1.8e-15. Note `torch._C._jit_set_profiling_executor(False)` makes it WORSE,
    because the legacy executor builds the fused graph on call 1 instead of call 2.

    `e3nn.set_optimization_defaults(jit_script_fx=False)` is kept because it is
    harmless and reduces codegen, but it is NOT what fixes this: e3nn's
    `CodeGenMixin.__setstate__` unconditionally calls `torch.jit.load`, so a pickled
    checkpoint always restores 18 RecursiveScriptModules regardless of the flag.
    """
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    e3nn.set_optimization_defaults(jit_script_fx=False)


configure_e3nn_for_batched_vjp()


class MaceMHCAdapter(BaseMHCAdapter):
    def __init__(self, model, *, device="cuda", dtype: torch.dtype = torch.float32):
        self.model = model.to(device=device, dtype=dtype).eval()
        self._device = torch.device(device)
        self._dtype = dtype
        self._heads = list(model.heads)
        self._committee = torch.tensor(
            [i for i, h in enumerate(self._heads) if "committee-" in h],
            dtype=torch.long, device=self._device,
        )
        if len(self._committee) == 0:  # a committee-less checkpoint: use all heads
            self._committee = torch.arange(len(self._heads), device=self._device)

    # --- construction -----------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device="cuda",
                        dtype: torch.dtype = torch.float32,
                        expect_sha256: str | None = None) -> "MaceMHCAdapter":
        """Load a pickled ScaleShiftMACE.

        `weights_only=False` is unavoidable: these checkpoints are pickled
        nn.Modules carrying embedded e3nn TorchScript codegen, not plain tensors.
        Mitigate by verifying the published digest BEFORE unpickling -- pass
        `expect_sha256`. We only ever load artifacts from Zenodo record 17829635.
        """
        path = Path(path)
        if expect_sha256 is not None:
            from forcesketch.utils.fetch import sha256_file

            got = sha256_file(path)
            if got != expect_sha256:
                raise RuntimeError(f"{path.name}: sha256 {got} != expected {expect_sha256}")
        model = torch.load(path, map_location="cpu", weights_only=False)  # noqa: S614
        if not hasattr(model, "heads"):
            raise TypeError(f"{path} is not a multi-head MACE model")
        return cls(model, device=device, dtype=dtype)

    # --- MHCAdapter surface ------------------------------------------------
    @property
    def num_heads(self) -> int:
        return len(self._committee)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def prepare(self, batch: dict) -> dict:
        batch = {
            k: (v.to(self._device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        for k, v in batch.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                batch[k] = v.to(self._dtype)
        batch["positions"] = batch["positions"].detach().requires_grad_(True)
        return batch

    def num_structures(self, batch: dict) -> int:
        return int(batch["batch"].max().item()) + 1

    def batch_index(self, batch: dict) -> Tensor:
        return batch["batch"]

    def positions(self, batch: dict) -> Tensor:
        return batch["positions"]

    def energies(self, batch: dict, *, create_graph: bool = False) -> Tensor:
        """[B, M] committee head energies."""
        out = self.model(batch, training=False, compute_force=False,
                         committee_heads=self._committee)
        heads = out.get("heads")
        if heads is None or heads.get("energy") is None:
            raise RuntimeError(
                "model returned heads=None. The pinned fork hardcodes loss='dpose' in "
                "ScaleShiftMACE.forward, which makes the committee branch unreachable. "
                "Apply environment/mace-fork.patch."
            )
        return heads["energy"][:, self._committee]

    # --- spec SS26 item-1 baseline, as shipped by the reference paper -------
    def member_forces_reference(self, batch: dict) -> Tensor:
        """M sequential reverse passes -- MACE's own `get_outputs_committee` loop,
        which is exactly spec SS26's 'explicit per-head force loop'. Returns
        [M, N_total, 3]."""
        E = self.energies(batch)
        pos = self.positions(batch)
        return torch.stack(
            [-torch.autograd.grad(E[:, m].sum(), pos, retain_graph=True)[0]
             for m in range(self.num_heads)]
        )
