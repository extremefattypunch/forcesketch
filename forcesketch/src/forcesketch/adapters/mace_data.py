"""extxyz -> MACE batches (plan Task 0.3/1.4).

Kept beside the MACE adapter because it is equally backbone-specific: nothing
outside `adapters/` should know what an AtomicData is.
"""

from __future__ import annotations

from pathlib import Path

import ase.io
import numpy as np
import torch


def load_frames(path: str | Path, *, limit: int | None = None) -> list:
    frames = ase.io.read(path, index=":", format="extxyz")
    return frames[:limit] if limit else frames


def reference_forces(frames: list) -> torch.Tensor:
    """[N_total, 3] DFT reference forces, concatenated in frame order."""
    key = "forces_ref" if "forces_ref" in frames[0].arrays else None
    out = [f.arrays[key] if key else f.get_forces() for f in frames]
    return torch.from_numpy(np.concatenate(out, axis=0))


def make_loader(frames: list, model, *, batch_size: int = 4, r_max: float | None = None,
                shuffle: bool = False):
    """Build a MACE dataloader whose z_table and heads come FROM THE MODEL, so a
    checkpoint can never be evaluated against a mismatched element table."""
    from mace import data
    from mace.tools import AtomicNumberTable, torch_geometric

    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cutoff = float(r_max if r_max is not None else model.r_max)
    heads = list(model.heads)
    configs = [data.config_from_atoms(a) for a in frames]
    dataset = [
        data.AtomicData.from_config(c, z_table=z_table, cutoff=cutoff, heads=heads)
        for c in configs
    ]
    return torch_geometric.dataloader.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
    )
