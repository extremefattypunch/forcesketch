"""Reference labels from extxyz frames, in one place.

The 3BPA ``*_ref.xyz`` and water files carry ``energy_ref`` / ``forces_ref``;
the plain 3BPA ``test_*.xyz`` and the MPtraj/OMat24 pools carry ``energy`` /
``forces``, which ASE parks on a SinglePointCalculator. Both conventions are
read here so that no cache builder has to know which it was given.
"""

from __future__ import annotations

import numpy as np
import torch


def reference_energy(atoms) -> float:
    if "energy_ref" in atoms.info:
        return float(atoms.info["energy_ref"])
    if "REF_energy" in atoms.info:
        return float(atoms.info["REF_energy"])
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    return float(atoms.get_potential_energy())


def reference_energies(frames: list) -> torch.Tensor:
    """[S] reference total energies in eV, frame order."""
    return torch.tensor([reference_energy(a) for a in frames], dtype=torch.float64)


def species(frames: list) -> torch.Tensor:
    """[S, A] atomic numbers (fixed-size systems only)."""
    return torch.from_numpy(np.stack([a.numbers for a in frames])).to(torch.int64)
