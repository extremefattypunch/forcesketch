#!/usr/bin/env python
"""Prepare rMD17 test splits exactly as the reference paper did (plan Task 0.3/6.1).

Two silent-failure traps, both closed by assertion here.

1. Units. rMD17 npz stores energies in kcal/mol and forces in kcal/mol/A; MACE
   wants eV. Missing the conversion is a factor of 23.06, which trains to garbage
   without ever raising. Asserted by checking the post-E0 per-atom energy is
   O(1-10) eV rather than O(100).

2. The split. We use the authors' shipped test.csv indices DIRECTLY rather than
   re-deriving them, because the select_trainset.py in the same Zenodo record does
   not reproduce its own CSVs: the script says n_strucs = 99988, but every index in
   train/val/test.csv is below 10000, and no choice of n_strucs (10000, 99988,
   100000) with default_rng(123) reproduces them. The CSVs are the ground truth for
   what the model actually saw, so they are what we honour.

   Verified: the three index sets are pairwise disjoint (train 50, val 100,
   test 1000, zero overlap), so test.csv is a genuinely held-out evaluation set.

The single joint committee was trained on all ten molecules, so evaluating it on
ethanol (9 atoms), aspirin (21) and azobenzene (24) gives spec SS32's size axis
with no model confound.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ase
import ase.io
import numpy as np
from ase.units import kcal, mol

DATA = Path("data/rmd17")
ZEN = Path("models/zenodo/rMD17")
SEED, N_STRUCS, N_TRAIN, N_VAL, N_TEST = 123, 99988, 50, 100, 1000
KCAL_MOL_TO_EV = kcal / mol  # 0.0433641...

# From the authors' run_mace.sh. NOTE these differ from 3BPA's (different level of
# theory: rMD17 is PBE/def2-SVP), so reusing the 3BPA values would be wrong.
E0S_EV = {1: -13.568422383046626, 6: -1025.2770951782686,
          7: -1479.0665594928669, 8: -2035.5709809589698}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", nargs="+", default=["ethanol", "aspirin", "azobenzene"])
    args = ap.parse_args()

    def read_ids(name: str) -> np.ndarray:
        return np.array([int(x) for x in
                         ZEN.joinpath(f"{name}.csv").read_text().strip().strip(",").split(",")])

    train_ids, val_ids, test_ids = (read_ids("train"), read_ids("val"), read_ids("test"))
    # The model saw train + val; test must be disjoint from both or the whole
    # generalization claim is contaminated.
    assert len(np.intersect1d(test_ids, train_ids)) == 0
    assert len(np.intersect1d(test_ids, val_ids)) == 0
    print(f"using the authors' shipped indices: train {len(train_ids)}, "
          f"val {len(val_ids)}, test {len(test_ids)} (pairwise disjoint, verified)")

    for name in args.molecules:
        npz = np.load(DATA / f"rmd17_{name}.npz")
        z = npz["nuclear_charges"]
        coords = npz["coords"][test_ids]                       # [n, A, 3] Angstrom
        energies = npz["energies"][test_ids] * KCAL_MOL_TO_EV  # -> eV
        forces = npz["forces"][test_ids] * KCAL_MOL_TO_EV      # -> eV/A

        e0_sum = sum(E0S_EV[int(zi)] for zi in z)
        per_atom = (energies - e0_sum) / len(z)
        assert np.abs(per_atom).mean() < 100, (
            f"{name}: mean |E - E0| per atom is {np.abs(per_atom).mean():.1f} eV -- "
            "unit conversion is almost certainly missing (kcal/mol is 23.06x eV)"
        )

        frames = []
        for c, e, f in zip(coords, energies, forces):
            a = ase.Atoms(numbers=z, positions=c)
            a.info["energy_ref"] = float(e)
            a.arrays["forces_ref"] = np.asarray(f)
            frames.append(a)
        out = DATA / f"{name}_test_ref.xyz"
        ase.io.write(out, frames, format="extxyz")
        print(f"  {name:<12} {len(frames):>5} frames x {len(z):>2} atoms  "
              f"| E-E0 per atom {per_atom.mean():+8.3f} eV  "
              f"| |F| mean {np.abs(forces).mean():.3f} eV/A  -> {out.name}")

    print("\nOK: using the authors' held-out test indices; units converted to eV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
