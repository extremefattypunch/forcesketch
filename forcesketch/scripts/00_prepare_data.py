#!/usr/bin/env python
"""Prepare 3BPA splits exactly as the reference paper did (plan Task 0.3).

Two things here are easy to get wrong and both fail SILENTLY.

1. The key rename. Raw 3BPA carries `energy=` and a `forces` array, but the
   authors' run_mace.sh passes --energy_key=energy_ref --forces_key=forces_ref.
   MACE's config_from_atoms does `atoms.info.get(energy_key, None)` with NO
   fallback, so training against the raw files would silently see energy=None on
   every config and fit forces only.

2. The split. Reproduced from the authors' own select_trainsets.py
   (numpy default_rng(123), shuffle, sort) and then CHECKED against the
   train_100.csv / val.csv index lists shipped in the Zenodo record. If the
   indices do not match, the model we evaluate was not trained on the data we
   think it was.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ase.io
import numpy as np

DATA = Path("data/3bpa")
ZEN = Path("models/zenodo/3BPA")
SEED, N_TRAIN, N_VAL = 123, 100, 300


def add_ref_keys(frames: list) -> list:
    """energy -> info['energy_ref'], forces -> arrays['forces_ref']."""
    for a in frames:
        a.info["energy_ref"] = float(a.get_potential_energy())
        a.arrays["forces_ref"] = a.get_forces().copy()
        a.calc = None  # stop ase re-emitting energy=/forces on write
    return frames


def read_csv_ids(path: Path) -> np.ndarray:
    return np.array([int(x) for x in path.read_text().strip().strip(",").split(",")])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    print("reading raw 3BPA ...")
    train_all = ase.io.read(DATA / "train_300K.xyz", index=":", format="extxyz")
    assert len(train_all) == 500, f"expected 500 train configs, got {len(train_all)}"

    # --- reproduce the authors' split -------------------------------------
    ids = np.arange(len(train_all))
    np.random.default_rng(SEED).shuffle(ids)
    train_ids = np.sort(ids[:N_TRAIN])
    val_ids = np.sort(ids[N_TRAIN:N_TRAIN + N_VAL])

    ref_train = read_csv_ids(ZEN / "train_100.csv")
    ref_val = read_csv_ids(ZEN / "val.csv")
    ok_t = np.array_equal(train_ids, ref_train)
    ok_v = np.array_equal(val_ids, ref_val)
    print(f"  train split matches authors' train_100.csv : {ok_t}  ({len(train_ids)} ids)")
    print(f"  val   split matches authors' val.csv       : {ok_v}  ({len(val_ids)} ids)")
    if not (ok_t and ok_v):
        print(f"    ours[:8]    {train_ids[:8]}")
        print(f"    authors[:8] {ref_train[:8]}")
        raise SystemExit("split reproduction FAILED -- do not proceed")
    if args.check_only:
        return 0

    # --- write the _ref-keyed files ---------------------------------------
    for name, frames in [
        ("train_300K_100.xyz", [train_all[i] for i in train_ids]),
        ("val_300K_300.xyz", [train_all[i] for i in val_ids]),
    ]:
        ase.io.write(DATA / name, add_ref_keys(frames), format="extxyz")
        print(f"  wrote {name}  ({len(frames)} frames)")

    for split in ("test_300K", "test_600K", "test_1200K"):
        frames = ase.io.read(DATA / f"{split}.xyz", index=":", format="extxyz")
        ase.io.write(DATA / f"{split}_ref.xyz", add_ref_keys(frames), format="extxyz")
        print(f"  wrote {split}_ref.xyz  ({len(frames)} frames)")

    # --- assert the trap is actually closed --------------------------------
    check = ase.io.read(DATA / "train_300K_100.xyz", index=":", format="extxyz")
    n_e = sum("energy_ref" in a.info for a in check)
    n_f = sum("forces_ref" in a.arrays for a in check)
    print(f"\n  energy_ref present on {n_e}/{len(check)} frames")
    print(f"  forces_ref present on {n_f}/{len(check)} frames")
    assert n_e == n_f == len(check), "key rename incomplete -- MACE would train on forces only"
    print("\nOK: splits reproduce the authors' indices and both _ref keys are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
