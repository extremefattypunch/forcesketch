#!/usr/bin/env python
"""Rebuild the three rMD17 head-force caches with reference energies.

The workshop caches for ethanol, aspirin and azobenzene (frozen tree) carry
per-head forces and reference forces but no energies, so the accuracy table
could not report an energy error for them. This script (1) prepares the test
frames exactly as the frozen `scripts/00_prepare_rmd17.py` did (authors'
shipped test indices, kcal/mol -> eV) and (2) evaluates the jointly trained
rMD17 disjoint committee on them with the same cache builder as the 3BPA
temperature ladder. The new caches are checked against the frozen ones: F and
f_ref must agree to float32-rounding, or the rebuild is rejected.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from forcesketch_journal.paths import CACHES, DATA, FROZEN, FROZEN_CACHES, MODELS, SCRATCH  # noqa: E402
sys.path.insert(0, str(FROZEN / "src"))

import ase  # noqa: E402
import ase.io  # noqa: E402
from ase.units import kcal, mol  # noqa: E402
from forcesketch.adapters.mace_data import load_frames  # noqa: E402
from forcesketch.adapters.mace_mhc import MaceMHCAdapter  # noqa: E402
from forcesketch.utils.reproducibility import checkpoint_hash, pin_numerics  # noqa: E402
from j5_3bpa_temperature_ladder import build_cache  # noqa: E402

KCAL_MOL_TO_EV = kcal / mol
ZEN = MODELS / "rMD17"
CKPT = ZEN / "full_trainset/multihead-disjoint/multihead_committee_stagetwo.model"
NPZ = SCRATCH / "rmd17"
XYZ = DATA / "data/rmd17"
E0S_EV = {1: -13.568422383046626, 6: -1025.2770951782686,
          7: -1479.0665594928669, 8: -2035.5709809589698}


def read_ids(name: str) -> np.ndarray:
    return np.array([int(x) for x in (ZEN / f"{name}.csv").read_text().strip().strip(",").split(",")])


def prepare(mol_name: str) -> pathlib.Path:
    out = XYZ / f"{mol_name}_test_ref.xyz"
    if out.exists():
        return out
    train_ids, val_ids, test_ids = read_ids("train"), read_ids("val"), read_ids("test")
    assert len(np.intersect1d(test_ids, train_ids)) == 0 and len(np.intersect1d(test_ids, val_ids)) == 0
    npz = np.load(NPZ / f"rmd17_{mol_name}.npz")
    z = npz["nuclear_charges"]
    coords = npz["coords"][test_ids]
    energies = npz["energies"][test_ids] * KCAL_MOL_TO_EV
    forces = npz["forces"][test_ids] * KCAL_MOL_TO_EV
    per_atom = (energies - sum(E0S_EV[int(zi)] for zi in z)) / len(z)
    assert np.abs(per_atom).mean() < 100, "unit conversion missing"
    frames = []
    for c, e, f in zip(coords, energies, forces):
        a = ase.Atoms(numbers=z, positions=c)
        a.info["energy_ref"] = float(e)
        a.arrays["forces_ref"] = np.asarray(f)
        frames.append(a)
    XYZ.mkdir(parents=True, exist_ok=True)
    ase.io.write(out, frames, format="extxyz")
    print(f"prepared {out} ({len(frames)} frames x {len(z)} atoms)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", nargs="+", default=["ethanol", "aspirin", "azobenzene"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--outdir", default=str(CACHES / "rmd17"))
    a = ap.parse_args()
    pin_numerics()
    outdir = pathlib.Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ad = MaceMHCAdapter.from_checkpoint(CKPT, device="cuda", dtype=torch.float64)
    chash = checkpoint_hash(ad.model)
    records = []
    for m in a.molecules:
        xyz = prepare(m)
        frames = load_frames(xyz)
        t0 = time.time()
        c = build_cache(ad, frames, batch_size=a.batch_size)
        dt = time.time() - t0
        tag = f"rmd17-disjoint_{m}"
        old = torch.load(FROZEN_CACHES / f"head_forces_{tag}.pt", weights_only=False, map_location="cpu")
        relF = float((old["F"].double() - c["F"]).abs().max() / c["F"].abs().max())
        relR = float((old["f_ref"].double() - c["f_ref"]).abs().max() / c["f_ref"].abs().max())
        if not (relF < 1e-5 and relR < 1e-10):
            raise SystemExit(f"{tag}: rebuilt cache disagrees with the frozen one (F {relF:.2e}, f_ref {relR:.2e})")
        delta = c["F"].mean(-1) - c["f_ref"]
        rec = {"experiment_id": "j5_rmd17_caches", "variant": "rmd17-disjoint", "split": m,
               "checkpoint_hash": chash, "n_structures": c["frames"], "num_atoms": c["A"],
               "force_rmse_mean_mev_A": float(delta.pow(2).mean().sqrt()) * 1000,
               "energy_mae_mev_atom": float(((c["E"].mean(-1) - c["E_ref"]) / c["A"]).abs().mean()) * 1000,
               "frozen_cache_rel_diff_F": relF, "frozen_cache_rel_diff_f_ref": relR, "seconds": dt}
        records.append(rec)
        torch.save({**c, "variant": "rmd17-disjoint", "split": m, "checkpoint_hash": chash},
                   outdir / f"head_forces_{tag}.pt")
        print(f"[{tag}] S={c['frames']} A={c['A']} F-RMSE {rec['force_rmse_mean_mev_A']:.1f} "
              f"E-MAE {rec['energy_mae_mev_atom']:.2f} meV/atom  (vs frozen: F {relF:.1e}) ({dt:.0f}s)")
    with (ROOT / "results/records/j5_rmd17_caches.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
