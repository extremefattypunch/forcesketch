#!/usr/bin/env python
"""J12a -- model accuracy and uncertainty calibration for every committee.

Both referees of the first submission asked for the numbers a reader of an
uncertainty paper expects before any ranking metric: energy and force errors of
the models, and how the committee spread compares with the error it is meant to
predict. Nothing here is a new experiment; it is read off the per-head force
caches (and, for the materials pools, the per-structure scan records).

Per system this writes
  force_rmse / force_mae          of the committee-mean force, meV/A
  force_rmse_head_mean            mean over heads of the per-head RMSE
  energy_mae_atom / energy_rmse_atom  of the committee-mean energy, meV/atom
  disagreement_median             median over atoms of the per-atom head std
  overconfidence                  force_rmse / disagreement_median (as in J6)
  z_var                           variance of z = error / sigma per component
                                  (1 for a calibrated Gaussian; >1 overconfident)
  spearman_atom                   Spearman rho of per-atom |error| vs per-atom sigma
  spearman_struct                 Spearman rho of per-structure e_max vs max-atom sigma
  reliability                     equal-count bins of per-atom sigma: (mean sigma, RMSE)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from forcesketch_journal.paths import CACHES, FROZEN_CACHES, NAIVE_CACHES  # noqa: E402

EV_TO_MEV = 1000.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def reliability(sigma: np.ndarray, err: np.ndarray, nbins: int = 10) -> list[dict]:
    order = np.argsort(sigma)
    out = []
    for chunk in np.array_split(order, nbins):
        out.append({"sigma_mean": float(sigma[chunk].mean()),
                    "rmse": float(np.sqrt(np.mean(err[chunk] ** 2))),
                    "mae": float(np.abs(err[chunk]).mean()), "n": int(chunk.size)})
    return out


def from_cache(path: pathlib.Path) -> dict:
    d = torch.load(path, weights_only=False, map_location="cpu")
    F, fref = d["F"].double(), d["f_ref"].double()                     # [S,A,3,M], [S,A,3]
    S, A, _, M = F.shape
    fbar = F.mean(-1)
    delta = fbar - fref                                               # [S,A,3]
    sig_c = F.std(dim=-1, unbiased=True)                              # [S,A,3] per component
    sig_a = sig_c.mean(-1)                                            # [S,A] per atom (Beck et al. Eq. 4)
    err_a = delta.norm(dim=-1)                                        # [S,A]
    row = {"tag": path.stem.replace("head_forces_", ""), "variant": d.get("variant"),
           "split": d.get("split"), "n_structures": S, "n_atoms": A, "n_heads": M,
           "construction": "independent" if "naive" in path.stem else "shared-trunk",
           "force_rmse": float(delta.pow(2).mean().sqrt()) * EV_TO_MEV,
           "force_mae": float(delta.abs().mean()) * EV_TO_MEV,
           "force_rmse_head_mean": float((F - fref[..., None]).pow(2).mean(dim=(0, 1, 2)).sqrt().mean()) * EV_TO_MEV,
           "disagreement_median": float(sig_a.median()) * EV_TO_MEV,
           "z_var": float((delta / sig_c.clamp(min=1e-12)).pow(2).mean()),
           "z_var_atom": float((err_a / (sig_a.clamp(min=1e-12) * np.sqrt(3))).pow(2).mean()),
           "spearman_atom": spearman(sig_a.flatten().numpy(), err_a.flatten().numpy()),
           "spearman_struct": spearman(sig_a.max(dim=1).values.numpy(), err_a.max(dim=1).values.numpy()),
           "reliability_atom": reliability(sig_a.flatten().numpy() * EV_TO_MEV,
                                           err_a.flatten().numpy() * EV_TO_MEV / np.sqrt(3))}
    row["overconfidence"] = row["force_rmse"] / row["disagreement_median"]
    if "E_ref" in d and "E" in d:
        E, Eref = d["E"].double(), d["E_ref"].double()
        de = (E.mean(-1) - Eref) / A
        row["energy_mae_atom"] = float(de.abs().mean()) * EV_TO_MEV
        row["energy_rmse_atom"] = float(de.pow(2).mean().sqrt()) * EV_TO_MEV
        row["energy_std_median_atom"] = float(E.std(dim=-1, unbiased=True).median() / A) * EV_TO_MEV
    return row


def from_scan(path: pathlib.Path) -> dict:
    """Materials pools: per-structure scalars only (ragged structures)."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    n = z["natoms"].astype(float)
    e_rmse, e_max, sig_ma = z["e_rmse"], z["e_max"], z["exact_maxatom"]
    row = {"tag": path.stem.replace("_perstructure", ""), "variant": meta.get("model"),
           "split": meta.get("subset", "mptraj_pool"), "n_structures": int(n.size),
           "n_atoms_median": float(np.median(n)), "n_heads": 8, "construction": "foundation readouts",
           "force_rmse": float(np.sqrt(np.sum(e_rmse ** 2 * 3 * n) / np.sum(3 * n))) * EV_TO_MEV,
           "spearman_struct": spearman(sig_ma, e_max),
           "reliability_struct": reliability(sig_ma * EV_TO_MEV, e_max * EV_TO_MEV),
           "ref_force_max_median": float(np.median(z["ref_force_max"])) * EV_TO_MEV,
           "frac_ref_force_max_below_0.05": float(np.mean(z["ref_force_max"] < 0.05)),
           "frac_single_atom": float(np.mean(n == 1))}
    if "E_ref" in z.files and "E_mean" in z.files:
        de = (z["E_mean"] - z["E_ref"]) / n
        row["energy_mae_atom"] = float(np.abs(de).mean()) * EV_TO_MEV
        row["energy_rmse_atom"] = float(np.sqrt(np.mean(de ** 2))) * EV_TO_MEV
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/records/j12_accuracy.json")
    a = ap.parse_args()
    rows = []
    seen = set()
    for d in (CACHES, NAIVE_CACHES, FROZEN_CACHES):
        for p in sorted(d.glob("head_forces_*.pt")):
            tag = p.stem
            if tag in seen:
                continue
            seen.add(tag)
            rows.append(from_cache(p))
            print(f"{rows[-1]['tag']:<34} F-RMSE {rows[-1]['force_rmse']:7.1f}  "
                  f"E-MAE {rows[-1].get('energy_mae_atom', float('nan')):6.2f}  "
                  f"overconf {rows[-1]['overconfidence']:5.1f}x  z_var {rows[-1]['z_var']:7.1f}  "
                  f"rho_atom {rows[-1]['spearman_atom']:.2f}")
    for p in sorted((ROOT / "results/records").glob("j6b_foundation_scan_perstructure.npz")) + \
             sorted((ROOT / "results/records/j10").glob("j10_*_perstructure.npz")):
        if "smoke" in p.name:
            continue
        rows.append(from_scan(p))
        print(f"{rows[-1]['tag']:<34} F-RMSE {rows[-1]['force_rmse']:7.1f}  "
              f"E-MAE {rows[-1].get('energy_mae_atom', float('nan')):6.2f}  rho_struct {rows[-1]['spearman_struct']:.2f}")
    (ROOT / a.out).write_text(json.dumps({"experiment_id": "j12_accuracy", "rows": rows}, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
