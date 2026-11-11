#!/usr/bin/env python
"""J12b -- per-atom error-versus-uncertainty data for the calibration figure.

Reads the head-force caches and writes a small record of subsampled per-atom
points (committee std vs. error norm) plus equal-count reliability bins for a
fixed panel of systems, so that tools/make_figures.py draws from a committed
record rather than from 400 MB of caches.
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
from forcesketch_journal.paths import CACHES, NAIVE_CACHES  # noqa: E402

PANEL = {  # label -> cache file
    "3BPA 1200 K, disjoint heads": CACHES / "head_forces_3bpa-disjoint_test_1200K.pt",
    "3BPA 1200 K, identical data": CACHES / "head_forces_3bpa-same_test_1200K.pt",
    "3BPA 1200 K, independent": NAIVE_CACHES / "head_forces_3bpa-naive-same_test_1200K.pt",
    "water PIMD, overlapping heads": CACHES / "head_forces_water-overlapping_pimd_T300K.pt",
    "water PIMD, independent": NAIVE_CACHES / "head_forces_water-naive_pimd_T300K.pt",
    "water MD, disjoint heads": CACHES / "head_forces_water-disjoint_md_T300K.pt",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--out", default="results/records/j12_calibration_points.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = {"experiment_id": "j12_calibration_data", "seed": a.seed, "systems": {}}
    for label, path in PANEL.items():
        d = torch.load(path, weights_only=False, map_location="cpu")
        F, fref = d["F"].double(), d["f_ref"].double()
        sig_a = (F.std(dim=-1, unbiased=True).mean(-1)).flatten().numpy() * 1000       # meV/A
        err_a = ((F.mean(-1) - fref).norm(dim=-1)).flatten().numpy() * 1000 / np.sqrt(3)
        order = np.argsort(sig_a)
        bins = [{"sigma_mean": float(sig_a[c].mean()), "rmse": float(np.sqrt(np.mean(err_a[c] ** 2))),
                 "n": int(c.size)} for c in np.array_split(order, 20)]
        pick = rng.choice(sig_a.size, size=min(a.n_points, sig_a.size), replace=False)
        out["systems"][label] = {"cache": path.name, "n_atoms_total": int(sig_a.size),
                                 "sigma": np.round(sig_a[pick], 3).tolist(),
                                 "error": np.round(err_a[pick], 3).tolist(),
                                 "reliability": bins,
                                 "spearman": float(np.corrcoef(np.argsort(np.argsort(sig_a)), np.argsort(np.argsort(err_a)))[0, 1])}
        print(f"{label:<32} n={sig_a.size:>8d} rho={out['systems'][label]['spearman']:.2f}")
    (ROOT / a.out).write_text(json.dumps(out) + "\n")
    print(f"wrote {a.out} ({(ROOT / a.out).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
