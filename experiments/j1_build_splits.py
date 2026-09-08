#!/usr/bin/env python
"""J1 -- build and hash the split manifests, and measure what blocking costs.

Two schemes are generated for every system: `contiguous_block` (the defensible
one for trajectory data) and `random_frame` (what the workshop used). Reporting
both turns "your splits leak" from a reviewer's objection into a measurement.

Guard bands come from the measured integrated autocorrelation time of the
primary exact acquisition score along the cache's row order, which for these
caches is genuine trajectory time order: 3BPA rows follow `test_1200K.xyz` file
order, and rMD17 row i is trajectory frame `test_ids[i]` where `test.csv` is
sorted ascending (verified).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from forcesketch_journal.data.splits import (  # noqa: E402
    alpha_feasible, contiguous_block_split, coverage_law, random_frame_split, write_manifest,
)
from forcesketch_journal.evaluation.autocorr import block_length, integrated_autocorr_time  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLIT_SEED = 20260901  # configs/seeds.yaml: calibration_split_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.10, 0.05, 0.01])
    args = ap.parse_args()

    out = ROOT / "manifests/splits"
    summary = []
    for p in sorted((ROOT / "results/perstructure/j2a").glob("*.pt")):
        d = torch.load(p, weights_only=True)
        tag = d["tag"]
        score = d["sig_exact_maxcomp"].numpy()
        ac = integrated_autocorr_time(score)
        guard = block_length(ac["tau_int"])
        n = len(score)

        blocked = contiguous_block_split(n, system=tag, seed=SPLIT_SEED,
                                         tau_int=ac["tau_int"], guard=guard)
        randomf = random_frame_split(n, system=tag, seed=SPLIT_SEED)
        for s in (blocked, randomf):
            s.assert_valid()
            write_manifest(s, out / f"{tag}__{s.scheme}.json",
                           split_seed=SPLIT_SEED, autocorr=ac,
                           frame_index_source=("file_order" if "1200K" in tag
                                               else "rmd17_test_csv_ascending"))

        power = {f"alpha_{a}": coverage_law(len(blocked.cal), a) for a in args.alphas}
        summary.append({
            "system": tag, "n": n, "tau_int": ac["tau_int"], "rho_1": ac["rho_1"],
            "time_ordered": ac["time_ordered"], "guard_frames": guard,
            "blocked": {"n_design": len(blocked.design), "n_cal": len(blocked.cal),
                        "n_test": len(blocked.test), "n_dropped": len(blocked.dropped),
                        "sha256": blocked.sha256[:12]},
            "random": {"n_design": len(randomf.design), "n_cal": len(randomf.cal),
                       "n_test": len(randomf.test), "sha256": randomf.sha256[:12]},
            "calibration_power": power,
        })

    (ROOT / "results/records").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/records/j1_splits.json").write_text(json.dumps(summary, indent=1))

    print(f"{'system':26s}{'n':>6s}{'tau':>6s}{'grd':>5s}{'design':>8s}{'cal':>6s}"
          f"{'test':>6s}{'drop':>6s}   sha256")
    print("-" * 88)
    for s in summary:
        b = s["blocked"]
        print(f"{s['system']:26s}{s['n']:6d}{s['tau_int']:6.2f}{s['guard_frames']:5d}"
              f"{b['n_design']:8d}{b['n_cal']:6d}{b['n_test']:6d}{b['n_dropped']:6d}   {b['sha256']}")

    print(f"\nCalibration power (blocked split):")
    print(f"{'system':26s}{'n_cal':>7s}" + "".join(f"{'a=%.2f' % a:>22s}" for a in args.alphas))
    print("-" * (33 + 22 * len(args.alphas)))
    for s in summary:
        row = f"{s['system']:26s}{s['blocked']['n_cal']:7d}"
        for a in args.alphas:
            law = s["calibration_power"][f"alpha_{a}"]
            row += (f"{'INFEASIBLE':>22s}" if not law["feasible"]
                    else f"{law['mean']:.3f} [{law['q05']:.3f},{law['q95']:.3f}]".rjust(22))
        print(row)
    print(f"\nwrote {2*len(summary)} manifests to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
