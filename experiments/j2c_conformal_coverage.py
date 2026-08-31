#!/usr/bin/env python
"""J2c -- realised finite-sample coverage of the calibrated gate's constant.

The manuscript states that "realised coverage is verified against the
Beta(k, n+1-k) law by simulation rather than asserted" and then reports no
number, so a reader cannot check the claim. This closes that: it produces the
record the macros derive from.

Two design choices worth stating.

First, the calibration-set sizes are READ FROM THE SPLIT MANIFESTS rather than
typed. `conformal.coverage_simulation` is exercised at the smallest and largest
n_cal the project actually uses, because the finite-sample guarantee is weakest
at the smallest calibration set and that is the binding case. Typing the range in
by hand is how the ~70 wrong claims in results/CONSISTENCY_AUDIT.md happened.

Second, doing so corrects the range. `calibration/conformal.py` says "the
smallest n_cal in this project is 92 (water, blocked split)"; the manifests say
84 (water-disjoint_md_T300K, contiguous_block). The docstring's *conclusion*
survives -- at the shipped alpha=0.05, k = ceil(85*0.95) = 81 <= 84, so every
split is still feasible and no published number was affected -- but the stated
minimum was wrong, and the feasibility margin is thinner than advertised.

Runs on CPU in seconds; no model execution, no CUDA, no download.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forcesketch_journal.calibration.conformal import (  # noqa: E402
    conformal_c,
    coverage_simulation,
)

MANIFESTS = ROOT / "manifests" / "splits"
OUT = ROOT / "results" / "records" / "j2c_conformal_coverage.json"


def calibration_sizes() -> dict[str, int]:
    """n_cal for every registered split, read from the manifests."""
    sizes: dict[str, int] = {}
    for p in sorted(MANIFESTS.glob("*.json")):
        d = json.loads(p.read_text())
        roles = d.get("roles", d)
        for key in ("cal", "calibration"):
            if isinstance(roles, dict) and key in roles:
                v = roles[key]
                sizes[p.stem] = len(v) if isinstance(v, list) else int(v)
                break
    return sizes


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sizes = calibration_sizes()
    if not sizes:
        print("no split manifests found", file=sys.stderr)
        return 1
    n_min, n_max = min(sizes.values()), max(sizes.values())
    who_min = min(sizes, key=lambda k: sizes[k])

    sims = {}
    for label, n in (("min", n_min), ("max", n_max)):
        sims[label] = coverage_simulation(n, args.alpha,
                                          n_trials=args.n_trials, seed=args.seed)

    # Feasibility of every registered split at the shipped alpha and at the
    # stricter one the module docstring calls out.
    import math
    feas = {}
    for a in (args.alpha, 0.01):
        infeasible = [s for s, n in sizes.items()
                      if math.ceil((n + 1) * (1 - a)) > n]
        feas[f"alpha_{a}"] = {"n_infeasible": len(infeasible),
                              "infeasible": sorted(infeasible)}

    # The +inf contract, exercised rather than asserted: an infeasible alpha must
    # make the gate skip nothing.
    import torch
    cal = torch.rand(n_min, dtype=torch.float64) + 0.5
    c_infeasible = conformal_c(cal, torch.ones_like(cal), 1.0 / (n_min + 2))

    rec = {
        "experiment_id": "j2c_conformal_coverage",
        "git_commit": git_commit(),
        "alpha": args.alpha,
        "n_trials": args.n_trials,
        "seed": args.seed,
        "n_cal_min": n_min,
        "n_cal_max": n_max,
        "n_cal_min_split": who_min,
        "n_splits": len(sizes),
        "coverage": sims,
        "feasibility": feas,
        "c_alpha_infeasible_is_inf": c_infeasible == float("inf"),
        "note": ("Calibration sizes read from manifests/splits, not typed. "
                 "Corrects calibration/conformal.py's stated minimum of 92: the "
                 "manifests give 84. Conclusion unchanged -- every split is "
                 "feasible at alpha=0.05."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2) + "\n")

    print(f"n_cal range {n_min}-{n_max} over {len(sizes)} splits "
          f"(min: {who_min})")
    for label, s in sims.items():
        print(f"  {label:3s} n_cal={s['n_cal']:4d}  realised={s['realised']:.4f}  "
              f"Beta(k,n+1-k) mean={s['beta_mean']:.4f}  k={s['k']}")
    for a, f in feas.items():
        print(f"  {a}: {f['n_infeasible']} of {len(sizes)} splits infeasible")
    print(f"  infeasible alpha returns +inf: {rec['c_alpha_infeasible_is_inf']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
