#!/usr/bin/env python
"""Score the active-learning round (J11) against protocols/j11_active_learning.yaml.

Reads results/records/j11/j11_{rule}_s{seed}.json and writes
results/records/j11_verdict.json with per-rule means and across-seed standard
deviations on each evaluation set, plus the five pre-registered predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
RULES = ["none", "global", "maxcomp", "maxatom", "force_norm", "energy_std", "random"]
SETS = ["heldout_1200K", "test_600K", "test_300K"]
METRICS = ["force_rmse_mev_A", "force_mae_mev_A", "energy_mae_mev_atom", "e_max_p95_mev_A"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="protocols/j11_active_learning.yaml")
    ap.add_argument("--out", default="results/records/j11_verdict.json")
    a = ap.parse_args()
    pp = ROOT / a.protocol
    got = hashlib.sha256(pp.read_bytes()).hexdigest()
    reg = json.loads((ROOT / "results/records/j11_protocol_registration.json").read_text())
    if got != reg["protocol_sha256"]:
        raise SystemExit("REFUSING TO SCORE: protocol changed since registration")
    proto = yaml.safe_load(pp.read_text())
    recs = {}
    for p in sorted((ROOT / "results/records/j11").glob("j11_*_s*.json")):
        r = json.loads(p.read_text()); recs[(r["selector"], r["seed"])] = r
    table = {}
    for rule in RULES:
        seeds = sorted(s for (r, s) in recs if r == rule)
        if not seeds:
            continue
        table[rule] = {"seeds": seeds, "n_train": recs[(rule, seeds[0])]["n_train"]}
        for st in SETS:
            for m in METRICS:
                v = np.array([recs[(rule, s)]["eval"][st][m] for s in seeds])
                table[rule][f"{st}/{m}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                                            "per_seed": v.tolist()}
        print(f"{rule:<11} n_train={table[rule]['n_train']:3d}  "
              + "  ".join(f"{st[:8]} F-RMSE {table[rule][f'{st}/force_rmse_mev_A']['mean']:6.1f}" for st in SETS))
    key = "heldout_1200K/force_rmse_mev_A"
    mean = {r: table[r][key]["mean"] for r in table}
    sd = {r: table[r][key]["sd"] for r in table}
    passed, detail = {}, {}
    if "none" in mean:
        acq = [r for r in table if r != "none"]
        passed["P1"] = all(mean[r] < mean["none"] for r in acq); detail["P1"] = {"none": mean["none"], **{r: mean[r] for r in acq}}
    if "random" in mean:
        informed = [r for r in ("global", "maxcomp", "maxatom", "force_norm") if r in mean]
        passed["P2"] = all(mean[r] < mean["random"] for r in informed); detail["P2"] = {"random": mean["random"], **{r: mean[r] for r in informed}}
    comm = [r for r in ("global", "maxcomp", "maxatom") if r in mean]
    if comm and "force_norm" in mean:
        best = min(comm, key=lambda r: mean[r])
        rel = (mean["force_norm"] - mean[best]) / mean[best]
        passed["P3"] = rel <= 0.10; detail["P3"] = {"best_committee_rule": best, "relative_gap": rel}
    if "global" in mean and "maxcomp" in mean:
        gap = abs(mean["global"] - mean["maxcomp"]); tol = max(sd["global"] or 0, sd["maxcomp"] or 0)
        passed["P4"] = gap < tol; detail["P4"] = {"gap": gap, "max_sd": tol}
    key3 = "test_300K/force_rmse_mev_A"
    if "none" in table:
        rel3 = {r: (table["none"][key3]["mean"] - table[r][key3]["mean"]) / table["none"][key3]["mean"] for r in table if r != "none"}
        passed["P5"] = all(v <= 0.05 for v in rel3.values()); detail["P5"] = rel3
    out = {"protocol_sha256": got, "protocol_version": proto["version"], "passed": passed,
           "n_passed": int(sum(passed.values())), "n_scored": len(passed), "detail": detail,
           "table": table, "n_heldout": json.loads((ROOT / "results/records/j11/selection_summary.json").read_text())["n_heldout"]
           if (ROOT / "results/records/j11/selection_summary.json").exists() else None}
    (ROOT / a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(passed), "->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
