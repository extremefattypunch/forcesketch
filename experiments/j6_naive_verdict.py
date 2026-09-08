#!/usr/bin/env python
"""Score the naive-committee results against `protocols/j6_naive_committee.yaml`.

Same discipline as the J8 scorer after its rewrite: this file knows nothing about
what the answers ought to be. Thresholds, comparisons and the required pass count
all come from the protocol, and the protocol's sha256 is checked against the
registration record so a prediction cannot be edited after the fact.

The one quantity computed here rather than read from a record is the
overconfidence ratio, and it is computed for BOTH sides of every pair by the same
function -- a matched comparison whose two halves came from different definitions
would be worthless, and that is exactly the class of defect this project has hit
before.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
CACHE_DIRS = [ROOT / "results/processed/naive", ROOT / "results/processed",
              FROZEN / "results/processed"]
EV_TO_MEV = 1000.0


def find_cache(tag: str) -> pathlib.Path:
    for d in CACHE_DIRS:
        p = d / f"head_forces_{tag}.pt"
        if p.exists():
            return p
    raise SystemExit(f"no cache for {tag} in {[str(d) for d in CACHE_DIRS]}")


def overconfidence(tag: str) -> dict:
    """committee force RMSE / median per-atom disagreement, both in meV/A.

    Identical code for naive and shared-trunk systems by construction.
    """
    c = torch.load(find_cache(tag), map_location="cpu", weights_only=False)
    F, fref = c["F"].double(), c["f_ref"].double()
    rmse = float((F.mean(-1) - fref).pow(2).mean().sqrt()) * EV_TO_MEV
    disag = float(F.std(dim=-1, unbiased=True).mean(dim=-1).median()) * EV_TO_MEV
    return {"force_rmse_mev_A": rmse, "median_disagreement_mev_A": disag,
            "overconfidence_ratio": rmse / disag}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="protocols/j6_naive_committee.yaml")
    ap.add_argument("--out", default="results/records/j6_naive_verdict.json")
    a = ap.parse_args()

    proto_path = ROOT / a.protocol
    proto = yaml.safe_load(proto_path.read_text())
    got = hashlib.sha256(proto_path.read_bytes()).hexdigest()
    reg = json.loads((ROOT / "results/records/j6_protocol_registration.json").read_text())
    if got != reg["protocol_sha256"]:
        print(f"REFUSING TO SCORE: {proto_path.name} has changed since registration\n"
              f"  registered {reg['protocol_sha256']}\n  now        {got}")
        return 2
    print(f"protocol : {proto_path.name} v{proto['version']} ({proto['status']})")
    print(f"sha256   : {got[:16]}...  registered {reg['registered_utc']}\n")

    thr = {c["metric"]: c for p in proto["predictions"] for c in p["criteria"]}

    spec_n = {d["system"]: d for d in json.loads(
        (ROOT / "results/records/j6_naive_spectrum.json").read_text())}
    spec_s = {d["system"]: d for d in json.loads(
        (ROOT / "results/records/j5_spectrum.json").read_text())}
    lo = {d["system"]: d for d in json.loads(
        (ROOT / "results/records/j6_naive_leading_only_L5.json").read_text())}
    panel = collections.defaultdict(dict)
    for r in (json.loads(l) for l in
              (ROOT / "results/records/j6_naive_oracle_panel.jsonl").open()):
        panel[(r["cache_tag"], r["error_score"])][r["signal"]] = r

    def resolve(name: str, universe) -> str:
        """Protocol name -> cache tag.

        The registered protocol names two 3BPA systems by variant alone
        (`3bpa-naive-same`) while the caches carry the split too
        (`3bpa-naive-same_test_1200K`). The protocol is hash-frozen, so the
        resolution happens here; it is required to be UNAMBIGUOUS, because
        silently picking one of several matching splits is how a prediction
        gets scored against a system it did not name.
        """
        if name in universe:
            return name
        # `name + "_"`, not a bare prefix: "3bpa-naive" must not swallow
        # "3bpa-naive-same", which is a different committee entirely.
        hits = sorted(t for t in universe if t.startswith(name + "_"))
        if len(hits) != 1:
            raise SystemExit(f"protocol system {name!r} resolves to {hits or 'nothing'}; "
                             "it must resolve to exactly one cache tag")
        return hits[0]

    pairs = [(resolve(s["naive"], spec_n), resolve(s["matched"], spec_s))
             for s in proto["systems"]]

    res, detail = {}, {}

    # ---- P1: naive head space is flatter than its shared-trunk counterpart ----
    rows = [{"naive": n, "matched": m,
             "srank_naive": spec_n[n]["stable_rank_FQ"],
             "srank_matched": spec_s[m]["stable_rank_FQ"]} for n, m in pairs]
    for r in rows:
        r["passed"] = r["srank_naive"] > r["srank_matched"]
    res["P1"] = all(r["passed"] for r in rows)
    detail["P1"] = rows

    # ---- P2: the srank ~ 2.5 rule predicts the sign of LO - CV ----------------
    t_srank = thr["sign_lo_minus_cv"]["threshold_srank"]
    rows = []
    for n, _ in pairs:
        s, d = spec_n[n]["stable_rank_FQ"], lo[n]["delta"]
        predicted = "+" if s > t_srank else "<=0"
        rows.append({"naive": n, "srank": s, "predicted": predicted, "delta": d,
                     "ci": [lo[n]["ci_lo"], lo[n]["ci_hi"]],
                     "significant": lo[n]["significant"],
                     "passed": (d > 0) if s > t_srank else (d <= 0)})
    res["P2"] = sum(r["passed"] for r in rows) >= 3
    detail["P2"] = rows

    # ---- P3: global beats max-component ---------------------------------------
    cells = [{"system": t, "error_score": e,
              "delta": d["exact_global"]["auroc_top05"] - d["exact_maxcomp"]["auroc_top05"]}
             for (t, e), d in sorted(panel.items())]
    for c in cells:
        c["passed"] = c["delta"] > 0
    frac = sum(c["passed"] for c in cells) / len(cells)
    res["P3"] = frac >= thr["auroc_global_minus_maxcomp"]["threshold"]
    detail["P3"] = {"positive_fraction": frac, "n_cells": len(cells), "cells": cells}

    # ---- P4: naive committees are less overconfident ---------------------------
    rows = []
    for n, m in pairs:
        on, om = overconfidence(n), overconfidence(m)
        rows.append({"naive": n, "matched": m,
                     "naive_ratio": on["overconfidence_ratio"],
                     "matched_ratio": om["overconfidence_ratio"],
                     "naive_detail": on, "matched_detail": om,
                     "passed": on["overconfidence_ratio"] < om["overconfidence_ratio"]})
    res["P4"] = all(r["passed"] for r in rows)
    detail["P4"] = rows

    # ---- P5: Factor A at a useful level ---------------------------------------
    t = thr["auroc_top05_exact_global"]["threshold"]
    rows = [{"system": n, "auroc": panel[(n, "e_max")]["exact_global"]["auroc_top05"]}
            for n, _ in pairs]
    for r in rows:
        r["passed"] = r["auroc"] >= t
    res["P5"] = all(r["passed"] for r in rows)
    detail["P5"] = {"threshold": t, "systems": rows}

    for p in proto["predictions"]:
        pid = p["id"]
        print(f"{pid}  {' '.join(p['statement'].split())[:92]}")
        print(f"      confidence stated in advance: {p['confidence']}")
        d = detail[pid]
        if pid == "P1":
            for r in d:
                print(f"      {r['naive']:28s} {r['srank_naive']:6.3f} vs "
                      f"{r['srank_matched']:6.3f} ({r['matched']}) "
                      f"-> {'PASS' if r['passed'] else 'FAIL'}")
        elif pid == "P2":
            for r in d:
                print(f"      {r['naive']:28s} srank {r['srank']:6.3f} predicts "
                      f"{r['predicted']:4s} | delta {r['delta']:+.4f} "
                      f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
                      f"-> {'PASS' if r['passed'] else 'FAIL'}")
            print(f"      {sum(r['passed'] for r in d)}/4 signs matched (need 3)")
        elif pid == "P3":
            print(f"      {int(d['positive_fraction'] * d['n_cells'])}/{d['n_cells']} cells "
                  f"positive = {d['positive_fraction']:.3f} "
                  f">= {thr['auroc_global_minus_maxcomp']['threshold']}")
            for c in d["cells"]:
                if not c["passed"]:
                    print(f"        negative cell: {c['system']} {c['error_score']} "
                          f"{c['delta']:+.4f}")
        elif pid == "P4":
            for r in d:
                print(f"      {r['naive']:28s} {r['naive_ratio']:5.2f}x vs "
                      f"{r['matched_ratio']:6.2f}x ({r['matched']}) "
                      f"-> {'PASS' if r['passed'] else 'FAIL'}")
        else:
            for r in d["systems"]:
                print(f"      {r['system']:28s} AUROC {r['auroc']:.3f} "
                      f">= {d['threshold']} -> {'PASS' if r['passed'] else 'FAIL'}")
        print(f"   => {pid} {'PASS' if res[pid] else 'FAIL'}\n")

    n = sum(res.values())
    print("=" * 74)
    print(f"VERDICT: {n}/{len(res)} pre-registered predictions confirmed  "
          f"({', '.join(k for k, v in sorted(res.items()) if v)} passed)")

    out = ROOT / a.out
    out.write_text(json.dumps({"protocol_sha256": got, "protocol_version": proto["version"],
                               "passed": res, "n_passed": n, "detail": detail},
                              indent=1, default=float))
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
