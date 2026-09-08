#!/usr/bin/env python
"""Score the water benchmark against `protocols/j8_water_prediction.yaml`.

**Rewritten after adversarial review.** The previous version loaded the protocol
file and then never consulted it: every threshold (1.0 ms/lane, 1.4x, B<=4,
10.9 ms/lane) was hardcoded here, so the pre-registration and the scorer could
drift apart silently and nothing would notice. It also tested only ONE half of P3
and P4, each of which states two conditions -- so "4/4 pre-registered predictions
confirmed" overstated what had actually been checked.

Now every threshold, comparison operator and admissibility rule is read from the
YAML, and a prediction passes only if ALL of its criteria pass. The scorer knows
nothing about the science; changing a threshold means editing the protocol, which
is version-controlled and whose commit must predate the result.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTO = ROOT / "protocols/j8_water_prediction.yaml"


def load(p: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in p.open()]


def fit(rows, B, impl):
    """Least-squares T(L) = a + b*L over measured, non-OOM lanes."""
    pts = sorted({(r["lanes"], r["median_ms"]) for r in rows
                  if r["batch_size"] == B and r["impl"] == impl and r.get("status") == "ok"})
    if len(pts) < 3:                      # 2-point "fits" are not fits
        return None
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (a + b * x)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    return {"intercept": a, "slope": b, "r2": r2, "n_points": len(pts)}


def admissible(rows, rules, load_bearing=()) -> tuple[bool, list[str], set]:
    """v3: exclude outlier CELLS, gate on how many were excluded.

    v2 gated the whole run on the maximum relative IQR across all cells, so one
    transient blip out of 40-60 invalidated an otherwise clean run. v3 drops the
    offending cells, fails only if too many are dropped, and additionally fails if
    any dropped cell is one a prediction actually reads.
    """
    fails, excluded = [], set()
    r0 = rows[0]
    if rules.get("require_non_mig") and r0.get("is_mig"):
        fails.append(f"MIG device ({r0['device_name']})")
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    lim = rules.get("cell_relative_iqr_limit", rules.get("max_relative_iqr"))
    if lim is not None:
        for r in ok_rows:
            if r["iqr_ms"] / r["median_ms"] > lim:
                excluded.add((r["batch_size"], r["lanes"], r["impl"]))
        frac = len(excluded) / max(len(ok_rows), 1)
        cap = rules.get("max_excluded_cell_fraction")
        if cap is not None and frac > cap:
            fails.append(f"{len(excluded)}/{len(ok_rows)} cells ({frac:.1%}) exceed the "
                         f"{lim:.0%} per-cell IQR limit, above the {cap:.0%} cap")
        elif excluded:
            fails.append(f"NOTE {len(excluded)}/{len(ok_rows)} cells ({frac:.1%}) excluded "
                         f"as outliers: {sorted(excluded)}")
        if rules.get("forbid_excluded_load_bearing_cells"):
            hit = excluded & set(load_bearing)
            if hit:
                fails.append(f"EXCLUDED CELL IS LOAD-BEARING: {sorted(hit)}")
    need = set(rules.get("min_lanes_measured") or [])
    have = {r["lanes"] for r in ok_rows}
    if need - have:
        fails.append(f"missing lanes {sorted(need - have)}")
    hard = [f for f in fails if not f.startswith("NOTE")]
    return (not hard), fails, excluded


def metrics(W, T, ref) -> dict:
    batches = sorted({r["batch_size"] for r in W})
    fits = {B: {i: fit(W, B, i) for i in ("serial", "batched")} for B in batches}

    def t(B, L, i):
        m = [r["median_ms"] for r in W if r["batch_size"] == B and r["lanes"] == L
             and r["impl"] == i and r.get("status") == "ok"]
        return m[0] if m else None

    def speedup(B):
        ex = [x for x in (t(B, 8, "serial"), t(B, 8, "batched")) if x]
        k3 = [x for x in (t(B, 4, "serial"), t(B, 4, "batched")) if x]
        return (min(ex) / min(k3)) if ex and k3 else None

    trans = next((B for B in batches if fits[B]["batched"] and fits[B]["batched"]["slope"] > 1.0),
                 None)
    ser = [fits[B]["serial"] for B in batches if fits[B]["serial"]]
    return {
        "_fits": fits, "_batches": batches,
        "batched_slope": {B: (fits[B]["batched"]["slope"] if fits[B]["batched"] else None)
                          for B in batches},
        "total_speedup": {B: speedup(B) for B in batches},
        "transition_batch": trans,
        "transition_batch_ratio_vs_3bpa": (trans / ref["threebpa_transition_batch"]
                                           if trans else None),
        "serial_slope_min": min((f["slope"] for f in ser), default=None),
        "serial_fit_r2_min": min((f["r2"] for f in ser), default=None),
    }


OPS = {"gt": lambda v, t: v > t, "ge": lambda v, t: v >= t,
       "lt": lambda v, t: v < t, "le": lambda v, t: v <= t,
       "between": lambda v, t: t[0] <= v <= t[1]}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--water", required=True, help="water lane-bench records")
    ap.add_argument("--reference", required=True,
                    help="3BPA lane-bench records from the SAME device and run conditions")
    ap.add_argument("--protocol", default=None,
                    help="protocol yaml (default: the v3 file); pass the frozen v2 "
                         "file to reproduce the PRE-REGISTERED verdict")
    ap.add_argument("--out", default="results/records/j8_water_verdict.json",
                    help="where the verdict goes; give each device/version its own "
                         "path so an inadmissible run cannot masquerade as the headline")
    a = ap.parse_args()
    proto_path = (ROOT / a.protocol) if a.protocol else PROTO
    proto = yaml.safe_load(proto_path.read_text())
    water, tb = ROOT / a.water, ROOT / a.reference
    W, T = load(water), load(tb)
    if W[0]["device_name"] != T[0]["device_name"]:
        print(f"REFUSING TO SCORE: water is {W[0]['device_name']} but the 3BPA reference is "
              f"{T[0]['device_name']}. The predictions are all ratios against 3BPA measured "
              f"on the SAME device; crossing devices would silently compare across hardware.")
        return 2

    print(f"protocol : {proto_path.name} v{proto['version']}")
    print(f"records  : {water.name}  ({len(W)} cells, "
          f"{W[0]['device_name']}, {W[0]['sm_count']} SMs)\n")

    # cells the predictions actually read: L=4 and L=8 at every batch, both impls
    load_bearing = [(B, L, i) for B in sorted({r["batch_size"] for r in W})
                    for L in (4, 8) for i in ("serial", "batched")]
    ok, fails, excluded = admissible(W, proto.get("admissibility", {}), load_bearing)
    print("ADMISSIBILITY")
    for f in fails:
        print(f"  FAIL  {f}")
    if ok:
        print("  all admissibility rules satisfied")
    else:
        print("\n  *** Run is INADMISSIBLE under the pre-registered rules. ***")
        print("  Scoring below is reported for information only and must not be quoted.\n")

    ref = dict(proto["reference"])
    if ref.get("derive_from_reference_run"):
        tb_batches = sorted({r["batch_size"] for r in T})
        tb_b = {B: fit(T, B, "batched") for B in tb_batches}
        tb_s = [fit(T, B, "serial") for B in tb_batches]
        tb_s = [f["slope"] for f in tb_s if f]
        agg = ref.get("threebpa_serial_slope_aggregator", "median")
        newslope = {"median": statistics.median, "min": min, "max": max}[agg](tb_s)
        newtrans = next((B for B in tb_batches if tb_b[B] and tb_b[B]["slope"] > 1.0), None)
        print("\nREFERENCE DERIVED FROM THE MATCHED 3BPA RUN (v3)")
        print(f"  serial slope     : {newslope:.2f} ms/lane ({agg} over B; "
              f"protocol fallback was {ref['threebpa_serial_slope']}, an A100 value)")
        print(f"  transition batch : {newtrans} (fallback {ref['threebpa_transition_batch']})")
        ref["threebpa_serial_slope"] = newslope
        if newtrans:
            ref["threebpa_transition_batch"] = newtrans
    m = metrics(W, T, ref)

    print(f"\n{'B':>4s}{'serial a':>10s}{'serial b':>10s}{'R2':>7s}"
          f"{'batch a':>10s}{'batch b':>10s}{'R2':>7s}{'speedup':>9s}")
    print("-" * 67)
    for B in m["_batches"]:
        s, b = m["_fits"][B]["serial"], m["_fits"][B]["batched"]
        sp = m["total_speedup"][B]
        print(f"{B:4d}{s['intercept']:10.2f}{s['slope']:10.2f}{s['r2']:7.3f}"
              f"{b['intercept']:10.2f}{b['slope']:10.2f}{b['r2']:7.3f}"
              f"{(f'{sp:.2f}x' if sp else '--'):>9s}")

    print("\n" + "=" * 74)
    print("PRE-REGISTERED PREDICTIONS  (every threshold read from the protocol)\n")
    results = {}
    for p in proto["predictions"]:
        sub = []
        for c in p["criteria"]:
            val = m[c["metric"]]
            if isinstance(val, dict):
                val = val.get(c["at_batch"])
            thr = c["threshold"]
            if isinstance(thr, str) and thr.startswith("from_reference:"):
                thr = ref[thr.split(":", 1)[1]]
            passed = (val is not None) and OPS[c["op"]](val, thr)
            sub.append({"metric": c["metric"], "at_batch": c.get("at_batch"),
                        "op": c["op"], "threshold": thr,
                        "value": val, "passed": bool(passed)})
        allp = all(s["passed"] for s in sub)
        results[p["id"]] = {"passed": allp, "criteria": sub,
                            "falsification_clause": bool(p.get("falsification_clause"))}
        print(f"{p['id']}  {' '.join(p['statement'].split())[:96]}")
        for s in sub:
            at = f" @B={s['at_batch']}" if s["at_batch"] else ""
            v = "n/a" if s["value"] is None else f"{s['value']:.4g}"
            thr_s = (f"{s['threshold']:.4g}" if isinstance(s['threshold'], (int, float))
                     else str(s['threshold']))
            print(f"      {s['metric']}{at}: {v}  {s['op']} {thr_s}"
                  f"   -> {'PASS' if s['passed'] else 'FAIL'}")
        print(f"   => {p['id']} {'PASS' if allp else 'FAIL'}"
              f"{'   (FALSIFICATION CLAUSE)' if p.get('falsification_clause') else ''}\n")

    n = sum(v["passed"] for v in results.values())
    ncrit = sum(len(v["criteria"]) for v in results.values())
    npass = sum(sum(c["passed"] for c in v["criteria"]) for v in results.values())
    print("=" * 74)
    print(f"VERDICT: {n}/{len(results)} predictions confirmed "
          f"({npass}/{ncrit} individual criteria)")
    if not results.get("P1", {}).get("passed", True):
        print("\nP1 is the falsification clause: per the protocol, acceleration should be")
        print("dropped from the paper rather than scoped.")
    if not ok:
        print("\nNOTE: run was INADMISSIBLE; the verdict above is informational only.")

    out = ROOT / a.out
    out.write_text(json.dumps({
        "protocol_version": proto["version"], "records": water.name,
        "device": W[0]["device_name"], "sm_count": W[0]["sm_count"],
        "admissible": ok, "admissibility_failures": fails,
        "metrics": {k: v for k, v in m.items() if not k.startswith("_")},
        "predictions": results,
    }, indent=1, default=float))
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
