#!/usr/bin/env python
"""Every scored verdict in the project, emitted as data.

The counterpart to `tools/audit.py`. The audit answers "is the pipeline sound?"
and exits non-zero when it is not. This answers "what did the experiments say?"
and **always exits zero** — a scientific outcome is not an error condition, and a
tool that returns failure when a prediction was refuted creates pressure to
arrange for predictions not to be refuted.

Everything below is read from a record. Nothing is transcribed, and a
pre-registered verdict is reported next to the confidence that was stated for it
in advance, so a confident miss is as visible as a hedged hit.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REC = ROOT / "results/records"


def _load(name: str):
    p = REC / name
    return json.loads(p.read_text()) if p.exists() else None


def stated_confidence(protocol: str) -> dict[str, str]:
    p = ROOT / "protocols" / protocol
    if not p.exists():
        return {}
    d = yaml.safe_load(p.read_text())
    return {q["id"]: q.get("confidence", "unstated") for q in d.get("predictions", [])}


def collect() -> list[dict]:
    out = []

    v = _load("j6_naive_verdict.json")
    if v:
        conf = stated_confidence("j6_naive_committee.yaml")
        out.append({
            "experiment": "J6 naive committees",
            "kind": "pre-registered",
            "score": f"{v['n_passed']}/{len(v['passed'])}",
            "predictions": [{"id": k, "passed": bool(p), "stated_confidence": conf.get(k, "?")}
                            for k, p in sorted(v["passed"].items())],
        })

    v = _load("j6b_foundation_verdict.json")
    if v:
        conf = stated_confidence("j6b_foundation_acquisition.yaml")
        out.append({
            "experiment": "J6B foundation acquisition (MPtraj)",
            "kind": "pre-registered",
            "score": f"{v['n_passed']}/{len(v['passed'])}",
            "predictions": [{"id": k, "passed": bool(p), "stated_confidence": conf.get(k, "?")}
                            for k, p in sorted(v["passed"].items())],
        })

    v = _load("j6c_size_normalisation.json")
    if v:
        panel = v.get("vs_maxcomp_all_scores", {})
        out.append({
            "experiment": "J6C size normalisation",
            "kind": "post-hoc repair, criteria pre-specified",
            "score": f"{v['n_passed']}/{len(v['passed'])}",
            "predictions": [{"id": k, "passed": bool(p)} for k, p in sorted(v["passed"].items())],
            "caveat": ("N4's criterion is a bare point comparison; across the four "
                       "error scores the result is "
                       f"{sum(c['delta'] > 0 and c['significant'] for c in panel.values())} "
                       "significant wins, "
                       f"{sum(not c['significant'] for c in panel.values())} ties, "
                       f"{sum(c['delta'] < 0 and c['significant'] for c in panel.values())} "
                       "significant losses. Do not quote the score without this."),
        })

    # The PRE-REGISTERED verdict is v2 on the matched Blackwell pair. Until this
    # was written the only J8 verdict record on disk was an INADMISSIBLE H200 run
    # scored under v3, which reads 4/4 -- so the record set said 4/4 while the
    # paper says 2/4. Both matched-pair versions are now recorded separately and
    # the H200 file is named for what it is.
    # Report the CLEAN matched pairs (both systems in one exclusive job at
    # identical settings). The Blackwell pair is still unmatched and is reported
    # last, labelled, so it cannot be mistaken for the headline.
    for dev, tag in (("H200", "NVIDIA_H200"),
                     ("A100-80GB", "NVIDIA_A100-SXM4-80GB"),
                     ("RTX PRO 6000 Blackwell", "NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition")):
        a = _load(f"j8_verdict_pair_{tag}_v2.json")
        b = _load(f"j8_verdict_pair_{tag}_v3.json")
        if not a:
            continue
        na = sum(p["passed"] for p in a["predictions"].values())
        nb = sum(p["passed"] for p in b["predictions"].values()) if b else None
        out.append({
            "experiment": f"J8 water systems prediction — clean matched pair, {dev}",
            "kind": "pre-registered (v2, frozen protocol)",
            "score": f"{na}/{len(a['predictions'])}",
            "predictions": [{"id": k, "passed": bool(p["passed"])}
                            for k, p in sorted(a["predictions"].items())],
            "caveat": (f"v3 scores {nb}/4"
                       + (" — IDENTICAL to v2, so the post-hoc protocol repair changes "
                          "nothing on this device." if nb == na else
                          " — the difference is P4, whose frozen threshold was an A100 "
                          "number and does not fit this device.")),
        })

    # The unmatched Blackwell run (separate jobs, different settings) is no longer
    # reported: its clean matched re-run above reproduces its verdict exactly, so
    # there is nothing the old record adds. It stays on disk as
    # j8_verdict_blackwell_v{2,3}.json.

    v = _load("j6d_complementarity.json")
    if v:
        m = v["marginal_over_force_norm"]["force_norm + committee"]
        n_sig = sum(r["significant"] and r["marginal"] > 0 for r in v["fixed_n_contrast"])
        out.append({
            "experiment": "J6D committee value over the free signal",
            "kind": "exploratory (not scored)",
            "score": "n/a",
            "finding": (f"MPtraj marginal {m['delta']:+.4f} "
                        f"[{m['ci_lo']:+.4f},{m['ci_hi']:+.4f}], "
                        f"significant={m['significant']}; fixed-N systems "
                        f"{n_sig}/{len(v['fixed_n_contrast'])} significant"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/records/verdicts.json")
    a = ap.parse_args()

    rows = collect()
    print("=" * 74)
    print("SCORED VERDICTS  (data, not a pass/fail gate -- this always exits 0)")
    print("=" * 74)
    for r in rows:
        print(f"\n{r['experiment']}   [{r['kind']}]   score: {r['score']}")
        for p in r.get("predictions", []):
            c = f"  (stated in advance: {p['stated_confidence']})" if "stated_confidence" in p else ""
            print(f"    {p['id']:4s} {'PASS' if p['passed'] else 'FAIL'}{c}")
        if "finding" in r:
            print(f"    {r['finding']}")
        if "caveat" in r:
            print(f"    CAVEAT: {' '.join(r['caveat'].split())}")

    scored = [r for r in rows if r["score"] != "n/a"]
    tot = sum(int(r["score"].split("/")[0]) for r in scored)
    den = sum(int(r["score"].split("/")[1]) for r in scored)
    print(f"\n{'=' * 74}\nAcross all scored protocols: {tot}/{den} predictions confirmed.")
    print("Reported as-is. Refuted predictions are results, and several of this")
    print("project's most useful findings came from them.")

    (ROOT / a.json).write_text(json.dumps(
        {"verdicts": rows, "total_confirmed": tot, "total_predictions": den}, indent=1))
    print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
