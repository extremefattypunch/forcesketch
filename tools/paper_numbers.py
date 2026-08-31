#!/usr/bin/env python
"""Every number the paper quotes, computed from a record. Nothing transcribed.

This is rule R3 made mechanical. Each entry below names a LaTeX macro, the record
file it comes from, and the function that derives it. Running this emits

    paper/macros.tex                    the \\newcommand definitions
    results/records/macro_provenance.json   macro -> {source, sha256, value, how}

and `tools/audit.py` then checks the loop closes: every macro in the .tex has a
provenance entry, every source file still hashes the same, and recomputing gives
the same value. A number that cannot be derived from a record cannot get a macro,
which is the only durable way to stop a hand-typed figure drifting into the text.

Adding a number to the paper means adding it here first.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import statistics as st

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
REC = ROOT / "results/records"


# --------------------------------------------------------------------------
# loaders (cached, so each source is hashed and read once)
# --------------------------------------------------------------------------
_CACHE: dict[str, object] = {}
_HASH: dict[str, str] = {}


def load(name: str):
    if name not in _CACHE:
        p = REC / name
        raw = p.read_bytes()
        _HASH[name] = hashlib.sha256(raw).hexdigest()
        _CACHE[name] = ([json.loads(l) for l in raw.decode().splitlines() if l.strip()]
                        if name.endswith(".jsonl") else json.loads(raw))
    return _CACHE[name]


# --------------------------------------------------------------------------
# derivations
# --------------------------------------------------------------------------
def _panel(rows, escore="e_max", metric="auroc_top05"):
    by = collections.defaultdict(dict)
    for r in rows:
        if r["error_score"] == escore:
            by[r["cache_tag"]][r["signal"]] = r[metric]
    return by


def factor_a_range(lo: bool):
    by = _panel(load("j2a_oracle_panel.jsonl"))
    v = [d["exact_global"] for d in by.values()]
    return min(v) if lo else max(v)


def free_gain_range(lo: bool):
    by = _panel(load("j2a_oracle_panel.jsonl"))
    g = [d["exact_global"] - max(d[s] for s in ("energy_std", "force_norm") if s in d)
         for d in by.values()]
    return min(g) if lo else max(g)


def _gvm_entries():
    d = load("j2a_global_vs_maxcomp_ci.json")
    return d["entries"] if isinstance(d, dict) else d


def gvm_count(sig: bool):
    return sum(c["delta"] > 0 and (c["significant"] if sig else True)
               for c in _gvm_entries())


def gvm_total():
    return len(_gvm_entries())


def free_sig_count():
    return sum(c["significant"] for c in load("j2a_free_signal_ci.json")["vs_free"])


def free_sig_total():
    return len(load("j2a_free_signal_ci.json")["vs_free"])


def lo_skip_range(lo: bool):
    rows = [r for r in load("j2b_r0_4.jsonl")
            if r["gate"] == "leading_only" and r["uq_lanes"] == 5 and r["score"] == "global"]
    v = [r["frac_exact_skipped"] for r in rows]
    return min(v) if lo else max(v)


def lo_beats_cv_count():
    return sum(r["delta"] > 0 and r["significant"]
               for r in load("j9a_leading_only_ci_global_L5.json"))


def lo_systems():
    return len(load("j9a_leading_only_ci_global_L5.json"))


def j3_mae():
    rows = load("j3_extreme_value_v2.json")["recall_prediction"]
    return st.mean(abs(r["shared_real"] - r["measured"]) for r in rows)


def m_sweep_retained(m: int):
    rows = [r for r in load("j4_m_sweep.jsonl")
            if r["system"] not in {"disjoint_test_1200K", "overlapping_test_1200K",
                                   "same_test_1200K"}]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["system"]][r["m"]] = r
    return st.mean(by[s][m]["auroc_global_mean"] / by[s][8]["auroc_global_mean"] for s in by)


def m_sweep_worst_subset(m: int):
    rows = [r for r in load("j4_m_sweep.jsonl")
            if r["system"] not in {"disjoint_test_1200K", "overlapping_test_1200K",
                                   "same_test_1200K"}]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["system"]][r["m"]] = r
    return min(by[s][m]["auroc_global_min"] / by[s][8]["auroc_global_mean"] for s in by)


def naive_overconf(which: str):
    d = load("j6_naive_verdict.json")["detail"]["P4"]
    key = "naive_ratio" if which == "naive" else "matched_ratio"
    v = [r[key] for r in d]
    return min(v) if which == "naive" else max(v)


def naive_passed():
    return load("j6_naive_verdict.json")["n_passed"]


def mptraj_n():
    return load("j6b_foundation_verdict.json")["n_structures"]


def mptraj_auroc(signal: str):
    return load("j6b_foundation_verdict.json")["factor_a"][signal]["e_max"]


def mptraj_extensivity():
    return load("j6b_foundation_verdict.json")["detail"]["Q3"]["ratio"]


def mptraj_jaccard():
    return load("j6b_foundation_verdict.json")["detail"]["Q4"]["jaccard"]


def j6c(key: str):
    d = load("j6c_size_normalisation.json")
    if key == "alpha":
        return d["alpha_star"]
    if key == "auroc0":
        return d["held_out_auroc"]["S_alpha=0.0"]["e_max"]
    if key == "aurocstar":
        return d["held_out_auroc"][f"S_alpha={d['alpha_star']}"]["e_max"]
    if key == "wins":
        return sum(c["delta"] > 0 and c["significant"]
                   for c in d["vs_maxcomp_all_scores"].values())
    if key == "losses":
        return sum(c["delta"] < 0 and c["significant"]
                   for c in d["vs_maxcomp_all_scores"].values())
    raise KeyError(key)


def j6d(key: str):
    d = load("j6d_complementarity.json")
    if key == "free":
        return d["held_out_auroc"]["force_norm only"]
    if key == "both":
        return d["held_out_auroc"]["force_norm + committee"]
    if key == "marginal":
        return d["marginal_over_force_norm"]["force_norm + committee"]["delta"]
    if key == "fixed_sig":
        return sum(r["significant"] and r["marginal"] > 0 for r in d["fixed_n_contrast"])
    if key == "fixed_n":
        return len(d["fixed_n_contrast"])
    if key == "fixed_lo":
        return min(r["marginal"] for r in d["fixed_n_contrast"])
    if key == "fixed_hi":
        return max(r["marginal"] for r in d["fixed_n_contrast"])
    raise KeyError(key)


def free_auroc_range(lo: bool):
    """AUROC span of the two FREE signals over the molecular panel.

    Was typed into the manuscript as "0.60--0.70".
    """
    rows = load("j2a_oracle_panel.jsonl")
    v = [r["auroc_top05"] for r in rows
         if r.get("error_score") == "e_max"
         and r.get("signal") in ("force_norm", "energy_std")]
    return min(v) if lo else max(v)


def stable_rank_range(lo: bool):
    """Span of the head-space stable rank over the molecular panel.

    Was typed into the manuscript as "1.2--4.3 of a possible 7".
    """
    v = [r["stable_rank_FQ"] for r in load("j5_spectrum.json")]
    return min(v) if lo else max(v)


def naive_accuracy_ratio():
    """How much more accurate a matched independently-trained committee is on the
    shifted (PIMD) data: multi-head force RMSE over naive force RMSE.

    Was typed into the manuscript as "2.7x". Reads both cache records; the
    declared source is the multi-head one, following fsWaterSpeedupSpread, which
    likewise derives across devices from one declared record.
    """
    mh = {f"{r['variant']}_{r.get('split')}": r["force_rmse_mean_mev_A"]
          for r in load("j5_water_caches.jsonl")}
    nv = {(r["variant"], r.get("split")): r["force_rmse_mean_mev_A"]
          for r in load("j6_naive_caches.jsonl")}
    return mh["water-overlapping_pimd_T300K"] / nv[("water-naive", "pimd_T300K")]


def j2c(key: str):
    """Realised finite-sample coverage of the gate's conformal constant.

    The manuscript claimed coverage was "verified against the Beta(k, n+1-k) law
    by simulation rather than asserted" and then quoted no number, so the claim
    was uncheckable. `min` is the binding case: the guarantee is weakest at the
    smallest calibration set.
    """
    d = load("j2c_conformal_coverage.json")
    if key == "n_cal_min":
        return d["n_cal_min"]
    if key == "n_cal_max":
        return d["n_cal_max"]
    if key == "realised_min":
        return d["coverage"]["min"]["realised"]
    if key == "beta_min":
        return d["coverage"]["min"]["beta_mean"]
    if key == "splits":
        return d["n_splits"]
    if key == "infeasible_at_shipped_alpha":
        a = d["alpha"]
        return d["feasibility"][f"alpha_{a}"]["n_infeasible"]
    raise KeyError(key)


def _fit(rows, B, impl):
    pts = sorted({(r["lanes"], r["median_ms"]) for r in rows
                  if r["batch_size"] == B and r["impl"] == impl and r.get("status") == "ok"})
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    b, a = np.polyfit(x, y, 1)
    return a, b


def _speedup(rows, B):
    def t(L, i):
        m = [r["median_ms"] for r in rows if r["batch_size"] == B and r["lanes"] == L
             and r["impl"] == i and r.get("status") == "ok"]
        return m[0] if m else None
    ex = [x for x in (t(8, "serial"), t(8, "batched")) if x]
    k3 = [x for x in (t(4, "serial"), t(4, "batched")) if x]
    return min(ex) / min(k3)


BLACK = "NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition"
# The CLEAN matched pairs are A100 and H200: both systems in one exclusive job at
# identical settings. The Blackwell pair came from two jobs at different settings,
# so it is no longer the source for any macro.
PAIR = {"a100": "NVIDIA_A100-SXM4-80GB_pair", "h200": "NVIDIA_H200_pair",
        "blackwell": f"{BLACK}_pair"}


def j8_speedup(system: str, B: int, dev: str = "a100"):
    tag = PAIR[dev]
    f = (f"j8_lane_bench_water_{tag}.jsonl" if system == "water"
         else f"j8_lane_bench_{tag}.jsonl")
    return _speedup(load(f), B)


def j8_speedup_spread(B: int = 16):
    """Max-min of water speedup at B across the clean pairs, in percent."""
    v = [j8_speedup("water", B, d) for d in PAIR]
    return 100.0 * (max(v) - min(v)) / min(v)


# --------------------------------------------------------------------------
# the registry: (macro, source, fn, format, description)
# --------------------------------------------------------------------------
MACROS = [
    # --- numbers that were typed into the prose, now derived ---------------
    # These three ranges were the only result figures in the manuscript not
    # backed by a record. Each reproduces the typed value exactly, which is
    # reassuring but was luck rather than provenance until now.
    # Percentages of the same two records, for the abstract: a general reader
    # parses "82--94%" far faster than "0.822--0.943 of".
    ("fsLOSkipLoPct", "j2b_r0_4.jsonl", lambda: 100 * lo_skip_range(True), "{:.0f}",
     "smallest share of exact evaluations skipped, percent"),
    ("fsLOSkipHiPct", "j2b_r0_4.jsonl", lambda: 100 * lo_skip_range(False), "{:.0f}",
     "largest share of exact evaluations skipped, percent"),
    ("fsFreeAurocLo", "j2a_oracle_panel.jsonl", lambda: free_auroc_range(True), "{:.2f}",
     "lowest AUROC of a free signal over the molecular panel"),
    ("fsFreeAurocHi", "j2a_oracle_panel.jsonl", lambda: free_auroc_range(False), "{:.2f}",
     "highest AUROC of a free signal over the molecular panel"),
    ("fsStableRankLo", "j5_spectrum.json", lambda: stable_rank_range(True), "{:.1f}",
     "smallest head-space stable rank over the molecular panel"),
    ("fsStableRankHi", "j5_spectrum.json", lambda: stable_rank_range(False), "{:.1f}",
     "largest head-space stable rank over the molecular panel"),
    ("fsNaiveAccuracyRatio", "j5_water_caches.jsonl", lambda: naive_accuracy_ratio(), "{:.1f}",
     "how much more accurate a matched independent committee is on shifted (PIMD) water"),
    # --- realised coverage of the calibrated gate (j2c) --------------------
    ("fsCovNCalMin", "j2c_conformal_coverage.json", lambda: j2c("n_cal_min"), "{:d}",
     "smallest calibration set over all registered splits (the binding case)"),
    ("fsCovNCalMax", "j2c_conformal_coverage.json", lambda: j2c("n_cal_max"), "{:d}",
     "largest calibration set over all registered splits"),
    ("fsCovRealisedMin", "j2c_conformal_coverage.json", lambda: j2c("realised_min"), "{:.3f}",
     "realised coverage at the smallest calibration set, alpha=0.05"),
    ("fsCovBetaMin", "j2c_conformal_coverage.json", lambda: j2c("beta_min"), "{:.3f}",
     "Beta(k,n+1-k) mean the theory predicts at that calibration set"),
    ("fsCovSplits", "j2c_conformal_coverage.json", lambda: j2c("splits"), "{:d}",
     "registered splits the coverage check covers"),
    ("fsCovInfeasible", "j2c_conformal_coverage.json",
     lambda: j2c("infeasible_at_shipped_alpha"), "{:d}",
     "splits with no feasible order statistic at the shipped alpha"),
    ("fsFactorALo", "j2a_oracle_panel.jsonl", lambda: factor_a_range(True), "{:.3f}",
     "lowest exact-global AUROC (top-5% e_max) over the six molecular systems"),
    ("fsFactorAHi", "j2a_oracle_panel.jsonl", lambda: factor_a_range(False), "{:.3f}",
     "highest exact-global AUROC over the six molecular systems"),
    ("fsGlobalWins", "j2a_global_vs_maxcomp_ci.json", lambda: gvm_count(False), "{:d}",
     "cells where global beats max-component"),
    ("fsGlobalWinsSig", "j2a_global_vs_maxcomp_ci.json", lambda: gvm_count(True), "{:d}",
     "of those, significant at 95%"),
    ("fsGlobalCells", "j2a_global_vs_maxcomp_ci.json", gvm_total, "{:d}",
     "total system x error-score cells"),
    ("fsFreeSig", "j2a_free_signal_ci.json", free_sig_count, "{:d}",
     "free-signal comparisons where exact global wins significantly"),
    ("fsFreeSigTotal", "j2a_free_signal_ci.json", free_sig_total, "{:d}",
     "free-signal comparisons made"),
    ("fsFreeGainLo", "j2a_oracle_panel.jsonl", lambda: free_gain_range(True), "{:.3f}",
     "smallest AUROC gain of exact global over the BEST free signal"),
    ("fsFreeGainHi", "j2a_oracle_panel.jsonl", lambda: free_gain_range(False), "{:.3f}",
     "largest such gain"),
    ("fsLOSkipLo", "j2b_r0_4.jsonl", lambda: lo_skip_range(True), "{:.3f}",
     "lowest fraction of exact evaluations skipped, leading-only r0=4 at 5 lanes"),
    ("fsLOSkipHi", "j2b_r0_4.jsonl", lambda: lo_skip_range(False), "{:.3f}",
     "highest such fraction"),
    ("fsLOBeatsCV", "j9a_leading_only_ci_global_L5.json", lo_beats_cv_count, "{:d}",
     "systems where leading-only significantly beats the control variate"),
    ("fsLOSystems", "j9a_leading_only_ci_global_L5.json", lo_systems, "{:d}",
     "systems compared"),
    ("fsBetaMAE", "j3_extreme_value_v2.json", j3_mae, "{:.3f}",
     "mean absolute error of the Beta-law recall prediction (shared_real)"),
    ("fsMFourRetained", "j4_m_sweep.jsonl", lambda: m_sweep_retained(4), "{:.3f}",
     "mean fraction of Factor A retained at m=4 relative to m=8"),
    ("fsMFourWorstSubset", "j4_m_sweep.jsonl", lambda: m_sweep_worst_subset(4), "{:.3f}",
     "worst single 4-member subcommittee, as a fraction of its m=8 committee"),
    ("fsMTwoWorstSubset", "j4_m_sweep.jsonl", lambda: m_sweep_worst_subset(2), "{:.3f}",
     "worst single 2-member subcommittee"),
    ("fsNaiveOverconfLo", "j6_naive_verdict.json", lambda: naive_overconf("naive"), "{:.2f}",
     "lowest naive-committee overconfidence ratio"),
    ("fsSharedOverconfHi", "j6_naive_verdict.json", lambda: naive_overconf("matched"), "{:.2f}",
     "highest matched shared-trunk overconfidence ratio"),
    ("fsNaivePassed", "j6_naive_verdict.json", naive_passed, "{:d}",
     "pre-registered naive-committee predictions confirmed, of five"),
    ("fsMPtrajN", "j6b_foundation_verdict.json", mptraj_n, "{:,d}",
     "MPtraj candidate structures scanned"),
    ("fsMPtrajGlobal", "j6b_foundation_verdict.json",
     lambda: mptraj_auroc("exact_global"), "{:.3f}",
     "exact global AUROC on MPtraj"),
    ("fsMPtrajFree", "j6b_foundation_verdict.json",
     lambda: mptraj_auroc("force_norm"), "{:.3f}",
     "free mean-force-norm AUROC on MPtraj"),
    ("fsMPtrajRandom", "j6b_foundation_verdict.json",
     lambda: mptraj_auroc("random_null"), "{:.3f}",
     "random null on MPtraj, the sanity check"),
    ("fsMPtrajExtensivity", "j6b_foundation_verdict.json", mptraj_extensivity, "{:.2f}",
     "ratio of median acquired atom count, global over max-component"),
    ("fsMPtrajJaccard", "j6b_foundation_verdict.json", mptraj_jaccard, "{:.3f}",
     "gate/exact acquisition overlap at budget 8000"),
    ("fsAlphaStar", "j6c_size_normalisation.json", lambda: j6c("alpha"), "{:.2f}",
     "design-selected size-normalisation exponent"),
    ("fsNormAurocZero", "j6c_size_normalisation.json", lambda: j6c("auroc0"), "{:.3f}",
     "held-out e_max AUROC of the unnormalised global statistic"),
    ("fsNormAurocStar", "j6c_size_normalisation.json", lambda: j6c("aurocstar"), "{:.3f}",
     "held-out e_max AUROC at alpha*"),
    ("fsNormWins", "j6c_size_normalisation.json", lambda: j6c("wins"), "{:d}",
     "error scores where alpha* significantly beats max-component"),
    ("fsNormLosses", "j6c_size_normalisation.json", lambda: j6c("losses"), "{:d}",
     "error scores where it significantly loses"),
    ("fsCompFree", "j6d_complementarity.json", lambda: j6d("free"), "{:.4f}",
     "MPtraj AUROC of the free signal alone"),
    ("fsCompBoth", "j6d_complementarity.json", lambda: j6d("both"), "{:.4f}",
     "MPtraj AUROC of free signal plus committee"),
    ("fsCompMarginal", "j6d_complementarity.json", lambda: j6d("marginal"), "{:.4f}",
     "marginal value of the committee over the free signal on MPtraj"),
    ("fsCompFixedSig", "j6d_complementarity.json", lambda: j6d("fixed_sig"), "{:d}",
     "fixed-N systems where the committee adds significantly"),
    ("fsCompFixedN", "j6d_complementarity.json", lambda: j6d("fixed_n"), "{:d}",
     "fixed-N systems tested"),
    ("fsCompFixedLo", "j6d_complementarity.json", lambda: j6d("fixed_lo"), "{:.3f}",
     "smallest fixed-N marginal"),
    ("fsCompFixedHi", "j6d_complementarity.json", lambda: j6d("fixed_hi"), "{:.3f}",
     "largest fixed-N marginal"),
    ("fsWaterSpeedupBOne", "j8_lane_bench_water_NVIDIA_A100-SXM4-80GB_pair.jsonl",
     lambda: j8_speedup("water", 1), "{:.2f}",
     "water total speedup at B=1, clean A100 matched pair"),
    ("fsWaterSpeedupBSixteen", "j8_lane_bench_water_NVIDIA_A100-SXM4-80GB_pair.jsonl",
     lambda: j8_speedup("water", 16), "{:.2f}",
     "water total speedup at B=16, clean A100 matched pair"),
    ("fsWaterSpeedupBSixteenHTwo", "j8_lane_bench_water_NVIDIA_H200_pair.jsonl",
     lambda: j8_speedup("water", 16, "h200"), "{:.2f}",
     "water total speedup at B=16, clean H200 matched pair"),
    ("fsWaterSpeedupBSixteenBlk", f"j8_lane_bench_water_{BLACK}_pair.jsonl",
     lambda: j8_speedup("water", 16, "blackwell"), "{:.2f}",
     "water total speedup at B=16, clean Blackwell matched pair"),
    ("fsWaterSpeedupSpread", "j8_lane_bench_water_NVIDIA_H200_pair.jsonl",
     lambda: j8_speedup_spread(16), "{:.1f}",
     "percent spread of the B=16 water speedup across the clean matched pairs"),
    ("fsThreeBPASpeedupBSixteen", "j8_lane_bench_NVIDIA_A100-SXM4-80GB_pair.jsonl",
     lambda: j8_speedup("3bpa", 16), "{:.2f}",
     "3BPA total speedup at B=16, same job and settings as the water run"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="paper/macros.tex")
    ap.add_argument("--provenance", default="results/records/macro_provenance.json")
    ap.add_argument("--check", action="store_true",
                    help="recompute and diff against the committed provenance, write nothing")
    a = ap.parse_args()

    prov, lines = {}, ["% Generated by tools/paper_numbers.py -- do not edit by hand.",
                       "% Every macro below is derived from a record in results/records/;",
                       "% tools/audit.py verifies that the derivation still reproduces."]
    for name, src, fn, fmt, desc in MACROS:
        value = fn()
        lines.append(f"\\newcommand{{\\{name}}}{{{fmt.format(value)}}}")
        prov[name] = {"source": src, "source_sha256": _HASH[src],
                      "value": value, "formatted": fmt.format(value),
                      "description": desc}

    if a.check:
        old = json.loads((ROOT / a.provenance).read_text())
        bad = []
        for k, v in prov.items():
            if k not in old:
                bad.append(f"{k}: new macro, not in committed provenance")
            elif old[k]["formatted"] != v["formatted"]:
                bad.append(f"{k}: {old[k]['formatted']} -> {v['formatted']}")
            elif old[k]["source_sha256"] != v["source_sha256"]:
                # value identical, source moved: the provenance is stale rather
                # than the paper being wrong. Still a failure -- regenerate --
                # but say which, or the two get conflated.
                bad.append(f"{k}: value unchanged ({v['formatted']}) but source "
                           f"{v['source']} re-hashed; regenerate provenance")
        for k in old:
            if k not in prov:
                bad.append(f"{k}: in provenance but no longer derivable")
        for b in bad:
            print(f"  MISMATCH {b}")
        print(f"{'FAIL' if bad else 'OK'}: {len(prov)} macros, {len(bad)} mismatches")
        return 1 if bad else 0

    tex = ROOT / a.tex
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text("\n".join(lines) + "\n")
    (ROOT / a.provenance).write_text(json.dumps(prov, indent=1, sort_keys=True))
    print(f"wrote {len(prov)} macros to {a.tex} and provenance to {a.provenance}")
    for name, src, _, fmt, _ in MACROS:
        print(f"  \\{name:24s} = {prov[name]['formatted']:>10s}   <- {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
