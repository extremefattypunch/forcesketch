"""Tables 1-3 (spec SS54) and LaTeX macros, generated from results/raw only.

Spec SS68: "Do not manually transcribe results into tables." So the manuscript
\\input{macros.tex} and contains no literal number; re-running this script after a
re-measurement updates every figure in the paper at once. `check_cherry_picking`
enforces spec SS70's rule that no configuration may be silently dropped.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW = Path("results/raw")
PROC = Path("results/processed")

# Measured lane-cost model, refit by `fit_cost_model` from lane_scaling_*.jsonl.
# Fallback only; the real value is fitted by `fit_cost_model`. Kept equal to
# LaneCostModel's cached defaults, which tests/test_cost_model.py enforces.
DEFAULT_COST = (10.38, 11.78)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def fit_cost_model(system: str = "disjoint", batch_size: int = 1) -> tuple[float, float]:
    """Least-squares fit of T(L) = a + b*L to the measured lane scan."""
    path = RAW / f"lane_scaling_{system}.jsonl"
    if not path.exists():
        return DEFAULT_COST
    recs = [r for r in load_jsonl(path) if r["batch_size"] == batch_size]
    if len(recs) < 2:
        return DEFAULT_COST
    L = np.array([r["lanes"] for r in recs], dtype=float)
    T = np.array([r["median_ms"] for r in recs], dtype=float)
    b, a = np.polyfit(L, T, 1)
    return float(a), float(b)


def mean_by(records: list[dict], keys: tuple[str, ...], fields: tuple[str, ...]) -> dict:
    """Group by `keys`, average `fields`, and keep the seed count for provenance."""
    acc: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        acc[tuple(r[k] for k in keys)].append(r)
    out = {}
    for k, rs in acc.items():
        out[k] = {f: float(np.mean([r[f] for r in rs])) for f in fields if f in rs[0]}
        out[k]["n_seeds"] = len(rs)
    return out


def check_cherry_picking(fidelity: list[dict], expected_seeds: int) -> list[str]:
    """Spec SS70: the analysis must not discard failed configurations."""
    problems = []
    grouped = mean_by(fidelity, ("method", "K", "with_mean_lane"), ("top5_recall",))
    for key, v in sorted(grouped.items()):
        if v["n_seeds"] != expected_seeds:
            problems.append(f"{key}: {v['n_seeds']} seeds, expected {expected_seeds}")
    methods = {k[0] for k in grouped}
    for m in methods:
        ks = sorted({k[1] for k in grouped if k[0] == m})
        # The grid was extended to K=5,6 to settle whether recall reaches 0.90
        # below the exact basis (it does not). Either grid is acceptable; a
        # grid MISSING points is not, which is what this guards.
        if m in ("gaussian", "rademacher", "pairwise") and ks not in ([1, 2, 3, 4, 7],
                                                                     [1, 2, 3, 4, 5, 6, 7]):
            problems.append(f"{m}: K grid is {ks}, expected [1,2,3,4,7] or [1,2,3,4,5,6,7]")
    return problems


def table1_primary(system_tag: str, cost=None) -> str:
    """Spec SS54 Table 1. Rows are NEVER filtered by outcome."""
    a, b = cost or fit_cost_model()
    fid = load_jsonl(RAW / f"03_sketch_fidelity_{system_tag}.jsonl")
    g = mean_by(fid, ("method", "K", "with_mean_lane", "total_lanes"),
                ("spearman", "top5_recall", "top5_jaccard", "p90_rel_err_S",
                 "same_max_atom_frac"))
    t_exact = a + b * 8
    lines = ["| method | K | lanes | Spearman | top-5% recall | top-5% Jaccard "
             "| incr. UQ ms | incr. speedup | total ms | total speedup |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (m, K, wm, L), v in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        incr = (a + b * L) - (a + b * 1)
        incr_ex = t_exact - (a + b * 1)
        name = m.replace("_", " ") + (" +mean" if wm else "")
        lines.append(
            f"| {name} | {K} | {L} | {v['spearman']:.3f} | {v['top5_recall']:.3f} "
            f"| {v['top5_jaccard']:.3f} | {incr:.1f} | {incr_ex/incr:.2f}x "
            f"| {a+b*L:.1f} | {t_exact/(a+b*L):.2f}x |")
    return "\n".join(lines)


def table2_scaling() -> str:
    """Spec SS54 Table 2: system, atoms, batch size, exact vs ForceSketch latency.

    Answers what the saving actually tracks. Columns are measured medians, not a
    fitted model, so the fit in `fit_cost_model` can be checked against them.
    """
    import glob
    lines = ["| system | atoms/struct | B | total atoms | exact UQ ms (L=8) "
             "| FS K=3 ms (L=4) | incr. speedup | K=2 speedup |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for f in sorted(glob.glob(str(RAW / "lane_scaling_*.jsonl"))):
        recs = load_jsonl(Path(f))
        by = {(r["batch_size"], r["lanes"]): r for r in recs}
        a_per = recs[0]["num_atoms"] // recs[0]["batch_size"]
        for B in sorted({r["batch_size"] for r in recs}):
            if (B, 8) not in by:
                continue
            t1, t3, t4, t8 = (by[(B, l)]["median_ms"] for l in (1, 3, 4, 8))
            lines.append(
                f"| {recs[0]['system']} | {a_per} | {B} | {by[(B,8)]['num_atoms']} "
                f"| {t8:.2f} | {t4:.2f} | {(t8-t1)/(t4-t1):.2f}x | {(t8-t1)/(t3-t1):.2f}x |")
    return "\n".join(lines)


def table_gate_baselines(score: str = "maxcomp") -> str:
    """The screening-gate comparison: is ForceSketch's gate better than free ones?"""
    f = RAW / f"07_gate_baselines_{score}.jsonl"
    if not f.exists():
        return ""
    recs = load_jsonl(f)
    order = ["energy (free)", "head-exact-mean K=4", "haar K=4", "control-variate K=4"]
    lines = ["| system | gate | lanes | exact skipped | high-UQ recall | speedup |",
             "|---|---|---:|---:|---:|---:|"]
    for sysname in ("3bpa", "ethanol", "aspirin", "azobenzene"):
        for g in order:
            r = next((x for x in recs if x["system"] == sysname and x["gate"] == g), None)
            if r is None:
                continue
            lines.append(
                f"| {sysname} | {g} | {r['total_lanes']} "
                f"| {r['frac_exact_skipped']:.3f} [{r['skip_ci_lo']:.3f}, {r['skip_ci_hi']:.3f}] "
                f"| {r['high_uq_recall']:.3f} [{r['recall_ci_lo']:.3f}, {r['recall_ci_hi']:.3f}] "
                f"| {r['screening_speedup']:.2f}x |")
    return "\n".join(lines)


def table3_screening(system_tags: list[str]) -> str:
    """Spec SS54 Table 3, across systems."""
    lines = ["| system | method | K | alpha | exact skipped | high-UQ recall "
             "| FNR | screening speedup |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for tag in system_tags:
        p = RAW / f"06_screening_{tag}.jsonl"
        if not p.exists():
            continue
        g = mean_by(load_jsonl(p), ("method", "K", "r0", "alpha"),
                    ("frac_exact_skipped", "high_uq_recall", "false_negative_rate",
                     "screening_speedup"))
        for (m, K, r0, alpha), v in sorted(g.items()):
            name = f"{m.replace('_',' ')}" + (f" r0={r0}" if r0 else "")
            lines.append(
                f"| {tag} | {name} | {K} | {alpha:.2f} | {v['frac_exact_skipped']:.3f} "
                f"| {v['high_uq_recall']:.3f} | {v['false_negative_rate']:.3f} "
                f"| {v['screening_speedup']:.2f}x |")
    return "\n".join(lines)


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def emit_macros(out: Path) -> dict:
    """One \\newcommand per frozen number. The manuscript cites these, never literals."""
    a, b = fit_cost_model()
    macros, vals = [], {}

    def put(name, value, fmt="{:.3f}"):
        s = fmt.format(value) if not isinstance(value, str) else value
        macros.append(_macro(name, s))
        vals[name] = value

    def put_pct(name, value):
        """Percent form of a fraction, for the lay-readable abstract.

        The abstract is written for readers new to molecular potentials, who read
        "84%" far more easily than "0.836". Emitted as a macro like everything
        else so the two forms cannot drift apart.
        """
        put(name, round(100 * value), "{:d}")

    put("fsCostIntercept", a, "{:.2f}")
    put("fsCostSlope", b, "{:.2f}")

    # LaTeX control sequences must be letters only -- no digits in macro names.
    WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 7: "Seven",
            16: "Sixteen", 64: "SixtyFour"}

    fid = load_jsonl(RAW / "03_sketch_fidelity_disjoint_test_1200K.jsonl")
    g = mean_by(fid, ("method", "K"), ("top5_recall", "spearman"))
    WORD.update({5: "Five", 6: "Six"})
    for K in (2, 3, 4, 5, 6):
        if ("haar", K) in g:
            put(f"fsHaarRecallK{WORD[K]}", g[("haar", K)]["top5_recall"])
            put(f"fsHaarSpearmanK{WORD[K]}", g[("haar", K)]["spearman"])
    put("fsHeadSubRecallKThree", g[("head_subsample", 3)]["top5_recall"])
    put("fsGaussRecallKThree", g[("gaussian", 3)]["top5_recall"])

    # lane-scaling ceilings
    ls = RAW / "lane_scaling_disjoint.jsonl"
    if ls.exists():
        recs = {(r["batch_size"], r["lanes"]): r["median_ms"] for r in load_jsonl(ls)}
        # Spec SS45 defines these differently and forbids conflating them:
        #   incremental UQ speedup = [T(L=8) - T(L=1)] / [T(L=1+K) - T(L=1)]
        #   total workflow speedup =  T(L=8) / T(L=1+K)
        # The raw ratio INCLUDES the mean-force lane, so it is the total, not the
        # incremental, figure.
        for B in (1, 64):
            if (B, 8) in recs:
                t1, t3, t4, t8 = (recs[(B, l)] for l in (1, 3, 4, 8))
                put(f"fsTotalKThreeB{WORD[B]}", t8 / t4, "{:.2f}")
                put(f"fsTotalKTwoB{WORD[B]}", t8 / t3, "{:.2f}")
                put(f"fsIncrKThreeB{WORD[B]}", (t8 - t1) / (t4 - t1), "{:.2f}")
                put(f"fsIncrKTwoB{WORD[B]}", (t8 - t1) / (t3 - t1), "{:.2f}")

    # screening headline: control variate r0=2, K=4, alpha=0.05, per system
    for tag, short in [("disjoint_test_1200K", "ThreeBPA"), ("rmd17-disjoint_ethanol", "Ethanol"),
                       ("rmd17-disjoint_aspirin", "Aspirin"),
                       ("rmd17-disjoint_azobenzene", "Azobenzene")]:
        p = RAW / f"06_screening_{tag}.jsonl"
        if not p.exists():
            continue
        gg = mean_by(load_jsonl(p), ("method", "K", "r0", "alpha"),
                     ("frac_exact_skipped", "high_uq_recall", "false_negative_rate",
                      "screening_speedup"))
        key = ("control_variate", 4, 2, 0.05)
        if key in gg:
            put(f"fsGateSkip{short}", gg[key]["frac_exact_skipped"])
            put(f"fsGateRecall{short}", gg[key]["high_uq_recall"])
            put(f"fsGateSpeedup{short}", gg[key]["screening_speedup"], "{:.2f}")

    # --- bootstrap CIs (spec SS47). The manuscript must never quote a bare mean.
    bs = RAW / "04_bootstrap_disjoint_test_1200K.jsonl"
    if bs.exists():
        recs = load_jsonl(bs)
        nb = {int(r["top5_recall_n_boot"]) for r in recs
              if r["experiment_id"] == "04_bootstrap_ci"}
        put("fsNBoot", f"{sorted(nb)[0]:,}" if len(nb) == 1 else "MIXED")
        by = {(r["method"], r["K"], r["r0"], r.get("with_mean_lane", False)): r
              for r in recs if r["experiment_id"] == "04_bootstrap_ci"}
        for K in (2, 3, 4):
            r = by.get(("haar", K, 0, False))
            if r:
                put(f"fsHaarRecallK{WORD[K]}CI",
                    f"[{r['top5_recall_ci_lo']:.3f}, {r['top5_recall_ci_hi']:.3f}]")
                put(f"fsHaarSpearmanK{WORD[K]}CI",
                    f"[{r['spearman_ci_lo']:.3f}, {r['spearman_ci_hi']:.3f}]")
        r = by.get(("control_variate", 4, 2, False))
        if r:
            put("fsCVRecallKFour", r["top5_recall"])
            put("fsCVRecallKFourCI",
                f"[{r['top5_recall_ci_lo']:.3f}, {r['top5_recall_ci_hi']:.3f}]")
            put("fsCVSpearmanKFour", r["spearman"])
        # paired differences -- the statistic that actually decides SS49
        for rec in recs:
            if rec["experiment_id"] != "04_bootstrap_diff":
                continue
            if "+exact mean" in rec["comparison"]:
                nm = "fsDeltaVsHeadSub"
            elif "gaussian" in rec["comparison"]:
                nm = "fsDeltaOrtho"
            elif "control_variate" in rec["comparison"]:
                nm = "fsDeltaCV"
            else:
                continue
            put(nm, rec["delta_top5_recall"])
            put(nm + "CI", f"[{rec['ci_lo']:.3f}, {rec['ci_hi']:.3f}]")

    # --- batched vs serial exact baseline (spec SS17 items 3-4) ------------
    bv = RAW / "02d_batched_vs_serial.jsonl"
    if bv.exists():
        d = {(r["batch_size"], r["lanes"], r["impl"]): r["median_ms"]
             for r in load_jsonl(bv)}
        # Read the skip fraction from the gate records rather than restating it:
        # it was hardcoded at 0.860 here while the measured 3BPA value moved to
        # 0.844, so the two disagreed silently.
        _gb = RAW / "07_gate_baselines_maxcomp.jsonl"
        SKIP = 0.844
        if _gb.exists():
            _cv = [r for r in load_jsonl(_gb)
                   if r["gate"] == "control-variate K=4" and r["system"] == "3bpa"]
            if _cv:
                SKIP = _cv[0]["frac_exact_skipped"]
        for B in (1, 16):
            best = {L: min(d[(B, L, "serial")], d[(B, L, "batched")])
                    for L in (1, 3, 4, 5, 8)}
            put(f"fsBestTotalKThreeB{WORD[B]}", best[8] / best[4], "{:.2f}")
            put(f"fsBestIncrKThreeB{WORD[B]}",
                (best[8] - best[1]) / (best[4] - best[1]), "{:.2f}")
            # Uniform fallback accounting (see gate_metrics): the gate has already
            # evaluated 4 independent centered directions, so completing the exact
            # basis costs r-4 = 3 further lanes, not a fresh 8.
            gate = best[5] + (1 - SKIP) * best[3]
            put(f"fsBestGateSpeedupB{WORD[B]}", best[8] / gate, "{:.2f}")
        put("fsBatchedFlatLaneOne", d[(1, 1, "batched")], "{:.1f}")
        put("fsBatchedFlatLaneEight", d[(1, 8, "batched")], "{:.1f}")

    # --- partial torch.compile (spec SS17 item 5) --------------------------
    cp = RAW / "02c_compiled.jsonl"
    if cp.exists():
        recs = [r for r in load_jsonl(cp) if r.get("compiled")]
        if recs:
            sp_ups = [r["speedup"] for r in recs]
            put("fsCompileSpeedupLo", min(sp_ups), "{:.2f}")
            put("fsCompileSpeedupHi", max(sp_ups), "{:.2f}")
            put("fsCompileRelErr", f"{recs[0]['correctness_rel_err']:.1e}"
                .replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}")

    # --- gate baselines (revision Stage C): energy / head-exact-mean / haar / CV
    gb = RAW / "07_gate_baselines_maxcomp.jsonl"
    if gb.exists():
        recs = load_jsonl(gb)
        SHORT = {"energy (free)": "Energy", "head-exact-mean K=4": "HeadExact",
                 "haar K=4": "Haar", "control-variate K=4": "CV"}
        for gate, short in SHORT.items():
            rs = [r for r in recs if r["gate"] == gate]
            if not rs:
                continue
            put(f"fsGate{short}SkipLo", min(r["frac_exact_skipped"] for r in rs))
            put(f"fsGate{short}SkipHi", max(r["frac_exact_skipped"] for r in rs))
            put(f"fsGate{short}RecallLo", min(r["high_uq_recall"] for r in rs))
            put(f"fsGate{short}RecallHi", max(r["high_uq_recall"] for r in rs))
            put(f"fsGate{short}SpeedLo", min(r["screening_speedup"] for r in rs), "{:.2f}")
            put(f"fsGate{short}SpeedHi", max(r["screening_speedup"] for r in rs), "{:.2f}")
        # per-system rows, so the paper can state the ONE case that is not a
        # Pareto domination (ethanol: the free energy gate attains recall 1.000)
        SYS = {"3bpa": "ThreeBPA", "ethanol": "Ethanol",
               "aspirin": "Aspirin", "azobenzene": "Azobenzene"}
        for gate, short in SHORT.items():
            for sysname, sysshort in SYS.items():
                rs = [r for r in recs if r["gate"] == gate and r["system"] == sysname]
                if not rs:
                    continue
                r = rs[0]
                put(f"fsGate{short}Skip{sysshort}", r["frac_exact_skipped"])
                put(f"fsGate{short}Recall{sysshort}", r["high_uq_recall"])
                put(f"fsGate{short}Speed{sysshort}", r["screening_speedup"], "{:.2f}")
                put(f"fsGate{short}Recall{sysshort}CI",
                    f"[{r['recall_ci_lo']:.3f}, {r['recall_ci_hi']:.3f}]")
                put(f"fsGate{short}Skip{sysshort}CI",
                    f"[{r['skip_ci_lo']:.3f}, {r['skip_ci_hi']:.3f}]")
        # Split sizes and realised test prevalence. The reviewer's question -- is
        # the conformal calibration split really disjoint from the split that fit
        # Q_{r0}? -- has to be answerable from the PDF, not from the code.
        for sysname, sysshort in SYS.items():
            rs = [r for r in recs if r["system"] == sysname]
            if not rs or "n_design" not in rs[0]:
                continue
            put(f"fsNDesign{sysshort}", str(rs[0]["n_design"]))
            put(f"fsNCal{sysshort}", str(rs[0]["n_cal"]))
            put(f"fsNTest{sysshort}", str(rs[0]["n_test"]))
        prev = [r["test_prevalence"] for r in recs if "test_prevalence" in r]
        if prev:
            put("fsTestPrevalenceLo", min(prev))
            put("fsTestPrevalenceHi", max(prev))
        lc = {r.get("lanes_complete") for r in recs if "lanes_complete" in r}
        if lc:
            put("fsFallbackLanesSketch", f"{int(min(lc))}")
            put("fsFallbackLanesEnergy", f"{int(max(lc))}")

        # how many of the 12 (4 systems x 3 baselines) the control variate dominates
        ndom = 0
        for sysname in SYS:
            cvr = [r for r in recs if r["gate"] == "control-variate K=4" and r["system"] == sysname]
            if not cvr:
                continue
            cv = cvr[0]
            for gate in SHORT:
                if gate == "control-variate K=4":
                    continue
                b = [r for r in recs if r["gate"] == gate and r["system"] == sysname]
                if b and cv["frac_exact_skipped"] > b[0]["frac_exact_skipped"] \
                        and cv["high_uq_recall"] > b[0]["high_uq_recall"]:
                    ndom += 1
        for nm in ("fsGateCVSkipLo", "fsGateCVSkipHi",
                   "fsGateCVRecallLo", "fsGateCVRecallHi"):
            if nm in vals:
                put_pct(nm + "Pct", vals[nm])
        put("fsGateCVDominates", str(ndom))
        put("fsGateCVComparisons", str(len(SYS) * (len(SHORT) - 1)))

    import glob as _glob

    # --- max-component acquisition (the PRIMARY rule; revision Stage B) --------
    # Ranges are across all four systems, so the paper never presents 3BPA as
    # universal. `rank_bias_ratio` quantifies the extreme-value bias that the
    # marginal finite-K correction provably does NOT remove (Stage B3).
    mc = sorted(_glob.glob(str(RAW / "03_sketch_fidelity_*_maxcomp.jsonl")))
    if mc:
        per_sys = [mean_by(load_jsonl(Path(f)), ("method", "K"),
                           ("top5_recall", "spearman", "rank_bias_ratio")) for f in mc]
        put("fsMaxNSystems", str(len(per_sys)))
        for K in (2, 3, 4, 5, 6):
            rows = [g[("haar", K)] for g in per_sys if ("haar", K) in g]
            if not rows:
                continue
            put(f"fsMaxRecallK{WORD[K]}Lo", min(r["top5_recall"] for r in rows))
            put(f"fsMaxRecallK{WORD[K]}Hi", max(r["top5_recall"] for r in rows))
            put(f"fsMaxSpearmanK{WORD[K]}Lo", min(r["spearman"] for r in rows))
            put(f"fsMaxSpearmanK{WORD[K]}Hi", max(r["spearman"] for r in rows))
        for K in (1, 4):
            rows = [g[("haar", K)] for g in per_sys if ("haar", K) in g]
            if rows:
                put(f"fsMaxBiasK{WORD[K]}Lo", min(r["rank_bias_ratio"] for r in rows), "{:.2f}")
                put(f"fsMaxBiasK{WORD[K]}Hi", max(r["rank_bias_ratio"] for r in rows), "{:.2f}")
        for nm in ("fsMaxRecallKSixLo", "fsMaxRecallKSixHi",
                   "fsMaxRecallKFourLo", "fsMaxRecallKFourHi"):
            if nm in vals:
                put_pct(nm + "Pct", vals[nm])
        # the exactness check: Haar at K=r is the exact basis, so bias must be 1
        rows = [g[("haar", 7)] for g in per_sys if ("haar", 7) in g]
        if rows:
            put("fsMaxBiasExact", max(r["rank_bias_ratio"] for r in rows), "{:.3f}")

    # --- SS5.3 prose numbers (revision F2b): serial-only comparison, and the
    # kernel-family composition, both of which were literals in the manuscript.
    bvs = RAW / "02d_batched_vs_serial.jsonl"
    if bvs.exists():
        d = {(r["batch_size"], r["lanes"], r["impl"]): r["median_ms"]
             for r in load_jsonl(bvs)}
        for B in sorted({k[0] for k in d}):
            if (B, 8, "serial") in d and (B, 4, "serial") in d:
                put(f"fsSerialTotalKThreeB{WORD[B]}",
                    d[(B, 8, "serial")] / d[(B, 4, "serial")], "{:.2f}")
        # Where does batched reverse mode STOP winning? The manuscript previously
        # asserted an out-of-memory failure at B=64; the records show it completes
        # and is simply slower, so we report the measured crossover instead.
        for B in sorted({k[0] for k in d}):
            if (B, 8, "batched") in d and (B, 8, "serial") in d:
                put(f"fsBatchedWinB{WORD[B]}",
                    d[(B, 8, "serial")] / d[(B, 8, "batched")], "{:.2f}")
        lost = [B for B in sorted({k[0] for k in d})
                if all(d.get((B, L, "batched"), 0) > d.get((B, L, "serial"), float("inf"))
                       for L in range(1, 9))]
        if lost:
            put("fsBatchedLosesBatch", str(lost[0]))
            put("fsBatchedRatioLost",
                d[(lost[0], 8, "batched")] / d[(lost[0], 8, "serial")], "{:.2f}")

    kb = PROC / "kernel_breakdown.json"
    if kb.exists():
        k = json.loads(kb.read_text())

        def frac(tag, fam):
            f = k[tag]["families"]
            return 100.0 * f[fam]["ms"] / sum(v["ms"] for v in f.values())

        def launches(tag):
            return sum(v["launches"] for v in k[tag]["families"].values())

        sketch = [t for t in k if t != "exact"]
        put("fsKernElemExact", frac("exact", "elementwise / activation"), "{:.1f}")
        put("fsKernElemSketchLo", min(frac(t, "elementwise / activation") for t in sketch), "{:.1f}")
        put("fsKernElemSketchHi", max(frac(t, "elementwise / activation") for t in sketch), "{:.1f}")
        put("fsKernTPExact", frac("exact", "tensor product / e3nn"), "{:.1f}")
        put("fsKernTPSketchLo", min(frac(t, "tensor product / e3nn") for t in sketch), "{:.1f}")
        put("fsKernTPSketchHi", max(frac(t, "tensor product / e3nn") for t in sketch), "{:.1f}")
        put("fsKernDistinct", str(sum(v["kernels"] for v in k["exact"]["families"].values())))
        # Launch COUNTS are totals over the profiled window, whose iteration count
        # the record does not carry -- so we report the ratio, which is invariant
        # to it, rather than a per-step figure we would have to assume.
        if "haar_K3" in k:
            put("fsKernLaunchRatio", launches("exact") / launches("haar_K3"), "{:.2f}")
            put("fsKernLaneRatio", k["exact"]["lanes"] / k["haar_K3"]["lanes"], "{:.2f}")

    # --- per-system paired differences the prose quotes individually (F2b) -----
    NAMED = {"rmd17-disjoint_aspirin": "Aspirin", "disjoint_test_1200K": "ThreeBPA"}
    for tag, short in NAMED.items():
        f = RAW / f"04_bootstrap_{tag}.jsonl"
        if not f.exists():
            continue
        for rec in load_jsonl(f):
            c = rec.get("comparison", "")
            if "+exact mean" in c:
                nm = f"fsDeltaVsHeadSub{short}"
            elif "head_subsample K=4" in c:
                nm = f"fsDeltaVsHeadSubNoMean{short}"
            else:
                continue
            put(nm, rec["delta_top5_recall"])
            put(nm + "CI", f"[{rec['ci_lo']:.3f}, {rec['ci_hi']:.3f}]")

    # --- seed-to-seed dispersion, the other literal range in SS5.1 -------------
    fid_files = sorted(_glob.glob(str(RAW / "03_sketch_fidelity_*_maxcomp.jsonl")))
    sds = []
    for f in fid_files:
        recs = load_jsonl(Path(f))
        for K in (2, 3, 4):
            seed_vals = [r["top5_recall"] for r in recs
                         if r["method"] == "haar" and r["K"] == K]
            if len(seed_vals) > 1:
                sds.append(float(np.std(seed_vals, ddof=1)))
    if sds:
        put("fsSeedSdLo", min(sds), "{:.2f}")
        put("fsSeedSdHi", max(sds), "{:.2f}")

    sps = sorted(_glob.glob(str(PROC / "spectrum_*.json")))
    if sps:
        specs = [json.loads(Path(f).read_text()) for f in sps]
        sr = [x["stable_rank_FQ"] for x in specs if "stable_rank_FQ" in x]
        t1 = [x["top1_fraction"] for x in specs]
        if sr:
            put("fsSrankLo", min(sr), "{:.2f}")
            put("fsSrankHi", max(sr), "{:.2f}")
        put("fsTopEigLo", min(t1))
        put("fsTopEigHi", max(t1))
        iso = specs[0].get("isotropic_top1_fraction", 1 / 7)
        put("fsTopEigIsotropic", iso)
        put("fsTopEigRatioLo", min(t1) / iso, "{:.1f}")
        put("fsTopEigRatioHi", max(t1) / iso, "{:.1f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("% Generated by analysis/tables.py -- do not edit by hand.\n"
                   + "\n".join(macros) + "\n")
    return vals


def table_batched_vs_serial(batch_sizes=(1, 16), lanes=(1, 4, 8)) -> str:
    """The SS5.3 latency table, generated (revision F2b).

    This table was hand-written into the manuscript, which is precisely how a
    paper drifts from its own data: nine literal decimals with no path back to a
    record. Emitting it means the numbers cannot silently go stale.
    """
    recs = load_jsonl(RAW / "02d_batched_vs_serial.jsonl")
    d = {(r["batch_size"], r["lanes"], r["impl"]): r["median_ms"] for r in recs}
    atoms = {r["batch_size"]: r["num_atoms"] for r in recs}
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{$B={B}$ ({atoms[B]} atoms)}}" for B in batch_sizes)
    rows = [f"\\begin{{tabular}}{{r{'rr' * len(batch_sizes)}}}", "\\toprule",
            f" & {head} \\\\",
            "lanes $L$ & " + " & ".join("serial & batched" for _ in batch_sizes) + " \\\\",
            "\\midrule"]
    for L in lanes:
        cells = []
        for B in batch_sizes:
            for impl in ("serial", "batched"):
                t = d.get((B, L, impl))
                if t is None:
                    cells.append("---")
                    continue
                other = d.get((B, L, "batched" if impl == "serial" else "serial"))
                win = other is not None and t < other
                cells.append(f"\\textbf{{{t:.1f}}}" if win else f"{t:.1f}")
        rows.append(f"{L} & " + " & ".join(cells) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows)
