#!/usr/bin/env python
"""Manuscript tables as generated .tex fragments, derived from records.

Same rule as the macros and figures: a table typed by hand is a table that drifts.
Each fragment is \\input by main.tex, so the numbers in the typeset table and the
numbers in the records cannot disagree.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REC = ROOT / "results/records"
OUT = ROOT / "paper/tables"


def rd(n):
    raw = (REC / n).read_text()
    return ([json.loads(l) for l in raw.splitlines() if l.strip()]
            if n.endswith(".jsonl") else json.loads(raw))


def table_mptraj():
    """Signal comparison on the foundation-model pool -- the scoping result."""
    fa = rd("j6b_foundation_verdict.json")["factor_a"]
    ph = rd("j6b_foundation_verdict.json")["post_hoc"]
    rows = [("exact max-atom", "exact_maxatom"), ("exact max-component", "exact_maxcomp"),
            ("exact global", "exact_global"), ("leading-only, 5 lanes", "leading_only"),
            (r"$\|\bar{f}\|$ \emph{(zero cost)}", "force_norm"),
            ("energy std \\emph{(zero cost)}", "energy_std"), ("random null", "random_null")]
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"& \multicolumn{4}{c}{AUROC by error score} & \multicolumn{2}{c}{stratified (post hoc)}\\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
         r"signal & $e_{\max}$ & $e_{\mathrm{rmse}}$ & $e_{\max\mathrm{c}}$ & $e_{q95}$"
         r" & by size & by force \\", r"\midrule"]
    for lab, k in rows:
        c = fa[k]
        ss = ph["size_stratified_auroc_e_max"].get(k, {}).get("mean")
        fs = ph["force_stratified_auroc_e_max"].get(k, {}).get("mean")
        bold = (lambda x: rf"\textbf{{{x:.3f}}}") if k == "force_norm" else (lambda x: f"{x:.3f}")
        L.append(f"{lab} & " + " & ".join(bold(c[e]) for e in
                 ("e_max", "e_rmse", "e_maxcomp", "e_q95"))
                 + f" & {'--' if ss is None else bold(ss)}"
                 + f" & {'--' if fs is None else bold(fs)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_mptraj.tex").write_text("\n".join(L) + "\n")
    return "table_mptraj"


def table_pairs():
    """Matched-pair verdicts and speedups, by device."""
    # Two tag forms, because the filenames differ: verdict records are named by
    # DEVICE, lane-bench records by DEVICE_pair. Conflating them cost a run once.
    DEV = [("H200", "NVIDIA_H200"), ("A100-SXM4-80GB", "NVIDIA_A100-SXM4-80GB"),
           ("RTX PRO 6000 Blackwell", "NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition")]

    def sp(rows, B):
        t = lambda L, i: [r["median_ms"] for r in rows if r["batch_size"] == B
                          and r["lanes"] == L and r["impl"] == i and r.get("status") == "ok"]
        ex, k3 = t(8, "serial") + t(8, "batched"), t(4, "serial") + t(4, "batched")
        return min(ex) / min(k3) if ex and k3 else float("nan")

    L = [r"\begin{tabular}{lccccccc}", r"\toprule",
         r"& \multicolumn{2}{c}{verdict} & \multicolumn{4}{c}{pre-registered predictions}"
         r" & speedup \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-8}",
         r"device & v2 & v3 & P1 & P2 & P3 & P4 & water, $B{=}16$ \\", r"\midrule"]
    for lab, tag in DEV:
        a = rd(f"j8_verdict_pair_{tag}_v2.json")
        b = rd(f"j8_verdict_pair_{tag}_v3.json")
        na = sum(p["passed"] for p in a["predictions"].values())
        nb = sum(p["passed"] for p in b["predictions"].values())
        marks = " & ".join(r"\checkmark" if a["predictions"][k]["passed"] else r"$\times$"
                           for k in ("P1", "P2", "P3", "P4"))
        w = sp(rd(f"j8_lane_bench_water_{tag}_pair.jsonl"), 16)
        L.append(rf"{lab} & \textbf{{{na}/4}} & {nb}/4 & {marks} & {w:.3f}$\times$ \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_pairs.tex").write_text("\n".join(L) + "\n")
    return "table_pairs"


def table_accuracy():
    """Accuracy and calibration of every committee (reviewer 1, point 3).

    One row per (system, construction, split): force RMSE/MAE of the committee
    mean, energy MAE per atom where a reference energy is cached, the median
    per-atom committee std, the overconfidence ratio and the per-atom Spearman
    correlation between std and error (per structure for the materials pools).
    """
    rows = rd("j12_accuracy.json")["rows"]
    NAMES = {  # tag -> (system, construction, split)
        "3bpa-disjoint_test_300K": ("3BPA", "disjoint heads", "300 K"),
        "3bpa-disjoint_test_600K": ("3BPA", "disjoint heads", "600 K"),
        "3bpa-disjoint_test_1200K": ("3BPA", "disjoint heads", "1200 K"),
        "3bpa-overlapping_test_300K": ("3BPA", "overlapping heads", "300 K"),
        "3bpa-overlapping_test_600K": ("3BPA", "overlapping heads", "600 K"),
        "3bpa-overlapping_test_1200K": ("3BPA", "overlapping heads", "1200 K"),
        "3bpa-same_test_300K": ("3BPA", "identical data", "300 K"),
        "3bpa-same_test_600K": ("3BPA", "identical data", "600 K"),
        "3bpa-same_test_1200K": ("3BPA", "identical data", "1200 K"),
        "3bpa-naive-same_test_1200K": ("3BPA", "independent, identical data", "1200 K"),
        "3bpa-naive_test_1200K": ("3BPA", "independent, resampled data", "1200 K"),
        "rmd17-disjoint_ethanol": ("ethanol", "disjoint heads", "test"),
        "rmd17-disjoint_aspirin": ("aspirin", "disjoint heads", "test"),
        "rmd17-disjoint_azobenzene": ("azobenzene", "disjoint heads", "test"),
        "water-disjoint_md_T300K": ("water", "disjoint heads", "MD"),
        "water-disjoint_pimd_T300K": ("water", "disjoint heads", "PIMD"),
        "water-overlapping_md_T300K": ("water", "overlapping heads", "MD"),
        "water-overlapping_pimd_T300K": ("water", "overlapping heads", "PIMD"),
        "water-naive_md_T300K": ("water", "independent", "MD"),
        "water-naive_pimd_T300K": ("water", "independent", "PIMD"),
        "j6b_foundation_scan": ("MPtraj pool", "foundation, QbC-8000 heads", "in-distribution"),
        "j10_qbc8000_aimd-from-PBE-1000-nvt": ("OMat24 AIMD NVT", "foundation, QbC-8000 heads", "out-of-distribution"),
        "j10_qbc8000_aimd-from-PBE-1000-npt": ("OMat24 AIMD NpT", "foundation, QbC-8000 heads", "out-of-distribution"),
        "j10_qbc8000_rattled-1000-subsampled": ("OMat24 rattled", "foundation, QbC-8000 heads", "out-of-distribution"),
    }
    by = {r["tag"]: r for r in rows}

    def f(x, n=1):
        return "--" if x is None else f"{x:.{n}f}"

    L = [r"\begin{tabular}{lllrrrrrr}", r"\toprule",
         r"system & construction & split & $F$ RMSE & $F$ MAE & $E$ MAE & $\tilde\sigma_a$ & overconf. & $\rho_s$ \\",
         r" & & & \multicolumn{2}{c}{(meV/\AA)} & (meV/atom) & (meV/\AA) & & \\", r"\midrule"]
    last = None
    for tag, (sys_, con, split) in NAMES.items():
        r = by.get(tag)
        if r is None:
            continue
        if last and last != sys_:
            L.append(r"\addlinespace[2pt]")
        last = sys_
        rho = r.get("spearman_atom", r.get("spearman_struct"))
        oc = "--" if r.get("overconfidence") is None else f(r["overconfidence"]) + r"$\times$"
        L.append(f"{sys_} & {con} & {split} & {f(r.get('force_rmse'))} & {f(r.get('force_mae'))} & "
                 f"{f(r.get('energy_mae_atom'), 2)} & {f(r.get('disagreement_median'))} & {oc} & {f(rho, 2)} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_accuracy.tex").write_text("\n".join(L) + "\n")
    return "table_accuracy"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (table_mptraj, table_pairs, table_accuracy):
        print(f"  wrote paper/tables/{f()}.tex")
