#!/usr/bin/env python
"""Manuscript figures, each derived from a record. Nothing drawn by hand.

Same discipline as `tools/paper_numbers.py`: a figure that cannot be regenerated
from `results/records/` does not belong in the paper. Each function declares the
records it reads, and `--provenance` writes figure -> {records, sha256} so
`tools/audit.py` can verify the link still holds.

DESIGN NOTES (why it looks the way it does)

* Form is chosen from the data's job, not from habit. Factor A is a comparison of
  many signals within each system -> a dot plot, not grouped bars, because bars
  spend ink on a baseline that carries no meaning for AUROC. The gate comparison
  is a signed difference with an interval -> a forest plot, which shows sign and
  significance directly rather than making the reader subtract two bar heights.
* Colour uses the validated reference palette's categorical slots in FIXED order
  (blue, orange, aqua, yellow, ...), never cycled or reassigned by rank. The
  palette validator ships as a node script and node is not available on this
  cluster, so rather than eyeball a new palette we do not deviate from the
  published slots.
* PRINT AND GRAYSCALE: every series carries a distinct MARKER and LINESTYLE as
  well as a hue, so identity never depends on colour. That is required anyway for
  a journal that may print in grayscale, and it satisfies the secondary-encoding
  rule for any pair near the CVD separation floor.
* Text is ink-coloured, never series-coloured. Grid and spines are recessive. No
  dual axes anywhere. Reference lines (the random null at 0.5, the pool median)
  are drawn once and labelled, since a reader cannot be expected to remember them.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
REC = ROOT / "results/records"
OUT = ROOT / "paper/figures"

# validated reference palette, categorical slots in fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
_USED: dict[str, set] = collections.defaultdict(set)


def rd(name: str):
    _USED[name]  # touch
    p = REC / name
    raw = p.read_bytes()
    _USED[name].add(hashlib.sha256(raw).hexdigest())
    return ([json.loads(l) for l in raw.decode().splitlines() if l.strip()]
            if name.endswith(".jsonl") else json.loads(raw))


def style(ax, *, xlabel="", ylabel="", title=""):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, width=0.8)
    ax.grid(True, color="#e6e5e0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
    if title:
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=6)


SHORT = {"disjoint_test_1200K": "3BPA disjoint", "overlapping_test_1200K": "3BPA overlapping",
         "same_test_1200K": "3BPA same", "rmd17-disjoint_ethanol": "ethanol",
         "rmd17-disjoint_aspirin": "aspirin", "rmd17-disjoint_azobenzene": "azobenzene"}
ORDER = list(SHORT)


# ---------------------------------------------------------------- figure 1
def fig_factor_a():
    """Dot plot: AUROC by signal within each system, with the random null marked."""
    rows = rd("j2a_oracle_panel.jsonl")
    by = collections.defaultdict(dict)
    for r in rows:
        if r["error_score"] == "e_max":
            by[r["cache_tag"]][r["signal"]] = r["auroc_top05"]
    sigs = [("exact_global", "exact global"), ("exact_maxcomp", "exact max-component"),
            ("exact_maxatom", "exact max-atom"), ("force_norm", r"$\|\bar{f}\|$ (free)"),
            ("energy_std", "energy std (free)")]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    y = np.arange(len(ORDER))[::-1]
    ax.axvline(0.5, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(0.502, y.max() + 0.55, "random null", fontsize=7.5, color=INK2, va="bottom")
    for i, (key, lab) in enumerate(sigs):
        xs = [by[s].get(key, np.nan) for s in ORDER]
        ax.scatter(xs, y, s=42, marker=MARKERS[i], facecolor=C[i], edgecolor="white",
                   linewidth=0.8, label=lab, zorder=3)
    ax.set_yticks(y, [SHORT[s] for s in ORDER], fontsize=8)
    style(ax, xlabel="AUROC, top-5% highest-error structures")
    ax.set_xlim(0.42, 0.90)
    # Legend BELOW the axes: inside, it sat on the aspirin and azobenzene rows.
    ax.legend(fontsize=7.6, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), labelcolor=INK2, handletextpad=0.3,
              columnspacing=1.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_factor_a.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_gate_forest():
    """Forest plot of leading-only minus control-variate skip, three budgets."""
    D = {L: {r["system"]: r for r in rd(f"j9a_leading_only_ci_global_L{L}.json")}
         for L in (4, 5, 6)}
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y0 = np.arange(len(ORDER))[::-1] * 1.0
    off = {4: +0.26, 5: 0.0, 6: -0.26}
    ax.axvline(0.0, color=INK2, lw=1.0, zorder=2)
    for i, L in enumerate((4, 5, 6)):
        for j, s in enumerate(ORDER):
            r = D[L][s]
            yy = y0[j] + off[L]
            ax.plot([r["ci_lo"], r["ci_hi"]], [yy, yy], color=C[i], lw=1.8,
                    solid_capstyle="round", zorder=3)
            ax.scatter([r["delta"]], [yy], s=26, marker=MARKERS[i], color=C[i],
                       edgecolor="white", linewidth=0.7, zorder=4,
                       label=f"{L} lanes" if j == 0 else None)
    ax.set_yticks(y0, [SHORT[s] for s in ORDER], fontsize=8)
    style(ax, xlabel="leading-only $-$ control variate, fraction of exact evaluations skipped")
    ax.text(0.004, y0.max() + 0.62, "leading-only better →", fontsize=7.5, color=INK2)
    ax.text(-0.004, y0.max() + 0.62, "← control variate better", fontsize=7.5,
            color=INK2, ha="right")
    ax.legend(fontsize=7.6, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), labelcolor=INK2, handletextpad=0.3,
              columnspacing=1.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_gate_forest.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_extensivity():
    """Two panels: acquired system size, and rank correlation with system size."""
    v = rd("j6b_foundation_verdict.json")
    med = v["acquisition_median_atoms"]
    labs = [("exact_global", "exact global"), ("energy_std", "energy std"),
            ("leading_only", "leading-only"), ("exact_maxatom", "exact max-atom"),
            ("exact_maxcomp", "exact max-comp"), ("force_norm", r"$\|\bar{f}\|$"),
            ("random_null", "random")]
    pool = v["detail"]["Q3"]["median_atoms_pool"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"wspace": 0.42})

    ax = axes[0]
    y = np.arange(len(labs))[::-1]
    for i, (k, lab) in enumerate(labs):
        for b, mk, al in ((1000, MARKERS[0], 1.0), (8000, MARKERS[1], 1.0)):
            val = med.get(f"{k}_{b}")
            if val is None:
                continue
            ax.scatter([val], [y[i]], s=40, marker=mk, color=C[0] if b == 1000 else C[1],
                       edgecolor="white", linewidth=0.8, zorder=3,
                       label=(f"budget {b}" if i == 0 else None), alpha=al)
    ax.axvline(pool, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(pool + 2.0, y.min() - 0.05, f"pool median ({pool:.0f})", fontsize=7,
            color=INK2, va="bottom", ha="left")
    ax.set_yticks(y, [l for _, l in labs], fontsize=7.6)
    style(ax, xlabel="median atoms in the acquired set", title="a  what gets acquired")
    ax.legend(fontsize=7.2, frameon=False, loc="lower right", labelcolor=INK2,
              handletextpad=0.3)

    ax = axes[1]
    n = rd("j6c_size_normalisation.json")
    alphas = [float(k.split("=")[1]) for k in n["held_out_auroc"] if k.startswith("S_alpha")]
    au = [n["held_out_auroc"][f"S_alpha={a}"]["e_max"] for a in alphas]
    ax.plot(alphas, au, color=C[0], lw=1.8, marker=MARKERS[0], ms=5,
            markeredgecolor="white", markeredgewidth=0.7, zorder=3, label=r"$S_\alpha$")
    mc = n["held_out_auroc"]["exact_maxcomp"]["e_max"]
    ax.axhline(mc, color=C[1], lw=1.5, ls=LINES[1], zorder=2, label="exact max-component")
    astar = n["alpha_star"]
    ax.scatter([astar], [n["held_out_auroc"][f"S_alpha={astar}"]["e_max"]], s=95,
               marker="*", color=C[3], edgecolor=INK, linewidth=0.5, zorder=5,
               label=rf"$\alpha^*={astar}$ (design-selected)")
    style(ax, xlabel=r"size-normalisation exponent $\alpha$", ylabel="held-out AUROC",
          title="b  the repair, and its ceiling")
    ax.legend(fontsize=7.2, frameon=False, loc="lower center", labelcolor=INK2,
              handletextpad=0.4)
    fig.savefig(OUT / "fig3_extensivity.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_speedup():
    """Speedup vs batch size, three devices, water and 3BPA."""
    DEV = [("H200", "NVIDIA_H200_pair"), ("A100-80GB", "NVIDIA_A100-SXM4-80GB_pair"),
           ("RTX PRO 6000 Blackwell", "NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition_pair")]

    def sp(rows, B):
        t = lambda L, i: [r["median_ms"] for r in rows if r["batch_size"] == B
                          and r["lanes"] == L and r["impl"] == i and r.get("status") == "ok"]
        ex, k3 = t(8, "serial") + t(8, "batched"), t(4, "serial") + t(4, "batched")
        return min(ex) / min(k3) if ex and k3 else None

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for panel, (sysname, pretty) in enumerate((("water", "water, 192 atoms, periodic"),
                                               ("3bpa", "3BPA, 27 atoms"))):
        ax = axes[panel]
        for i, (lab, tag) in enumerate(DEV):
            f = (f"j8_lane_bench_water_{tag}.jsonl" if sysname == "water"
                 else f"j8_lane_bench_{tag}.jsonl")
            rows = rd(f)
            bs = sorted({r["batch_size"] for r in rows})
            xy = [(b, sp(rows, b)) for b in bs]
            xy = [(b, s) for b, s in xy if s]
            ax.plot([b for b, _ in xy], [s for _, s in xy], color=C[i], lw=1.8,
                    ls=LINES[i], marker=MARKERS[i], ms=5, markeredgecolor="white",
                    markeredgewidth=0.7, zorder=3, label=lab if panel == 0 else None)
        ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log", base=2)
        style(ax, xlabel="batch size", ylabel="total speedup" if panel == 0 else "",
              title=f"{'ab'[panel]}  {pretty}")
        ax.set_ylim(0.95, 2.0)
    axes[0].legend(fontsize=7.4, frameon=False, loc="lower right", labelcolor=INK2,
                   handletextpad=0.5)
    fig.savefig(OUT / "fig4_speedup.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_beta_law():
    """Predicted vs measured recall; the identity line is the claim."""
    rows = rd("j3_extreme_value_v2.json")["recall_prediction"]
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    lo, hi = 0.30, 1.0
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1,
            label="perfect prediction")
    for i, (mode, lab) in enumerate((("shared_real", "real geometry"),
                                     ("shared_iso", "isotropic null"))):
        ax.scatter([r["measured"] for r in rows], [r[mode] for r in rows], s=34,
                   marker=MARKERS[i], facecolor=C[i], edgecolor="white",
                   linewidth=0.7, zorder=3, label=lab)
    style(ax, xlabel="measured top-5% recall", ylabel="predicted from the Beta law")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=7.2, frameon=False, loc="upper left", labelcolor=INK2,
              handletextpad=0.3)
    fig.savefig(OUT / "fig5_beta_law.pdf", bbox_inches="tight")
    plt.close(fig)


FIGS = {"fig1_factor_a": fig_factor_a, "fig2_gate_forest": fig_gate_forest,
        "fig3_extensivity": fig_extensivity, "fig4_speedup": fig_speedup,
        "fig5_beta_law": fig_beta_law}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--provenance", default="results/records/figure_provenance.json")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prov = {}
    for name, fn in FIGS.items():
        if a.only and name not in a.only:
            continue
        _USED.clear()
        fn()
        prov[name] = {"records": {k: sorted(v)[0] for k, v in _USED.items()}}
        print(f"  {name}.pdf  <- {', '.join(sorted(_USED))}")
    p = ROOT / a.provenance
    if p.exists() and a.only:
        old = json.loads(p.read_text())
        old.update(prov)
        prov = old
    p.write_text(json.dumps(prov, indent=1, sort_keys=True))
    print(f"wrote {len(FIGS)} figures to {OUT} and provenance to {a.provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
