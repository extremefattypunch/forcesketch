"""Figures 2-5 (spec SS29, SS30, SS53). Generated from results/raw only."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The ONE lane-cost model, fitted from results/raw/lane_scaling_disjoint.jsonl.
# Restating it here as literals is what let the paper carry four different slopes.
from forcesketch.screening.fallback_gate import LaneCostModel as _LCM

INTERCEPT, SLOPE = _LCM().intercept_ms, _LCM().slope_ms_per_lane
STYLE = {
    "haar": ("o", "#1f77b4"), "gaussian": ("s", "#d62728"),
    "rademacher": ("^", "#2ca02c"), "pairwise": ("D", "#9467bd"),
    "head_subsample": ("v", "#ff7f0e"), "control_variate": ("*", "#17becf"),
}


def lane_ms(lanes: int) -> float:
    return INTERCEPT + SLOPE * lanes


def incremental_uq_ms(lanes_total: int) -> float:
    """T_UQ = T(L=1+K) - T(L=1): the mean-force lane is not uncertainty cost (spec SS45)."""
    return lane_ms(lanes_total) - lane_ms(1)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def pareto(fidelity_jsonl: Path, out: Path, *, title: str) -> None:
    """Figure 2 (spec SS29): incremental force-UQ latency vs top-5% recall."""
    recs = load_jsonl(fidelity_jsonl)
    agg: dict[tuple, list] = defaultdict(list)
    for r in recs:
        if r["method"] == "head_subsample" and not r["with_mean_lane"]:
            continue  # the +mean variant is the like-for-like comparison
        agg[(r["method"], r["K"], r["total_lanes"])].append(r["top5_recall"])

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for method in STYLE:
        pts = sorted((incremental_uq_ms(L), float(np.mean(v)), K)
                     for (m, K, L), v in agg.items() if m == method)
        if not pts:
            continue
        marker, colour = STYLE[method]
        x, y, ks = zip(*pts)
        ax.plot(x, y, marker=marker, color=colour, label=method.replace("_", " "),
                lw=1.4, ms=7, alpha=0.9)
        for xi, yi, k in pts:
            ax.annotate(f"{k}", (xi, yi), textcoords="offset points",
                        xytext=(5, -9), fontsize=7, color=colour)

    ax.axhline(0.90, ls="--", c="0.4", lw=1)
    ax.text(ax.get_xlim()[1], 0.905, "H2 target 0.90", ha="right", fontsize=8, color="0.35")
    ax.set_xlabel("incremental force-UQ latency  $T(L{=}1{+}K) - T(L{=}1)$   [ms]")
    ax.set_ylabel("recall of exact top-5% uncertainty set")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def screening_curve(screening_jsonl: Path, out: Path, *, title: str) -> None:
    """Figure 5 (spec SS34): fraction of exact evaluations skipped vs high-UQ recall."""
    recs = load_jsonl(screening_jsonl)
    agg: dict[tuple, list] = defaultdict(list)
    for r in recs:
        key = (r["method"], r["K"], r["r0"], r["alpha"])
        agg[key].append((r["frac_exact_skipped"], r["high_uq_recall"]))

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    series: dict[tuple, list] = defaultdict(list)
    for (m, K, r0, alpha), v in agg.items():
        a = np.mean(v, axis=0)
        series[(m, K, r0)].append((alpha, a[0], a[1]))
    for (m, K, r0), pts in sorted(series.items()):
        pts.sort()
        _, x, y = zip(*pts)
        marker, colour = STYLE["control_variate" if r0 else m]
        label = f"{m}{f' r0={r0}' if r0 else ''} K={K}"
        ax.plot(x, y, marker=marker, ms=6, lw=1.3, alpha=0.85, label=label, color=colour)

    ax.axhline(0.95, ls="--", c="0.4", lw=1)
    ax.axvline(0.50, ls=":", c="0.6", lw=1)
    ax.text(0.51, 0.9505, "H5 targets", fontsize=8, color="0.35")
    ax.set_xlabel("fraction of exact UQ evaluations skipped")
    ax.set_ylabel("recall of exact high-uncertainty configurations")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def method_schematic(out: Path, *, M: int = 8, K: int = 4, r0: int = 2) -> None:
    """Figure 1 (spec SS53): what ForceSketch replaces, and where the gate sits.

    The spec's sketch stops at "exact vs approximate force-UQ". Since the result
    that holds is screening rather than replacement, this figure carries one step
    further: the approximate score feeds a calibrated gate that either clears a
    structure or falls back to the exact computation. Drawing it any other way
    would advertise a claim the data does not support.
    """
    import matplotlib.patches as mp

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x0, y0, x1, y1, fc, ec="0.3"):
        ax.add_patch(mp.FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                       boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=1.1))

    def arrow(x1, y1, x2, y2, c="0.35", lw=1.2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                    shrinkA=0, shrinkB=0))

    C_TRUNK, C_HEAD = "#d6e4f2", "#eef4fa"
    C_EXACT, C_FS, C_GATE = "#f7dede", "#dcefe4", "#fdf1d8"
    MID = 50.0

    # --- shared trunk: ONE forward ---------------------------------------
    box(28, 86, 72, 96, C_TRUNK)
    ax.text(MID, 91, "shared message-passing trunk", ha="center", va="center",
            fontsize=9.5, weight="bold")
    ax.text(MID, 98.5, "one forward pass  \u2014  all $M$ head energies", ha="center",
            fontsize=8, style="italic", color="0.4")
    arrow(MID, 86, MID, 82.5)

    # --- M heads ----------------------------------------------------------
    w, gap = 4.0, 1.0
    total = M * w + (M - 1) * gap
    x0 = MID - total / 2
    for i in range(M):
        box(x0 + i * (w + gap), 75.5, x0 + i * (w + gap) + w, 81.5, C_HEAD)
    ax.text(MID, 72, f"$M={M}$ scalar energy heads,  $e(x)\\in\\mathbb{{R}}^{{{M}}}$",
            ha="center", fontsize=8.5)

    # --- the reverse-mode split -------------------------------------------
    ax.text(MID, 68.8, "forces require reverse mode: one pass per head-space direction",
            ha="center", fontsize=8, style="italic", color="0.4")
    ax.plot([MID, MID], [66.5, 63.5], color="0.35", lw=1.2)
    ax.plot([23, 77], [63.5, 63.5], color="0.35", lw=1.2)
    arrow(23, 63.5, 23, 61); arrow(77, 63.5, 77, 61)

    # --- the two paths ----------------------------------------------------
    box(3, 40, 45, 61, C_EXACT)
    ax.text(24, 58, "EXACT", ha="center", fontsize=9.5, weight="bold", color="#8c2f2f")
    ax.text(24, 50.5,
            "1 mean direction  $s_0=\\frac{1}{M}\\mathbf{1}$\n"
            f"+ ${M-1}$ centered directions $q_j$",
            ha="center", va="center", fontsize=8.5)
    ax.text(24, 43, f"$\\mathbf{{{M}}}$ reverse lanes", ha="center", fontsize=9,
            weight="bold")

    box(55, 40, 97, 61, C_FS)
    ax.text(76, 58, "FORCESKETCH", ha="center", fontsize=9.5, weight="bold",
            color="#2f6b4a")
    ax.text(76, 50.5,
            "1 mean direction  $s_0$\n"
            f"+ $r_0={r0}$ leading directions (exact)\n"
            f"+ ${K-r0}$ Haar residual directions",
            ha="center", va="center", fontsize=8.5)
    ax.text(76, 43, f"$\\mathbf{{{K+1}}}$ reverse lanes", ha="center", fontsize=9,
            weight="bold")

    ax.text(24, 36.5, "exact  $v_d$,  $S(x)$", ha="center", fontsize=8.5)
    ax.text(76, 36.5, "unbiased  $\\hat v_d$,  $\\hat S(x)$", ha="center", fontsize=8.5)
    arrow(76, 40, 76, 30.5)

    # --- the screening gate ----------------------------------------------
    box(52, 10, 97, 29, C_GATE)
    ax.text(74.5, 25.5, "screening gate", ha="center", fontsize=9, weight="bold",
            color="#8a6d1f")
    ax.text(74.5, 17.5,
            "$U(x)=c_\\alpha\\,\\hat S(x)$   (calibrated, conservative)\n\n"
            "$U(x)<\\tau$   $\\Rightarrow$   skip exact force-UQ\n"
            "otherwise   $\\Rightarrow$   fall back to exact",
            ha="center", va="center", fontsize=8.5)

    arrow(52, 19.5, 26, 19.5, c="#b03030", lw=1.5)
    ax.text(38, 21.6, "fallback", ha="center", fontsize=8.5, color="#b03030",
            weight="bold")
    ax.plot([24, 24], [36.8, 19.5], color="#b03030", lw=1.2, ls=":")

    ax.text(MID, 3.5,
            f"$M={M}$ reverse lanes  $\\rightarrow$  $K+1={K+1}$, "
            "with the exact path run only where the calibrated bound\n"
            "cannot rule out the high-uncertainty threshold $\\tau$",
            ha="center", fontsize=8.5, color="0.2")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def k_tradeoff(fidelity_jsonl: Path, out: Path, *, title: str,
               cost=(INTERCEPT, SLOPE)) -> None:
    """Figure 3 (spec SS53): recall, Spearman and latency against K, per method.

    Three panels sharing the K axis, because the spec asks for exactly this
    comparison and because the curves disagree in an informative way -- ranking
    saturates far earlier than tail selection does.
    """
    a, b = cost
    recs = load_jsonl(fidelity_jsonl)
    agg: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["method"] == "head_subsample" and not r["with_mean_lane"]:
            continue
        agg[(r["method"], r["K"])]["recall"].append(r["top5_recall"])
        agg[(r["method"], r["K"])]["rho"].append(r["spearman"])
        agg[(r["method"], r["K"])]["lanes"].append(r["total_lanes"])

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    panels = [("recall", "top-5% recall of exact set", 0.90, "H2 target"),
              ("rho", "Spearman with exact $S$", 0.85, r"$\rho$ gate"),
              ("lanes", "total force+UQ latency [ms]", None, None)]

    for ax, (field, ylabel, hline, hlabel) in zip(axes, panels):
        for method in STYLE:
            pts = sorted((K, float(np.mean(v[field])), float(np.std(v[field])))
                         for (m, K), v in agg.items() if m == method)
            if not pts:
                continue
            marker, colour = STYLE[method]
            x, y, sd = zip(*pts)
            if field == "lanes":
                y = [a + b * yi for yi in y]
                sd = [0.0] * len(y)
            ax.errorbar(x, y, yerr=sd, marker=marker, color=colour, lw=1.4, ms=6,
                        capsize=2, alpha=0.9, label=method.replace("_", " "))
        if hline is not None:
            ax.axhline(hline, ls="--", c="0.45", lw=1)
            ax.text(ax.get_xlim()[1], hline + 0.012, hlabel, ha="right", fontsize=7.5,
                    color="0.35")
        ax.set_xlabel("$K$ (uncertainty directions)")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_xticks([1, 2, 3, 4, 7])
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def gate_pareto(gate_jsonl: Path, out: Path, *, title: str) -> None:
    """Skip fraction vs high-UQ recall for every gate, all systems (revision C4).

    This is the figure the paper's positive claim actually rests on, so it plots
    the comparison rather than ForceSketch alone: a calibrated gate is a generic
    wrapper, and the claim only stands if this gate beats the cheaper gates a
    reader would reach for at matched reverse-lane budget.

    Up-and-right is better in both axes, so Pareto dominance is visible directly
    rather than inferred from a table. Error bars are the 95% paired bootstrap
    intervals, which matter here: a top-5% positive set holds only ~50-110
    structures, so recall is the noisier of the two axes by some margin.
    """
    recs = load_jsonl(gate_jsonl)
    STYLE = {
        "energy (free)":       ("tab:gray",   "o", "energy disagreement (0 lanes, free)"),
        "head-exact-mean K=4": ("tab:orange", "s", "head subsampling + exact mean (5)"),
        "haar K=4":            ("tab:blue",   "^", "Haar sketch (5)"),
        "control-variate K=4": ("tab:red",    "*", "control variate (5)"),
    }
    SYSMARK = {"3bpa": 0, "ethanol": 1, "aspirin": 2, "azobenzene": 3}

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for gate, (colour, marker, label) in STYLE.items():
        rs = [r for r in recs if r["gate"] == gate]
        if not rs:
            continue
        x = [r["frac_exact_skipped"] for r in rs]
        y = [r["high_uq_recall"] for r in rs]
        xerr = np.abs(np.array([[r["frac_exact_skipped"] - r["skip_ci_lo"] for r in rs],
                                [r["skip_ci_hi"] - r["frac_exact_skipped"] for r in rs]]))
        yerr = np.abs(np.array([[r["high_uq_recall"] - r["recall_ci_lo"] for r in rs],
                                [r["recall_ci_hi"] - r["high_uq_recall"] for r in rs]]))
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=colour,
                    alpha=0.35, elinewidth=1, capsize=2, zorder=1)
        ax.scatter(x, y, c=colour, marker=marker,
                   s=180 if marker == "*" else 55, label=label,
                   edgecolors="black", linewidths=0.5, zorder=3)
        if gate == "control-variate K=4":
            # The control-variate points cluster tightly (that IS the result), so
            # the labels must be staggered or they overprint one another.
            OFFSETS = {"3bpa": (10, 10), "ethanol": (-58, 10),
                       "aspirin": (10, -18), "azobenzene": (-74, -18)}
            for r in rs:
                ax.annotate(r["system"], (r["frac_exact_skipped"], r["high_uq_recall"]),
                            textcoords="offset points",
                            xytext=OFFSETS.get(r["system"], (6, -10)),
                            fontsize=9, color="0.15", fontweight="semibold")

    ax.axhline(0.95, ls=":", c="0.5", lw=1)
    ax.text(0.02, 0.951, "0.95 recall", fontsize=7, color="0.4")
    ax.set_xlabel("fraction of exact force-UQ evaluations skipped  (higher is better)")
    ax.set_ylabel("high-UQ recall  (higher is better)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def maxcomp_recall(out: Path, *, pattern: str = "03_sketch_fidelity_*_maxcomp.jsonl",
                   raw: Path = Path("results/raw")) -> None:
    """Top-5% recall vs K under the PRIMARY max-component rule, all systems.

    The paper declares max-component acquisition primary but previously showed it
    only in prose, while both figures plotted the easier global statistic. This is
    the negative result made visible: recall climbs steadily with K and still does
    not reach the 0.90 line at K=6, on any system -- the exact basis is the only
    budget that recovers the tail.
    """
    import glob

    LABEL = {"disjoint_test_1200K": "3BPA @ 1200 K", "rmd17-disjoint_ethanol": "ethanol",
             "rmd17-disjoint_aspirin": "aspirin", "rmd17-disjoint_azobenzene": "azobenzene"}
    MARK = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for i, f in enumerate(sorted(glob.glob(str(raw / pattern)))):
        recs = [r for r in load_jsonl(Path(f)) if r["method"] == "haar"]
        if not recs:
            continue
        tag = Path(f).stem.replace("03_sketch_fidelity_", "").replace("_maxcomp", "")
        by = defaultdict(list)
        for r in recs:
            by[r["K"]].append(r["top5_recall"])
        Ks = sorted(by)
        mu = np.array([np.mean(by[k]) for k in Ks])
        sd = np.array([np.std(by[k], ddof=1) if len(by[k]) > 1 else 0.0 for k in Ks])
        ax.errorbar(Ks, mu, yerr=sd, marker=MARK[i % len(MARK)], capsize=3,
                    lw=1.6, ms=5, label=LABEL.get(tag, tag))

    ax.axhline(0.90, ls="--", c="crimson", lw=1.2)
    ax.text(6.9, 0.915, "0.90 = replacement quality", color="crimson", fontsize=8,
            ha="right")
    ax.set_xlabel("head-space directions $K$   (exact basis at $K=7$)")
    ax.set_ylabel(r"top-5% recall under $u^{\max\mathrm{-comp}}$")
    ax.set_title("Primary acquisition rule: the tail is not recovered\nuntil the budget is exact",
                 fontsize=10)
    ax.set_ylim(0, 1.04)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
