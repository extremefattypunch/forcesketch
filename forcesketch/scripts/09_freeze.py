#!/usr/bin/env python
"""Freeze the experiment set (spec SS71 "freeze", SS68 provenance).

Produces `results/manifests/freeze.json`: a single record that pins the git
commit, the environment, every checkpoint and dataset by hash, the frozen seeds,
and a hash of every raw result file. Regenerates the tables, macros and figures
from those raw files so the manuscript cannot drift from the data.

Also runs the audits that would otherwise be promises rather than checks:
  * spec SS47 -- seeds.yaml must be marked frozen, and every stochastic method must
    carry all ten of them (no seed dropped, good or bad).
  * spec SS70 -- no configuration silently absent from the grid.
  * spec SS68 -- every result record carries git_commit.

Exits non-zero if any audit fails, so "frozen" means checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import platform
import subprocess
from pathlib import Path

import yaml

from analysis.tables import (
    RAW,
    check_cherry_picking,
    emit_macros,
    fit_cost_model,
    load_jsonl,
    table1_primary,
    table3_screening,
)
from forcesketch.utils.reproducibility import git_commit

SYSTEMS = ["disjoint_test_1200K", "rmd17-disjoint_ethanol",
           "rmd17-disjoint_aspirin", "rmd17-disjoint_azobenzene"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-n-boot", type=int, default=10000)
    args = ap.parse_args()

    problems: list[str] = []
    cfg = yaml.safe_load(Path("configs/seeds.yaml").read_text())

    # --- SS47: seeds frozen, and all of them used -------------------------
    if not cfg.get("frozen"):
        problems.append("configs/seeds.yaml is not marked frozen")
    n_seeds = len(cfg["sketch_seeds"])
    if n_seeds != 10:
        problems.append(f"expected 10 frozen seeds, found {n_seeds}")

    # --- SS70: no dropped configurations ----------------------------------
    for tag in SYSTEMS:
        f = RAW / f"03_sketch_fidelity_{tag}.jsonl"
        if not f.exists():
            problems.append(f"missing fidelity results for {tag}")
            continue
        problems += [f"{tag}: {p}" for p in check_cherry_picking(load_jsonl(f), n_seeds)]

    # --- SS47: CIs must be at the frozen resample count -------------------
    boot_counts = set()
    for tag in SYSTEMS:
        f = RAW / f"04_bootstrap_{tag}.jsonl"
        if not f.exists():
            problems.append(f"missing bootstrap results for {tag}")
            continue
        for r in load_jsonl(f):
            n = r.get("top5_recall_n_boot") or r.get("n_boot")
            if n:
                boot_counts.add(int(n))
    if boot_counts and boot_counts != {args.expect_n_boot}:
        problems.append(f"bootstrap resample counts {sorted(boot_counts)} != "
                        f"{args.expect_n_boot} (spec SS47 freeze)")

    # --- SS68: every record carries provenance ----------------------------
    for f in sorted(RAW.glob("*.jsonl")):
        recs = load_jsonl(f)
        if not recs:
            problems.append(f"{f.name} is empty")
        elif not all("git_commit" in r for r in recs):
            problems.append(f"{f.name}: some records lack git_commit")

    # --- double-blind: a named build must not be freezable -----------------
    # SIMBIOCHEM II is double-blind and states that "failing to submit an
    # anonymised paper will cause desk rejection". That is unrecoverable -- there
    # is no rebuttal phase -- so it is checked mechanically rather than by eye,
    # and checked in the RENDERED PDF as well as the source, because the author
    # block is set by a style file the source never spells out.
    _anon = yaml.safe_load(Path("configs/anonymity.yaml").read_text())
    IDENTIFYING = [re.compile(p_, re.I) for p_ in _anon["needles"]]
    _ALLOWED = re.compile("|".join(_anon["allowed"]), re.I)
    paper = Path("paper")
    for f in sorted(paper.glob("*.tex")):
        text = f.read_text(errors="ignore")
        for needle in IDENTIFYING:
            m = needle.search(text)
            if m and not _ALLOWED.search(m.group(0)):
                problems.append(f"ANONYMITY: {f.name} contains {m.group(0)!r}")
    for opt in ("final", "preprint"):
        if f"[{opt}]{{neurips_2026}}" in (paper / "main.tex").read_text():
            problems.append(f"ANONYMITY: neurips_2026 loaded with '{opt}' -- de-anonymises")

    pdf = paper / "main.pdf"
    if not pdf.exists():
        problems.append("ANONYMITY: paper/main.pdf absent -- cannot verify the rendered build")
    else:
        try:
            rendered = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                                      text=True, check=True).stdout
            for needle in IDENTIFYING:
                m = needle.search(rendered)
                if m and not _ALLOWED.search(m.group(0)):
                    problems.append(f"ANONYMITY: rendered PDF contains {m.group(0)!r}")
            # A macro at end of line swallows the newline, so "\\fsFoo\nwhile"
            # renders as "1.000while". Invisible in the source, obvious in the PDF.
            glued = re.findall(r"[0-9]\.[0-9]{2,3}[a-zA-Z]{2,}", rendered)
            for g in sorted(set(glued)):
                problems.append(f"RENDER: macro ate a space -- {g!r} in the PDF")
            if "Anonymous Author" not in rendered:
                problems.append("ANONYMITY: rendered PDF lacks the anonymous author block")
            # Identifying links are called out explicitly by the call for papers.
            # The allowlist lives in configs/anonymity.yaml so there is ONE policy,
            # not one here and a different one in the supplement builder.
            # pdftotext wraps long URLs across lines, so join them before matching
            # -- otherwise every wrapped link looks like an unknown host.
            joined = re.sub(r"[\s\u00ad]+", "", rendered)
            for m in re.findall(r"https?://[^\s)\]}]+", joined):
                if not _ALLOWED.search(m):
                    problems.append(f"ANONYMITY: rendered PDF links to {m[:80]}")
            # A placeholder artifact URL is worse than none: it looks like a real
            # link and 404s for the reviewer who follows it.
            if "PLACEHOLDER" in rendered:
                problems.append(
                    "ARTIFACT: \\fsCodeUrl is still the PLACEHOLDER. Publish the "
                    "artifact and set the real anonymised URL in paper/main.tex.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            problems.append("ANONYMITY: pdftotext unavailable -- cannot verify the rendered build")

    # --- regenerate every derived artifact --------------------------------
    a, b = fit_cost_model()
    Path("results/processed").mkdir(parents=True, exist_ok=True)
    Path("results/processed/table1_3bpa.md").write_text(table1_primary(SYSTEMS[0]))
    Path("results/processed/table3_screening.md").write_text(table3_screening(SYSTEMS))
    from analysis.tables import table2_scaling
    Path("results/processed/table2_scaling.md").write_text(table2_scaling())
    macros = emit_macros(Path("paper/macros.tex"))

    from analysis.figures import (
        gate_pareto,
        k_tradeoff,
        method_schematic,
        pareto,
        screening_curve,
    )

    TITLES = {
        "disjoint_test_1200K": "3BPA@1200K, disjoint committee (M=8)",
        "rmd17-disjoint_ethanol": "rMD17 ethanol (9 atoms), joint committee (M=8)",
        "rmd17-disjoint_aspirin": "rMD17 aspirin (21 atoms), joint committee (M=8)",
        "rmd17-disjoint_azobenzene": "rMD17 azobenzene (24 atoms), joint committee (M=8)",
    }
    method_schematic(Path("paper/figures/fig1_schematic.png"))
    for tag in SYSTEMS:
        t = TITLES.get(tag, tag)
        if (RAW / f"03_sketch_fidelity_{tag}.jsonl").exists():
            pareto(RAW / f"03_sketch_fidelity_{tag}.jsonl",
                   Path(f"paper/figures/fig2_pareto_{tag}.png"), title=t)
        if (RAW / f"06_screening_{tag}.jsonl").exists():
            screening_curve(RAW / f"06_screening_{tag}.jsonl",
                            Path(f"paper/figures/fig5_screening_{tag}.png"), title=t)
    gb = RAW / "07_gate_baselines_maxcomp.jsonl"
    if gb.exists():
        gate_pareto(gb, Path("paper/figures/fig5_gate_pareto.png"),
                    title="Screening gates at matched budget, max-component rule, 4 systems")
    k_tradeoff(RAW / f"03_sketch_fidelity_{SYSTEMS[0]}.jsonl",
               Path("paper/figures/fig3_k_tradeoff.png"),
               title=TITLES[SYSTEMS[0]] + ", 10 seeds; error bars are seed sd")

    sha, dirty = git_commit()
    if dirty:
        problems.append("working tree is dirty -- commit before freezing")

    manifest = {
        "frozen": len(problems) == 0,
        "git_commit": sha,
        "git_dirty": dirty,
        "n_bootstrap_resamples": args.expect_n_boot,
        "seeds": cfg,
        "cost_model_ms": {"intercept": a, "slope_per_lane": b},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "lockfile_sha256": sha256(Path("environment/lockfile.txt")),
            "smoke_test": json.loads(Path("environment/smoke_test.json").read_text())["passed"],
        },
        "checkpoints": json.loads(Path("results/manifests/checkpoints.json").read_text())
        if Path("results/manifests/checkpoints.json").exists() else {},
        "datasets": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                     for p in sorted(Path("data/3bpa").glob("*.xyz"))},
        "raw_results": {p.name: {"records": len(load_jsonl(p)), "sha256": sha256(p)}
                        for p in sorted(RAW.glob("*.jsonl"))},
        "n_macros": len(macros),
        "audit_problems": problems,
    }
    out = Path("results/manifests/freeze.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))

    print(f"git commit      {sha[:12]}{' (DIRTY)' if dirty else ''}")
    print(f"cost model      T(L) = {a:.2f} + {b:.2f} L ms")
    print(f"raw result sets {len(manifest['raw_results'])}")
    print(f"macros emitted  {len(macros)}")
    print(f"bootstrap       {args.expect_n_boot} resamples")
    if problems:
        print(f"\nFREEZE BLOCKED -- {len(problems)} audit problem(s):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nall audits clean; freeze.json written")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
