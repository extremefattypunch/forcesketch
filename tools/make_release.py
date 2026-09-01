#!/usr/bin/env python
"""Assemble the deposit archive the Data and Software Availability statement promises.

JCTC's policy (effective May 2026) asks for the materials needed to reproduce key
results, and gpt_plan.md section 18 lists the release package: source, environment
lockfile, dependency versions, dataset acquisition scripts, dataset and checkpoint
hashes, split manifests, training configs, probe seeds, raw predictions, raw
timings, figure and table scripts, test suite, worked example.

Two decisions worth stating.

**Only git-tracked files go in.** The ~384 MB of `.pt` caches are deliberately
gitignored and regenerable, and `tools/reproduce.py` is explicit that a fresh
clone has the six frozen head-force caches but not the water, naive or MPtraj
ones. Shipping regenerable binaries would make the archive large and no more
reproducible; shipping their rebuild commands does. So the archive carries what
git carries, and the manifest says so rather than implying completeness.

**Every file is hashed into MANIFEST.sha256.** An availability statement whose
archive cannot be checked against itself is decoration. This also lets a reader
verify the deposit was not silently modified after the DOI was minted.

The script REPORTS which release-package categories it satisfied and which it did
not, and exits non-zero if a category the manuscript claims is missing. It does
not create the deposit: minting a DOI is the author's act, not a script's.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parent

# category -> at least one tracked path must match one of these globs
REQUIRED = {
    "source code": ["src/**/*.py", "experiments/*.py", "tools/*.py"],
    "dataset acquisition + hashes": ["results/STAGE2_ENVIRONMENT.md"],
    "split manifests": ["manifests/splits/*.json"],
    "pre-registered protocols": ["protocols/*"],
    "probe seeds / configs": ["manifests/**/*.json", "protocols/*"],
    "raw predictions + timings": ["results/records/*.jsonl", "results/records/*.json"],
    "figure + table scripts": ["tools/make_figures.py", "tools/make_tables.py"],
    "test suite": ["tests/*.py"],
    "worked example / repro entry point": ["tools/reproduce.py"],
    "manuscript source": ["paper/*.tex", "paper/references.bib"],
}


def tracked() -> list[pathlib.Path]:
    out = subprocess.run(["git", "ls-files", ROOT.name],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    rels = []
    for line in out.splitlines():
        if not line.strip():
            continue
        p = REPO / line
        if p.is_file():
            rels.append(pathlib.Path(line).relative_to(ROOT.name))
    return rels


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "release" / "forcesketch-artifact.zip"))
    ap.add_argument("--format", choices=["zip", "dir"], default="zip")
    args = ap.parse_args()

    files = tracked()
    if not files:
        print("no tracked files found -- is this a git checkout?", file=sys.stderr)
        return 1

    # which release-package categories are actually satisfied
    missing = []
    print(f"tracked files: {len(files)}\n")
    print("release package coverage:")
    for cat, globs in REQUIRED.items():
        hits = sum(1 for f in files if any(f.match(g) for g in globs))
        mark = "ok  " if hits else "MISS"
        print(f"  [{mark}] {cat:38s} {hits:4d} file(s)")
        if not hits:
            missing.append(cat)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = ["# sha256 of every file in this deposit.",
                "# Regenerable binaries (~384 MB of .pt caches) are NOT included:",
                "# they are gitignored and rebuilt by the commands in .gitignore and",
                "# results/STAGE2_ENVIRONMENT.md. tools/reproduce.py --tier offline",
                "# runs entirely from what IS here.",
                ""]
    for f in sorted(files):
        manifest.append(f"{sha256(ROOT / f)}  {f}")
    manifest_text = "\n".join(manifest) + "\n"

    if args.format == "dir":
        d = out.with_suffix("")
        if d.exists():
            shutil.rmtree(d)
        for f in files:
            dest = d / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, dest)
        (d / "MANIFEST.sha256").write_text(manifest_text)
        n = sum(1 for _ in d.rglob("*") if _.is_file())
        mb = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e6
        print(f"\nwrote {d}/ ({n} files, {mb:.1f} MB)")
    else:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(files):
                z.write(ROOT / f, str(pathlib.Path("forcesketch-artifact") / f))
            z.writestr("forcesketch-artifact/MANIFEST.sha256", manifest_text)
        print(f"\nwrote {out} ({len(files) + 1} entries, {out.stat().st_size / 1e6:.1f} MB)")

    if missing:
        print(f"\n{len(missing)} release-package category(ies) MISSING: {missing}")
        print("The manuscript's availability statement claims these. Fix before depositing.")
        return 1

    print("\nevery category the manuscript claims is present.")
    print("Next, and only the author can do this:")
    print("  1. deposit the archive on Zenodo (or Figshare)")
    print("  2. take the CONCEPT doi, which resolves to the latest version")
    print("  3. put it in paper/artifact.tex as \\fsArtifactDoi")
    print("  4. rebuild: paper/build_jctc.sh and paper/build_iclr.sh both refuse")
    print("     while the locator is still a placeholder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
