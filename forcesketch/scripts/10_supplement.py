#!/usr/bin/env python
"""Build the anonymous code supplement (revision F4).

The call for papers requires full anonymisation and calls out identifying links
specifically -- a GitHub URL or an absolute home directory in a committed file is
enough for desk rejection, and there is no rebuttal phase in which to fix it. So
this script does not merely zip the tree: it scans every file it is about to add
and REFUSES to write the archive if anything identifying survives.

What goes in: the estimator library, the experiment scripts, the analysis code,
the tests, the frozen seeds and configs, every raw result record, and the freeze
manifest. What stays out: checkpoints and datasets (public, pinned by SHA-256 in
the manifest), profiler binaries, and the manuscript sources.
"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

INCLUDE = [
    ("src/forcesketch", "**/*.py"),
    ("analysis", "*.py"),
    ("scripts", "*.py"),
    ("tests", "*.py"),
    ("configs", "*.yaml"),
    ("environment", "*"),
    ("results/raw", "*.jsonl"),
    ("results/manifests", "*.json"),
    ("results/processed", "*.json"),
    ("results/processed", "*.md"),
]

ANON_CFG = Path("configs/anonymity.yaml")


def load_anonymity() -> tuple[list, re.Pattern, list]:
    """Needles, allowed patterns and substitutions live OUTSIDE the shipped code.

    The first version of this script hard-coded the author's name as a detection
    pattern -- and then shipped itself, putting the name in front of exactly the
    reviewer it was meant to hide it from. The config file is excluded from the
    archive for that reason.
    """
    import yaml

    cfg = yaml.safe_load(ANON_CFG.read_text())
    needles = [re.compile(p, re.I) for p in cfg["needles"]]
    allowed = re.compile("|".join(cfg["allowed"]), re.I)
    subs = [(re.compile(s["pattern"], re.I), s["replace"])
            for s in cfg.get("substitutions", [])]
    return needles, allowed, subs


def scrub(text: str, subs: list) -> str:
    for pat, rep in subs:
        text = pat.sub(rep, text)
    return text


def scan(text: str, needles: list, allowed: re.Pattern) -> list[str]:
    hits = []
    for line in text.splitlines():
        if allowed.search(line):
            continue
        for pat in needles:
            m = pat.search(line)
            if m:
                hits.append(f"{m.group(0)!r} in: {line.strip()[:100]}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="supplement_anonymous.zip")
    ap.add_argument("--format", choices=["zip", "dir"], default="zip",
                    help="'dir' writes a clean, standalone directory ready to become "
                         "its own git repository -- for venues that take an artifact "
                         "URL rather than a supplementary upload")
    ap.add_argument("--force", action="store_true",
                    help="write the archive even if the anonymity scan fails")
    args = ap.parse_args()

    needles, allowed, subs = load_anonymity()

    files: list[Path] = []
    for root, pattern in INCLUDE:
        base = Path(root)
        if not base.exists():
            print(f"  (skipping absent {root})")
            continue
        files += sorted(f for f in base.glob(pattern) if f.is_file())
    files = sorted(set(f for f in files if f.resolve() != ANON_CFG.resolve()))

    payload: dict[Path, str | bytes] = {}
    problems: dict[str, list[str]] = {}
    for f in files:
        try:
            text = scrub(f.read_text(errors="strict"), subs)
        except (UnicodeDecodeError, OSError):
            payload[f] = f.read_bytes()      # binary: shipped as-is, not scannable
            continue
        payload[f] = text
        hits = scan(text, needles, allowed)  # scan the SCRUBBED text, not the original
        if hits:
            problems[str(f)] = hits

    if problems:
        print(f"ANONYMITY SCAN FAILED -- {len(problems)} file(s):")
        for name, hits in sorted(problems.items()):
            print(f"  {name}")
            for h in hits[:3]:
                print(f"      {h}")
        if not args.force:
            print("\nRefusing to write the supplement. Fix these or pass --force.")
            return 1

    out = Path(args.out)
    if args.format == "dir":
        # A standalone tree, not a nested one: this becomes the repository root,
        # so paths must work with no "forcesketch/" prefix in front of them.
        if out.exists():
            shutil.rmtree(out)
        for f, blob in payload.items():
            dest = out / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(blob) if isinstance(blob, str) else dest.write_bytes(blob)
        (out / "README.md").write_text(README)
        (out / ".gitignore").write_text(
            "__pycache__/\n*.py[cod]\n.pytest_cache/\ndata/\nmodels/\n"
            "results/profile/\n*.nsys-rep\n*.ncu-rep\n")
        n_files = sum(1 for _ in out.rglob("*") if _.is_file())
        mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
        print(f"anonymity scan clean over {len(payload)} files "
              f"({sum(1 for v in payload.values() if isinstance(v, str))} scanned)")
        print(f"wrote {out}/ ({n_files} files, {mb:.1f} MB)")
        return 0

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f, blob in payload.items():
            arc = str(Path("forcesketch") / f)
            z.writestr(arc, blob) if isinstance(blob, str) else z.writestr(arc, blob)
        z.writestr("forcesketch/README_SUPPLEMENT.md", README)

    mb = out.stat().st_size / 1e6
    print(f"anonymity scan clean over {len(payload)} files "
          f"({sum(1 for v in payload.values() if isinstance(v, str))} scanned)")
    print(f"wrote {out} ({len(files) + 1} entries, {mb:.1f} MB)")
    return 0


README = """# ForceSketch -- anonymous supplement

Estimator library, experiment scripts, analysis code, frozen seeds and every raw
result record behind the paper's numbers.

Not included: model checkpoints and datasets. Both are public and are pinned by
SHA-256 in `results/manifests/freeze.json`; `scripts/00_environment.py` fetches
and verifies them.

## Reproducing the paper's numbers

    python scripts/00_environment.py        # fetch + verify checkpoints and data
    python scripts/01_exact_reproduction.py # cache per-head forces F and energies E
    python scripts/03_sketch_fidelity.py --score maxcomp   # primary acquisition rule
    python scripts/07_gate_baselines.py --score maxcomp --n-boot 10000
    python scripts/09_freeze.py             # regenerate every table, macro and figure

Everything downstream of step 2 is offline linear algebra on the cached force
matrix -- no GPU and no model execution needed. The timing experiments
(`02b`, `02d`, `02c`, `02e`) do need the GPU.

## One thing worth knowing before you run anything

`forcesketch.adapters.mace_mhc.configure_e3nn_for_batched_vjp()` disables the
TensorExpr fuser at import time. Without it,
`torch.autograd.grad(..., is_grads_batched=True)` fails on this stack -- but only
from the THIRD call onwards, so a quick test will appear to pass. The appendix of
the paper explains why.
"""


if __name__ == "__main__":
    raise SystemExit(main())
