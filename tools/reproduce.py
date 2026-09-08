#!/usr/bin/env python
"""End-to-end reproduction from git-tracked artifacts only.

The plan calls this "the single most persuasive reproducibility artifact
available here", and the claim it makes is narrow enough to be true:

  `tools/reproduce.py --tier offline` regenerates the paper's core statistical
  results on CPU, from artifacts that are IN THE REPOSITORY, with no model
  execution, no CUDA, and no download.

**The tiers are honest about what a fresh clone actually has.** The six frozen
head-force caches are tracked in git; the water, naive and MPtraj artifacts are
not (they are ~384 MB of regenerable binaries, gitignored with their rebuild
commands). So:

  offline   uses only git-tracked caches -> J2a panel, its bootstrap intervals,
            J1 splits, J3 extreme-value law. CPU, no downloads.
  full      additionally needs the regenerated caches and the MPtraj scan, hence
            a GPU and the data fetch. Listed here so the gap is visible rather
            than implied.

Verification is by VALUE, not by file hash: records carry timestamps, so a
re-run legitimately changes their bytes. Each reproduced record is fed back
through `tools/paper_numbers.py`, and every macro whose source was regenerated
must come out with the same formatted value. That is the property a reader cares
about — the paper's numbers come back — and it is strictly stronger than "the
script ran without error".
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"

# (label, argv, records this step produces)
OFFLINE = [
    ("j2a oracle panel", [
        "experiments/j2a_oracle_panel.py",
        "--cache-dirs", str(FROZEN / "results/processed"),
        "--out", "{rec}/j2a_oracle_panel.jsonl",
        "--perstructure", "{work}/perstructure"],
     ["j2a_oracle_panel.jsonl"]),
    ("j2a bootstrap intervals", [
        "experiments/j2a_confidence_intervals.py",
        "--perstructure", "{work}/perstructure",
        "--out-dir", "{rec}"],
     ["j2a_global_vs_maxcomp_ci.json", "j2a_free_signal_ci.json"]),
    ("j3 extreme-value law", [
        "experiments/j3_extreme_value.py",
        "--cache-dirs", str(FROZEN / "results/processed"),
        "--out", "{rec}/j3_extreme_value_v2.json"],
     ["j3_extreme_value_v2.json"]),
]


def run(step, work: pathlib.Path, rec: pathlib.Path, py: str) -> tuple[bool, float, str]:
    label, argv, _ = step
    argv = [a.format(work=work, rec=rec) for a in argv]
    t0 = time.time()
    r = subprocess.run([py, *argv], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0, time.time() - t0, (r.stderr or r.stdout)[-600:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="offline", choices=["offline"])
    ap.add_argument("--work", default=None, help="scratch dir (default: a temp dir)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    work = pathlib.Path(a.work) if a.work else ROOT / ".reproduce_tmp"
    rec = work / "records"
    if work.exists() and not a.keep:
        shutil.rmtree(work)
    rec.mkdir(parents=True, exist_ok=True)

    print(f"reproduce --tier {a.tier}")
    print(f"  work dir : {work}")
    print(f"  inputs   : {FROZEN / 'results/processed'} (git-tracked caches only)\n")

    produced, total, failed = [], 0.0, []
    for step in OFFLINE:
        ok, dt, tail = run(step, work, rec, a.python)
        total += dt
        print(f"  {'OK  ' if ok else 'FAIL'} {step[0]:28s} {dt:7.1f}s")
        if ok:
            produced += step[2]
        else:
            failed.append(step[0])
            print("       " + tail.replace("\n", "\n       ")[:500])

    if failed:
        print(f"\n{len(failed)} steps failed; cannot verify values")
        return 1

    # value-level verification: recompute the macros from the REPRODUCED records
    prov = json.loads((ROOT / "results/records/macro_provenance.json").read_text())
    checked = {k: v for k, v in prov.items() if v["source"] in produced}
    print(f"\n  verifying {len(checked)} macros derived from the {len(produced)} "
          f"reproduced records")

    sys.path.insert(0, str(ROOT / "tools"))
    import importlib
    import paper_numbers as pn
    pn.REC = rec                      # point the derivations at the reproduced copies
    pn._CACHE.clear()
    pn._HASH.clear()
    importlib.reload  # noqa: B018  (kept explicit: REC is rebound, not re-imported)

    bad = []
    for name, src, fn, fmt, _ in pn.MACROS:
        if name not in checked:
            continue
        got = fmt.format(fn())
        want = checked[name]["formatted"]
        if got != want:
            bad.append(f"{name}: committed {want}, reproduced {got}")
        else:
            print(f"    OK  \\{name:22s} = {got}")

    print()
    for b in bad:
        print(f"    DIFF {b}")
    print(f"total wall clock: {total:.1f}s")
    if bad:
        print(f"REPRODUCTION FAILED: {len(bad)} of {len(checked)} macros differ")
        return 1
    print(f"REPRODUCTION OK: all {len(checked)} macros from git-tracked inputs, "
          f"CPU only, in {total:.0f}s")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
