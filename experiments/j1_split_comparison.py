#!/usr/bin/env python
"""Factor A under blocked versus random-frame test splits.

**Regenerated.** The table in `J1_SPLIT_REPAIR.md` was computed before the split
seed was mixed with the system name, and was never recomputed when the manifests
were. A cross-document audit caught it: the document's own text says the splits
were regenerated, while its comparison table still described the old ones.

The claim this table supports -- that the workshop's random-frame split was not
materially leaky -- is exactly the kind that must be recomputed rather than
carried, because if blocking mattered it would show up here first.

Writes `results/records/j1_split_comparison.json` so the numbers are derivable,
and `tools/paper_numbers.py` takes the headline from it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forcesketch_journal.evaluation.tail_metrics import auroc, top_p_mask  # noqa: E402

PANEL = ["disjoint_test_1200K", "overlapping_test_1200K", "same_test_1200K",
         "rmd17-disjoint_ethanol", "rmd17-disjoint_aspirin", "rmd17-disjoint_azobenzene"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perstructure", default="results/perstructure/j2a")
    ap.add_argument("--out", default="results/records/j1_split_comparison.json")
    ap.add_argument("--error-p", type=float, default=0.05)
    a = ap.parse_args()

    per = {}
    for p in sorted((ROOT / a.perstructure).glob("*.pt")):
        d = torch.load(p, weights_only=True)
        per[d["tag"]] = d

    rows = []
    for sysname in PANEL:
        if sysname not in per:
            continue
        r = per[sysname]
        sig, err = r["sig_exact_global"].double(), r["err_e_max"].double()
        vals = {}
        for scheme in ("contiguous_block", "random_frame"):
            man = ROOT / f"manifests/splits/{sysname}__{scheme}.json"
            idx = torch.tensor(json.loads(man.read_text())["roles"]["test"])
            vals[scheme] = float(auroc(sig[idx], top_p_mask(err[idx], a.error_p)))
        rows.append({"experiment_id": "j1_split_comparison", "system": sysname,
                     "blocked": vals["contiguous_block"],
                     "random": vals["random_frame"],
                     "difference": vals["contiguous_block"] - vals["random_frame"]})

    diffs = [x["difference"] for x in rows]
    summary = {"mean_abs_difference": float(np.mean(np.abs(diffs))),
               "max_abs_difference": float(np.max(np.abs(diffs))),
               "n_positive": int(sum(d > 0 for d in diffs)),
               "n_negative": int(sum(d < 0 for d in diffs)),
               "n_systems": len(rows)}

    print(f"{'system':28s}{'blocked':>10s}{'random':>10s}{'diff':>10s}")
    for x in rows:
        print(f"{x['system']:28s}{x['blocked']:10.4f}{x['random']:10.4f}"
              f"{x['difference']:+10.4f}")
    print(f"\nmean |difference| = {summary['mean_abs_difference']:.4f}   "
          f"max = {summary['max_abs_difference']:.4f}   "
          f"{summary['n_positive']} positive / {summary['n_negative']} negative")

    (ROOT / a.out).write_text(json.dumps(
        {"experiment_id": "j1_split_comparison", "error_p": a.error_p,
         "systems": rows, "summary": summary}, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
