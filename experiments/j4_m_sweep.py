#!/usr/bin/env python
"""J4 (reduced) -- how much of this depends on M = 8?

Every result in the project is M = 8, which the Stage 1 gate listed as one of its
three open limitations. The plan replaces the expensive version of J4 (train new
committees at M in {4, 16} x 3 seeds) with a **labelled subcommittee sweep**: for
each committee, evaluate every subset of size m = 2..8 and read the M-dependence
off directly. That answers "is M = 8 more than you need?" exactly, and "what would
M = 16 give?" only by extrapolation -- a limit stated here rather than papered
over.

**This is exploratory, not pre-registered.** No prediction was registered before
running it and none is scored; labelling a descriptive sweep as confirmatory would
be worse than not registering it at all. It is reported as a trend with the
subset-to-subset spread attached, because the spread is the interesting part: at
m = 2 there are 28 different committees you could have trained, and how much they
disagree is the honest measure of how lucky a small committee has to be.

Both constructions are swept, so the M-dependence of shared-trunk and naive
committees can be compared on matched systems.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch_journal.evaluation.tail_metrics import auroc  # noqa: E402

EV_TO_MEV = 1000.0


def sweep_system(F: torch.Tensor, f_ref: torch.Tensor, *, top_p: float) -> list[dict]:
    """All subcommittees of every size, for one cache."""
    S, A, _, M = F.shape
    F = F.double()
    delta_full = F.mean(dim=-1) - f_ref.double()
    e_max_full = delta_full.norm(dim=-1).max(dim=1).values                  # [S]
    k = max(1, int(round(top_p * S)))
    # Positives are fixed by the FULL committee's reference error: the question is
    # whether a smaller committee can still find the same bad structures, not
    # whether it can redefine which structures are bad.
    thresh = e_max_full.topk(k).values[-1]
    positive = e_max_full >= thresh

    out = []
    for m in range(2, M + 1):
        subsets = list(itertools.combinations(range(M), m))
        idx = torch.tensor(subsets, dtype=torch.long)                       # [n, m]
        sub = F[..., idx]                                                   # [S, A, 3, n, m]
        v = sub.var(dim=-1, unbiased=True)                                  # [S, A, 3, n]
        glob = v.sum(dim=(1, 2)).T                                          # [n, S]
        maxcomp = v.sqrt().flatten(1, 2).max(dim=1).values.T                # [n, S]
        a_glob = auroc(glob, positive.expand_as(glob))
        a_maxc = auroc(maxcomp, positive.expand_as(maxcomp))
        # accuracy and calibration of the subcommittee mean force
        d = sub.mean(dim=-1) - f_ref.double().unsqueeze(-1)                 # [S, A, 3, n]
        rmse = d.pow(2).mean(dim=(0, 1, 2)).sqrt() * EV_TO_MEV              # [n]
        disag = sub.std(dim=-1, unbiased=True).mean(dim=2).flatten(0, 1).median(dim=0).values
        out.append({
            "m": m, "n_subsets": len(subsets),
            "auroc_global_mean": float(a_glob.mean()),
            "auroc_global_sd": float(a_glob.std(unbiased=True)) if len(subsets) > 1 else 0.0,
            "auroc_global_min": float(a_glob.min()), "auroc_global_max": float(a_glob.max()),
            "auroc_maxcomp_mean": float(a_maxc.mean()),
            "global_minus_maxcomp_mean": float((a_glob - a_maxc).mean()),
            "global_beats_maxcomp_frac": float((a_glob > a_maxc).double().mean()),
            "force_rmse_mev_A_mean": float(rmse.mean()),
            "overconfidence_mean": float((rmse / (disag * EV_TO_MEV)).mean()),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dirs", nargs="+",
                    default=["results/processed/naive", str(FROZEN / "results/processed"),
                             "results/processed"])
    ap.add_argument("--top-p", type=float, default=0.05)
    ap.add_argument("--out", default="results/records/j4_m_sweep.jsonl")
    a = ap.parse_args()

    seen, records = set(), []
    for cache in sorted({p for d in a.cache_dirs
                         for p in (ROOT / d if not str(d).startswith("/") else pathlib.Path(d))
                         .glob("head_forces_*.pt")}):
        tag = cache.stem.replace("head_forces_", "")
        if tag in seen:
            continue
        seen.add(tag)
        c = torch.load(cache, map_location="cpu", weights_only=False)
        rows = sweep_system(c["F"], c["f_ref"], top_p=a.top_p)
        for r in rows:
            records.append({"experiment_id": "j4_m_sweep", "system": tag,
                            "construction": "naive" if "naive" in tag else "shared_trunk",
                            **r})
        full = rows[-1]
        half = next(r for r in rows if r["m"] == 4)
        print(f"{tag:30s} m=8 {full['auroc_global_mean']:.3f} | "
              f"m=4 {half['auroc_global_mean']:.3f} +- {half['auroc_global_sd']:.3f} "
              f"[{half['auroc_global_min']:.3f},{half['auroc_global_max']:.3f}] | "
              f"retained {half['auroc_global_mean'] / full['auroc_global_mean']:.1%}")

    p = ROOT / a.out
    with p.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"\nwrote {len(records)} rows to {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
