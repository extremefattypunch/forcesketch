#!/usr/bin/env python
"""J2a -- the oracle panel: does EXACT multi-head committee disagreement predict
reference-force error at all?

This is Factor A of the two-factor decomposition, and it is deliberately the first
thing the journal programme runs. Factor B (does the ForceSketch gate preserve
whatever the exact estimator decides?) is a claim about fidelity to a chosen
oracle and is logically independent. Factor A asks whether that oracle is worth
anything, and its answer determines the paper's identity:

    A >= 0.75          the application claim stands as written
    0.60 <= A < 0.75   the claim becomes regime-scoped ("when is force UQ worth
                       computing?"), which is itself a useful contribution
    A <  0.60          the paper restructures around the acquisition rule being
                       both hard to estimate AND a weak predictor

Nothing here is fitted. There is no basis, no conformal constant, no threshold --
only rank correlations between quantities that already exist in the cache. So
there is no train/test leakage risk and the full evaluation set is used, which is
also the highest-power estimate. Once J1's split manifests exist this is re-run on
the test split alone to confirm the numbers do not move.

Two free baselines are included because they can embarrass the whole enterprise
and cost nothing: the head-energy spread (one forward pass, no backward at all)
and the mean-force magnitude (already computed). If either ranks reference error
as well as force disagreement does, the case for computing force uncertainty
collapses, and that is worth knowing on day one rather than in review.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from forcesketch_journal.evaluation.tail_metrics import (  # noqa: E402
    auprc, auroc, enrichment_at, recall_at, risk_coverage, top_p_mask,
)

FROZEN = pathlib.Path(__file__).resolve().parents[2] / "forcesketch"
EV_PER_A_TO_MEV = 1000.0


def git_commit(repo: pathlib.Path) -> tuple[str, bool]:
    def run(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain"))


def error_scores(F: torch.Tensor, f_ref: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reference-force error, four ways. `delta` is [S, A, 3] in eV/A.

    e_max is primary: it is the max over ATOMS of the error vector norm, matching
    how a practitioner asks "how wrong is the worst atom in this structure?".
    e_maxcomp is the max over the 3N COMPONENTS and is included because the
    primary uncertainty score is also a max over components -- comparing a
    component-max signal to an atom-max error would confound the reduction with
    the physics.
    """
    delta = F.mean(dim=-1) - f_ref                                   # [S, A, 3]
    return {
        "e_max": delta.norm(dim=-1).max(dim=1).values,               # max_a |delta_a|_2
        "e_rmse": delta.pow(2).mean(dim=(1, 2)).sqrt(),              # RMS over 3N
        "e_maxcomp": delta.abs().flatten(1).max(dim=1).values,       # max_d |delta_d|
        "e_q95": delta.abs().flatten(1).quantile(0.95, dim=1),
    }


def uncertainty_signals(F: torch.Tensor, E: torch.Tensor | None, seed: int) -> dict[str, torch.Tensor]:
    """Every per-structure scalar a practitioner might rank by.

    v and sigma follow the frozen definitions exactly (unbiased variance over the
    head axis, as in scripts/03_sketch_fidelity.py); the three exact acquisition
    scalars are `ranking()`'s maxcomp / maxatom / global.
    """
    v = F.var(dim=-1, unbiased=True)                                 # [S, A, 3]
    sigma = v.sqrt()
    out = {
        "exact_maxcomp": sigma.flatten(1).max(dim=1).values,         # PRIMARY oracle
        "exact_maxatom": sigma.mean(dim=-1).max(dim=1).values,       # max_a u^MHC_a
        "exact_global": v.sum(dim=(1, 2)),                           # easier statistic
        # --- free signals: no extra backward pass at all ---
        "force_norm": F.mean(dim=-1).norm(dim=-1).max(dim=1).values,
        "random_null": torch.rand(F.shape[0], generator=torch.Generator().manual_seed(seed),
                                  dtype=torch.float64),
    }
    if E is not None:
        out["energy_std"] = E.std(dim=-1, unbiased=True)             # one forward pass
    return out


def evaluate(signal: torch.Tensor, err: torch.Tensor) -> dict:
    row = {}
    for p in (0.01, 0.05, 0.10):
        pos = top_p_mask(err, p)
        tag = f"{int(p*100):02d}"
        row[f"auroc_top{tag}"] = float(auroc(signal, pos))
        row[f"recall_top{tag}"] = float(recall_at(signal, pos, p))
        if p == 0.05:
            row["auprc_top05"] = float(auprc(signal, pos))
            row["enrichment_top05"] = float(enrichment_at(signal, pos, p))
    rc = risk_coverage(signal, err)
    row["aurc_excess"] = float(rc["aurc_excess"])
    rc_max = risk_coverage(signal, err, reduce="max")
    row["aurc_excess_max"] = float(rc_max["aurc_excess"])
    # Spearman = Pearson on ranks; reuse the frozen tree's definition.
    sys.path.insert(0, str(FROZEN))
    from analysis.metrics import kendall, spearman  # noqa: E402
    row["spearman"] = float(spearman(signal, err))
    row["kendall"] = float(kendall(signal, err))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/records/j2a_oracle_panel.jsonl")
    ap.add_argument("--perstructure", default="results/perstructure/j2a")
    ap.add_argument("--random-seed", type=int, default=20260902)
    ap.add_argument("--cache-dirs", nargs="+", default=None,
                    help="directories of head_forces_*.pt; defaults to the frozen tree")
    args = ap.parse_args()

    sha, dirty = git_commit(FROZEN.parent)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ps_dir = pathlib.Path(args.perstructure)
    ps_dir.mkdir(parents=True, exist_ok=True)

    records = []
    dirs = [pathlib.Path(d) for d in (args.cache_dirs or [FROZEN / "results/processed"])]
    caches = sorted({p for d in dirs for p in d.glob("head_forces_*.pt")})
    for path in caches:
        tag = path.stem.replace("head_forces_", "")
        d = torch.load(path, weights_only=True, map_location="cpu")
        F, f_ref = d["F"].double(), d["f_ref"].double()
        E = d["E"].double() if "E" in d else None

        errs = error_scores(F, f_ref)
        sigs = uncertainty_signals(F, E, args.random_seed)

        torch.save({"tag": tag, **{f"err_{k}": v for k, v in errs.items()},
                    **{f"sig_{k}": v for k, v in sigs.items()}}, ps_dir / f"{tag}.pt")

        for sname, signal in sigs.items():
            for ename, err in errs.items():
                records.append({
                    "experiment_id": "j2a_oracle_panel",
                    "experiment_version": 1,
                    "git_commit": sha, "git_dirty": dirty,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "cache_tag": tag,
                    "checkpoint_hash": d.get("checkpoint_hash"),
                    "n_structures": int(F.shape[0]), "number_of_atoms": int(F.shape[1]),
                    "M": int(F.shape[3]), "split_role": "full_evaluation_set",
                    "signal": sname, "error_score": ename,
                    "precision": "float64",
                    "median_error_mev_A": float(err.median()) * EV_PER_A_TO_MEV,
                    "median_signal": float(signal.median()),
                    **evaluate(signal, err),
                })

    with out_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"wrote {len(records)} records to {out_path}")
    print(f"wrote per-structure tensors for {len(caches)} caches to {ps_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
