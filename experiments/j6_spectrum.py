#!/usr/bin/env python
"""Centred head-space spectrum for any cache directory.

Written for J6 (P1: does the shared trunk concentrate the head space?), but it
also closes a provenance hole: `results/records/j5_spectrum.json` was quoted in
J2B's srank table and had **no generating script** in the repository. A record
nobody can regenerate is a number on trust, which is exactly what R3 forbids.

So this script runs in two modes. `--validate` recomputes the ten shared-trunk
systems and diffs against the committed record; only if that passes are the naive
numbers it produces worth anything, because it proves this implementation is the
one that produced the table already in the paper.

Definitions, on the pooled centred head-space Gram
`G = (F Qc)^T (F Qc)` with `Qc` the M x r Helmert basis and rows ranging over all
S x D (structure, coordinate) pairs, eigenvalues mu_1 >= ... >= mu_r:

    top1_fraction   = mu_1 / sum(mu)          1/r when isotropic
    stable_rank_FQ  = sum(mu) / mu_1          r   when isotropic, 1 when rank-one
    stable_rank_A   = sum(mu^2) / mu_1^2      the same measure one power up, i.e.
                                              for A = F^T F rather than for F Qc
    effective_rank  = (sum mu)^2 / sum(mu^2)  the participation ratio, also r when
                                              isotropic and 1 when rank-one
    n_for_90pct     = smallest k with cumulative p >= 0.9

Recovering `effective_rank` took one wrong guess worth recording: the entropy
exponential exp(-sum p log p) is the other common "effective rank" and it
reproduces nothing here (3.066 against the recorded 2.101 on water PIMD). The
participation ratio matches to 1e-15 on all ten systems. The two disagree most
exactly where the spectrum is most concentrated, which is the regime the J2B
srank table is about -- so this was worth pinning down rather than assuming.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch.exact.centered_basis import helmert_basis  # noqa: E402


def spectrum(F: torch.Tensor) -> dict:
    """F [S, A, 3, M] -> pooled centred head-space spectrum summary."""
    M = F.shape[-1]
    r = M - 1
    Qc = helmert_basis(M, dtype=torch.float64)                        # [M, r]
    X = torch.einsum("sadm,mr->sadr", F.double(), Qc).flatten(0, 2)   # [S*D, r]
    G = X.T @ X
    mu = torch.linalg.eigvalsh(G).flip(0).clamp_min(0)                # descending
    tot = mu.sum()
    p = mu / tot
    return {
        "stable_rank_FQ": float(tot / mu[0]),
        "stable_rank_A": float((mu.pow(2).sum()) / mu[0].pow(2)),
        "effective_rank": float(tot.pow(2) / mu.pow(2).sum()),
        "isotropic_top1_fraction": 1.0 / r,
        "top1_fraction": float(p[0]),
        "n_for_90pct": float((p.cumsum(0) < 0.9).sum() + 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dirs", nargs="+", default=["results/processed/naive"])
    ap.add_argument("--out", default="results/records/j6_naive_spectrum.json")
    ap.add_argument("--validate", default=None,
                    help="recompute and diff against an existing spectrum record")
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args()

    if a.validate:
        ref = {d["system"]: d for d in json.loads((ROOT / a.validate).read_text())}
        worst, checked = 0.0, 0
        for cache in sorted({p for d in a.cache_dirs
                             for p in (ROOT / d).glob("head_forces_*.pt")}):
            tag = cache.stem.replace("head_forces_", "")
            if tag not in ref:
                continue
            got = spectrum(torch.load(cache, map_location="cpu", weights_only=False)["F"])
            for k, v in got.items():
                dv = abs(v - ref[tag][k])
                worst = max(worst, dv)
                print(f"  {tag:30s} {k:26s} {v:12.6f} vs {ref[tag][k]:12.6f}  d={dv:.2e}")
            checked += 1
        print(f"\nvalidated {checked} systems against {a.validate}; worst |diff| = {worst:.3e}")
        if checked == 0:
            print("NOTHING VALIDATED -- no cache tag matched the reference record")
            return 2
        return 0 if worst < a.tol else 1

    out = []
    for cache in sorted({p for d in a.cache_dirs for p in (ROOT / d).glob("head_forces_*.pt")}):
        tag = cache.stem.replace("head_forces_", "")
        c = torch.load(cache, map_location="cpu", weights_only=False)
        s = {"system": tag, **spectrum(c["F"])}
        out.append(s)
        print(f"{tag:30s} srank_FQ={s['stable_rank_FQ']:.3f}  top1={s['top1_fraction']:.3f}  "
              f"eff_rank={s['effective_rank']:.3f}")
    p = ROOT / a.out
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {len(out)} systems to {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
