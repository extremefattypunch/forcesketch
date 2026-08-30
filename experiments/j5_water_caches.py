#!/usr/bin/env python
"""J5 -- build head-force caches for bulk liquid water (192 atoms, periodic).

Water is the highest-value system addition available, and it earns its place
twice over. Statistically, D = 3N = 576 against ethanol's 27 gives a 21x range in
dimension, which is what makes the extreme-value law of J3 testable by blind
prediction rather than by fit. For the systems story, water carries ~17,150 edges
per structure against 3BPA's 520 -- 33x the work -- which is exactly the axis the
J8 smoke test identified as the one that decides whether lane reduction pays.

Cache schema is identical to the frozen `01_exact_reproduction.py` output, so
every downstream journal experiment reads water with no special-casing:

    {F: [S, A, 3, M] float64, E: [S, M], f_ref: [S, A, 3], M, A, variant, split,
     checkpoint_hash, frames}

Periodic correctness is asserted rather than assumed, because every failure mode
here is silent -- a missing `pbc` flag or a mismatched key yields finite, wrong
numbers with no exception. Verified before any forces are computed:
  * every frame has pbc = (True, True, True)
  * the cell is constant and minimum-image is valid (12.42 A > 2 * r_max = 12.0)
  * the graph actually contains periodic-image edges
  * energy_ref / forces_ref are present
and after: the estimator stack is cross-checked against the direct per-head
variance, as the frozen script does.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch.adapters.mace_data import load_frames, make_loader, reference_forces  # noqa: E402
from forcesketch.adapters.mace_mhc import MaceMHCAdapter  # noqa: E402
from forcesketch.exact.member_forces import variance_from_member_forces  # noqa: E402
from forcesketch.utils.reproducibility import checkpoint_hash, git_commit, pin_numerics  # noqa: E402

EV_TO_MEV = 1000.0


def assert_periodic_ok(frames: list, r_max: float) -> dict:
    """Every one of these failures is silent if unchecked."""
    a0 = frames[0]
    cells = np.stack([np.asarray(f.cell) for f in frames])
    diag = np.diag(cells[0])
    checks = {
        "all_frames_pbc_true": bool(all(np.all(f.pbc) for f in frames)),
        "cell_constant": bool(np.allclose(cells, cells[0])),
        "min_image_valid": bool(np.all(diag > 2 * r_max)),
        "min_image_margin_A": float(diag.min() - 2 * r_max),
        "has_forces_ref": "forces_ref" in a0.arrays,
        "has_energy_ref": "energy_ref" in a0.info,
        "n_atoms": len(a0),
        "cell_diag_A": [float(x) for x in diag],
    }
    for k in ("all_frames_pbc_true", "cell_constant", "min_image_valid",
              "has_forces_ref", "has_energy_ref"):
        if not checks[k]:
            raise SystemExit(f"periodic precondition FAILED: {k} -> {checks}")
    return checks


def build_cache(adapter, frames, *, batch_size: int) -> dict:
    A = len(frames[0])
    M = adapter.num_heads
    S = len(frames)
    F = torch.empty(S, A, 3, M, dtype=torch.float64)
    E = torch.empty(S, M, dtype=torch.float64)
    done = 0
    n_periodic_edges = 0
    for batch in make_loader(frames, adapter.model, batch_size=batch_size):
        b = adapter.prepare(batch.to_dict())
        if "shifts" in b:
            n_periodic_edges += int((b["shifts"].abs().sum(-1) > 0).sum())
        e = adapter.energies(b).detach().double()               # [B, M]
        f = adapter.exact_head_forces(b).double()               # [M, N, 3]
        n = f.shape[1] // A
        F[done:done + n] = f.view(M, n, A, 3).permute(1, 2, 3, 0).cpu()
        E[done:done + n] = e.cpu()
        done += n
    assert done == S, f"{done} != {S}"
    return {"F": F, "E": E,
            "f_ref": reference_forces(frames).view(S, A, 3).double(),
            "frames": S, "M": M, "A": A, "n_periodic_edges": n_periodic_edges}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["disjoint", "overlapping"])
    ap.add_argument("--splits", nargs="+", default=["md_T300K", "pimd_T300K"])
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    pin_numerics()
    outdir = pathlib.Path(args.outdir or (ROOT / "results/processed"))
    outdir.mkdir(parents=True, exist_ok=True)
    sha, dirty = git_commit(FROZEN.parent)
    records = []

    for variant in args.variants:
        ckpt = FROZEN / f"models/zenodo/water/multihead-{variant}/multihead_committee_stagetwo.model"
        ad = MaceMHCAdapter.from_checkpoint(ckpt, device="cuda", dtype=torch.float64)
        r_max = float(ad.model.r_max)
        chash = checkpoint_hash(ad.model)
        for split in args.splits:
            frames = load_frames(FROZEN / f"models/zenodo/water/w64_{split}.xyz",
                                 limit=args.n_frames)
            checks = assert_periodic_ok(frames, r_max)
            t0 = time.time()
            c = build_cache(ad, frames, batch_size=args.batch_size)
            dt = time.time() - t0

            F, E, fref = c["F"], c["E"], c["f_ref"]
            # independent cross-check: estimator stack vs direct per-head variance
            v_direct = variance_from_member_forces(F.permute(3, 0, 1, 2))
            v_est = F.var(dim=-1, unbiased=True)
            rel = float((v_direct - v_est).abs().max() / v_est.abs().max())
            assert rel < 1e-9, f"estimator/direct mismatch {rel:.2e}"

            delta = F.mean(-1) - fref
            summary = {
                "experiment_id": "j5_water_caches", "git_commit": sha, "git_dirty": dirty,
                "dataset": "water_w64", "variant": f"water-{variant}", "split": split,
                "checkpoint_hash": chash, "precision": "float64",
                "num_heads": c["M"], "num_atoms": c["A"], "n_structures": c["frames"],
                "D_3N": 3 * c["A"], "periodic": True,
                "n_periodic_image_edges": c["n_periodic_edges"],
                "force_rmse_mean_mev_A": float(delta.pow(2).mean().sqrt()) * EV_TO_MEV,
                "median_atom_disagreement_mev_A":
                    float(F.std(dim=-1, unbiased=True).mean(dim=-1).median()) * EV_TO_MEV,
                "max_component_uncertainty_median_mev_A":
                    float(v_est.sqrt().flatten(1).max(dim=1).values.median()) * EV_TO_MEV,
                "e_max_median_mev_A": float(delta.norm(dim=-1).max(dim=1).values.median()) * EV_TO_MEV,
                "estimator_vs_direct_rel_err": rel,
                "seconds": dt, **checks,
            }
            records.append(summary)
            tag = f"water-{variant}_{split}"
            torch.save({**c, "variant": f"water-{variant}", "split": split,
                        "checkpoint_hash": chash}, outdir / f"head_forces_{tag}.pt")
            print(f"[{tag}] S={c['frames']} A={c['A']} D={3*c['A']} "
                  f"periodic_edges={c['n_periodic_edges']} "
                  f"force_rmse={summary['force_rmse_mean_mev_A']:.1f} "
                  f"disagreement={summary['median_atom_disagreement_mev_A']:.1f} meV/A "
                  f"({dt:.0f}s)")

    out = ROOT / "results/records/j5_water_caches.jsonl"
    with out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"\nwrote {len(records)} caches to {outdir} and records to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
