#!/usr/bin/env python
"""Experiment 1: exact MHC reproduction, and the head-force cache (plan Task 1.4).

Spec SS25 asks for the exact force-UQ behaviour of the pretrained model on 3BPA:
per structure E_m, f_m, exact v_d, u_a^RMS, u_a^MHC, global S, max atom
uncertainty, max component uncertainty.

This script also produces the artifact everything downstream depends on: a cache
of the per-head force matrix F [n_struct, A, 3, M]. Every ForceSketch estimator
is a LINEAR functional of F (g_k = F w_k), so once F is cached, the whole SS27/SS28
fidelity sweep, the SS36/SS37 ablations, the SS33 screening gate and the SS38
spectrum can be evaluated exactly and offline, with no further model execution.
Only the timing experiments (SS26, SS31, SS32) need the live model.

Cached in float64: the head-space combination cancels O(|mean force|) down to
O(|disagreement|), and here the mean force is ~6x the disagreement, so a float32
cache would put round-off into the reference itself (plan resolution R5).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from forcesketch.adapters.mace_data import load_frames, make_loader, reference_forces
from forcesketch.adapters.mace_mhc import MaceMHCAdapter
from forcesketch.estimators.scores import uncertainty_scores
from forcesketch.exact.centered_basis import exact_seed_bundle
from forcesketch.utils.reproducibility import checkpoint_hash, git_commit, pin_numerics

VARIANTS = {
    "disjoint": "models/zenodo/3BPA/trainset_100/multihead-disjoint",
    "overlapping": "models/zenodo/3BPA/trainset_100/multihead-overlapping",
    "same": "models/zenodo/3BPA/trainset_100/multihead-same",
    # The rMD17 committee is a single JOINT model trained on all ten molecules, so
    # evaluating it on ethanol/aspirin/azobenzene gives spec SS32's molecule-size
    # axis with no model confound.
    "rmd17-disjoint": "models/zenodo/rMD17/full_trainset/multihead-disjoint",
    "rmd17-overlapping": "models/zenodo/rMD17/full_trainset/multihead-overlapping",
}

DATA_PATHS = {
    "test_1200K": "data/3bpa/test_1200K_ref.xyz",
    "test_600K": "data/3bpa/test_600K_ref.xyz",
    "test_300K": "data/3bpa/test_300K_ref.xyz",
    "ethanol": "data/rmd17/ethanol_test_ref.xyz",
    "aspirin": "data/rmd17/aspirin_test_ref.xyz",
    "azobenzene": "data/rmd17/azobenzene_test_ref.xyz",
}


def build_cache(adapter: MaceMHCAdapter, frames: list, *, batch_size: int) -> dict:
    """-> {'F': [S, A, 3, M] head forces, 'E': [S, M] head energies, 'f_ref': [S, A, 3]}.

    Head ENERGIES are cached alongside the forces because they are the input to the
    free energy-disagreement screening baseline: all M come from the single forward
    pass the model already runs, so that gate costs zero reverse lanes. Without it
    the paper cannot answer the obvious reviewer question -- does a calibrated gate
    on the free signal do just as well?
    """
    A = len(frames[0])
    assert all(len(f) == A for f in frames), "3BPA is fixed-size; cache assumes it"
    M = adapter.num_heads
    out = torch.empty(len(frames), A, 3, M, dtype=torch.float64)
    out_E = torch.empty(len(frames), M, dtype=torch.float64)

    done = 0
    for batch in make_loader(frames, adapter.model, batch_size=batch_size):
        b = adapter.prepare(batch.to_dict())
        E = adapter.energies(b).detach().double()          # [B, M]
        F = adapter.exact_head_forces(b).double()          # [M, N, 3]
        n_struct = F.shape[1] // A
        F = F.view(M, n_struct, A, 3).permute(1, 2, 3, 0)  # [n_struct, A, 3, M]
        out[done:done + n_struct] = F.cpu()
        out_E[done:done + n_struct] = E.cpu()
        done += n_struct
    assert done == len(frames)
    return {"F": out, "E": out_E,
            "f_ref": reference_forces(frames).view(len(frames), A, 3).double()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="disjoint", choices=list(VARIANTS))
    ap.add_argument("--split", default="test_1200K")
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    pin_numerics()
    ckpt = Path(VARIANTS[args.variant]) / "multihead_committee_stagetwo.model"
    adapter = MaceMHCAdapter.from_checkpoint(ckpt, dtype=torch.float64)
    frames = load_frames(DATA_PATHS[args.split], limit=args.n_frames)
    M, A = adapter.num_heads, len(frames[0])
    print(f"{args.variant} committee, M={M}, {len(frames)} structures x {A} atoms")

    cache = build_cache(adapter, frames, batch_size=args.batch_size)
    F = cache["F"]  # [S, A, 3, M]

    # --- exact spec SS7 quantities, straight from F -------------------------
    v = F.var(dim=-1, unbiased=True)                    # [S, A, 3]  v_d
    sigma = v.sqrt()
    scores = {
        "S_global": v.sum(dim=(1, 2)),                  # [S]
        "u_atom_mhc": sigma.mean(dim=-1),               # [S, A]
        "u_atom_rms": v.mean(dim=-1).sqrt(),            # [S, A]
    }
    max_atom = scores["u_atom_mhc"].amax(dim=-1)
    max_comp = sigma.amax(dim=(1, 2))
    mean_force = F.mean(dim=-1)                         # [S, A, 3]
    force_err = (mean_force - cache["f_ref"]).pow(2).mean(dim=-1).sqrt()  # [S, A] RMSE/atom

    # cross-check the estimator stack against this direct route
    B0 = min(8, len(frames))
    b0 = adapter.prepare(next(iter(make_loader(frames[:B0], adapter.model,
                                               batch_size=B0))).to_dict())
    bundle = exact_seed_bundle(M, B0, dtype=torch.float64, device=adapter.device)
    lanes = adapter.vjp_for_seeds(b0, bundle.seeds, batched=False)
    sc = uncertainty_scores(lanes, bundle, batch_index=adapter.batch_index(b0),
                            n_structures=B0)
    err = (sc.global_trace.cpu() - scores["S_global"][:B0]).abs().max() / scores["S_global"][:B0].max()
    print(f"  estimator-stack vs direct-from-F, global S: max rel err {err:.2e}")
    assert err < 1e-9, "estimator stack disagrees with the direct route"

    # rho = |mean force| / sqrt(v): the fp32 error-amplification factor (R5)
    rho = float((mean_force.abs().mean() / v.sqrt().mean()))
    print(f"  rho = |mean force| / sigma = {rho:.1f}   "
          f"(fp32 max rel err would be ~{1.4 * 1.19e-7 * rho * np.sqrt(2*np.log(v.numel())):.1e})")

    outdir = Path("results/processed")
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.variant}_{args.split}"
    torch.save({"F": F, "E": cache["E"], "f_ref": cache["f_ref"],
                "frames": len(frames), "M": M, "A": A,
                "variant": args.variant, "split": args.split,
                "checkpoint_hash": checkpoint_hash(adapter.model)},
               outdir / f"head_forces_{tag}.pt")

    sha, dirty = git_commit()
    summary = {
        "experiment_id": "01_exact_reproduction", "git_commit": sha, "git_dirty": dirty,
        "dataset": "rmd17" if args.variant.startswith("rmd17") else "3bpa", "split": args.split, "variant": args.variant,
        "checkpoint_hash": checkpoint_hash(adapter.model), "precision": "float64",
        "num_heads": M, "num_atoms": A, "n_structures": len(frames),
        "S_global": {"median": float(scores["S_global"].median()),
                     "p90": float(scores["S_global"].quantile(0.9)),
                     "max": float(scores["S_global"].max())},
        "u_atom_mhc_median_mev_A": float(scores["u_atom_mhc"].median() * 1000),
        "max_atom_uncertainty_median_mev_A": float(max_atom.median() * 1000),
        "max_component_uncertainty_median_mev_A": float(max_comp.median() * 1000),
        "force_rmse_mean_mev_A": float(force_err.mean() * 1000),
        "rho_mean_force_over_sigma": rho,
    }
    raw = Path("results/raw"); raw.mkdir(parents=True, exist_ok=True)
    with open(raw / f"01_exact_reproduction_{tag}.jsonl", "w") as fh:
        fh.write(json.dumps(summary) + "\n")

    print(f"\n  global S           median {summary['S_global']['median']:.4g}"
          f"  p90 {summary['S_global']['p90']:.4g}")
    print(f"  u_atom^MHC         median {summary['u_atom_mhc_median_mev_A']:.1f} meV/A")
    print(f"  max-atom uncert.   median {summary['max_atom_uncertainty_median_mev_A']:.1f} meV/A")
    print(f"  mean-force RMSE    {summary['force_rmse_mean_mev_A']:.1f} meV/A")
    print(f"\nwrote {outdir / f'head_forces_{tag}.pt'} "
          f"({F.numel() * 8 / 1e6:.1f} MB) and the SS68 summary record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
