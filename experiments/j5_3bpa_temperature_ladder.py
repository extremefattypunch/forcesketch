#!/usr/bin/env python
"""J5 — the 3BPA temperature ladder: a controlled test of the distribution-shift
hypothesis, and a fix for the missing head energies.

J5's water result suggested that Factor A (how well exact committee disagreement
predicts reference-force error) tracks *degree of distribution shift* rather than
system size or chemistry: water MD in-distribution scored 0.647–0.689, water PIMD
under quantum-nuclear shift scored 0.970–0.976. But water changes the shift and
the physics together, and rMD17 sits above water MD despite both being
in-distribution, so shift is not established as the driver.

3BPA isolates it. The committees were trained on 300 K data; the release ships
test sets at 300 K, 600 K and 1200 K of the same molecule. Holding the model,
chemistry, system size and D fixed while varying only how far the evaluation
distribution sits from training is the controlled experiment the hypothesis needs.
The workshop used only the 1200 K set.

This also closes a gap found in Stage 1: the committed `overlapping` and `same`
caches predate the addition of head energies, so `E` is absent and the free
`energy_std` baseline covers only four of ten system-regimes. Regenerating all
three committees at all three temperatures fixes that in the same pass.

Output schema matches `01_exact_reproduction.py` exactly, so every downstream
journal experiment reads these with no special-casing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch.adapters.mace_data import load_frames, make_loader, reference_forces  # noqa: E402
from forcesketch.adapters.mace_mhc import MaceMHCAdapter  # noqa: E402
from forcesketch.exact.member_forces import variance_from_member_forces  # noqa: E402
from forcesketch.utils.reproducibility import checkpoint_hash, git_commit, pin_numerics  # noqa: E402

EV_TO_MEV = 1000.0
VARIANTS = {"disjoint": "multihead-disjoint",
            "overlapping": "multihead-overlapping",
            "same": "multihead-same"}


def build_cache(adapter, frames, *, batch_size: int) -> dict:
    A = len(frames[0])
    M = adapter.num_heads
    S = len(frames)
    F = torch.empty(S, A, 3, M, dtype=torch.float64)
    E = torch.empty(S, M, dtype=torch.float64)
    done = 0
    for batch in make_loader(frames, adapter.model, batch_size=batch_size):
        b = adapter.prepare(batch.to_dict())
        e = adapter.energies(b).detach().double()
        f = adapter.exact_head_forces(b).double()
        n = f.shape[1] // A
        F[done:done + n] = f.view(M, n, A, 3).permute(1, 2, 3, 0).cpu()
        E[done:done + n] = e.cpu()
        done += n
    assert done == S, f"{done} != {S}"
    return {"F": F, "E": E,
            "f_ref": reference_forces(frames).view(S, A, 3).double(),
            "frames": S, "M": M, "A": A}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--temperatures", nargs="+", default=["300K", "600K", "1200K"])
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    pin_numerics()
    outdir = pathlib.Path(args.outdir or (ROOT / "results/processed"))
    outdir.mkdir(parents=True, exist_ok=True)
    sha, dirty = git_commit(FROZEN.parent)
    records = []

    for variant in args.variants:
        ckpt = FROZEN / f"models/zenodo/3BPA/trainset_100/{VARIANTS[variant]}/multihead_committee_stagetwo.model"
        ad = MaceMHCAdapter.from_checkpoint(ckpt, device="cuda", dtype=torch.float64)
        chash = checkpoint_hash(ad.model)
        for temp in args.temperatures:
            frames = load_frames(FROZEN / f"data/3bpa/test_{temp}_ref.xyz", limit=args.n_frames)
            t0 = time.time()
            c = build_cache(ad, frames, batch_size=args.batch_size)
            dt = time.time() - t0

            F, E, fref = c["F"], c["E"], c["f_ref"]
            v_direct = variance_from_member_forces(F.permute(3, 0, 1, 2))
            rel = float((v_direct - F.var(dim=-1, unbiased=True)).abs().max()
                        / F.var(dim=-1, unbiased=True).abs().max())
            assert rel < 1e-9, f"estimator/direct mismatch {rel:.2e}"

            delta = F.mean(-1) - fref
            rmse = float(delta.pow(2).mean().sqrt()) * EV_TO_MEV
            disagr = float(F.std(dim=-1, unbiased=True).mean(dim=-1).median()) * EV_TO_MEV
            records.append({
                "experiment_id": "j5_3bpa_temperature_ladder",
                "git_commit": sha, "git_dirty": dirty,
                "dataset": "3bpa", "variant": variant, "split": f"test_{temp}",
                "temperature_K": int(temp.rstrip("K")),
                "trained_at_K": 300, "checkpoint_hash": chash, "precision": "float64",
                "num_heads": c["M"], "num_atoms": c["A"], "n_structures": c["frames"],
                "force_rmse_mean_mev_A": rmse,
                "median_atom_disagreement_mev_A": disagr,
                "overconfidence_ratio": rmse / disagr,
                "estimator_vs_direct_rel_err": rel, "seconds": dt,
            })
            tag = f"3bpa-{variant}_test_{temp}"
            torch.save({**c, "variant": f"3bpa-{variant}", "split": f"test_{temp}",
                        "checkpoint_hash": chash}, outdir / f"head_forces_{tag}.pt")
            print(f"[{tag}] S={c['frames']} rmse={rmse:7.1f} disagr={disagr:5.1f} "
                  f"overconf={rmse/disagr:5.1f}x ({dt:.0f}s)")

    out = ROOT / "results/records/j5_3bpa_temperature_ladder.jsonl"
    with out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"\nwrote {len(records)} caches to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
