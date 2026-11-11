#!/usr/bin/env python
"""J11 -- one round of active learning on 3BPA, scored by what it buys.

Pre-registered in `protocols/j11_active_learning.yaml`.

Why. Everything else in this work ranks structures by how wrong the model's
forces are on them. Reviewer 1 of the first submission pointed out that
identifying high-error structures is not the same as demonstrating a better
acquisition strategy, because nothing was retrained. This experiment closes
that gap for the one system where it is cheap enough to do properly.

What. The pool is the 3BPA 1200 K test set (2,139 frames with DFT labels). Each
selection rule picks k = 50 pool frames using the shared-trunk disjoint
committee trained on `train_300K_100` (the per-head force cache from the
temperature ladder, so the selection is a deterministic function of a committed
artifact). A fresh single-head MACE is then trained on the 100 training frames
plus the 50 selected ones with exactly the recipe Beck et al. used for their
independent ("naive") committee members, and evaluated on the pool frames that
NO rule selected (a common held-out set), on the 600 K test set, and on the
300 K test set. Three model seeds per rule; the `random` rule draws a different
subset per seed; `none` trains on the 100 frames alone and is the control.

Selection rules: global (sum of per-coordinate head variance), maxcomp (largest
per-component head std), maxatom (largest per-atom head std), force_norm (max
over atoms of the predicted mean-force norm; zero cost), energy_std (std of the
head energies; one forward pass), random, none.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from forcesketch_journal.paths import CACHES, DATA, FROZEN, PY_GPU, SCRATCH  # noqa: E402
sys.path.insert(0, str(FROZEN / "src"))

import ase.io  # noqa: E402

AL = SCRATCH / "al"
REC = ROOT / "results/records/j11"
POOL_XYZ = DATA / "data/3bpa/test_1200K_ref.xyz"
TRAIN_XYZ = DATA / "data/3bpa/train_300K_100.xyz"
VALID_XYZ = DATA / "data/3bpa/val_300K_300.xyz"
TEST_XYZ = {"test_600K": DATA / "data/3bpa/test_600K_ref.xyz",
            "test_300K": DATA / "data/3bpa/test_300K_ref.xyz"}
CACHE = CACHES / "head_forces_3bpa-disjoint_test_1200K.pt"
SELECTORS = ["none", "global", "maxcomp", "maxatom", "force_norm", "energy_std", "random"]
# Verbatim from models/zenodo/3BPA/trainset_100/naive-committee/run_mace.sh,
# except --valid_file (a fixed validation set, as naive-committee-same used, so
# the training set is exactly the 100 + k frames and not 80% of them).
E0S = "{1: -13.587222780835477, 6: -1029.4889999855063, 7: -1484.9814568572233, 8: -2041.9816003861047}"
MACE_ARGS = ["--model=MACE", "--num_interactions=2", "--num_channels=32", "--max_L=1",
             "--correlation=3", "--r_max=6.0", "--forces_weight=1000", "--energy_weight=10",
             "--energy_key=energy_ref", "--forces_key=forces_ref", "--batch_size=10",
             "--valid_batch_size=10", "--max_num_epochs=625", "--start_swa=469",
             "--eval_interval=8", "--ema", "--swa", "--error_table=PerAtomRMSE",
             "--default_dtype=float32", "--device=cuda"]


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ select
def signals(d: dict) -> dict[str, torch.Tensor]:
    F, E = d["F"].double(), d["E"].double()                 # [S,A,3,M], [S,M]
    v = F.var(dim=-1, unbiased=True)                         # [S,A,3]
    return {"global": v.sum(dim=(1, 2)),
            "maxcomp": v.sqrt().flatten(1).max(dim=1).values,
            "maxatom": v.sqrt().mean(dim=-1).max(dim=1).values,
            "force_norm": F.mean(-1).norm(dim=-1).max(dim=1).values,
            "energy_std": E.std(dim=-1, unbiased=True)}


def select(k: int, seeds: list[int]) -> dict:
    d = torch.load(CACHE, weights_only=False, map_location="cpu")
    S = d["F"].shape[0]
    sig = signals(d)
    sel: dict[str, dict[str, list[int]]] = {}
    for name, s in sig.items():
        top = torch.topk(s, k).indices.sort().values.tolist()
        sel[name] = {str(seed): top for seed in seeds}
    sel["random"] = {str(seed): sorted(np.random.default_rng(seed).choice(S, k, replace=False).tolist())
                     for seed in seeds}
    sel["none"] = {str(seed): [] for seed in seeds}
    used = sorted({i for v in sel.values() for ids in v.values() for i in ids})
    heldout = [i for i in range(S) if i not in set(used)]
    out = {"k": k, "seeds": seeds, "pool": str(POOL_XYZ), "pool_sha256": sha256(POOL_XYZ),
           "cache": str(CACHE), "cache_checkpoint_hash": d.get("checkpoint_hash"),
           "n_pool": S, "n_heldout": len(heldout), "selection": sel, "heldout": heldout,
           "overlap": {a: {b: len(set(sel[a][str(seeds[0])]) & set(sel[b][str(seeds[0])]))
                           for b in sel} for a in sel}}
    AL.mkdir(parents=True, exist_ok=True)
    (AL / "selection.json").write_text(json.dumps(out, indent=1))
    pool = ase.io.read(POOL_XYZ, index=":", format="extxyz")
    train = ase.io.read(TRAIN_XYZ, index=":", format="extxyz")
    (AL / "sets").mkdir(exist_ok=True)
    for name, per_seed in sel.items():
        for seed, ids in per_seed.items():
            frames = train + [pool[i] for i in ids]
            for i, a in zip([-1] * len(train) + ids, frames):
                a.info["pool_index"] = int(i)
            ase.io.write(AL / "sets" / f"{name}_s{seed}_train.xyz", frames, format="extxyz")
    ase.io.write(AL / "sets" / "heldout_1200K.xyz", [pool[i] for i in heldout], format="extxyz")
    print(f"selected k={k} for {len(sel)} rules; common held-out set {len(heldout)} of {S}")
    for a in sel:
        print(f"  {a:<11}", {b: out['overlap'][a][b] for b in sel if b != a})
    return out


# ------------------------------------------------------------------- train
def train(selector: str, seed: int) -> pathlib.Path:
    run = AL / "runs" / f"{selector}_s{seed}"
    model = run / "mlp_stagetwo.model"
    if model.exists():
        print(f"[{selector} s{seed}] model exists, skipping training")
        return model
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    cmd = [str(PY_GPU.parent / "mace_run_train"), "--name=mlp", f"--seed={seed}",
           f"--train_file={AL / 'sets' / f'{selector}_s{seed}_train.xyz'}",
           f"--valid_file={VALID_XYZ}", f"--E0s={E0S}",
           f"--work_dir={run}", f"--model_dir={run}", f"--log_dir={run}",
           f"--checkpoints_dir={run / 'checkpoints'}", f"--results_dir={run}", *MACE_ARGS]
    t0 = time.time()
    with (run / "train.stdout").open("w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(run))
    (run / "train_seconds").write_text(f"{time.time() - t0:.1f}\n")
    if r.returncode != 0 or not model.exists():
        raise SystemExit(f"training failed for {selector} s{seed}: see {run / 'train.stdout'}")
    shutil.rmtree(run / "checkpoints", ignore_errors=True)   # regenerable, large
    print(f"[{selector} s{seed}] trained in {time.time() - t0:.0f}s")
    return model


# ---------------------------------------------------------------- evaluate
def evaluate(selector: str, seed: int, device: str = "cuda") -> dict:
    sys.path.insert(0, str(ROOT / "experiments"))
    from j6_naive_caches import make_loader  # same loader as the naive-committee caches

    run = AL / "runs" / f"{selector}_s{seed}"
    model = torch.load(run / "mlp_stagetwo.model", map_location="cpu", weights_only=False)
    if not hasattr(model, "heads"):
        model.heads = ["default"]
    model = model.to(device=device, dtype=torch.float64).eval()
    rec = {"experiment_id": "j11_active_learning", "selector": selector, "seed": seed,
           "model_sha256": sha256(run / "mlp_stagetwo.model"),
           "train_seconds": float((run / "train_seconds").read_text()) if (run / "train_seconds").exists() else None,
           "train_file_sha256": sha256(AL / "sets" / f"{selector}_s{seed}_train.xyz"),
           "n_train": sum(1 for _ in ase.io.iread(AL / "sets" / f"{selector}_s{seed}_train.xyz", format="extxyz")),
           "eval": {}}
    sets = {"heldout_1200K": AL / "sets" / "heldout_1200K.xyz", **TEST_XYZ}
    per_structure = {}
    for name, xyz in sets.items():
        frames = ase.io.read(xyz, index=":", format="extxyz")
        A = len(frames[0])
        fp, ep = [], []
        for batch in make_loader(frames, model, batch_size=16):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.to_dict().items()}
            for k, v in b.items():
                if torch.is_tensor(v) and torch.is_floating_point(v):
                    b[k] = v.to(torch.float64)
            b["positions"] = b["positions"].detach().requires_grad_(True)
            out = model(b, training=False, compute_force=True)
            fp.append(out["forces"].detach().double().cpu()); ep.append(out["energy"].detach().double().cpu())
        fp = torch.cat(fp).view(-1, A, 3); ep = torch.cat(ep)
        fr = torch.from_numpy(np.stack([a.arrays["forces_ref"] for a in frames])).double()
        er = torch.tensor([float(a.info["energy_ref"]) for a in frames], dtype=torch.float64)
        d = fp - fr
        rec["eval"][name] = {
            "n": len(frames),
            "force_rmse_mev_A": float(d.pow(2).mean().sqrt()) * 1000,
            "force_mae_mev_A": float(d.abs().mean()) * 1000,
            "e_max_median_mev_A": float(d.norm(dim=-1).max(dim=1).values.median()) * 1000,
            "e_max_p95_mev_A": float(d.norm(dim=-1).max(dim=1).values.quantile(0.95)) * 1000,
            "energy_mae_mev_atom": float((ep - er).abs().mean() / A) * 1000,
            "energy_rmse_mev_atom": float((ep - er).pow(2).mean().sqrt() / A) * 1000,
        }
        per_structure[name] = {"e_max": d.norm(dim=-1).max(dim=1).values, "e_rmse": d.pow(2).mean(dim=(1, 2)).sqrt(),
                               "de_per_atom": (ep - er) / A}
        print(f"[{selector} s{seed}] {name}: F-RMSE {rec['eval'][name]['force_rmse_mev_A']:.1f} meV/A  "
              f"E-MAE {rec['eval'][name]['energy_mae_mev_atom']:.2f} meV/atom")
    torch.save(per_structure, run / "eval_perstructure.pt")
    REC.mkdir(parents=True, exist_ok=True)
    (REC / f"j11_{selector}_s{seed}.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["select", "train", "evaluate", "all"], default="all")
    ap.add_argument("--selectors", nargs="+", default=SELECTORS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    reg = json.loads((ROOT / "results/records/j11_protocol_registration.json").read_text())
    if sha256(ROOT / reg["protocol"]) != reg["protocol_sha256"]:
        raise SystemExit("REFUSING TO RUN: protocols/j11_active_learning.yaml changed since registration")
    if a.phase in ("select", "all"):
        select(a.k, [0, 1, 2])          # always all three seeds: the held-out set must be common
    if a.phase in ("train", "all"):
        for s in a.seeds:
            for sel in a.selectors:
                train(sel, s)
    if a.phase in ("evaluate", "all"):
        for s in a.seeds:
            for sel in a.selectors:
                evaluate(sel, s, device=a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
