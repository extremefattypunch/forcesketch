#!/usr/bin/env python
"""J6 -- head-force caches for NAIVE committees (M independently trained models).

Pre-registered in `protocols/j6_naive_committee.yaml`; see that file for what is
being tested and why. In one line: every result in the project so far comes from
one shared-trunk multi-head model, and this is the control that says which of
those results are properties of committees in general and which are properties of
that architecture.

The output schema is byte-compatible with `01_exact_reproduction.py` and with
`j5_water_caches.py`:

    {F: [S, A, 3, M] float64, E: [S, M], f_ref: [S, A, 3], M, A, variant, split,
     checkpoint_hash, frames}

so every downstream experiment reads these caches through `--cache-dirs` with no
new analysis code. That is deliberate: if the naive numbers went through a
different code path they would not be comparable to the shared-trunk numbers, and
the entire point of this experiment is the comparison.

What differs from the multi-head path, and why each difference is checked:

  * M models instead of M heads. The M columns of F must correspond to the same
    structures in the same order, so ONE dataloader (built from model 0) feeds
    all M models rather than one loader per model. A per-model loader would be
    deterministic today and a silent misalignment the day someone adds shuffling.
  * The models must agree on the element table and cutoff, or column m would be
    evaluated on a different graph from column m'. Asserted, not assumed --
    a mismatched `atomic_numbers` ORDER silently permutes species embeddings and
    yields finite, wrong forces.
  * No `committee_heads` argument and no head-space cotangent: forces come from
    ordinary autograd on each model's total energy. Cross-checked against MACE's
    own `compute_force=True` output on the first batch, since these are two
    independent routes to the same tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch.adapters.mace_data import load_frames, reference_forces  # noqa: E402
from forcesketch.exact.member_forces import variance_from_member_forces  # noqa: E402
from forcesketch.utils.reproducibility import git_commit, pin_numerics  # noqa: E402

EV_TO_MEV = 1000.0
MODELS = pathlib.Path("/n/holylabs/hekstra_lab/Everyone/ianpoon/forcesketch-data/models/zenodo")

# (tag, committee dir, frames file, matched shared-trunk system tag)
SYSTEMS = {
    "3bpa-naive-same_test_1200K": (
        MODELS / "3BPA/trainset_100/naive-committee-same",
        MODELS.parent.parent / "data/3bpa/test_1200K.xyz", "same_test_1200K"),
    "3bpa-naive_test_1200K": (
        MODELS / "3BPA/trainset_100/naive-committee",
        MODELS.parent.parent / "data/3bpa/test_1200K.xyz", "overlapping_test_1200K"),
    "water-naive_md_T300K": (
        MODELS / "water/naive-committee",
        MODELS / "water/w64_md_T300K.xyz", "water-overlapping_md_T300K"),
    "water-naive_pimd_T300K": (
        MODELS / "water/naive-committee",
        MODELS / "water/w64_pimd_T300K.xyz", "water-overlapping_pimd_T300K"),
}


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_committee(d: pathlib.Path, *, device: str, dtype: torch.dtype):
    """Load nnp-0..7 and prove they are mutually comparable before use."""
    paths = sorted(d.glob("nnp-*.model"), key=lambda p: int(p.stem.split("-")[1]))
    if not paths:
        raise SystemExit(f"no nnp-*.model under {d}")
    models, hashes = [], []
    for p in paths:
        m = torch.load(p, map_location="cpu", weights_only=False)  # noqa: S614
        if not hasattr(m, "heads"):
            # plain MACE checkpoints predate the multi-head fork's `heads` list;
            # the dataloader needs one, and a single-head model has exactly one.
            m.heads = ["default"]
        models.append(m.to(device=device, dtype=dtype).eval())
        hashes.append(sha256_file(p))

    ref = models[0]
    for i, m in enumerate(models[1:], start=1):
        if not torch.equal(m.atomic_numbers.cpu(), ref.atomic_numbers.cpu()):
            raise SystemExit(f"{paths[i].name}: atomic_numbers differ from nnp-0 "
                             f"({m.atomic_numbers.tolist()} vs {ref.atomic_numbers.tolist()}). "
                             "Columns of F would be evaluated on different species maps.")
        if float(m.r_max) != float(ref.r_max):
            raise SystemExit(f"{paths[i].name}: r_max {float(m.r_max)} != {float(ref.r_max)}; "
                             "the graphs would differ between committee members.")
        if list(m.heads) != list(ref.heads):
            raise SystemExit(f"{paths[i].name}: heads {list(m.heads)} != {list(ref.heads)}")
    # Distinct weights are the whole premise -- two identical members would make
    # the committee silently smaller than M.
    if len(set(hashes)) != len(hashes):
        raise SystemExit(f"duplicate committee members in {d}: {hashes}")
    return models, [p.name for p in paths], hashes


def make_loader(frames, model, *, batch_size: int):
    from mace import data
    from mace.tools import AtomicNumberTable, torch_geometric

    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    configs = [data.config_from_atoms(a) for a in frames]
    dataset = [data.AtomicData.from_config(c, z_table=z_table, cutoff=float(model.r_max),
                                           heads=list(model.heads)) for c in configs]
    return torch_geometric.dataloader.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=False, drop_last=False)


def build_cache(models, frames, *, batch_size: int, device: str, dtype: torch.dtype) -> dict:
    A = len(frames[0])
    M, S = len(models), len(frames)
    F = torch.empty(S, A, 3, M, dtype=torch.float64)
    E = torch.empty(S, M, dtype=torch.float64)
    done, n_periodic_edges, force_route_rel = 0, 0, None

    for bi, batch in enumerate(make_loader(frames, models[0], batch_size=batch_size)):
        bd = batch.to_dict()
        for m, model in enumerate(models):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in bd.items()}
            for k, v in b.items():
                if torch.is_tensor(v) and torch.is_floating_point(v):
                    b[k] = v.to(dtype)
            b["positions"] = b["positions"].detach().requires_grad_(True)
            out = model(b, training=False, compute_force=True)
            e = out["energy"].detach().double()                      # [B]
            f = out["forces"].detach().double()                      # [N, 3]
            if bi == 0 and m == 0:
                # independent route: autograd on the total energy. MACE computes
                # `forces` internally the same way, so a disagreement here means
                # the model is not differentiating what we think it is.
                b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
                b2["positions"] = b["positions"].detach().clone().requires_grad_(True)
                o2 = model(b2, training=False, compute_force=False)
                g = -torch.autograd.grad(o2["energy"].sum(), b2["positions"])[0].double()
                force_route_rel = float((g - f).abs().max() / f.abs().max().clamp(min=1e-30))
            if m == 0 and "shifts" in b:
                n_periodic_edges += int((b["shifts"].abs().sum(-1) > 0).sum())
            n = f.shape[0] // A
            F[done:done + n, :, :, m] = f.view(n, A, 3).cpu()
            E[done:done + n, m] = e.cpu()
        done += n
    assert done == S, f"{done} != {S}"
    return {"F": F, "E": E,
            "f_ref": reference_forces(frames).view(S, A, 3).double(),
            "frames": S, "M": M, "A": A, "n_periodic_edges": n_periodic_edges,
            "force_route_rel_err": force_route_rel}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=sorted(SYSTEMS))
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default=str(ROOT / "results/processed/naive"))
    args = ap.parse_args()

    pin_numerics()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sha, dirty = git_commit(FROZEN.parent)
    proto = json.loads((ROOT / "results/records/j6_protocol_registration.json").read_text())
    records = []

    for tag in args.systems:
        cdir, xyz, matched = SYSTEMS[tag]
        models, names, hashes = load_committee(cdir, device=args.device, dtype=torch.float64)
        frames = load_frames(xyz, limit=args.n_frames)
        t0 = time.time()
        c = build_cache(models, frames, batch_size=args.batch_size,
                        device=args.device, dtype=torch.float64)
        dt = time.time() - t0

        Ft, fref = c["F"], c["f_ref"]
        v_direct = variance_from_member_forces(Ft.permute(3, 0, 1, 2))
        v_est = Ft.var(dim=-1, unbiased=True)
        rel = float((v_direct - v_est).abs().max() / v_est.abs().max())
        assert rel < 1e-9, f"estimator/direct mismatch {rel:.2e}"
        assert c["force_route_rel_err"] is None or c["force_route_rel_err"] < 1e-10, \
            f"MACE forces vs autograd disagree: {c['force_route_rel_err']:.2e}"

        delta = Ft.mean(-1) - fref
        rmse = float(delta.pow(2).mean().sqrt()) * EV_TO_MEV
        disag = float(Ft.std(dim=-1, unbiased=True).mean(dim=-1).median()) * EV_TO_MEV
        variant, split = tag.split("_", 1)
        records.append({
            "experiment_id": "j6_naive_caches", "git_commit": sha, "git_dirty": dirty,
            "protocol_sha256": proto["protocol_sha256"],
            "construction": "naive", "variant": variant, "split": split,
            "matched_shared_trunk_system": matched,
            "committee_dir": str(cdir), "member_files": names,
            "member_sha256": hashes, "checkpoint_hash": hashlib.sha256(
                "".join(hashes).encode()).hexdigest()[:12],
            "precision": "float64", "num_heads": c["M"], "num_atoms": c["A"],
            "n_structures": c["frames"], "D_3N": 3 * c["A"],
            "n_periodic_image_edges": c["n_periodic_edges"],
            "force_rmse_mean_mev_A": rmse,
            "median_atom_disagreement_mev_A": disag,
            "overconfidence_ratio": rmse / disag,
            "estimator_vs_direct_rel_err": rel,
            "mace_vs_autograd_rel_err": c["force_route_rel_err"],
            "seconds": dt,
        })
        torch.save({**{k: v for k, v in c.items() if k != "force_route_rel_err"},
                    "variant": variant, "split": split,
                    "checkpoint_hash": records[-1]["checkpoint_hash"]},
                   outdir / f"head_forces_{tag}.pt")
        print(f"[{tag}] S={c['frames']} A={c['A']} M={c['M']} "
              f"rmse={rmse:.1f} disagreement={disag:.1f} meV/A "
              f"overconfidence={rmse/disag:.2f}x ({dt:.0f}s)")

    out = ROOT / "results/records/j6_naive_caches.jsonl"
    with out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"\nwrote {len(records)} caches to {outdir} and records to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
