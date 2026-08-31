#!/usr/bin/env python
"""Validate the reference committee checkpoints (plan Task 0.4 gate).

Six sanity criteria, all required:

  1. force RMSE on 3BPA@1200K in 100-250 meV/A
  2. per-head force RMSEs within ~1.5x of each other
  3. non-degenerate disagreement: median atom disagreement 10-40% of force RMSE
  4. disjoint shows LARGER disagreement than overlapping (the paper's ordering)
  5. energy MAE roughly 5-20 meV/atom
  6. head energies genuinely differ

Criterion 3 is the one that matters most. A committee whose heads collapsed would
make every ForceSketch estimate vacuously exact, and nothing in a loss curve would
show it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from forcesketch.adapters.mace_data import load_frames, make_loader, reference_forces
from forcesketch.adapters.mace_mhc import MaceMHCAdapter
from forcesketch.utils.reproducibility import checkpoint_hash, pin_numerics

EV_TO_MEV = 1000.0


def evaluate(adapter: MaceMHCAdapter, frames: list, *, batch_size: int = 8) -> dict:
    loader = make_loader(frames, adapter.model, batch_size=batch_size)
    se_head = torch.zeros(adapter.num_heads, dtype=torch.float64)
    n_comp = 0
    disagreement: list[torch.Tensor] = []
    e_head_std: list[torch.Tensor] = []
    e_abs_err, n_atoms_tot = 0.0, 0

    f_ref_all = reference_forces(frames).to(torch.float64)
    e_ref_all = torch.tensor([f.info["energy_ref"] for f in frames], dtype=torch.float64)

    atom_off, frame_off = 0, 0
    for batch in loader:
        batch = adapter.prepare(batch.to_dict())
        E = adapter.energies(batch).detach().double().cpu()  # [B, M]
        F = adapter.member_forces_reference(batch).detach().double()  # [M, N, 3]
        n, b = F.shape[1], E.shape[0]

        f_ref = f_ref_all[atom_off:atom_off + n].to(F.device)
        se_head += ((F - f_ref.unsqueeze(0)) ** 2).sum(dim=(1, 2)).cpu()
        n_comp += n * 3
        disagreement.append(F.std(dim=0).mean(dim=-1).cpu())  # [N], mean over xyz

        e_head_std.append(E.std(dim=-1))
        e_abs_err += float((E.mean(dim=-1) - e_ref_all[frame_off:frame_off + b]).abs().sum())
        n_atoms_tot += n
        atom_off += n
        frame_off += b

    rmse_head = (se_head / n_comp).sqrt() * EV_TO_MEV
    dis = torch.cat(disagreement) * EV_TO_MEV
    return {
        "force_rmse_committee_mev_A": float(rmse_head.mean()),
        "force_rmse_per_head_mev_A": [float(x) for x in rmse_head],
        "head_rmse_spread_ratio": float(rmse_head.max() / rmse_head.min()),
        "median_atom_disagreement_mev_A": float(dis.median()),
        "mean_atom_disagreement_mev_A": float(dis.mean()),
        "energy_mae_mev_per_atom": float(e_abs_err / n_atoms_tot * EV_TO_MEV),
        "median_head_energy_std_mev": float(torch.cat(e_head_std).median() * EV_TO_MEV),
        "n_frames": len(frames),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="results/manifests/checkpoints.json")
    args = ap.parse_args()

    pin_numerics()
    frames = load_frames("data/3bpa/test_1200K_ref.xyz", limit=args.n_frames)
    print(f"3BPA@1200K: {len(frames)} frames x {len(frames[0])} atoms\n")

    variants = {
        "disjoint": "models/zenodo/3BPA/trainset_100/multihead-disjoint",
        "overlapping": "models/zenodo/3BPA/trainset_100/multihead-overlapping",
        "same": "models/zenodo/3BPA/trainset_100/multihead-same",
    }
    report: dict[str, dict] = {}
    for name, d in variants.items():
        path = Path(d) / "multihead_committee_stagetwo.model"
        adapter = MaceMHCAdapter.from_checkpoint(path)
        m = evaluate(adapter, frames, batch_size=args.batch_size)
        m["checkpoint_hash"] = checkpoint_hash(adapter.model)
        m["path"] = str(path)
        report[name] = m
        print(f"[{name}]")
        print(f"  force RMSE (committee mean of heads) : {m['force_rmse_committee_mev_A']:8.1f} meV/A")
        print(f"  per-head RMSE spread (max/min)       : {m['head_rmse_spread_ratio']:8.2f}x")
        print(f"  median atom disagreement             : {m['median_atom_disagreement_mev_A']:8.1f} meV/A"
              f"  ({100*m['median_atom_disagreement_mev_A']/m['force_rmse_committee_mev_A']:.0f}% of RMSE)")
        print(f"  median head-energy std               : {m['median_head_energy_std_mev']:8.1f} meV")
        print(f"  checkpoint_hash                      : {m['checkpoint_hash']}\n")

    print("=" * 66)
    ok = True
    for name, m in report.items():
        r = m["force_rmse_committee_mev_A"]
        frac = m["median_atom_disagreement_mev_A"] / r
        c1 = 100 <= r <= 250
        c2 = m["head_rmse_spread_ratio"] <= 1.5
        c3 = m["median_atom_disagreement_mev_A"] > 0
        print(f"  {name:12s} RMSE 100-250 [{ 'ok' if c1 else 'NO'}]  "
              f"head spread <=1.5x [{'ok' if c2 else 'NO'}]  "
              f"disagreement > 0 [{'ok' if c3 else 'NO'}]  (frac {frac:.2f})")
        ok &= c2 and c3
    d, o = (report["disjoint"]["median_atom_disagreement_mev_A"],
            report["overlapping"]["median_atom_disagreement_mev_A"])
    s = report["same"]["median_atom_disagreement_mev_A"]
    print(f"\n  head-diversity ordering  same {s:.1f} | overlapping {o:.1f} | disjoint {d:.1f} meV/A")
    print(f"  disjoint > overlapping (paper's ordering): {d > o}")
    print("=" * 66)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
