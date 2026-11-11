#!/usr/bin/env python
"""J10 -- score the foundation committees on OUT-OF-DISTRIBUTION materials.

Pre-registered in `protocols/j10_ood.yaml`.

Why this exists. The MPtraj pool scored in J6B is held out from the committee
READOUTS but not from the MACE-MP trunk, which was pretrained on all of MPtraj
(labels included), and the pool contains relaxed, near-equilibrium frames whose
reference forces are close to zero. Both reviewers of the first submission
raised this. OMat24 (Barroso-Luque et al., 2024) post-dates MACE-MP-0, was
generated with Materials-Project-compatible DFT settings, and is what Beck et
al. themselves used as the out-of-distribution test for the same committees.
Its AIMD subsets are far from equilibrium by construction.

What it does. For each requested foundation committee (the QbC-8000 one scored
in J6B and, because nothing here was in their training sets either, the
random-8000, max-mean-force-8000 and QbC-1000 ones), scan a seeded subsample of
each OMat24 validation subset with exactly the same `scan` function as J6B, so
every statistic is computed by the same code path. Per-structure outputs go to a
.pt under FORCESKETCH_CACHES/ood and a compact npz under results/records/j10.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from forcesketch_journal.paths import CACHES, FOUNDATION, FROZEN, SCRATCH  # noqa: E402
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch.adapters.mace_mhc import MaceMHCAdapter  # noqa: E402
from forcesketch.utils.reproducibility import pin_numerics  # noqa: E402
from forcesketch_journal.data.aselmdb import AseLMDB  # noqa: E402
from j6b_foundation_scan import scan, self_test, write_compact  # noqa: E402

MODELS = {
    "qbc8000": FOUNDATION / "qbc/dataset_8000/MACE_qbc_max_selection_stagetwo.model",
    "qbc1000": FOUNDATION / "qbc/dataset_1000/MACE_qbc_max_selection_stagetwo.model",
    "random8000": FOUNDATION / "random/MACE_stagetwo.model",
    "mmf8000": FOUNDATION / "max_mean_force/MACE_max_forces_stagetwo.model",
}
SUBSETS = ["aimd-from-PBE-1000-nvt", "aimd-from-PBE-1000-npt", "rattled-1000-subsampled"]


def sample_frames(db_path: pathlib.Path, n: int, seed: int, allowed: set[int]):
    """Seeded subsample of an .aselmdb, keeping only structures whose elements the
    model knows. Returns (frames, row_ids, sids, n_dropped_elements)."""
    db = AseLMDB(db_path)
    ids = np.array(db.ids())
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(ids, size=min(n, len(ids)), replace=False))
    frames, rows, sids, dropped = [], [], [], 0
    for rid in pick.tolist():
        atoms = db.get(rid)
        if not set(int(z) for z in atoms.numbers) <= allowed:
            dropped += 1
            continue
        atoms.arrays["REF_forces"] = atoms.get_forces()
        atoms.info["REF_energy"] = float(atoms.get_potential_energy())
        sids.append(str(atoms.info.get("data", {}).get("sid", "")))
        atoms.info = {"REF_energy": atoms.info["REF_energy"]}   # drop the dict payload
        atoms.calc = None
        frames.append(atoms)
        rows.append(rid)
    db.close()
    return frames, rows, sids, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--omat-root", default=str(SCRATCH / "omat24"))
    ap.add_argument("--n-per-subset", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=str(CACHES / "ood"))
    ap.add_argument("--smoke", action="store_true", help="32 structures, one subset, one model")
    a = ap.parse_args()

    pin_numerics()
    self_test()
    reg = json.loads((ROOT / "results/records/j10_protocol_registration.json").read_text())
    proto_sha = hashlib.sha256((ROOT / reg["protocol"]).read_bytes()).hexdigest()
    if proto_sha != reg["protocol_sha256"]:
        raise SystemExit("REFUSING TO SCAN: protocols/j10_ood.yaml changed since registration")

    out_dir = pathlib.Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rec_dir = ROOT / "results/records/j10"; rec_dir.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        a.models, a.subsets, a.n_per_subset = a.models[:1], a.subsets[:1], 32
    dtype = getattr(torch, a.dtype)
    summary = []
    for mname in a.models:
        ad = MaceMHCAdapter.from_checkpoint(MODELS[mname], device=a.device, dtype=dtype)
        allowed = set(int(z) for z in ad.model.atomic_numbers)
        msha = hashlib.sha256(MODELS[mname].read_bytes()).hexdigest()
        print(f"model {mname}: {MODELS[mname].name} heads={ad.num_heads} elements={len(allowed)}",
              flush=True)
        for sub in a.subsets:
            t0 = time.time()
            frames, rows, sids, dropped = sample_frames(
                pathlib.Path(a.omat_root) / sub / "data.aselmdb", a.n_per_subset, a.seed, allowed)
            print(f"  {sub}: {len(frames)} structures ({dropped} dropped for unknown elements) "
                  f"loaded in {time.time() - t0:.0f}s", flush=True)
            parts, edges = [], 0
            for i in range(0, len(frames), a.chunk):
                r = scan(ad, frames[i:i + a.chunk], batch_size=a.batch_size)
                edges += r.pop("n_periodic_edges")
                parts.append(r)
                el = time.time() - t0
                print(f"    {min(i + a.chunk, len(frames)):6d} scanned  {el:6.0f}s", flush=True)
            res = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
            res.update({"row_id": torch.tensor(rows), "n_periodic_edges": edges,
                        "model": mname, "model_path": str(MODELS[mname]), "model_sha256": msha,
                        "subset": sub, "seed": a.seed, "n_requested": a.n_per_subset,
                        "n_dropped_elements": dropped, "protocol_sha256": proto_sha,
                        "seconds": time.time() - t0, "sids": sids})
            tag = f"j10_{mname}_{sub}" + ("_smoke" if a.smoke else "")
            torch.save(res, out_dir / f"{tag}.pt")
            res_np = {k: v for k, v in res.items() if k != "sids"}
            write_compact(res_np, rec_dir / f"{tag}_perstructure.npz")
            np.savez_compressed(rec_dir / f"{tag}_sids.npz", sids=np.array(sids), row_id=np.array(rows))
            delta_e = (res["E_mean"] - res["E_ref"]) / res["natoms"]
            summary.append({
                "model": mname, "subset": sub, "n": len(frames), "dropped": dropped,
                "force_rmse_mev_A": float(res["e_rmse"].pow(2).mean().sqrt()) * 1000,
                "energy_mae_mev_atom": float(delta_e.abs().mean()) * 1000,
                "median_ref_force_max": float(res["ref_force_max"].median()),
                "seconds": res["seconds"]})
            print("    ", json.dumps(summary[-1]), flush=True)
    with (rec_dir / ("j10_scan_summary%s.json" % ("_smoke" if a.smoke else ""))).open("w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
