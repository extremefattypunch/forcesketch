#!/usr/bin/env python
"""J6B -- scan the MPtraj candidate pool with a foundation committee.

Pre-registered in `protocols/j6b_foundation_acquisition.yaml`.

Two things make this different from every other experiment here, and both are
handled rather than assumed away:

1. **Ragged structures.** The pool spans 1-444 atoms, so the [S, A, 3, M] cache
   schema does not exist. Each structure is reduced on the fly to its r x r
   centred head-space Gram

       A_s = Qc^T ( sum_{d in s} F_d F_d^T ) Qc          (7 x 7, any atom count)

   which is sufficient for every global-statistic estimator:
       exact global   = tr(A_s) / (M-1)
       leading-only   = tr(Q_r0^T A_s Q_r0) / (M-1)
   Max-component is *not* a function of A_s -- it needs per-coordinate values --
   so it is accumulated separately as a scatter-max. `--self-test` asserts the
   identity against the direct computation on a fixed-N cache; it runs first and
   the scan refuses to start if it fails, because a wrong compression here would
   produce entirely plausible numbers.

2. **Contamination.** The scoring model is the one for which the pool is fully
   held out. See the protocol: the pool is the complement of the QBC selection,
   so the QBC model never saw any of it, while the `random` and
   `max_mean_force` models were trained on 7,498 and 6,236 pool structures.

Nothing here uses `utils/layout.py`'s untested variable-N path -- per-structure
reductions are done directly against MACE's own batch index.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch.adapters.mace_data import make_loader  # noqa: E402
from forcesketch.adapters.mace_mhc import MaceMHCAdapter  # noqa: E402
from forcesketch.exact.centered_basis import helmert_basis  # noqa: E402
from forcesketch.utils.reproducibility import pin_numerics  # noqa: E402

FOUNDATION = pathlib.Path("/n/holylabs/hekstra_lab/Everyone/ianpoon/forcesketch-data/"
                          "models/zenodo/foundation")
POOL = FOUNDATION / "qbc/mp_traj_qbc_79th_not_selected.xyz"


def self_test() -> None:
    """Assert the A_s compression reproduces the direct global statistic."""
    cache = FROZEN / "results/processed/head_forces_disjoint_test_1200K.pt"
    if not cache.exists():
        raise SystemExit(f"self-test needs {cache}")
    F = torch.load(cache, map_location="cpu", weights_only=False)["F"].double()[:64]
    S, A, _, M = F.shape
    Qc = helmert_basis(M, dtype=torch.float64)
    direct = F.var(dim=-1, unbiased=True).sum(dim=(1, 2))              # [S]
    X = F.flatten(1, 2)                                                # [S, D, M]
    G = torch.einsum("sdm,sdn->smn", X, X)
    A_s = torch.einsum("mr,smn,nq->srq", Qc, G, Qc)
    via_gram = A_s.diagonal(dim1=-2, dim2=-1).sum(-1) / (M - 1)
    rel = float((via_gram - direct).abs().max() / direct.abs().max())
    if not rel < 1e-12:
        raise SystemExit(f"A_s compression is WRONG: rel err {rel:.3e}")

    # and the leading-only identity, against an explicit projection
    r0 = 4
    Q_r0 = torch.linalg.qr(torch.randn(M - 1, r0, generator=
                                       torch.Generator().manual_seed(0),
                                       dtype=torch.float64))[0]
    a = torch.einsum("sdm,mr->sdr", X, Qc)                             # [S, D, r]
    lo_direct = torch.einsum("sdr,rk->sdk", a, Q_r0).pow(2).sum((1, 2)) / (M - 1)
    lo_gram = torch.einsum("rk,srq,qk->s", Q_r0, A_s, Q_r0) / (M - 1)
    rel2 = float((lo_gram - lo_direct).abs().max() / lo_direct.abs().max())
    if not rel2 < 1e-12:
        raise SystemExit(f"leading-only identity is WRONG: rel err {rel2:.3e}")
    print(f"self-test OK: global {rel:.2e}, leading-only {rel2:.2e} (n=64 structures)")


def read_frames(path: pathlib.Path, keep: set[int] | None):
    """Stream extxyz, materialising only the frames we keep.

    ASE parses this file's `forces` column into a SinglePointCalculator rather
    than into `atoms.arrays`, while MACE's `config_from_atoms` reads
    `arrays["REF_forces"]` and silently substitutes ZEROS when it is absent. The
    two conventions do not meet, and the failure is invisible: reference error
    then equals the predicted force exactly, so `e_max` comes back identical to
    the free `force_norm` signal and every downstream AUROC is a comparison of a
    signal against itself. The first smoke run hit precisely this. Copying the
    calculator's forces into the key MACE actually reads is the fix; the guard in
    `scan` is what makes a recurrence loud.
    """
    import ase.io
    out, idx = [], []
    for i, atoms in enumerate(ase.io.iread(str(path), format="extxyz")):
        if keep is None or i in keep:
            atoms.arrays["REF_forces"] = atoms.get_forces()
            out.append(atoms)
            idx.append(i)
            if keep is not None and len(out) == len(keep):
                break
    return out, idx


def iter_chunks(path: pathlib.Path, keep: set[int] | None, chunk: int):
    """Yield (frames, indices) in bounded batches.

    The pool is 547 MB / 136,923 structures; materialising it all before the
    first forward pass wastes both memory and the GPU's time. Streaming also
    means a job that dies partway has still printed real progress.
    """
    import ase.io
    buf, idxbuf = [], []
    for i, atoms in enumerate(ase.io.iread(str(path), format="extxyz")):
        if keep is not None and i not in keep:
            continue
        atoms.arrays["REF_forces"] = atoms.get_forces()
        buf.append(atoms)
        idxbuf.append(i)
        if len(buf) == chunk:
            yield buf, idxbuf
            buf, idxbuf = [], []
    if buf:
        yield buf, idxbuf


def scan(adapter, frames, *, batch_size: int) -> dict:
    M = adapter.num_heads
    Qc = helmert_basis(M, dtype=torch.float64).to(adapter.device)
    acc = {k: [] for k in ("A_s", "exact_maxcomp", "exact_maxatom", "exact_global",
                           "force_norm", "energy_std", "natoms",
                           "e_max", "e_rmse", "e_maxcomp", "e_q95")}
    n_periodic_edges = 0
    for batch in make_loader(frames, adapter.model, batch_size=batch_size):
        bd = batch.to_dict()
        b = adapter.prepare(bd)
        B = adapter.num_structures(b)
        bidx = adapter.batch_index(b)                                  # [N] atom -> structure
        f = adapter.exact_head_forces(b).double()                      # [M, N, 3]
        E = adapter.energies(b).detach().double()                      # [B, M]
        if "shifts" in b:
            n_periodic_edges += int((b["shifts"].abs().sum(-1) > 0).sum())

        N = f.shape[1]
        x = f.permute(1, 2, 0).reshape(N * 3, M)                       # [D, M]
        didx = bidx.repeat_interleave(3)                               # [D]
        G = torch.zeros(B, M, M, dtype=torch.float64, device=f.device)
        G.index_add_(0, didx, x.unsqueeze(2) * x.unsqueeze(1))
        A_s = torch.einsum("mr,smn,nq->srq", Qc, G, Qc)                # [B, r, r]

        v = f.var(dim=0, unbiased=True)                                # [N, 3]
        vflat = v.reshape(-1)
        g = torch.zeros(B, dtype=torch.float64, device=f.device).index_add_(0, didx, vflat)
        mc = torch.full((B,), -1.0, dtype=torch.float64, device=f.device) \
            .scatter_reduce(0, didx, vflat.sqrt(), reduce="amax")
        ma = torch.full((B,), -1.0, dtype=torch.float64, device=f.device) \
            .scatter_reduce(0, bidx, v.sqrt().mean(-1), reduce="amax")

        fbar = f.mean(0)                                               # [N, 3] mean force
        fref = b["forces"].double() if "forces" in b else None
        if fref is None or not torch.any(fref != 0):
            raise SystemExit(
                "reference forces are absent or identically zero. MACE substitutes "
                "zeros for a missing REF_forces key, which would make e_max equal "
                "the predicted force norm and every AUROC self-referential.")
        fn = torch.full((B,), -1.0, dtype=torch.float64, device=f.device) \
            .scatter_reduce(0, bidx, fbar.norm(dim=-1), reduce="amax")

        acc["A_s"].append(A_s.cpu())
        acc["exact_global"].append(g.cpu())
        acc["exact_maxcomp"].append(mc.cpu())
        acc["exact_maxatom"].append(ma.cpu())
        acc["force_norm"].append(fn.cpu())
        acc["energy_std"].append(E.std(dim=-1, unbiased=True).cpu())
        acc["natoms"].append(torch.bincount(bidx, minlength=B).cpu())

        if fref is None:
            raise SystemExit("batch carries no reference forces; check the forces key")
        d = fbar - fref                                                # [N, 3]
        an = d.norm(dim=-1)                                            # [N]
        acc["e_max"].append(torch.full((B,), -1.0, dtype=torch.float64, device=f.device)
                            .scatter_reduce(0, bidx, an, reduce="amax").cpu())
        acc["e_maxcomp"].append(torch.full((B,), -1.0, dtype=torch.float64, device=f.device)
                                .scatter_reduce(0, didx, d.reshape(-1).abs(),
                                                reduce="amax").cpu())
        sq = torch.zeros(B, dtype=torch.float64, device=f.device) \
            .index_add_(0, didx, d.reshape(-1).pow(2))
        cnt = torch.zeros(B, dtype=torch.float64, device=f.device) \
            .index_add_(0, didx, torch.ones_like(d.reshape(-1)))
        acc["e_rmse"].append((sq / cnt).sqrt().cpu())
        # q95 needs the whole per-structure vector; do it per structure (cheap)
        q = torch.stack([d.reshape(-1)[didx == s].abs().quantile(0.95) for s in range(B)])
        acc["e_q95"].append(q.cpu())

    out = {k: torch.cat(v) for k, v in acc.items()}
    out["n_periodic_edges"] = n_periodic_edges
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(FOUNDATION /
                    "qbc/dataset_8000/MACE_qbc_max_selection_stagetwo.model"))
    ap.add_argument("--n-sample", type=int, default=136923)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--out", default="results/records/j6b_foundation_scan.pt")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--first-n", type=int, default=None,
                    help="restrict sampling to the first N pool structures (smoke tests only)")
    a = ap.parse_args()

    pin_numerics()
    self_test()                                   # always, not only with the flag
    if a.self_test:
        return 0

    proto = json.loads((ROOT / "results/records/j6b_protocol_registration.json").read_text())
    n_pool = a.first_n or 136923
    rng = np.random.default_rng(a.seed)
    keep = set(rng.choice(n_pool, size=min(a.n_sample, n_pool), replace=False).tolist())
    print(f"sampling {len(keep)} of {n_pool} pool structures (seed {a.seed})")

    dtype = getattr(torch, a.dtype)
    ad = MaceMHCAdapter.from_checkpoint(a.model, device=a.device, dtype=dtype)
    print(f"model {pathlib.Path(a.model).name}  heads={ad.num_heads}  dtype={a.dtype}",
          flush=True)

    t0 = time.time()
    parts, idx, edges = [], [], 0
    for frames, chunk_idx in iter_chunks(POOL, keep, a.chunk):
        r = scan(ad, frames, batch_size=a.batch_size)
        edges += r.pop("n_periodic_edges")
        parts.append(r)
        idx.extend(chunk_idx)
        el = time.time() - t0
        print(f"  {len(idx):7d} scanned  {len(idx) / el:6.1f}/s  "
              f"eta {(len(keep) - len(idx)) / max(len(idx) / el, 1e-9) / 60:6.1f} min",
              flush=True)
        # Partial save every chunk: a multi-hour scan that dies at the end with
        # nothing on disk is a multi-hour scan wasted.
        if len(parts) % 5 == 0:
            partial = {k: torch.cat([q[k] for q in parts]) for k in parts[0]}
            partial["pool_index"] = torch.tensor(idx)
            partial["partial"] = True
            torch.save(partial, str(ROOT / a.out) + ".partial")
    res = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    res["n_periodic_edges"] = edges
    dt = time.time() - t0
    res["pool_index"] = torch.tensor(idx)
    res["model"] = a.model
    res["model_sha256"] = hashlib.sha256(pathlib.Path(a.model).read_bytes()).hexdigest()
    res["protocol_sha256"] = proto["protocol_sha256"]
    res["seed"] = a.seed
    res["seconds"] = dt
    out = ROOT / a.out
    torch.save(res, out)
    print(f"scanned {len(idx)} structures in {dt:.0f}s "
          f"({len(idx) / dt:.1f}/s, {res['n_periodic_edges']} periodic edges)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
