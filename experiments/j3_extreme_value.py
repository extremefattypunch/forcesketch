#!/usr/bin/env python
"""J3-lite -- can the negative result be PREDICTED rather than merely measured?

The workshop paper attributes the top-5% recall failure to extreme-value bias in
`max_d sigma_hat_d`. That cannot be the whole story: recall is a rank statistic,
and a common multiplicative bias leaves every rank unchanged. This script
decomposes the mechanism into its three parts and tests which one actually does
the damage.

Step 1 -- verify the distributional law. For a Haar sketch of K directions drawn
inside the r-dimensional centred head space,

    v_hat_d / v_d  =  (r/K) * ||P_S a_d||^2 / ||a_d||^2,   ||P_S a||^2/||a||^2 ~ Beta(K/2, (r-K)/2)

where P_S projects onto the random K-subspace and a_d = Q^T F^T e_d. Checked
against the shipped `haar_seeds` generator, not a re-implementation.

Step 2 -- predict recall from the law alone. Take the EXACT per-structure
coordinate variances from the caches, inject noise drawn from the Beta law, and
recompute top-5% recall. Two variants isolate the coupling:

    independent : every coordinate gets its own Beta draw
    shared      : one random subspace per structure, all D coordinates share it
                  (what the real estimator does)

If `shared` reproduces the measured recall, the mechanism is fully explained by
estimator variance plus the extreme-value reduction -- no appeal to anything the
model does. If `independent` also reproduces it, coordinate coupling is
irrelevant and the story is purely marginal.

Step 3 -- the budget arithmetic. A Chernoff bound on the Gaussian sketch gives
K >= 8 log(2D/delta) / eps^2 for uniform componentwise accuracy. Evaluated at
these D, it dwarfs the available budget K <= r = M-1 by orders of magnitude,
which turns "we measured low recall" into "uniform accuracy is unreachable at any
budget this architecture affords".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch.exact.centered_basis import helmert_basis  # noqa: E402
from forcesketch.sketches.registry import make_sketch_seeds  # noqa: E402
from forcesketch_journal.evaluation.tail_metrics import recall_at, top_p_mask  # noqa: E402


def verify_beta_law(r: int, K: int, n_draws: int, seed: int = 0) -> dict:
    """Empirical vs analytic moments of ||P_S a||^2/||a||^2 for a Haar K-frame."""
    M = r + 1
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n_draws, M, generator=g, dtype=torch.float64)
    Qc = helmert_basis(M, dtype=torch.float64)
    a = (Qc @ (Qc.T @ a.T)).T                                  # project into centred space

    b = make_sketch_seeds("haar", M=M, K=K, batch_size=n_draws, seed=seed, dtype=torch.float64)
    proj = torch.einsum("sm,ksm->ks", a, b.seeds)              # [K, n]
    frac = (proj ** 2).sum(0) / (a ** 2).sum(-1)

    alpha, beta = K / 2, (r - K) / 2
    return {
        "r": r, "K": K, "n_draws": n_draws,
        "empirical_mean": float(frac.mean()), "analytic_mean": alpha / (alpha + beta),
        "empirical_var": float(frac.var()),
        "analytic_var": alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)),
    }


def beta_noise(v: torch.Tensor, r: int, K: int, *, mode: str, gen: torch.Generator,
               F: torch.Tensor | None = None) -> torch.Tensor:
    """v [S, D] exact coordinate variances -> v_hat under the Haar law.

    Three modes, and the distinction between the last two is the point:

      "indep"       every coordinate gets its own Beta(K/2,(r-K)/2) draw.
      "shared_iso"  one Haar K-frame per structure applied to D coordinate
                    directions drawn INDEPENDENTLY and ISOTROPICALLY in head space.
      "shared_real" one Haar K-frame per structure applied to the ACTUAL
                    per-coordinate head-space directions a_d = Q^T F(x)^T e_d,
                    which is what the deployed estimator does.

    **Correction.** An earlier version had only "indep" and "shared_iso" and
    concluded from their agreement that coordinate coupling is irrelevant. That
    conclusion was vacuous: for a *fixed* subspace, independent isotropic
    directions give independent Beta draws, so the two modes are identical by
    construction, not as an empirical finding. Measured across-coordinate
    correlation was 0.0126 for "shared_iso" against 0.0121 for "indep" -- i.e.
    zero coupling either way. Only "shared_real" can test coupling, because only
    the real a_d are correlated with one another.
    """
    S, D = v.shape
    if mode == "indep":
        alpha, beta = K / 2, (r - K) / 2
        ga = torch._standard_gamma(torch.full((S, D), alpha, dtype=torch.float64), generator=gen)
        gb = torch._standard_gamma(torch.full((S, D), beta, dtype=torch.float64), generator=gen)
        frac = ga / (ga + gb)
    else:
        M = r + 1
        O = torch.linalg.qr(torch.randn(S, r, K, generator=gen, dtype=torch.float64))[0]
        if mode == "shared_iso":
            A = torch.randn(S, D, r, generator=gen, dtype=torch.float64)
        elif mode == "shared_real":
            if F is None:
                raise ValueError("shared_real needs F [S, A, 3, M]")
            Qc = helmert_basis(M, dtype=torch.float64)                 # [M, r]
            A = torch.einsum("sadm,mr->sadr", F.double(), Qc).flatten(1, 2)   # [S, D, r]
        else:
            raise ValueError(mode)
        A = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-300)
        frac = torch.einsum("sdr,srk->sdk", A, O).pow(2).sum(-1)
    return v * (r / K) * frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=10, help="noise replicates per structure")
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--out", default="results/records/j3_extreme_value.json")
    ap.add_argument("--cache-dirs", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    out = {"beta_law": [], "recall_prediction": [], "budget": []}

    # ---- step 1: the law -------------------------------------------------
    print("Step 1 -- Haar projection fraction is Beta(K/2, (r-K)/2)?  (r=7, 200k draws)\n")
    print(f"{'K':>3s}{'emp mean':>11s}{'Beta mean':>11s}{'emp var':>11s}{'Beta var':>11s}")
    print("-" * 47)
    for K in range(1, 7):
        d = verify_beta_law(7, K, 200_000)
        out["beta_law"].append(d)
        print(f"{K:3d}{d['empirical_mean']:11.5f}{d['analytic_mean']:11.5f}"
              f"{d['empirical_var']:11.5f}{d['analytic_var']:11.5f}")

    # ---- step 2: predict the measured recall -----------------------------
    print("\n\nStep 2 -- top-5% max-component recall: measured vs predicted from the law\n")
    print(f"{'system':26s}{'D':>5s}{'K':>3s}{'measured':>10s}{'sh_real':>9s}{'sh_iso':>9s}{'indep':>9s}")
    print("-" * 62)
    # One generator consumed across every (system, K, mode, replicate) makes the
    # output depend on HOW MANY systems were in the run and in WHAT ORDER -- so the
    # record could not be regenerated independently, and the offline reproduction
    # tier caught exactly that (MAE 0.017 committed vs 0.011 on a rerun over the
    # same six systems). The noise draw is now keyed to the item it belongs to, so
    # any subset reproduces bit-identically.
    def _gen(*key) -> torch.Generator:
        h = hashlib.blake2b("|".join(str(k) for k in (args.seed, *key)).encode(),
                            digest_size=8).digest()
        return torch.Generator().manual_seed(int.from_bytes(h, "big") % (2 ** 63))

    dirs=[pathlib.Path(d) for d in (args.cache_dirs or [FROZEN/"results/processed"])]
    for cache in sorted({p for d in dirs for p in d.glob("head_forces_*.pt")}):
        tag = cache.stem.replace("head_forces_", "")
        d = torch.load(cache, weights_only=True, map_location="cpu")
        F = d["F"].double()
        M = int(d["M"])
        r = M - 1
        v = F.var(dim=-1, unbiased=True).flatten(1)                 # [S, D]
        D = v.shape[1]
        s_exact = v.sqrt().max(dim=1).values
        pos = top_p_mask(s_exact, args.p)

        for K in (3, 4, 6):
            # measured: run the real estimator
            meas = []
            for rep in range(args.n_rep):
                b = make_sketch_seeds("haar", M=M, K=K, batch_size=F.shape[0],
                                      seed=1000003 + 37 * rep, dtype=torch.float64)
                G = torch.einsum("sadm,ksm->ksad", F, b.seeds)
                vh = (b.variance_scale * (G ** 2).sum(0)).flatten(1)
                sh = vh.clamp_min(0).sqrt().max(dim=1).values
                meas.append(float(recall_at(sh, pos, args.p)))
            pred = {}
            for name in ("shared_real", "shared_iso", "indep"):
                vals = []
                for rep in range(args.n_rep):
                    vh = beta_noise(v, r, K, mode=name,
                                    gen=_gen(tag, K, name, rep), F=F)
                    vals.append(float(recall_at(vh.sqrt().max(dim=1).values, pos, args.p)))
                pred[name] = sum(vals) / len(vals)
            row = {"system": tag, "D": D, "K": K,
                   "measured": sum(meas) / len(meas), **pred}
            out["recall_prediction"].append(row)
            print(f"{tag:26s}{D:5d}{K:3d}{row['measured']:10.3f}"
                  f"{row['shared_real']:9.3f}{row['shared_iso']:9.3f}{row['indep']:9.3f}")

    # ---- step 3: the budget arithmetic -----------------------------------
    print("\n\nStep 3 -- K required for uniform componentwise accuracy (Gaussian Chernoff)\n")
    print(f"{'D (=3N)':>9s}{'eps=0.5':>10s}{'eps=0.2':>10s}{'eps=0.1':>10s}   available K")
    print("-" * 55)
    for N, label in ((9, "ethanol"), (21, "aspirin"), (24, "azobenzene"),
                     (27, "3BPA"), (192, "water (J5)")):
        D = 3 * N
        ks = [math.ceil(8 * math.log(2 * D / 0.05) / e ** 2) for e in (0.5, 0.2, 0.1)]
        out["budget"].append({"system": label, "N": N, "D": D,
                              "K_eps50": ks[0], "K_eps20": ks[1], "K_eps10": ks[2],
                              "K_available": 7})
        print(f"{D:9d}{ks[0]:10d}{ks[1]:10d}{ks[2]:10d}   {'7 (= M-1)':>12s}   {label}")

    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
