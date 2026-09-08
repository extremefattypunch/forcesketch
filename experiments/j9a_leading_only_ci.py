#!/usr/bin/env python
"""J9.1 with intervals -- is the randomised residual sketch worth its lanes?

The sweep in j2b showed leading-only beating the control variate in 20/24
matched-budget cells. A count of wins is not a result until the per-cell margins
are resolved, so this recomputes the per-structure skip decisions and puts a
paired block-bootstrap interval on the difference.

Paired over test structures, because both estimators decide on the same
structures: a structure that is easy to clear is easy for both, and resampling
them independently would inflate the interval by throwing that away.

The control variate is averaged over its 10 frozen probe seeds *inside* each
bootstrap replicate, so the interval reflects structure sampling. Seed risk is
reported separately, as the seed-wise worst case, because a deployment draws one
sketch and not ten.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch.sketches.control_variate import leading_head_directions  # noqa: E402
from forcesketch_journal.evaluation.autocorr import (  # noqa: E402
    block_length, integrated_autocorr_time,
)
from forcesketch_journal.evaluation.bootstrap_block import moving_block_indices  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from j2b_factor_b import acquisition, conformal_c, est_cv, est_exact, est_leading_only  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="global", choices=["global", "maxcomp"])
    ap.add_argument("--lanes", type=int, default=5, help="total reverse lanes for both")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-p", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--cv-r0", type=int, default=2)
    ap.add_argument("--cache-dirs", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = yaml.safe_load((FROZEN / "configs/seeds.yaml").read_text())["sketch_seeds"]
    lo_r0 = args.lanes - 1          # leading-only: r0 exact directions + mean lane
    cv_K = args.lanes - 1           # control variate: K UQ lanes + mean lane

    rows = []
    print(f"J9.1 -- leading-only(r0={lo_r0}) vs control-variate(r0={args.cv_r0},K={cv_K}), "
          f"both {args.lanes} lanes, {args.score} statistic\n")
    print(f"{'system':26s}{'LO skip':>9s}{'CV skip':>9s}{'delta':>8s}{'95% CI':>18s}"
          f"{'sig':>6s}{'CV worst':>10s}")
    print("-" * 86)

    dirs=[pathlib.Path(d) for d in (args.cache_dirs or [FROZEN/"results/processed"])]
    for cache in sorted({p for d in dirs for p in d.glob("head_forces_*.pt")}):
        tag = cache.stem.replace("head_forces_", "")
        man = json.loads((ROOT / f"manifests/splits/{tag}__contiguous_block.json").read_text())
        d = torch.load(cache, weights_only=True, map_location="cpu")
        F = d["F"].double()
        M = int(d["M"])
        design = torch.tensor(man["roles"]["design"])
        cal = torch.tensor(man["roles"]["cal"])
        test = torch.tensor(man["roles"]["test"])

        v_ex, sig_ex = est_exact(F)
        s_exact = acquisition(v_ex, sig_ex, args.score)
        tau = float(s_exact[design].quantile(1.0 - args.target_p))

        Q_lo = leading_head_directions(F[design], lo_r0)
        Q_cv = leading_head_directions(F[design], args.cv_r0)

        v_lo, _ = est_leading_only(F, M, Q_lo, lo_r0)
        s_lo = acquisition(v_lo, v_lo.clamp_min(0).sqrt(), args.score)
        skip_lo = (conformal_c(s_exact[cal], s_lo[cal], args.alpha) * s_lo[test] < tau).double()

        skip_cv = []
        for sd in seeds:
            v_c, sg_c = est_cv(F, M, cv_K, sd, Q_cv, args.cv_r0)
            s_c = acquisition(v_c, sg_c, args.score)
            skip_cv.append((conformal_c(s_exact[cal], s_c[cal], args.alpha) * s_c[test] < tau).double())
        skip_cv = torch.stack(skip_cv)                       # [n_seeds, n_test]

        bl = block_length(integrated_autocorr_time(s_exact[test].numpy())["tau_int"])
        idx = moving_block_indices(test.numel(), args.n_boot, bl,
                                   torch.Generator().manual_seed(20260902))
        reps = skip_lo[idx].mean(-1) - skip_cv[:, idx].mean(-1).mean(0)
        lo_q, hi_q = torch.quantile(reps, torch.tensor([0.025, 0.975], dtype=reps.dtype))
        delta = float(skip_lo.mean() - skip_cv.mean())
        sig = bool(lo_q > 0 or hi_q < 0)
        rows.append({"system": tag, "score": args.score, "lanes": args.lanes,
                     "lo_r0": lo_r0, "cv_r0": args.cv_r0, "cv_K": cv_K,
                     "skip_lo": float(skip_lo.mean()), "skip_cv": float(skip_cv.mean()),
                     "delta": delta, "ci_lo": float(lo_q), "ci_hi": float(hi_q),
                     "significant": sig, "block_len": bl, "n_boot": args.n_boot,
                     "cv_skip_worst_seed": float(skip_cv.mean(-1).min())})
        print(f"{tag:26s}{float(skip_lo.mean()):9.3f}{float(skip_cv.mean()):9.3f}"
              f"{delta:+8.3f}  [{float(lo_q):+.3f},{float(hi_q):+.3f}]{str(sig):>6s}"
              f"{float(skip_cv.mean(-1).min()):10.3f}")

    # Explicit --out exists so a run over a different cache directory cannot
    # overwrite the shared-trunk record it is meant to be compared against.
    out = ROOT / (args.out or
                  f"results/records/j9a_leading_only_ci_{args.score}_L{args.lanes}.json")
    out.write_text(json.dumps(rows, indent=1))
    n_sig = sum(r["significant"] and r["delta"] > 0 for r in rows)
    print(f"\nleading-only significantly better on {n_sig}/{len(rows)} systems")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
