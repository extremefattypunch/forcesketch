#!/usr/bin/env python
"""Do the reported bootstrap CIs cover split-draw variability? (They do not.)

An adversarial review claimed that `j9a`'s 95% intervals quantify only the
resampling of test structures, and that alternative *split draws* of the same
quantity land outside them. If so, every "significant" call in J2b/J9a is
understated, because the split assignment is itself a random choice (a seeded
permutation of contiguous blocks) that the reported interval conditions on.

The test is direct: hold everything fixed except the split seed, redraw the
design/cal/test assignment N times, recompute the leading-only-minus-control-
variate skip difference each time, and compare the spread of those point estimates
against the bootstrap CI that j9a reports for the single canonical seed.

Three outcomes:
  * split spread comfortably inside the CI  -> finding refuted, CIs are adequate
  * split spread comparable to the CI       -> CIs roughly right by luck
  * split spread much wider than the CI     -> finding CONFIRMED; the reported
                                               intervals understate uncertainty and
                                               significance calls must be redone
                                               against a split-inclusive interval
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FROZEN / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from forcesketch.sketches.control_variate import leading_head_directions  # noqa: E402
from forcesketch_journal.data.splits import contiguous_block_split  # noqa: E402
from forcesketch_journal.evaluation.autocorr import block_length, integrated_autocorr_time  # noqa: E402
from j2b_factor_b import acquisition, conformal_c, est_cv, est_exact, est_leading_only  # noqa: E402

CANONICAL_SEED = 20260901


def delta_for_split(F, M, split, *, score, alpha, target_p, lo_r0, cv_r0, cv_K, seeds):
    design = torch.as_tensor(split.design.copy())
    cal = torch.as_tensor(split.cal.copy())
    test = torch.as_tensor(split.test.copy())
    v_ex, sig_ex = est_exact(F)
    s_exact = acquisition(v_ex, sig_ex, score)
    tau = float(s_exact[design].quantile(1.0 - target_p))

    Q_lo = leading_head_directions(F[design], lo_r0)
    Q_cv = leading_head_directions(F[design], cv_r0)

    v_lo, _ = est_leading_only(F, M, Q_lo, lo_r0)
    s_lo = acquisition(v_lo, v_lo.clamp_min(0).sqrt(), score)
    skip_lo = float((conformal_c(s_exact[cal], s_lo[cal], alpha) * s_lo[test] < tau).double().mean())

    sk = []
    for sd in seeds:
        v_c, sg_c = est_cv(F, M, cv_K, sd, Q_cv, cv_r0)
        s_c = acquisition(v_c, sg_c, score)
        sk.append(float((conformal_c(s_exact[cal], s_c[cal], alpha) * s_c[test] < tau).double().mean()))
    return skip_lo - float(np.mean(sk))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--lanes", type=int, default=5)
    ap.add_argument("--score", default="global")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-p", type=float, default=0.05)
    ap.add_argument("--cv-r0", type=int, default=2)
    ap.add_argument("--systems", nargs="+", default=[
        "disjoint_test_1200K", "overlapping_test_1200K",
        "rmd17-disjoint_ethanol", "water-disjoint_md_T300K"])
    args = ap.parse_args()

    probe_seeds = yaml.safe_load((FROZEN / "configs/seeds.yaml").read_text())["sketch_seeds"]
    lo_r0 = cv_K = args.lanes - 1
    reported = {r["system"]: r for r in json.load(
        open(ROOT / f"results/records/j9a_leading_only_ci_{args.score}_L{args.lanes}.json"))}

    print(f"Split-draw variability vs the reported bootstrap CI "
          f"({args.n_splits} redraws, L={args.lanes}, {args.score})\n")
    print(f"{'system':28s}{'canonical':>11s}{'reported 95% CI':>20s}"
          f"{'split mean':>12s}{'split sd':>10s}{'split 2.5-97.5%':>20s}{'ratio':>7s}")
    print("-" * 108)
    rows = []
    for tag in args.systems:
        cache = next((d / f"head_forces_{tag}.pt" for d in
                      (FROZEN / "results/processed", ROOT / "results/processed")
                      if (d / f"head_forces_{tag}.pt").exists()), None)
        if cache is None:
            print(f"{tag:28s} cache not found"); continue
        d = torch.load(cache, weights_only=True, map_location="cpu")
        F, M = d["F"].double(), int(d["M"])
        v_ex, sig_ex = est_exact(F)
        s_exact = acquisition(v_ex, sig_ex, args.score)
        guard = block_length(integrated_autocorr_time(s_exact.numpy())["tau_int"])
        tau_int = integrated_autocorr_time(s_exact.numpy())["tau_int"]

        deltas = []
        for i in range(args.n_splits):
            sp = contiguous_block_split(F.shape[0], system=tag, seed=CANONICAL_SEED + i,
                                        tau_int=tau_int, guard=guard)
            deltas.append(delta_for_split(F, M, sp, score=args.score, alpha=args.alpha,
                                          target_p=args.target_p, lo_r0=lo_r0,
                                          cv_r0=args.cv_r0, cv_K=cv_K, seeds=probe_seeds))
        deltas = np.array(deltas)
        rep = reported.get(tag)
        ci_w = (rep["ci_hi"] - rep["ci_lo"]) if rep else float("nan")
        q = np.quantile(deltas, [0.025, 0.975])
        split_w = q[1] - q[0]
        outside = int(((deltas < rep["ci_lo"]) | (deltas > rep["ci_hi"])).sum()) if rep else -1
        rows.append({"system": tag, "canonical_delta": rep["delta"] if rep else None,
                     "reported_ci": [rep["ci_lo"], rep["ci_hi"]] if rep else None,
                     "split_deltas": deltas.tolist(), "split_mean": float(deltas.mean()),
                     "split_sd": float(deltas.std(ddof=1)),
                     "split_q025": float(q[0]), "split_q975": float(q[1]),
                     "width_ratio": float(split_w / ci_w) if ci_w == ci_w else None,
                     "n_outside_reported_ci": outside, "n_splits": args.n_splits})
        print(f"{tag:28s}{rep['delta']:+11.3f}"
              f"  [{rep['ci_lo']:+.3f},{rep['ci_hi']:+.3f}]"
              f"{deltas.mean():+12.3f}{deltas.std(ddof=1):10.3f}"
              f"  [{q[0]:+.3f},{q[1]:+.3f}]{split_w/ci_w:7.1f}x")

    print(f"\n{'system':28s}{'redraws outside the reported CI':>34s}")
    print("-" * 62)
    for r in rows:
        print(f"{r['system']:28s}{r['n_outside_reported_ci']:>20d} / {r['n_splits']}")
    out = ROOT / f"results/records/j9b_split_draw_variability_L{args.lanes}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
