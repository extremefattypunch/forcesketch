#!/usr/bin/env python
"""Does Factor A track distribution shift? The controlled test.

The hypothesis, raised by the water MD-vs-PIMD result: exact committee force
uncertainty predicts reference-force error well precisely when the evaluation
distribution is far from training, and poorly in-distribution. Water could not
settle it, because MD and PIMD differ in physics as well as in shift, and rMD17
sat above water MD despite both being in-distribution.

The 3BPA ladder isolates the variable. Same committees (trained at 300 K), same
molecule, same 27 atoms, same D = 81 — only the evaluation temperature changes.
If the hypothesis holds, Factor A must rise monotonically with temperature, and it
must do so for all three committee constructions.

This is a directional prediction registered before the ladder caches existed
(see results/J5_WATER.md section 3, which states it as a hypothesis and names
this experiment as its test).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from forcesketch_journal.evaluation.autocorr import block_length, integrated_autocorr_time  # noqa: E402
from forcesketch_journal.evaluation.bootstrap_block import paired_bootstrap, paired_difference  # noqa: E402
from forcesketch_journal.evaluation.tail_metrics import auroc, top_p_mask  # noqa: E402

EV_TO_MEV = 1000.0


def metric(sig, err):
    return auroc(sig, top_p_mask(err, 0.05))


def signals(F, E):
    v = F.var(dim=-1, unbiased=True)
    out = {"exact_global": v.sum(dim=(1, 2)),
           "exact_maxcomp": v.sqrt().flatten(1).max(dim=1).values,
           "force_norm": F.mean(dim=-1).norm(dim=-1).max(dim=1).values}
    if E is not None:
        out["energy_std"] = E.std(dim=-1, unbiased=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    rows = []
    for variant in ("disjoint", "overlapping", "same"):
        for temp in ("300K", "600K", "1200K"):
            p = ROOT / f"results/processed/head_forces_3bpa-{variant}_test_{temp}.pt"
            if not p.exists():
                print(f"missing {p.name}"); continue
            d = torch.load(p, weights_only=True, map_location="cpu")
            F, fref = d["F"].double(), d["f_ref"].double()
            E = d["E"].double() if "E" in d else None
            delta = F.mean(-1) - fref
            e_max = delta.norm(dim=-1).max(dim=1).values
            bl = block_length(integrated_autocorr_time(e_max.numpy())["tau_int"])
            sig = signals(F, E)
            r = {"variant": variant, "temperature_K": int(temp.rstrip("K")),
                 "n": int(F.shape[0]), "block_len": bl,
                 "force_rmse_mev_A": float(delta.pow(2).mean().sqrt()) * EV_TO_MEV,
                 "median_disagreement_mev_A":
                     float(F.std(dim=-1, unbiased=True).mean(dim=-1).median()) * EV_TO_MEV}
            r["overconfidence"] = r["force_rmse_mev_A"] / r["median_disagreement_mev_A"]
            for name, s in sig.items():
                b = paired_bootstrap(metric, s, e_max, n_boot=args.n_boot,
                                     scheme="block", block_len=bl)
                r[f"auroc_{name}"] = b["point"]
                r[f"auroc_{name}_ci"] = [b["ci_lo"], b["ci_hi"]]
            if "energy_std" in sig:
                g = paired_difference(metric, sig["exact_global"], sig["force_norm"],
                                      e_max, n_boot=args.n_boot, scheme="block", block_len=bl)
                r["gain_over_force_norm"] = g
            rows.append(r)

    print("3BPA TEMPERATURE LADDER — committees trained at 300 K\n")
    print(f"{'variant':13s}{'T (K)':>7s}{'RMSE':>9s}{'disagr':>8s}{'overconf':>10s}"
          f"{'AUROC global':>14s}{'95% CI':>18s}{'force_norm':>12s}")
    print("-" * 92)
    for r in rows:
        ci = r["auroc_exact_global_ci"]
        print(f"{r['variant']:13s}{r['temperature_K']:7d}{r['force_rmse_mev_A']:9.1f}"
              f"{r['median_disagreement_mev_A']:8.1f}{r['overconfidence']:9.1f}x"
              f"{r['auroc_exact_global']:14.3f}  [{ci[0]:.3f},{ci[1]:.3f}]"
              f"{r['auroc_force_norm']:12.3f}")

    print("\n\nIS THE PREDICTION CONFIRMED?  Factor A must rise with temperature.\n")
    ok = 0
    for variant in ("disjoint", "overlapping", "same"):
        v = [r for r in rows if r["variant"] == variant]
        if len(v) < 2:
            continue
        a = [r["auroc_exact_global"] for r in sorted(v, key=lambda x: x["temperature_K"])]
        mono = all(a[i] <= a[i + 1] for i in range(len(a) - 1))
        ok += mono
        print(f"  {variant:13s} " + " -> ".join(f"{x:.3f}" for x in a)
              + f"   {'MONOTONE INCREASING' if mono else 'NOT monotone'}")
    print(f"\n{ok}/3 committee constructions show Factor A rising monotonically with shift.")

    out = ROOT / "results/records/j5_shift_hypothesis.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
