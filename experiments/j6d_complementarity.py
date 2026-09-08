#!/usr/bin/env python
"""J6D -- does committee UQ add anything ON TOP of the free signal?

**Exploratory. Not pre-registered, nothing scored.** J6B established that on the
MPtraj pool the free `max mean-force norm` outranks every committee statistic by a
wide margin. That makes one question unavoidable and it is not a question any
existing experiment answers: a practitioner already has `‖f̄‖` for nothing, so the
decision is not "committee or free signal" but "free signal, or free signal PLUS
eight backward passes". If the committee's information is a subset of the free
signal's, the method is dominated here; if it is complementary, it has a role.

Method. Fit a logistic regression on the DESIGN split to predict membership of the
top-5% e_max set, evaluate AUROC on the disjoint held-out split, and compare nested
feature sets. Features are logged because all of these signals span orders of
magnitude. The marginal value of the committee is the held-out AUROC difference
between the two-feature and one-feature models, with a paired bootstrap interval.

Fairness notes, since nested-model comparisons are easy to rig:
  * every model is fitted on design only and scored on held-out only;
  * the size feature is included as its own arm, so "the committee is just a
    proxy for atom count" is testable rather than assertable;
  * standardisation uses design-split statistics only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forcesketch_journal.evaluation.bootstrap_block import paired_difference  # noqa: E402
from forcesketch_journal.evaluation.tail_metrics import auroc, top_p_mask  # noqa: E402

M = 8
R = M - 1


def fit_logistic(X: torch.Tensor, y: torch.Tensor, *, steps: int = 400,
                 lr: float = 0.5, l2: float = 1e-4) -> torch.Tensor:
    """Plain L2-regularised logistic regression, LBFGS. Returns weights [d+1]."""
    n, d = X.shape
    Xb = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)
    w = torch.zeros(d + 1, dtype=X.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xb @ w
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) \
            + l2 * w[:-1].pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def fixed_n_contrast(error_p: float, n_boot: int) -> list[dict]:
    """The same question on the FIXED-N systems, using the project's own splits.

    The MPtraj answer alone would be over-read as "committee UQ is redundant".
    On molecules the free signal is much weaker, so the honest claim is a
    contrast, and a contrast has to be measured on both sides with one method.
    Design role fits, test role scores -- the manifests already define both.
    """
    FROZEN_P = ROOT.parents[0] / "forcesketch"
    dirs = [ROOT / "results/processed", ROOT / "results/processed/naive",
            FROZEN_P / "results/processed"]
    rows, seen = [], set()
    for cache in sorted({q for d in dirs for q in d.glob("head_forces_*.pt")}):
        tag = cache.stem.replace("head_forces_", "")
        man = ROOT / f"manifests/splits/{tag}__contiguous_block.json"
        if tag in seen or not man.exists():
            continue
        seen.add(tag)
        roles = json.loads(man.read_text())["roles"]
        des = torch.tensor(roles["design"]); tst = torch.tensor(roles["test"])
        c = torch.load(cache, map_location="cpu", weights_only=False)
        F = c["F"].double()
        v = F.var(dim=-1, unbiased=True)
        lg = lambda t: t.clamp_min(1e-30).log()  # noqa: E731
        committee = lg(v.sum(dim=(1, 2)))
        fbar = F.mean(-1)
        force_norm = lg(fbar.norm(dim=-1).max(dim=1).values)
        e_max = (fbar - c["f_ref"].double()).norm(dim=-1).max(dim=1).values

        y_d = top_p_mask(e_max[des], error_p).double()
        pos_t = top_p_mask(e_max[tst], error_p)
        if y_d.sum() < 5 or pos_t.sum() < 5:
            continue
        out = {}
        for name, cols in (("force_norm", [force_norm]),
                           ("committee", [committee]),
                           ("both", [force_norm, committee])):
            Xd = torch.stack([f[des] for f in cols], dim=1)
            mu, sd = Xd.mean(0), Xd.std(0).clamp_min(1e-12)
            w = fit_logistic((Xd - mu) / sd, y_d)
            Xt = (torch.stack([f[tst] for f in cols], dim=1) - mu) / sd
            out[name] = torch.cat([Xt, torch.ones(len(tst), 1, dtype=Xt.dtype)], 1) @ w
        ci = paired_difference(auroc, out["both"], out["force_norm"], pos_t,
                               n_boot=n_boot, scheme="iid")
        rows.append({"system": tag, "n_test": len(tst),
                     "auroc_force_norm": float(auroc(out["force_norm"], pos_t)),
                     "auroc_committee": float(auroc(out["committee"], pos_t)),
                     "auroc_both": float(auroc(out["both"], pos_t)),
                     "marginal": ci["delta"], "ci_lo": ci["ci_lo"],
                     "ci_hi": ci["ci_hi"], "significant": ci["significant"]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="results/records/j6b_foundation_scan.pt")
    ap.add_argument("--out", default="results/records/j6d_complementarity.json")
    ap.add_argument("--error-p", type=float, default=0.05)
    ap.add_argument("--design-frac", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=20260816)
    ap.add_argument("--alpha", type=float, default=0.75, help="from J6C design selection")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    d = torch.load(ROOT / a.scan, map_location="cpu", weights_only=False)
    n = d["exact_global"].shape[0]
    natoms = d["natoms"].double()
    e_max = d["e_max"].double()

    gen = torch.Generator().manual_seed(a.split_seed)
    perm = torch.randperm(n, generator=gen)
    n_design = int(a.design_frac * n)
    design, test = perm[:n_design], perm[n_design:]

    lg = lambda t: t.clamp_min(1e-30).log()  # noqa: E731
    feats = {
        "force_norm": lg(d["force_norm"].double()),
        "committee": lg(d["exact_global"].double() / (3 * natoms).pow(a.alpha)),
        "maxatom": lg(d["exact_maxatom"].double()),
        "natoms": lg(natoms),
        "energy_std": lg(d["energy_std"].double()),
    }
    y_all = top_p_mask(e_max, a.error_p).double()

    # positives defined WITHIN each split, so the label definition never crosses it
    y_d = top_p_mask(e_max[design], a.error_p).double()
    pos_t = top_p_mask(e_max[test], a.error_p)

    ARMS = {
        "force_norm only":            ["force_norm"],
        "committee only":             ["committee"],
        "natoms only":                ["natoms"],
        "force_norm + natoms":        ["force_norm", "natoms"],
        "force_norm + committee":     ["force_norm", "committee"],
        "force_norm + committee + natoms": ["force_norm", "committee", "natoms"],
        "force_norm + maxatom":       ["force_norm", "maxatom"],
        "all five":                   list(feats),
    }

    scores, table = {}, {}
    print(f"design {len(design)}  held-out {len(test)}  alpha={a.alpha}  "
          f"top-{a.error_p:.0%} e_max\n")
    print(f"{'model':36s}{'held-out AUROC':>16s}")
    for name, cols in ARMS.items():
        Xd = torch.stack([feats[c][design] for c in cols], dim=1)
        mu, sd = Xd.mean(0), Xd.std(0).clamp_min(1e-12)      # design statistics only
        w = fit_logistic((Xd - mu) / sd, y_d)
        Xt = (torch.stack([feats[c][test] for c in cols], dim=1) - mu) / sd
        s = torch.cat([Xt, torch.ones(len(test), 1, dtype=Xt.dtype)], 1) @ w
        scores[name] = s
        table[name] = float(auroc(s, pos_t))
        print(f"{name:36s}{table[name]:16.4f}")

    base = "force_norm only"
    print(f"\nmarginal value over `{base}` (paired bootstrap, held-out):")
    deltas = {}
    for name in ARMS:
        if name == base:
            continue
        ci = paired_difference(auroc, scores[name], scores[base], pos_t,
                               n_boot=a.n_boot, scheme="iid")
        deltas[name] = ci
        print(f"  {name:36s}{ci['delta']:+.4f}  "
              f"[{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}]  sig={ci['significant']}")

    fixed = fixed_n_contrast(a.error_p, a.n_boot)
    print(f"\nFIXED-N CONTRAST (project splits: design fits, test scores)")
    print(f"  {'system':30s}{'free':>8s}{'cmte':>8s}{'both':>8s}{'marginal':>10s}{'sig':>6s}")
    for r in sorted(fixed, key=lambda x: -x["marginal"]):
        print(f"  {r['system']:30s}{r['auroc_force_norm']:8.3f}{r['auroc_committee']:8.3f}"
              f"{r['auroc_both']:8.3f}{r['marginal']:+10.4f}{str(r['significant']):>6s}")
    if fixed:
        sig = sum(r["significant"] and r["marginal"] > 0 for r in fixed)
        print(f"  committee adds significantly on {sig}/{len(fixed)} fixed-N systems; "
              f"on MPtraj: {deltas['force_norm + committee']['delta']:+.4f} "
              f"(sig={deltas['force_norm + committee']['significant']})")

    out = ROOT / a.out
    out.write_text(json.dumps({
        "fixed_n_contrast": fixed,
        "alpha": a.alpha, "error_p": a.error_p, "n_design": len(design),
        "n_test": len(test), "held_out_auroc": table,
        "marginal_over_force_norm": deltas,
        "note": "exploratory; not pre-registered, nothing scored",
    }, indent=1, default=float))
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
