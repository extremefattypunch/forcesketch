#!/usr/bin/env python
"""J6C -- can size normalisation repair the global statistic at variable N?

Governed by `protocols/j6c_size_normalisation.yaml`, which fixes the estimator
family, the alpha grid, the selection procedure and the success criteria. Read
the header of that file first: this is a post-hoc REPAIR of a known failure, not
a blind prediction, and it is labelled as such throughout.

Two halves:

  * the variable-N half, on the 136,923-structure MPtraj scan, where alpha* is
    chosen on a design split and every reported number comes from the disjoint
    remainder;
  * the fixed-N side, over the molecular/water caches, where the normalisation
    must change absolutely nothing. See `fixed_n_invariance` -- an audit showed
    the original form of that check was vacuous, and it now separates the
    derivation from the test that can actually fail.
"""

from __future__ import annotations

import argparse
import hashlib
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

from forcesketch_journal.evaluation.bootstrap_block import paired_difference  # noqa: E402
from forcesketch_journal.evaluation.tail_metrics import auroc, top_p_mask  # noqa: E402

M = 8
R = M - 1


def normalised(stat: torch.Tensor, natoms: torch.Tensor, alpha: float) -> torch.Tensor:
    return stat / (3.0 * natoms.double()).pow(alpha)


def fixed_n_invariance(alphas, p: float) -> dict:
    """Checks on the normalisation, with the honest label on each.

    **An adversarial audit found the original version of this function vacuous
    and it was right.** It synthesised its own constant atom-count vector, so
    `normalised(g, na, alpha)` was `g` divided by one positive scalar, and AUROC
    is invariant under any increasing map. Three deliberately broken
    implementations -- an operator-precedence bug, N taken from the wrong
    structure, and NO NORMALISATION AT ALL -- every one returned
    max|dAUROC| = 0.000e+00 and "passed". The check protected nothing.

    What is reported now, separated by what it is worth:

      invariance_fixed_n   still computed, but labelled a DERIVATION. At fixed N,
                           3N is one constant per system; it cannot reorder
                           anything and the measurement cannot fail. It is kept
                           because the claim is load-bearing for adopting the
                           normalisation globally, not because it is evidence.
      reference_max_relerr a REAL test: on the variable-N atom counts, the
                           shipped `normalised` must agree elementwise with an
                           independently written reference. This is what catches
                           a wrong implementation.
      mutation_caught      proof the real test has teeth: the three broken
                           implementations above must all FAIL it.
    """
    dirs = [ROOT / "results/processed", ROOT / "results/processed/naive",
            FROZEN / "results/processed"]
    worst, n_nan, sysids = 0.0, 0, {}
    for cache in sorted({q for d in dirs for q in d.glob("head_forces_*.pt")}):
        c = torch.load(cache, map_location="cpu", weights_only=False)
        F = c["F"].double()
        S, A = F.shape[0], F.shape[1]
        if F.ndim != 4:
            raise SystemExit(f"{cache.name}: not a dense fixed-N cache")
        g = F.var(dim=-1, unbiased=True).sum(dim=(1, 2))
        delta = F.mean(-1) - c["f_ref"].double()
        pos = top_p_mask(delta.norm(dim=-1).max(dim=1).values, p)
        na = torch.full((S,), float(A), dtype=torch.float64)
        base = float(auroc(g, pos))
        # Distinct SYSTEM identity, not distinct filename: some caches are the
        # same system/committee under two naming conventions (frozen tree vs
        # journal tree). They differ only by float re-run noise (~6e-16
        # relative), so the key is QUANTISED -- exact equality would count them
        # as distinct and inflate the system count, which it did.
        sysids.setdefault((int(S), int(A),
                           f"{float(c['f_ref'].double().sum()):.10e}",
                           f"{float(g.sum()):.10e}"), cache.name)
        for a_ in alphas:
            got = float(auroc(normalised(g, na, a_), pos))
            if got != got or base != base:      # NaN: never silently 0.0
                n_nan += 1
                continue
            worst = max(worst, abs(got - base))

    # --- the check that can actually fail: variable N, independent reference ---
    gen = torch.Generator().manual_seed(4)
    nv = torch.randint(1, 445, (5000,), generator=gen).double()
    stat = torch.rand(5000, generator=gen, dtype=torch.float64) * 100 + 1e-6
    rel = 0.0
    for a_ in alphas:
        ref = torch.from_numpy(np.asarray(
            [float(x) / (3.0 * float(m)) ** a_ for x, m in zip(stat.tolist(), nv.tolist())]))
        got = normalised(stat, nv, a_)
        rel = max(rel, float(((got - ref).abs() / ref.abs().clamp_min(1e-300)).max()))

    def _mutants(st, m, a_):
        return {"precedence": st / 3.0 * m.double().pow(a_),
                "wrong_N": st / (3.0 * m.double().flip(0)).pow(a_),
                "absent": st}
    caught = {}
    for name, val in _mutants(stat, nv, 0.75).items():
        ref = normalised(stat, nv, 0.75)
        caught[name] = bool(float((val - ref).abs().max()) > 1e-12)

    return {"n_cache_files": len({q for d in dirs for q in d.glob("head_forces_*.pt")}),
            "n_distinct_systems": len(sysids),
            "invariance_fixed_n_max_abs_auroc_change": worst,
            "invariance_is_a_derivation_not_a_test": True,
            "n_undefined_auroc_skipped": n_nan,
            "reference_max_relerr_variable_n": rel,
            "mutation_caught": caught,
            "mutation_test_has_teeth": all(caught.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="results/records/j6b_foundation_scan.pt")
    ap.add_argument("--protocol", default="protocols/j6c_size_normalisation.yaml")
    ap.add_argument("--out", default="results/records/j6c_size_normalisation.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    pp = ROOT / a.protocol
    proto = yaml.safe_load(pp.read_text())
    got = hashlib.sha256(pp.read_bytes()).hexdigest()
    reg = json.loads((ROOT / "results/records/j6c_protocol_registration.json").read_text())
    if got != reg["protocol_sha256"]:
        print("REFUSING TO SCORE: protocol changed since registration")
        return 2
    st = proto["analysis"]["settings"]
    p, alphas = st["error_p"], proto["grid"]
    print(f"protocol : {pp.name} v{proto['version']} ({proto['status']})")
    print(f"grid     : {alphas}\n")

    # ---------------- N1: fixed-N invariance ----------------
    inv = fixed_n_invariance(alphas, p)
    print(f"N1 fixed-N invariance (a DERIVATION, cannot fail): "
          f"{inv['n_distinct_systems']} distinct systems in {inv['n_cache_files']} files, "
          f"max |AUROC change| = {inv['invariance_fixed_n_max_abs_auroc_change']:.3e}")
    print(f"   real check -- variable-N agreement with an independent reference: "
          f"max rel err {inv['reference_max_relerr_variable_n']:.3e}")
    print(f"   mutation test has teeth: {inv['mutation_test_has_teeth']} "
          f"({inv['mutation_caught']})")

    # ---------------- variable-N ----------------
    d = torch.load(ROOT / a.scan, map_location="cpu", weights_only=False)
    n = d["exact_global"].shape[0]
    natoms = d["natoms"]
    err = {k: d[k].double() for k in ("e_max", "e_rmse", "e_maxcomp", "e_q95")}
    g0 = d["exact_global"].double()
    maxcomp = d["exact_maxcomp"].double()
    maxatom = d["exact_maxatom"].double()
    fnorm = d["force_norm"].double()

    gen = torch.Generator().manual_seed(st["split_seed"])
    perm = torch.randperm(n, generator=gen)
    n_design = int(st["design_frac"] * n)
    design, test = perm[:n_design], perm[n_design:]

    # context the choice of alpha depends on: is the TARGET size-dependent?
    corr_target_N = {k: float(np.corrcoef(v.numpy(), natoms.double().numpy())[0, 1])
                     for k, v in err.items()}
    print("\ncorr(error score, N):  " +
          "  ".join(f"{k}={v:+.3f}" for k, v in corr_target_N.items()))

    # ---- selection on DESIGN only ----
    pos_d = top_p_mask(err["e_max"][design], p)
    design_auroc = {a_: float(auroc(normalised(g0, natoms, a_)[design], pos_d))
                    for a_ in alphas}
    alpha_star = min((a_ for a_ in alphas
                      if design_auroc[a_] == max(design_auroc.values())))
    print("\nDESIGN-split selection (this split is used for NOTHING else):")
    for a_ in alphas:
        print(f"    alpha={a_:<5} AUROC={design_auroc[a_]:.4f}"
              f"{'   <- alpha*' if a_ == alpha_star else ''}")

    # ---- everything below on the held-out split ----
    def A(sig, escore):
        return float(auroc(sig[test], top_p_mask(err[escore][test], p)))

    sig_star = normalised(g0, natoms, alpha_star)
    rows = []
    print(f"\nHELD-OUT ({len(test)} structures)")
    print(f"    {'estimator':22s}" + "".join(f"{e:>11s}" for e in err) + f"{'corr N':>9s}")
    cands = {f"S_alpha={a_}": normalised(g0, natoms, a_) for a_ in alphas}
    cands["exact_maxcomp"] = maxcomp
    cands["exact_maxatom"] = maxatom
    cands["force_norm (free)"] = fnorm
    table = {}
    for name, s in cands.items():
        table[name] = {e: A(s, e) for e in err}
        c = float(np.corrcoef(s[test].numpy(), natoms[test].double().numpy())[0, 1])
        star = "  <- alpha*" if name == f"S_alpha={alpha_star}" else ""
        print(f"    {name:22s}" + "".join(f"{table[name][e]:11.3f}" for e in err)
              + f"{c:9.3f}{star}")

    # ---- acquisition ----
    lo_r0 = st["lo_r0"]
    A_s = d["A_s"].double()
    Gd = A_s[design].sum(0)
    Q = torch.linalg.eigh(Gd)[1][:, -lo_r0:]
    lo0 = torch.einsum("rk,srq,qk->s", Q, A_s, Q) / R
    lo_star = normalised(lo0, natoms, alpha_star)

    def top(sig, k):
        return test[sig[test].topk(min(k, len(test))).indices]

    acq = {}
    print(f"\nACQUISITION from {len(test)} held-out candidates")
    print(f"    {'estimator':22s}{'budget':>8s}{'median atoms':>14s}{'Jaccard vs S0':>15s}")
    for b in st["budgets"]:
        ref0 = top(g0, b)
        for name, s in (("S_0 (global)", g0), (f"S_alpha*={alpha_star}", sig_star),
                        ("exact_maxcomp", maxcomp), ("force_norm (free)", fnorm)):
            t = top(s, b)
            acq[(name, b)] = t
            print(f"    {name:22s}{b:8d}{natoms[t].double().median():14.0f}"
                  f"{len(set(t.tolist()) & set(ref0.tolist())) / len(set(t.tolist()) | set(ref0.tolist())):15.3f}")

    def jac(x, y):
        sx, sy = set(x.tolist()), set(y.tolist())
        return len(sx & sy) / len(sx | sy)

    jn = {b: jac(top(lo_star, b), top(sig_star, b)) for b in st["budgets"]}
    j0 = {b: jac(top(lo0, b), top(g0, b)) for b in st["budgets"]}
    print(f"\n    gate overlap (leading-only vs exact), unnormalised: "
          + "  ".join(f"b={b}:{v:.3f}" for b, v in j0.items()))
    print(f"    gate overlap, normalised at alpha*:                "
          + "  ".join(f"b={b}:{v:.3f}" for b, v in jn.items()))

    # ---- scoring ----
    res = {}
    res["N1"] = (inv["invariance_fixed_n_max_abs_auroc_change"] <= 1e-12
             and inv["reference_max_relerr_variable_n"] <= 1e-12
             and inv["mutation_test_has_teeth"])
    res["N2"] = alpha_star > 0.0
    res["N3"] = all(table[f"S_alpha={alpha_star}"][e] > table["S_alpha=0.0"][e] for e in err)
    res["N4"] = table[f"S_alpha={alpha_star}"]["e_max"] > table["exact_maxcomp"]["e_max"]
    res["N5"] = float(natoms[acq[(f"S_alpha*={alpha_star}", 8000)]].double().median()) < 40
    res["N6"] = jn[8000] >= 0.90

    ci_vs_mc = paired_difference(auroc, sig_star[test], maxcomp[test],
                                 top_p_mask(err["e_max"][test], p),
                                 n_boot=a.n_boot, scheme="iid")
    ci_vs_g0 = paired_difference(auroc, sig_star[test], g0[test],
                                 top_p_mask(err["e_max"][test], p),
                                 n_boot=a.n_boot, scheme="iid")
    ci_vs_fn = paired_difference(auroc, sig_star[test], fnorm[test],
                                 top_p_mask(err["e_max"][test], p),
                                 n_boot=a.n_boot, scheme="iid")

    # vs max-component on ALL FOUR error scores, not only the one alpha was tuned
    # on. An audit found the single-score version generalised a tie into "the
    # deficit is repaired" while a significant deficit survived on e_maxcomp.
    panel = {}
    for e in err:
        panel[e] = paired_difference(auroc, sig_star[test], maxcomp[test],
                                     top_p_mask(err[e][test], p),
                                     n_boot=a.n_boot, scheme="iid")
    print("\nS_alpha* vs exact max-component, ALL FOUR error scores (held-out):")
    for e, c_ in panel.items():
        verdict = ("significant WIN" if c_["delta"] > 0 and c_["significant"] else
                   "significant LOSS" if c_["significant"] else "tie (n.s.)")
        print(f"    {e:10s} {c_['delta']:+.5f}  [{c_['ci_lo']:+.5f},{c_['ci_hi']:+.5f}]  {verdict}")
    n_win = sum(c_["delta"] > 0 and c_["significant"] for c_ in panel.values())
    n_loss = sum(c_["delta"] < 0 and c_["significant"] for c_ in panel.values())
    print(f"    => {n_win} significant wins, {4 - n_win - n_loss} ties, "
          f"{n_loss} significant losses")

    print("\n" + "=" * 74)
    for q in proto["predictions"]:
        print(f"{q['id']}  {' '.join(q['statement'].split())[:86]}")
        print(f"      -> {'PASS' if res[q['id']] else 'FAIL'}")
    print(f"\nalpha* = {alpha_star}")
    print(f"  vs S_0        : {ci_vs_g0['delta']:+.4f} "
          f"[{ci_vs_g0['ci_lo']:+.4f},{ci_vs_g0['ci_hi']:+.4f}] sig={ci_vs_g0['significant']}")
    print(f"  vs maxcomp    : {ci_vs_mc['delta']:+.4f} "
          f"[{ci_vs_mc['ci_lo']:+.4f},{ci_vs_mc['ci_hi']:+.4f}] sig={ci_vs_mc['significant']}")
    print(f"  vs force_norm : {ci_vs_fn['delta']:+.4f} "
          f"[{ci_vs_fn['ci_lo']:+.4f},{ci_vs_fn['ci_hi']:+.4f}] sig={ci_vs_fn['significant']}")
    print(f"\nVERDICT: {sum(res.values())}/{len(res)} criteria met "
          f"({', '.join(k for k, v in sorted(res.items()) if v) or 'none'})")

    out = ROOT / a.out
    out.write_text(json.dumps({
        "protocol_sha256": got, "alpha_star": alpha_star,
        "design_auroc": design_auroc, "held_out_auroc": table,
        "fixed_n_invariance": inv, "corr_error_natoms": corr_target_N,
        "n_test": len(test), "n_design": len(design),
        "gate_jaccard_unnormalised": j0, "gate_jaccard_normalised": jn,
        "acquisition_median_atoms": {f"{k[0]}|{k[1]}": float(natoms[v].double().median())
                                     for k, v in acq.items()},
        "ci_vs_S0": ci_vs_g0, "ci_vs_maxcomp": ci_vs_mc, "ci_vs_force_norm": ci_vs_fn,
        "vs_maxcomp_all_scores": panel,
        "N4_note": ("scored as a bare point comparison per the registered criterion; "
                    "the e_max margin is +0.001 with a CI straddling zero, and a "
                    "significant deficit survives on e_maxcomp. Read as parity on "
                    "e_max, not as a repair across the panel."),
        "passed": res, "n_passed": sum(res.values()),
    }, indent=1, default=float))
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
