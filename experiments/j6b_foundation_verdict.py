#!/usr/bin/env python
"""Score the foundation-model acquisition scan against its pre-registered protocol.

Reads `results/records/j6b_foundation_scan.pt` and answers Q1-Q5 from
`protocols/j6b_foundation_acquisition.yaml`. The protocol's sha256 is verified
against the registration record; the scorer refuses to run if it has changed.

Beyond the five scored predictions it reports the things that make the numbers
interpretable and that no prediction covers: what each signal's acquired set
looks like in atom count, how strongly each signal correlates with system size,
and the pairwise overlap of every acquisition. On fixed-size systems those
questions are meaningless -- which is exactly why the extensivity problem could
not have been seen anywhere else in this project.
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
sys.path.insert(0, str(ROOT / "src"))

from forcesketch_journal.evaluation.bootstrap_block import paired_difference  # noqa: E402
from forcesketch_journal.evaluation.tail_metrics import auroc, top_p_mask  # noqa: E402

M = 8
R = M - 1


def leading_basis(A_s: torch.Tensor, design: torch.Tensor, r0: int) -> torch.Tensor:
    """Top-r0 eigenvectors of the pooled DESIGN head-space Gram."""
    G = A_s[design].sum(0)
    w, V = torch.linalg.eigh(G)
    return V[:, -r0:]                                    # [r, r0], ascending eigenvalues


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="results/records/j6b_foundation_scan.pt")
    ap.add_argument("--protocol", default="protocols/j6b_foundation_acquisition.yaml")
    ap.add_argument("--out", default="results/records/j6b_foundation_verdict.json")
    ap.add_argument("--design-frac", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=20260816)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()

    pp = ROOT / a.protocol
    proto = yaml.safe_load(pp.read_text())
    got = hashlib.sha256(pp.read_bytes()).hexdigest()
    reg = json.loads((ROOT / "results/records/j6b_protocol_registration.json").read_text())
    if got != reg["protocol_sha256"]:
        print("REFUSING TO SCORE: protocol changed since registration")
        return 2

    d = torch.load(ROOT / a.scan, map_location="cpu", weights_only=False)
    if d.get("partial"):
        print("*** scan is PARTIAL -- informational only ***")
    n = d["exact_global"].shape[0]
    natoms = d["natoms"].double()
    print(f"protocol : {pp.name} v{proto['version']} ({proto['status']})")
    print(f"scan     : {n} structures, {int(natoms.min())}-{int(natoms.max())} atoms, "
          f"median {int(natoms.median())}\n")

    err = {k: d[k].double() for k in ("e_max", "e_rmse", "e_maxcomp", "e_q95")}
    sig = {k: d[k].double() for k in ("exact_global", "exact_maxcomp", "exact_maxatom",
                                      "force_norm", "energy_std")}
    sig["random_null"] = torch.rand(n, generator=torch.Generator().manual_seed(7),
                                    dtype=torch.float64)
    p = proto["analysis"]["settings"]["error_p"]

    # ---------- Factor A table ----------
    pos = top_p_mask(err["e_max"], p)
    print(f"{'signal':16s}" + "".join(f"{e:>12s}" for e in err))
    A = {}
    for s, v in sig.items():
        A[s] = {e: float(auroc(v, top_p_mask(err[e], p))) for e in err}
        print(f"{s:16s}" + "".join(f"{A[s][e]:12.3f}" for e in err))

    res, detail = {}, {}

    # ---------- Q1 ----------
    q1 = A["exact_global"]["e_max"]
    res["Q1"] = q1 >= 0.70
    detail["Q1"] = {"auroc": q1, "threshold": 0.70}

    # ---------- Q2 ----------
    cells = []
    for e in err:
        delta = A["exact_global"][e] - A["exact_maxcomp"][e]
        ci = paired_difference(auroc, sig["exact_global"], sig["exact_maxcomp"],
                               top_p_mask(err[e], p), n_boot=a.n_boot, scheme="iid")
        cells.append({"error_score": e, "delta": delta, **{k: ci[k] for k in
                      ("ci_lo", "ci_hi", "significant")}, "passed": delta > 0})
    res["Q2"] = sum(c["passed"] for c in cells) >= 3
    detail["Q2"] = cells

    # ---------- acquisition sets ----------
    budgets = proto["analysis"]["settings"]["budgets"]
    gen = torch.Generator().manual_seed(a.split_seed)
    perm = torch.randperm(n, generator=gen)
    n_design = int(a.design_frac * n)
    design, candidate = perm[:n_design], perm[n_design:]

    r0 = proto["analysis"]["settings"]["lo_r0"]
    Q = leading_basis(d["A_s"].double(), design, r0)
    lo_sig = torch.einsum("rk,srq,qk->s", Q, d["A_s"].double(), Q) / R
    sig["leading_only"] = lo_sig
    A["leading_only"] = {e: float(auroc(lo_sig, top_p_mask(err[e], p))) for e in err}
    print(f"{'leading_only':16s}" + "".join(f"{A['leading_only'][e]:12.3f}" for e in err))

    def top(sig_name: str, k: int) -> torch.Tensor:
        v = sig[sig_name][candidate]
        return candidate[v.topk(min(k, len(candidate))).indices]

    print(f"\nACQUISITION from {len(candidate)} candidates "
          f"({n_design} held out to fit the leading basis)")
    acq = {}
    for b in budgets:
        print(f"\n  budget {b}  ({b / len(candidate):.1%} of the pool)")
        print(f"    {'signal':16s}{'median atoms':>14s}{'mean atoms':>12s}"
              f"{'corr(sig,N)':>13s}{'Jaccard vs global':>19s}")
        ref = top("exact_global", b)
        for s in ("exact_global", "exact_maxcomp", "exact_maxatom", "leading_only",
                  "force_norm", "energy_std", "random_null"):
            t = top(s, b)
            acq[(s, b)] = t
            na = natoms[t]
            c = float(np.corrcoef(sig[s].numpy(), natoms.numpy())[0, 1])
            print(f"    {s:16s}{na.median():14.0f}{na.mean():12.1f}{c:13.3f}"
                  f"{jaccard(t, ref):19.3f}")

    # ---------- Q3 ----------
    b = 8000
    mg = float(natoms[acq[("exact_global", b)]].median())
    mm = float(natoms[acq[("exact_maxcomp", b)]].median())
    ratio = mg / mm
    res["Q3"] = ratio >= 1.25
    detail["Q3"] = {"median_atoms_global": mg, "median_atoms_maxcomp": mm,
                    "ratio": ratio, "threshold": 1.25,
                    "median_atoms_pool": float(natoms.median()),
                    "corr_global_natoms": float(np.corrcoef(
                        sig["exact_global"].numpy(), natoms.numpy())[0, 1]),
                    "corr_maxcomp_natoms": float(np.corrcoef(
                        sig["exact_maxcomp"].numpy(), natoms.numpy())[0, 1])}

    # ---------- Q4 ----------
    j = jaccard(acq[("leading_only", b)], acq[("exact_global", b)])
    res["Q4"] = j >= 0.90
    detail["Q4"] = {"jaccard": j, "threshold": 0.90, "budget": b, "lo_r0": r0,
                    "jaccard_1000": jaccard(acq[("leading_only", 1000)],
                                            acq[("exact_global", 1000)])}

    # ---------- Q5 ----------
    free = {s: A[s]["e_max"] for s in ("force_norm", "energy_std")}
    best = max(free, key=free.get)
    gap = A["exact_global"]["e_max"] - free[best]
    ci = paired_difference(auroc, sig["exact_global"], sig[best], pos,
                           n_boot=a.n_boot, scheme="iid")
    res["Q5"] = gap >= 0.05
    detail["Q5"] = {"best_free": best, "best_free_auroc": free[best], "gap": gap,
                    "threshold": 0.05, **{k: ci[k] for k in
                                          ("ci_lo", "ci_hi", "significant")}}

    # ---------- POST-HOC diagnostics, labelled as such ----------
    # Not pre-registered and not scored. Both exist to interpret Q2/Q5 rather
    # than to rescue them: if a free signal wins, the useful question is whether
    # it wins because it carries the same information or because the target is
    # confounded with something the free signal happens to encode.
    post = {}

    # (a) size-stratified: AUROC within atom-count quartiles, which removes the
    #     structure-size confound that an extensive signal exploits.
    qs = torch.quantile(natoms, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    bins = torch.bucketize(natoms, qs)
    strat = {}
    for s in ("exact_global", "exact_maxcomp", "force_norm", "energy_std", "leading_only"):
        vals = []
        for bi in range(4):
            m = bins == bi
            if m.sum() < 100:
                continue
            vals.append(float(auroc(sig[s][m], top_p_mask(err["e_max"][m], p))))
        strat[s] = {"per_quartile": vals, "mean": float(np.mean(vals))}
    post["size_stratified_auroc_e_max"] = strat

    # (b) relative error: does the committee know anything beyond "big forces"?
    rel = err["e_max"] / sig["force_norm"].clamp_min(1e-12)
    post["relative_error_auroc"] = {
        s: float(auroc(sig[s], top_p_mask(rel, p)))
        for s in ("exact_global", "exact_maxcomp", "force_norm", "energy_std",
                  "leading_only", "random_null")}
    post["corr_e_max_force_norm"] = float(np.corrcoef(
        err["e_max"].numpy(), sig["force_norm"].numpy())[0, 1])

    # (c) force-stratified: the decisive test of whether the committee carries
    #     information BEYOND "this structure has large forces". Within a narrow
    #     force-magnitude band the free signal has almost no variation left to
    #     exploit, so if the committee still ranks error there, it is adding
    #     something; if it drops to chance, it was only ever a force proxy.
    #     The relative-error diagnostic in (b) cannot answer this on its own,
    #     because force_norm sits in that ratio's denominator and is therefore
    #     mechanically anti-correlated with it.
    fq = torch.quantile(sig["force_norm"],
                        torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64))
    fbins = torch.bucketize(sig["force_norm"], fq)
    fstrat = {}
    for s in ("exact_global", "exact_maxcomp", "exact_maxatom", "force_norm",
              "energy_std", "leading_only", "random_null"):
        vals = []
        for bi in range(5):
            m = fbins == bi
            if m.sum() < 100:
                continue
            vals.append(float(auroc(sig[s][m], top_p_mask(err["e_max"][m], p))))
        fstrat[s] = {"per_quintile": vals, "mean": float(np.mean(vals))}
    post["force_stratified_auroc_e_max"] = fstrat
    print("\n  within force-magnitude quintiles (does the committee beat a force proxy"
          " once force is held fixed?)")
    for s, v in fstrat.items():
        print(f"    {s:16s}{v['mean']:8.3f}   " +
              " ".join(f"{x:.3f}" for x in v["per_quintile"]))

    print("\nPOST-HOC (not pre-registered, not scored)")
    print(f"  corr(e_max, force_norm) = {post['corr_e_max_force_norm']:.3f}")
    print(f"  {'signal':16s}{'AUROC e_max':>13s}{'size-stratified':>17s}"
          f"{'AUROC rel. error':>18s}")
    for s in ("exact_global", "exact_maxcomp", "leading_only", "force_norm", "energy_std"):
        print(f"  {s:16s}{A[s]['e_max']:13.3f}{strat[s]['mean']:17.3f}"
              f"{post['relative_error_auroc'][s]:18.3f}")

    print("\n" + "=" * 74)
    for q in proto["predictions"]:
        i = q["id"]
        print(f"{i}  {' '.join(q['statement'].split())[:88]}")
        print(f"      stated confidence: {q['confidence']}   -> "
              f"{'PASS' if res[i] else 'FAIL'}")
        print(f"      {json.dumps(detail[i], default=lambda x: round(float(x), 4))[:200]}\n")
    npass = sum(res.values())
    print(f"VERDICT: {npass}/{len(res)} pre-registered predictions confirmed "
          f"({', '.join(k for k, v in sorted(res.items()) if v) or 'none'})")

    out = ROOT / a.out
    out.write_text(json.dumps({
        "protocol_sha256": got, "n_structures": n, "partial": bool(d.get("partial")),
        "model": str(d.get("model")), "model_sha256": d.get("model_sha256"),
        "factor_a": A, "passed": res, "n_passed": npass, "detail": detail,
        "acquisition_median_atoms": {f"{s}_{b}": float(natoms[t].median())
                                     for (s, b), t in acq.items()},
        "post_hoc": post,
    }, indent=1, default=float))
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
