#!/usr/bin/env python
"""Score the out-of-distribution scan (J10) and the filtered in-distribution
re-analysis against protocols/j10_ood.yaml.

Reads the compact per-structure records written by experiments/j10_ood_scan.py
(results/records/j10/*_perstructure.npz) and by the MPtraj rescan
(results/records/j6b_foundation_scan_perstructure.npz). Writes
results/records/j10_ood_verdict.json. Refuses to run if the protocol changed.
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
from forcesketch_journal.evaluation.bootstrap_block import paired_bootstrap, paired_difference  # noqa: E402
from forcesketch_journal.evaluation.tail_metrics import auroc, top_p_mask  # noqa: E402

SIGNALS = {"exact_global": "global", "exact_maxcomp": "max-component", "exact_maxatom": "max-atom",
           "force_norm": "force norm (zero cost)", "energy_std": "energy spread (zero cost)"}
COMMITTEE = ["exact_global", "exact_maxcomp", "exact_maxatom"]
ERRORS = ["e_max", "e_rmse", "e_maxcomp", "e_q95"]
MODELS = ["qbc8000", "qbc1000", "random8000", "mmf8000"]
SUBSETS = ["aimd-from-PBE-1000-nvt", "aimd-from-PBE-1000-npt", "rattled-1000-subsampled"]
AIMD = SUBSETS[:2]


def load_npz(p: pathlib.Path) -> dict:
    z = np.load(p, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def panel(d: dict, mask: np.ndarray | None, seed: int, p: float = 0.05, n_boot: int = 1000) -> dict:
    """AUROC of every signal for every error target, with a random null and
    paired bootstrap CIs for free - best committee."""
    idx = np.arange(d["e_max"].size) if mask is None else np.flatnonzero(mask)
    out = {"n": int(idx.size), "auroc": {}, "free_minus_best_committee": {}}
    g = torch.Generator().manual_seed(seed)
    rnd = torch.rand(idx.size, generator=g, dtype=torch.float64)
    for e in ERRORS:
        err = torch.from_numpy(d[e][idx]).double()
        pos = top_p_mask(err, p)
        row = {s: float(auroc(torch.from_numpy(d[s][idx]).double(), pos)) for s in SIGNALS}
        row["random_null"] = float(auroc(rnd, pos))
        out["auroc"][e] = row
        best = max(COMMITTEE, key=lambda s: row[s])
        ci = paired_difference(lambda s, r: auroc(s, top_p_mask(r, p)),
                               torch.from_numpy(d["force_norm"][idx]).double(),
                               torch.from_numpy(d[best][idx]).double(), err,
                               n_boot=n_boot, scheme="block", block_len=1, seed=seed)
        out["free_minus_best_committee"][e] = {"best_committee": best, "delta": float(ci["delta"]),
                                               "ci95": [float(ci["ci_lo"]), float(ci["ci_hi"])]}
    return out


def decile_auroc(d: dict, mask: np.ndarray | None, p: float = 0.05, n_bins: int = 10) -> dict:
    """Mean AUROC within reference-force-magnitude deciles: does the signal
    carry information once the force magnitude itself is held roughly fixed?"""
    idx = np.arange(d["e_max"].size) if mask is None else np.flatnonzero(mask)
    order = idx[np.argsort(d["ref_force_max"][idx])]
    out = {}
    for s in list(SIGNALS) + ["random_null"]:
        vals = []
        for chunk in np.array_split(order, n_bins):
            err = torch.from_numpy(d["e_max"][chunk]).double()
            pos = top_p_mask(err, p)
            if pos.sum() == 0 or pos.sum() == len(pos):
                continue
            sig = torch.rand(len(chunk), generator=torch.Generator().manual_seed(1), dtype=torch.float64) \
                if s == "random_null" else torch.from_numpy(d[s][chunk]).double()
            vals.append(float(auroc(sig, pos)))
        out[s] = {"mean": float(np.mean(vals)), "per_decile": vals}
    return out


def compositions(subset: str, row_ids: np.ndarray) -> np.ndarray | None:
    """Element-count matrix [n, 119] for the scanned OMat24 rows, read back from
    the LMDB on CPU. Needed because OMat24 was computed with a different VASP
    pseudopotential set from MPtraj, so the raw energy error is dominated by
    per-element reference offsets that a one-off linear fit removes."""
    try:
        from forcesketch_journal.data.aselmdb import AseLMDB
        from forcesketch_journal.paths import SCRATCH
        db = AseLMDB(SCRATCH / "omat24" / subset / "data.aselmdb")
    except Exception as e:  # noqa: BLE001
        print(f"  (no compositions for {subset}: {e})")
        return None
    X = np.zeros((row_ids.size, 119))
    for i, rid in enumerate(row_ids.tolist()):
        np.add.at(X[i], db.get(rid).numbers, 1.0)
    db.close()
    return X


def accuracy(d: dict, X: np.ndarray | None = None) -> dict:
    n = d["natoms"].astype(float)
    de_tot = d["E_mean"] - d["E_ref"]
    out = {"force_rmse_mev_A": float(np.sqrt(np.sum(d["e_rmse"] ** 2 * 3 * n) / np.sum(3 * n))) * 1000,
           "force_mae_proxy_mev_A": None,
           "energy_mae_mev_atom": float(np.abs(de_tot / n).mean()) * 1000,
           "median_ref_force_max_eV_A": float(np.median(d["ref_force_max"])),
           "frac_ref_force_max_below_0.05": float(np.mean(d["ref_force_max"] < 0.05))}
    if X is not None:
        cols = np.flatnonzero(X.sum(0) > 0)
        coef, *_ = np.linalg.lstsq(X[:, cols], de_tot, rcond=None)
        resid = de_tot - X[:, cols] @ coef
        out["energy_mae_mev_atom_after_element_offsets"] = float(np.abs(resid / n).mean()) * 1000
        out["n_elements_fitted"] = int(cols.size)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="protocols/j10_ood.yaml")
    ap.add_argument("--out", default="results/records/j10_ood_verdict.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    pp = ROOT / a.protocol
    got = hashlib.sha256(pp.read_bytes()).hexdigest()
    reg = json.loads((ROOT / "results/records/j10_protocol_registration.json").read_text())
    if got != reg["protocol_sha256"]:
        raise SystemExit("REFUSING TO SCORE: protocol changed since registration")
    proto = yaml.safe_load(pp.read_text())
    seed = int(proto["analysis"]["settings"].get("bootstrap", "seed 20261111").split()[-1])
    thr = float(proto["analysis"]["settings"]["filtered_force_threshold_eV_A"])

    rec_dir = ROOT / "results/records/j10"
    ood, comp = {}, {}
    for m in MODELS:
        for sub in SUBSETS:
            p = rec_dir / f"j10_{m}_{sub}_perstructure.npz"
            if p.exists():
                d = load_npz(p)
                X = comp.setdefault(sub, compositions(sub, d["row_id"]))
                ood[(m, sub)] = {"panel": panel(d, None, seed, n_boot=a.n_boot),
                                 "deciles": decile_auroc(d, None), "accuracy": accuracy(d, X),
                                 "n_dropped_elements": d["meta"].get("n_dropped_elements")}
                print(f"{m:<10} {sub:<28} n={ood[(m, sub)]['panel']['n']:5d}  "
                      f"maxatom {ood[(m, sub)]['panel']['auroc']['e_max']['exact_maxatom']:.3f}  "
                      f"free {ood[(m, sub)]['panel']['auroc']['e_max']['force_norm']:.3f}  "
                      f"F-RMSE {ood[(m, sub)]['accuracy']['force_rmse_mev_A']:.0f}", flush=True)
    mp_path = ROOT / "results/records/j6b_foundation_scan_perstructure.npz"
    mptraj = None
    if mp_path.exists():
        d = load_npz(mp_path)
        keep = (d["ref_force_max"] >= thr) & (d["natoms"] > 1)
        mptraj = {"full": panel(d, None, seed, n_boot=a.n_boot),
                  "filtered": panel(d, keep, seed, n_boot=a.n_boot),
                  "n_excluded_low_force": int(np.sum(d["ref_force_max"] < thr)),
                  "n_excluded_single_atom": int(np.sum(d["natoms"] == 1)),
                  "n_kept": int(keep.sum()), "threshold_eV_A": thr,
                  "deciles_full": decile_auroc(d, None), "deciles_filtered": decile_auroc(d, keep),
                  "accuracy": accuracy(d),
                  "corr_log_e_max_log_ref_force_max": float(np.corrcoef(
                      np.log(d["e_max"] + 1e-6), np.log(d["ref_force_max"] + 1e-6))[0, 1])}
        print(f"MPtraj: kept {mptraj['n_kept']} of {d['e_max'].size}; free full "
              f"{mptraj['full']['auroc']['e_max']['force_norm']:.3f} filtered "
              f"{mptraj['filtered']['auroc']['e_max']['force_norm']:.3f}")

    # ---- predictions
    def g(m, sub, path):
        v = ood.get((m, sub))
        for k in path:
            v = v[k] if v is not None else None
        return v
    detail, passed = {}, {}
    q = [g("qbc8000", s, ("panel", "auroc", "e_max", "exact_maxatom")) for s in AIMD]
    detail["P1"] = {"auroc_maxatom_e_max": dict(zip(AIMD, q))}
    passed["P1"] = all(v is not None and v >= 0.70 for v in q)
    dl = [g("qbc8000", s, ("panel", "free_minus_best_committee", "e_max")) for s in AIMD]
    detail["P2"] = {"free_minus_best": dict(zip(AIMD, dl))}
    passed["P2"] = all(v is not None and v["delta"] >= -0.02 for v in dl)
    if mptraj is not None:
        f = mptraj["filtered"]["free_minus_best_committee"]["e_max"]
        detail["P3"] = f; passed["P3"] = f["delta"] > 0.0
    dd = [g("qbc8000", s, ("deciles", "exact_maxatom", "mean")) for s in AIMD]
    detail["P4"] = {"within_decile_maxatom": dict(zip(AIMD, dd))}
    passed["P4"] = all(v is not None and v > 0.55 for v in dd)
    rows = [(m, np.mean([g(m, s, ("accuracy", "force_rmse_mev_A")) for s in AIMD if g(m, s, ("accuracy",))]),
             np.mean([g(m, s, ("panel", "auroc", "e_max", "exact_maxatom")) for s in AIMD if g(m, s, ("panel",))]))
            for m in MODELS if any(g(m, s, ("panel",)) for s in AIMD)]
    if len(rows) >= 2:
        best_rmse = min(rows, key=lambda r: r[1])[0]; best_auroc = max(rows, key=lambda r: r[2])[0]
        detail["P5"] = {"by_model": {r[0]: {"force_rmse": float(r[1]), "auroc_maxatom": float(r[2])} for r in rows},
                        "lowest_rmse": best_rmse, "highest_auroc": best_auroc}
        passed["P5"] = best_rmse == best_auroc
    out = {"protocol_sha256": got, "protocol_version": proto["version"], "passed": passed,
           "n_passed": int(sum(passed.values())), "n_scored": len(passed), "detail": detail,
           "ood": {f"{m}|{s}": v for (m, s), v in ood.items()}, "mptraj": mptraj}
    (ROOT / a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(passed), "->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
