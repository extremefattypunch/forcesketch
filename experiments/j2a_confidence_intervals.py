#!/usr/bin/env python
"""Block-bootstrap intervals for the J2a oracle panel.

**Written to close a provenance hole.** `j2a_global_vs_maxcomp_ci.json` and
`j2a_free_signal_ci.json` back the paper's most-quoted claims -- "24 of 24
positive, 19 of 24 significant" and the +0.067 to +0.182 gain over the best free
signal -- and until now **no script in the repository produced them**. They were
made by an ad-hoc inline computation in an earlier session, which means R3 ("no
transcribed numbers") held only by luck: nobody could regenerate them to check.
This is the second such hole found (see `j6_spectrum.py` for the first).

`--validate` recomputes both records and diffs against the committed copies, so
the script must first prove it is the thing that produced them before it is
trusted to produce them again.

Block length comes from the measured integrated autocorrelation time of each
ERROR series, `ceil(2 tau_int)`, matching the J1 machinery -- not a guess, and not
shared across error scores, which is why the committed records carry different
block lengths for different scores on the same system.
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
from forcesketch_journal.evaluation.bootstrap_block import (  # noqa: E402
    paired_bootstrap, paired_difference,
)
from forcesketch_journal.evaluation.tail_metrics import auroc, top_p_mask  # noqa: E402

# Order matches the committed records so a regenerated file is directly
# comparable; the validation below keys on (system, score) regardless, because
# comparing two record sets by POSITION silently misaligns them -- which is
# exactly what happened on the first run of this validation and produced
# "significance flipped" on four entries that were in fact identical.
PANEL = ["disjoint_test_1200K", "overlapping_test_1200K", "same_test_1200K",
         "rmd17-disjoint_ethanol", "rmd17-disjoint_aspirin", "rmd17-disjoint_azobenzene"]
ERRORS = ["e_max", "e_maxcomp", "e_rmse", "e_q95"]
FREE = ["energy_std", "force_norm"]


def load_panel(d: pathlib.Path) -> dict[str, dict]:
    out = {}
    for p in sorted(d.glob("*.pt")):
        r = torch.load(p, weights_only=True)
        if r["tag"] in PANEL:
            out[r["tag"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perstructure", default="results/perstructure/j2a")
    ap.add_argument("--out-dir", default="results/records")
    ap.add_argument("--error-p", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--validate", action="store_true",
                    help="recompute and diff against the committed records; write nothing")
    a = ap.parse_args()

    panel = load_panel(ROOT / a.perstructure)
    missing = [s for s in PANEL if s not in panel]
    if missing:
        raise SystemExit(f"missing per-structure tensors for {missing}; run j2a first")

    gvm, vs_free, absolute = [], [], []
    for sysname in PANEL:
        r = panel[sysname]
        for e in ERRORS:
            err = r[f"err_{e}"].double()
            pos = top_p_mask(err, a.error_p)
            bl = block_length(integrated_autocorr_time(err.numpy())["tau_int"])
            ci = paired_difference(auroc, r["sig_exact_global"].double(),
                                   r["sig_exact_maxcomp"].double(), pos,
                                   n_boot=a.n_boot, scheme="block", block_len=bl,
                                   seed=a.seed)
            gvm.append({"system": sysname, "error_score": e, **ci})

        err = r["err_e_max"].double()
        pos = top_p_mask(err, a.error_p)
        bl = block_length(integrated_autocorr_time(err.numpy())["tau_int"])
        for f in FREE:
            if f"sig_{f}" not in r:
                continue
            ci = paired_difference(auroc, r["sig_exact_global"].double(),
                                   r[f"sig_{f}"].double(), pos,
                                   n_boot=a.n_boot, scheme="block", block_len=bl,
                                   seed=a.seed)
            vs_free.append({"system": sysname, "free_signal": f, **ci})
        # Section 1's per-system AUROC intervals had the same missing-provenance
        # problem as the paired comparisons; they are produced here too.
        row = {"system": sysname, "block_len": bl}
        for sname in ("exact_global", "exact_maxcomp", "exact_maxatom",
                      "force_norm", "energy_std", "random_null"):
            if f"sig_{sname}" not in r:
                continue
            row[sname] = float(auroc(r[f"sig_{sname}"].double(), pos))
            if sname == "exact_global":
                b = paired_bootstrap(auroc, r[f"sig_{sname}"].double(), pos,
                                     n_boot=a.n_boot, scheme="block", block_len=bl,
                                     seed=a.seed)
                row["exact_global_ci"] = [b["ci_lo"], b["ci_hi"]]
        absolute.append(row)

    out_gvm = {"path": "j2a_global_vs_maxcomp_ci.json", "data": gvm}
    out_free = {"path": "j2a_free_signal_ci.json",
                "data": {"vs_free": vs_free, "absolute": absolute}}

    if a.validate:
        bad = 0
        for o in (out_gvm, out_free):
            ref = json.loads((ROOT / "results/records" / o["path"]).read_text())
            new = o["data"]
            refs = ref if isinstance(ref, list) else ref["vs_free"]
            news = new if isinstance(new, list) else new["vs_free"]
            key = (lambda c: (c["system"], c.get("error_score") or c["free_signal"]))
            R = {key(c): c for c in refs}
            N = {key(c): c for c in news}
            if set(R) != set(N):
                print(f"  {o['path']}: key sets differ; only in committed "
                      f"{sorted(set(R) - set(N))}, only in recomputed {sorted(set(N) - set(R))}")
                bad += 1
                continue
            worst = 0.0
            for k_ in R:
                x, y = R[k_], N[k_]
                for f_ in ("delta", "ci_lo", "ci_hi"):
                    worst = max(worst, abs(x[f_] - y[f_]))
                if x.get("significant") != y.get("significant"):
                    print(f"  {o['path']}: significance flipped on {k_}")
                    bad += 1
            print(f"  {o['path']}: {len(refs)} entries, max |diff| = {worst:.3e}")
            if worst > 1e-12:
                bad += 1
        print("VALIDATION " + ("FAILED" if bad else "OK -- this script reproduces "
                               "the committed records exactly"))
        return 1 if bad else 0

    d = ROOT / a.out_dir
    meta = {"experiment_id": "j2a_confidence_intervals", "seed": a.seed,
            "n_boot": a.n_boot, "error_p": a.error_p,
            "note": ("Regenerated. The previous copies of these two records had no "
                     "generating script and no recorded seed, and could not be "
                     "reproduced by the documented procedure; they are kept under "
                     "results/records/superseded/. The seed is recorded here so this "
                     "cannot recur.")}
    (d / "j2a_global_vs_maxcomp_ci.json").write_text(
        json.dumps({"meta": meta, "entries": gvm}, indent=1))
    (d / "j2a_free_signal_ci.json").write_text(
        json.dumps({"meta": meta, "vs_free": vs_free, "absolute": absolute}, indent=1))
    pos_n = sum(c["delta"] > 0 for c in gvm)
    sig_n = sum(c["delta"] > 0 and c["significant"] for c in gvm)
    print(f"global beats max-component in {pos_n}/{len(gvm)} cells, "
          f"{sig_n} significant")
    print(f"free-signal comparisons: {sum(c['significant'] for c in vs_free)}/{len(vs_free)} "
          f"significant")
    print(f"wrote 2 records to {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
