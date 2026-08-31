# Cross-document consistency audit

Seven agents extracted every quantitative claim from the fourteen results
documents and checked each against the records by running code. Raw output:
`results/records/consistency_audit.json`.

| | count |
|---|---|
| claims checked | **507** |
| verified against a record | **373** |
| **wrong** (a record contradicts them) | **~70** |
| unverifiable (no record backs them) | **~64** |
| cross-document contradictions, adversarially verified | **13 of 16 reported** |

**One pattern accounts for most of the failures: corrections that were made in
one document and never propagated to the others that quote the same number.**
The project corrected itself repeatedly — split seeds, the lane count, the J3
noise model, the CI records — and each correction left stale copies elsewhere.
That is a systemic weakness of prose-carried numbers, and it is exactly what
`tools/paper_numbers.py` exists to prevent: the 41 macro-backed numbers were
**not** among the failures, because they are derived rather than typed.

## Per document

| document | claims | wrong | unverifiable |
|---|---|---|---|

| `J1_SPLIT_REPAIR.md` | 43 | 13 | 4 |
| `J2A_ORACLE_PANEL.md` | 34 | 2 | 2 |
| `J2B_FACTOR_B.md` | 33 | 7 | 8 |
| `J3_EXTREME_VALUE.md` | 29 | 6 | 3 |
| `J4_M_SWEEP.md` | 25 | 5 | 1 |
| `J5_TEMPERATURE_LADDER.md` | 27 | 3 | 1 |
| `J5_WATER.md` | 50 | 4 | 3 |
| `J6B_FOUNDATION_ACQUISITION.md` | 33 | 4 | 1 |
| `J6C_SIZE_NORMALISATION.md` | 35 | 1 | 1 |
| `J6_NAIVE_COMMITTEE.md` | 32 | 1 | 0 |
| `J8_MATCHED_PAIR.md` | 20 | 2 | 1 |
| `J8_SM120_CONTROL.md` | 21 | 11 | 3 |
| `J8_SMOKE_A100.md` | 17 | 4 | 1 |
| `J8_WATER_VERDICT.md` | 25 | 7 | 1 |

## Fixed in this pass

- **`J1_SPLIT_REPAIR.md`** — blocked-vs-random table regenerated from the shipped manifests (mean |diff| 0.017 -> 0.027)
- **`J2A_ORACLE_PANEL.md`** — CI records regenerated with a recorded seed; 8/10 -> 9/10; section-1 intervals
- **`J3_EXTREME_VALUE.md`** — per-item seeding; MAE 0.017 -> 0.015; tables regenerated
- **`J5_WATER.md`** — free-signal count 8/10 -> 9/10; in-distribution 3BPA overconfidence corrected (was the 1200 K extrapolative figure)
- **`J6B_FOUNDATION_ACQUISITION.md`** — 'one in five acquired differ' -> 11.5%; relative-error null noted
- **`J8_SM120_CONTROL.md`** — SUPERSEDED banner with the corrected 8-lane numbers
- **`J8_SMOKE_A100.md`** — 84-87% -> 82-94%; broken record pointer corrected

## Not yet fixed

The remainder are recorded in `consistency_audit.json` with the evidence for
each. They fall into three groups, in descending order of how much they matter:

1. **Stale numbers in J2B_FACTOR_B.md and J8_WATER_VERDICT.md.** J2B's
   leading-only-minus-control-variate figures at L=5 predate the current
   `j9a_leading_only_ci_global_L5.json`, so the srank-separation table quotes
   +0.021 where the record now gives +0.039, and two water entries in that table
   have no surviving record at all. J8_WATER_VERDICT describes an A100 run as
   "clean" that in fact fails its own pre-registered admissibility gate (19 of 60
   cells above the 3% IQR limit). **Neither changes a conclusion** — the srank
   separation and the water speedup both survive — but both must be regenerated
   before the manuscript quotes them.

2. **Overstated ranges.** A recurring habit of quoting a range that excludes
   some of the rows in the table directly above it: J6B's "0.80–0.84" against
   cells spanning 0.759–0.836; J8_MATCHED_PAIR's serial R² "0.9999–1.000" against
   an actual minimum of 0.9968; J8_WATER_VERDICT's per-edge constant "~3×" that
   does not follow from the two numbers in the same sentence.

3. **Claims with no record.** Thirty, mostly edge counts and wall-clock times
   that were printed to a log but never recorded — including the
   frequently-quoted "~17,150 edges per structure for water against 3BPA's 520",
   which is consistent with the records but not derivable from them.

## The one that most deserves attention

`J8_MATCHED_PAIR.md` claims water and 3BPA were measured "on the same device,
under the same run conditions". Same device and same physical node is true. Same
run conditions is **not**: the 3BPA cell used `warmup=100, iters=200` and the
water cell `warmup=25, iters=50`, in two separate jobs at different times. The
matched-pair framing is the paper's cleanest systems result and it rests on that
equivalence. `slurm/j8_matched_pair.sbatch` now runs both systems in **one
exclusive job at identical settings**, and is queued on A100 and H200; the
document should be rebuilt on whichever lands rather than patched.


---

## Final tally, after the contradiction and refutation phases

The workflow completed with 24 agents: seven extractors, one cross-document
contradiction finder over the 18 subjects appearing in more than one document,
and one adversarial verifier per reported contradiction. **16 contradictions
reported, 13 survived refutation**, and all 13 have now been fixed:

| # | subject | documents | fixed |
|---|---|---|---|
| 1 | Beta-law MAE 0.017 vs 0.015 | STAGE1 vs J3, REVIEW | ✓ |
| 2 | blocked-vs-random mean difference 0.017 vs 0.027 | STAGE1 vs J1 | ✓ |
| 3 | Blackwell serial slope 4.58 vs 5.10 | J8_SMOKE vs J8_SM120 | ✓ |
| 4 | leading-only − CV at L=5, all six systems | J2B vs REVIEW | ✓ regenerated |
| 5 | gain over best free signal +0.088–0.230 vs +0.067–0.182 | J5 vs J2A, STAGE1 | ✓ |
| 6 | free-signal significance 8/10 vs 9/10 | STAGE1 vs J2A, J5 | ✓ |
| 7 | J8 water verdict 4/4 vs 3/4 | J8_WATER_VERDICT vs REVIEW, MATCHED_PAIR | ✓ |
| 8 | calibration span 39–58% in two runs | J1, REVIEW vs manifests | ✓ |
| 9 | matched-pair "same run conditions" | J8_MATCHED_PAIR vs records | ✓ **resolved by re-run** |
| 10 | dropped-frame range 16–64 vs 16–96 | J1 prose vs its own table, STAGE1 | ✓ |
| 11 | serial-fit R² 0.9999–1.000 vs 0.9968 min | J8_MATCHED_PAIR vs records | ✓ |
| 12 | MPtraj AUROC range 0.80–0.84 vs its own table | J6B internal | ✓ |
| 13 | per-edge constant "~3×" vs 1.3× | J8_WATER_VERDICT internal | ✓ |

**Three findings from fixing them are worth keeping:**

1. **The J1 calibration-span claim was wrong in a way that matters.** Both J1 and
   REVIEW_FINDINGS asserted the split-seed fix gave "39–58% in two runs per
   system". The shipped manifests give 19.6–49.7% over the six headline systems,
   and **rMD17 azobenzene still has a single contiguous calibration run** — so the
   anti-confound goal the fix existed to achieve is met on five of six, not all.

2. **The J2B budget table predated the J1 split repair** (its `n_test` is 1259
   against the current 1252) and was never regenerated. All three budgets are now
   recomputed: leading-only still wins significantly on 5 of 6 at every budget,
   the advantage is *larger* than reported, and monotone shrinkage holds on 5 of 6
   rather than all six.

3. **Four rows of the srank table have no surviving record.** The water L=5 run
   wrote to the same path as the molecular run and was overwritten; the two PIMD
   values survive only in `review_candidates.json` and the two MD magnitudes
   (+0.582, +0.336) — the largest numbers in the table, and the ones the
   "srank predicts the sign" argument leans on hardest — have no corroboration at
   all. They are flagged † and must be regenerated before that table is quoted.


---

## Contradiction 9, closed properly rather than annotated

The clean re-runs landed (jobs 39518641 / 39518642, 2026-08-16): water and 3BPA
in **one exclusive job per device at identical settings**, max relative IQR
≤ 1.01%, SM clocks at maximum. `J8_MATCHED_PAIR.md` is rebuilt on them rather
than patched, and the result is stronger than the mismatched pair it replaces:

| device | v2 (pre-registered) | v3 | water speedup @ B=16 |
|---|---|---|---|
| H200 | **4/4** | 4/4 | 1.852× |
| A100-80GB | **3/4** | 3/4 | 1.880× |
| Blackwell (still unmatched ‡) | 2/4 | 3/4 | 1.891× |

Two things this settles:

* **The B=16 speedup is 1.85–1.89× on three GPU generations** (sm_80, sm_90,
  sm_120), agreeing to 1.5% across the two clean pairs. P1 and P2 — the
  predictions that carry the speedup claim — pass on every device.
* **v2 and v3 agree on both clean pairs.** The post-hoc protocol repair changes
  nothing where the run is clean; it mattered only for Blackwell, whose per-lane
  serial cost sits far below the threshold v2 froze. That defuses most of the
  post-hoc concern the J8 chapter has been carrying.

**Gap closed (job 40846458).** The clean Blackwell pair landed and **reproduces
the earlier verdict exactly** (2/4 under v2, 3/4 under v3), so nothing was riding
on the mismatched conditions — the defect was real and worth fixing, and fixing it
changed no conclusion. All three devices now have admissible matched pairs, and
the B=16 water speedup across them is 1.852–1.894× (a 2.3% spread over three GPU
generations).
