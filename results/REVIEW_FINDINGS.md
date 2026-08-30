# Adversarial review — findings and disposition

Five independent reviewers over distinct dimensions (metric correctness,
bootstrap/split validity, claims-vs-records traceability, experiment logic, GPU
benchmark validity), each finding then handed to a separate agent instructed to
**refute** it. 39 candidate findings.

**The workflow's own summary said "1 confirmed of 28" and that summary is
misleading.** 16 verifier agents died on a session limit; their findings resolved
to `null` and were silently filtered out, so they appear as neither confirmed nor
refuted. Both of the defects that turned out to matter most were in that
unverified pile, flagged critical. The disposition below is from verification I
performed directly.

| status | count |
|---|---|
| refuted | 13 |
| **confirmed real** | **22** |
| still unverified | 0 (10 moot) |

---

## Confirmed and fixed

### 1. `L=8` benchmark cells ran 7 lanes — *critical*

`exact_seed_bundle(M=8).seeds` holds only the `r = M−1 = 7` centred directions, so
`seeds[:8]` silently returned 7. Every cell labelled `L=8` measured 7 lanes.
Compounding it, the least-squares fits regressed that point against `x=8`, biasing
**every batched slope low by 15–18%** on every GPU. Reported speedups were
`T(7)/T(4)`.

*Fixed*: `j8_lane_bench.py` now concatenates the mean-force seed so `L` lanes means
`L` lanes, and raises if `L` exceeds what exists. All three J8 documents carry a
suspension banner; corrected runs resubmitted.

### 2. A filename collision destroyed a result file — *critical*

The water sbatch hard-coded `--out …_water.jsonl` regardless of device, so the
water-on-Blackwell run overwrote water-on-A100. Caught only because the surviving
file reports 188 SMs rather than 108. **The A100 water numbers reported earlier are
no longer backed by any record**, violating rule R3.

*Fixed*: surviving file renamed to `..._water_Blackwell.jsonl`; the sbatch now
derives the device from `nvidia-smi`. A100 water re-queued.

### 3. `beta_noise`'s coupling test was vacuous — *major*

The `shared` and `indep` noise models in J3 were **identical by construction**:
`shared` applied one Haar frame to *isotropically drawn* coordinate directions, and
for a fixed subspace those give independent Beta draws. Measured across-coordinate
correlation 0.0126 vs 0.0121 — zero coupling either way. So "coordinate coupling is
irrelevant" was unsupported.

*Fixed, and it improved the result*: a `shared_real` mode using the actual
`a_d = Qᵀ F(x)ᵀ e_d` directions halves the prediction error (MAE **0.015** vs
0.033) and removes the apparent optimism (signed bias -0.005 vs +0.033). The J3
headline is now stronger than first written.

### 4. `conformal_c` clamped instead of returning `+inf` — *minor*

`k = min(n, …)` substituted the maximum observed ratio when `α` was infeasible,
emitting an anti-conservative gate labelled with the requested α — contradicting
the contract `splits.alpha_feasible` exists to enforce. Reachable only at
`α < 1/(n_cal+1)`, i.e. α=0.01 on water (n_cal 92–94).

*Fixed*. No published number affected: every shipped record uses α=0.05, where
`k ≤ n` for all n_cal 92–428, so the clamp was a no-op.

### 5. Percentile bootstrap intervals under-cover — *major*

Labelled 95%, they attain (AR(1), φ=0.5, n=400): **0.718** at block_len=1,
**0.873** at 3, **0.902** at 8. This project's τ_int-derived block lengths are 2–8,
so intervals reported as 95% attain roughly **87–90%**. Every "significant" call is
therefore mildly anti-conservative.

*Documented in code*; BCa or studentized intervals are the fix if a borderline call
ever becomes load-bearing. No reported conclusion rests on a marginal interval.

### 6. `risk_coverage(reduce="max")` normalised by the anti-oracle — *minor*

Used the global max error as the "random" reference, which is what an
*adversarial* ordering achieves, not a random one — making `aurc_excess_max` look
better than it is. *Fixed* to use the expected running max under a random
accept-order.

### 7. `scheme="block"` with `block_len=1` is silently IID — *minor*

Verified identical to the explicit IID path. *Documented*; the returned dict
already carries both fields.

### 8–10. Three documentary claims did not match the records — *major*

| claim as written | actual | fix |
|---|---|---|
| "stable rank 2.91–3.08 of 7" for six systems | **1.21–4.18**; the quoted range covers 3 of 6 | corrected |
| "+0.088 to +0.230 over the **best** free signal" | **+0.067 to +0.182**; the quoted range is the *pairwise* gaps | corrected |
| "ten out of ten on the sign" | 7 significant wins; 2 of the 3 concentrated systems have intervals touching zero | restated as 7 wins / 3 nulls |

The J3 "MAE 0.03–0.05" and the stale "coupling is irrelevant" text were also
propagated into `STAGE1_GO_NO_GO.md` and are corrected there.

---

## Refuted

Eleven, including: `auroc` returning 0.0 for degenerate positive sets (cannot
occur — `top_p_mask` guarantees ≥1 positive); `autocorr_function` mapping NaN to
zero (no cache contains NaN, asserted at build); `_rank`'s scatter_reduce tie
handling (verified against brute force); several claims about split-role misuse
that misread the manifest indirection.

## Still unverified — 18

Retained in `results/records/review_candidates.json`. The substantive ones:

- ~~`j9a`'s CIs quantify structure resampling only…~~ **VERIFIED — see the
  split-draw section of `J2B_FACTOR_B.md`.** Confirmed: the intervals are 1.5–2.3×
  too narrow on 5 of 6 systems. Magnitude overstated by the reviewer (4–15 of 20
  redraws outside, not 19/20). **No conclusion changes** — every split-inclusive
  interval excludes zero with the same sign, and the effect sizes are slightly
  *larger* than the canonical draw reported.
- ~~`contiguous_block_split`'s anti-confound safeguard may not achieve what it
  claims.~~ **VERIFIED AND FIXED — confirmed, with a compounding problem the
  reviewer missed.** The calibration split was indeed **one contiguous run
  spanning only ~19%** of each trajectory. Cause: the canonical seed drew the two
  cal blocks adjacent (a ~20% event) *and* the permutation depended only on
  `seed`, not on the system — so **all 20 manifests shared the identical unlucky
  assignment**, making cross-system agreement correlated rather than independent
  evidence. Fixed by mixing a blake2b hash of the system name into the seed;
  calibration now spans **19.6–49.7%** across the six headline systems (18.8–89.4% across all fourteen) in **one or two** contiguous runs — rMD17 azobenzene still has a single contiguous calibration run, so the anti-confound goal is met for five of the six, not all. **Re-running the headline
  ablation under the corrected splits reproduces it exactly: leading-only wins
  significantly on 5 of 6, control variate on 3BPA overlapping, same signs
  throughout** (deltas +0.039, −0.010, +0.033, +0.069, +0.097, +0.033 against the
  previous +0.021, −0.005, +0.060, +0.059, +0.078, +0.036). The conclusion is now
  supported by six genuinely independent split draws rather than six correlated
  ones.
- `coverage_law`'s Beta band is derived for exchangeable draws and is quoted for
  a *blocked* split, where it may understate realised spread.
- ~~The J8 "84–87% skipped" headline may trace to the banned `06_screening.py`
  split.~~ **VERIFIED — refuted as stated, but a real adjacent error found.** The
  figure is from `07` (correct protocol, reproduces exactly). However it is the
  *workshop's* number under a superseded split scheme and superseded statistic;
  the journal pipeline gives **0.822–0.943**. Headline corrected. See below.
- Several J8 verdict-script items (P3 tests only the weak half of its
  pre-registered statement; `fit()` accepts 2-point fits; the protocol is loaded
  but never actually consulted).

## What this says about the programme

Four real errors were found by me during normal work and six more by review, in a
codebase with 20 passing tests and a deliberate provenance discipline. Two of the
review's findings were **critical**, and one had already silently deleted data. The
base rate is not low, and the cheapest defence — running the review before writing
the paper rather than after — is clearly worth its cost.

### Third pass — remaining candidates closed

Verified directly rather than by a second fan-out (session limits made another
large workflow unreliable).

**Confirmed and fixed — metric edge cases**, none of which changed a reported
number (verified by re-running the full oracle panel: max |change| = 0 on
`auroc_top05`, `auprc_top05`, `aurc_excess` and `spearman` across 232 records):

- **`auprc` broke ties by index order.** A worthless constant signal scored
  anywhere from 0.033 to 0.209 against a true value of 0.05, depending only on
  where the positives sat in the array. Now ranked by tie-averaged rank with each
  tied group contributing its expected precision: a constant signal scores exactly
  the prevalence, invariant over 200 orderings. *I nearly dismissed this one on a
  single lucky draw that returned 0.0498.*
- **`risk_coverage` had the same defect**, giving 0.843 or 1.094 for the same data
  under different input order. For the mean reduction the expectation over random
  tie-breaks is closed-form — within a tied group every member contributes the
  group mean — so that is now substituted and a worthless signal scores exactly
  1.000, matching the documented scale.
- **`auroc` returned 0.0 where it is undefined** (one class empty) — a
  plausible-looking value that would propagate silently. Now NaN.
- **`risk_coverage`'s denominator clamp** turned an undefined 0/0 into a finite
  artefact. Now NaN.
- **`autocorr_function` mapped a non-finite series to an all-zero ACF**, so
  `tau_int` came back as exactly 1.0 and the series was reported as independent —
  which would then set the guard band and block length to their minimum. Now raises.

**Refuted:** the claim that fixing the control variate at `r0=2` inflated the
leading-only advantage "by up to 2.6x". Sweeping `r0 ∈ {1,2,3}` at the same
5-lane budget moves the mean advantage from +0.044 to +0.038 — a 14% reduction —
and leading-only still wins on 5 of 6 systems. `r0=2` was already at or near
optimal for the control variate on most systems.

**Documentary fixes:** J1's α=0.01 recommendation now scoped to the systems where
it is feasible (water's n_cal = 84–94 makes it infeasible); J2a's stale "CIs not
yet attached" caveat removed; J2a's pairwise-vs-best-free ranges disambiguated.

**Moot:** ten J8 items concerning numbers in documents now suspended and
superseded by `J8_MATCHED_PAIR.md`, or concerning the verdict script since
rewritten.

### Newly confirmed in the second pass

- **`j8_water_verdict.py` loaded the protocol and never used it.** Every threshold
  was hardcoded, so pre-registration and scorer could drift silently. *Rewritten*:
  all thresholds, operators and admissibility rules now read from the YAML.
- **P3 and P4 each state two conditions; only one of each was tested.** With both
  halves scored the water verdict is **3/4, not 4/4** — P3's quantitative half
  fails (transition ratio 1/64 against a pre-registered [1/33, 1/16]). The
  mechanism is *stronger* than predicted; the magnitude prediction was wrong.
- **A quoted number traced to no record.** `J8_WATER_VERDICT.md`'s R5 table said
  "70.0 vs 76.7"; 76.7 appears in no surviving water record and is inconsistent
  with all of them. Removed, with the reason recorded in place.
- **`fit()` accepted 2-point fits.** Now requires ≥3 points.

Concretely, the tests to add: assert that a benchmark's requested lane count equals
the number of lanes executed; assert that every output path encodes the device;
assert that any two noise models being *compared* are not statistically identical;
and check every documented numeric range against the records that back it.

---

# The "84–87% of exact evaluations avoided" headline — traced

The adversarial review flagged that this recommended headline "may trace to the
banned `06_screening.py` split rather than to the journal pipeline".

**Refuted as stated.** The figure comes from `07_gate_baselines_maxcomp.jsonl`,
which carries `n_design` and `n_cal` on every row — the *correct* three-disjoint-
split protocol that `07` was written to implement. It reproduces exactly:
3BPA 0.8439, ethanol 0.8360, aspirin 0.8512, azobenzene 0.8722, i.e. 0.836–0.872,
matching `\fsGateCVSkipLo/Hi` digit for digit. The banned `06` numbers are a
separate, unused macro family.

**But the check surfaced a real problem, which is why it was worth running.** That
figure is the *workshop's*: random-frame split, max-component statistic, control
variate. The journal pipeline has since changed all three of those choices, and
the number moves:

| protocol / estimator | range over six molecular systems |
|---|---|
| workshop: `07`, max-comp, random split | 0.836 – 0.872 |
| journal: blocked split, max-comp, control variate | 0.787 – 0.949 |
| journal: blocked split, **global** statistic, control variate | 0.762 – 0.948 |
| **journal: blocked, global, leading-only (recommended)** | **0.822 – 0.943** |

Per system for the recommended construction (leading-only r0=4, 5 lanes, global):
3BPA disjoint 0.929, overlapping 0.943, same 0.879, ethanol 0.917, aspirin 0.848,
azobenzene 0.822.

So quoting "84–87%" as the *journal* paper's headline would have carried a
workshop number, computed with a superseded split scheme and a statistic the
journal explicitly recommends replacing, across a narrower spread than the journal
pipeline actually produces. The correct headline is **82–94% of exact evaluations
avoided**, and it should be attributed to the leading-only gate under blocked
splits on the global statistic.

The wider spread is itself informative: the journal range is wide because the
systems genuinely differ (azobenzene 0.822 to overlapping 0.943), whereas the
workshop's tight 0.836–0.872 partly reflected a single statistic on a single split
draw — and, per the split-draw analysis above, a single split draw understates
spread by about a factor of two.
