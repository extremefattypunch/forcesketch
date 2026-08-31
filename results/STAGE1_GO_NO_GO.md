# Stage 1 — go/no-go report

The gate the whole programme was staged around. Everything below was produced on
**CPU from committed artifacts**: no GPU, no model execution, no data download.
Total compute: well under a minute of wall time across four experiments.

`forcesketch/` is verified bit-identical to tag `workshop-v1.0`
(`git diff --quiet workshop-v1.0 -- forcesketch/`). 20 tests pass. 4,414 result
records across 31 files; 12 hashed split manifests.

---

## Verdict: **GO**, with a changed recommendation and a changed method

Three findings, in descending order of consequence.

### 1. The global statistic should replace max-component as primary — at fixed N

Exact committee force uncertainty predicts top-5% reference-force error at
AUROC 0.737–0.848 (global) versus 0.653–0.824 (max-component). The global
statistic wins in **24/24** system × error-score cells, **19/24 significantly**,
none significantly negative.

> **Later scoped by J6B.** Every system behind this gate has fixed N. On a
> variable-size pool (136,923 MPtraj crystals, 1–444 atoms) the ordering reverses
> on all four error scores, significantly, because the global statistic is
> extensive and preferentially selects large structures. The recommendation is
> sound as measured and must be stated as *"at fixed system size"*. See
> `J6B_FOUNDATION_ACQUISITION.md`.

This decides the regime call. Under max-component the project is R-B; **under the
global statistic it is R-A** on the pre-registered thresholds (3BPA ≥ 0.75, and
all three rMD17 molecules ≥ 0.70). The full application claim is available and
JCTC remains the target.

The switch is legitimate because the comparison was pre-registered in the approved
plan before any of it was run, not chosen after seeing results. One caveat kept in
the open: 3BPA's lower CI bound (0.732) sits just under 0.75, so R-A is met on
point estimates rather than uniformly on intervals.

### 2. The randomised residual sketch does not earn its lanes

At matched reverse-lane budget, the deterministic `leading_only` estimator — r0
exact leading eigen-directions of the design-set Gram matrix, no random residual —
beats the control variate on **5 of 6 systems at every budget tested (L = 4, 5, 6)**,
by +0.014 to +0.158 in skip fraction with paired block-bootstrap intervals
excluding zero. It also retains more high-uncertainty structures (worst-seed
recall 0.983–1.000 vs 0.895–1.000), has a Factor B gap of exactly 0.000 on all
six systems, and has **zero probe-seed variance by construction**.

The single exception is 3BPA overlapping, where the control variate wins
significantly but by 0.3–1.3 percentage points in a regime where both already
skip >93%.

This follows from the project's own spectral measurement and from the argument
already in `control_variate.py`'s docstring: spectral concentration is what
*hurts* a random projection.

> **Corrected after review.** An earlier version of this paragraph quoted
> "stable rank 2.91–3.08 of 7" as describing all six systems. That range is the
> workshop paper's, and it covers only 3 of the 6: measured srank(FQ) is
> 1.212 (3BPA overlapping), 2.908, 2.918, 2.993, 3.079, and 4.175 (3BPA same) —
> a range of **1.21–4.18**. The wider spread is what later made the
> spectral-concentration test possible, so the corrected number is the more useful
> one.

At fixed budget, spend every lane on exact leading directions. **Randomisation was
solving a problem a deterministic basis solves better.** The method section gets
simpler, not larger.

### 3. The negative result is now predicted, not merely measured

The Haar projection law `v̂_d/v_d = (r/K)·Beta(K/2,(r−K)/2)` is verified against
the shipped generator to 4–5 decimals in both moments. Injecting that noise into
the exact variances **reproduces the measured top-5% recall across six systems,
three budgets and a 3× range in D**, with no free parameters and mean absolute
error **0.015** (corrected — see below).

Two corrections to the published mechanism fall out. A common multiplicative bias
cannot move a rank statistic, so **the damage is estimator variance, not bias**.
> **Corrected after review.** This previously claimed that "coordinate coupling is
> irrelevant" because the shared-subspace and independent noise models agreed. The
> two models were **identical by construction** — the shared branch used isotropic
> coordinate directions, which give independent Beta draws — so their agreement
> established nothing. Using the *real* head-space directions halves the prediction
> error (MAE 0.015 vs 0.033) and removes the apparent optimism. Coupling matters.
> See `J3_EXTREME_VALUE.md`.

The budget bound makes it decisive: uniform componentwise accuracy at ε = 0.1
needs K ≈ 5,600–8,000 against an available `K ≤ M−1 = 7`. Even at ε = 0.5 it needs
~224–322. The result is *unreachable by two to three orders of magnitude*, not
merely unreached.

---

## Findings that constrain the claims

**The ceiling is low.** Factor B being ~0 says the gate is faithful, not that the
policy is good. Selecting by uncertainty enriches for high-error structures by
only **1.8–5.3×** over random, and on aspirin under max-component by **0.96× —
indistinguishable from random**. AUROC 0.74–0.85 describes bulk ranking;
acquisition is a tail decision and the tail is worse. This belongs in the abstract,
not buried.

**Force UQ does earn its backward passes** — on molecules — by **+0.067 to
+0.182** AUROC over the *best* free signal available for each system
(head-energy spread or mean-force magnitude, whichever is stronger there).

> **Corrected after review.** This previously read "+0.088 to +0.230 … significant
> in 8/10", which is the range of the *pairwise* gaps against each free signal
> taken separately. The gain over the **best** free signal is the honest quantity
> and it is smaller: +0.067 (ethanol) to +0.182 (3BPA overlapping). The pairwise significance count is **9 of 10** after the CI records were
> regenerated with a recorded seed (aspirin's `force_norm` resolves above zero);
> see the correction in `J2A_ORACLE_PANEL.md` §3.

Modest, real, and never previously quantified for multi-head committees — and
**it does not hold on water**, where the committee fails to significantly beat the
free signal in-distribution (see `J5_WATER.md`). Scope it to system class.

**Committee diversity runs inverse to error detection.** The `same` committee
(least diverse, ~26× overconfident in magnitude) ranks error *best* at 0.848; the
`disjoint` committee (most diverse) worst at 0.773. Magnitude calibration and rank
informativeness are different properties. Free, unplanned, and needs a mechanism
before it is claimed.

**Trajectory leakage is a non-issue here, and that is now provable.** Measured
τ_int is 1.0–4.0; blocked and random test splits differ by **0.027** mean absolute
(max 0.053) with no consistent sign — 3 positive, 3 negative, and the largest
difference favours the *blocked* split, which is the opposite of a leakage
signature. The workshop's random-frame split was not materially
leaky. Blocked splits are adopted anyway, since they cost 16–96 frames.

---

## Defects found and their status

| # | defect | status |
|---|---|---|
| 1 | `06_screening.py` fits τ, basis and `c_α` on one split | superseded; journal code uses hashed three-role manifests |
| 2 | `04_bootstrap_ci.py` CV numbers ~20% in-sample | superseded by J2b on held-out test |
| 3 | two divergent split implementations | replaced by one; both banned by AST test |
| 4 | random-frame split on trajectory data | measured, quantified, non-issue (above) |
| 5 | control variate has no max-component fidelity record | produced in J2b |
| 8 | no `workshop-v1.0` tag | created |
| **11** | `checkpoint_param_sha256` hashes `str(t.dtype)` despite documenting dtype-invariance, so **no cache hash matches `checkpoints.json`** | found while validating caches; one-line fix pending in journal package |

Defects 6, 7, 9, 10 are Stage-2/GPU-side and unchanged.

---

## What Stage 1 could not answer

- **`E` (head energies) is missing from the `overlapping` and `same` caches**, so
  the free `energy_std` baseline covers only four of six systems. ~2 GPU-minutes
  to regenerate once `fs-gpu` exists.
- **Every result is M = 8, one architecture, one spectral regime.** The
  leading-only advantage is predicted to shrink as the spectrum flattens — that is
  a falsifiable prediction, and the naive committee, water and larger M are
  exactly the tests.
- **No timing.** Every claim here is about decision quality. Whether skipping
  76–95% of exact evaluations saves wall-clock is Stage 2, and remains a genuine
  kill gate for the acceleration framing.

## Recommended next steps

1. **Stage 2 environment bring-up and the J8 smoke test**, early, because it is a
   kill gate rather than a week-13 deliverable. `torch 2.13.0+cu129` on the sm_120
   Blackwell nodes first.
2. **Regenerate the two 3BPA caches with `E`**, and add water — its D = 576 makes
   the extreme-value law testable by blind prediction over a 21× range in D, and
   it tests the leading-only prediction in a new spectral regime.
3. **The naive (independently trained) committee**, as the control for both the
   spectral-concentration prediction and the shared-trunk scope statement.
4. Only then active learning, on the global statistic and the leading-only gate.
