# J6B — foundation model, real materials pool: the scoping result

**Pre-registered** in `protocols/j6b_foundation_acquisition.yaml`, sha256
`2d8234629a937e97…`, registered with zero J6B records on disk. The scorer refuses
to run if the protocol has changed.

**Verdict: 2 of 5 predictions confirmed (Q1, Q3).** This is the harshest result in
the project and the most useful one, because it is the only experiment run in the
setting the method is actually pitched at.

*(gpt_plan calls active learning "J6"; this repository's J6 is the naive-committee
control, so the acquisition work is J6B.)*

## Setting

MACE-MP `agnesi_medium` fine-tuned with 8 disjoint committee heads and
`optimize_readouts_only: true` — a **frozen foundation trunk with committee
readouts**, which is the deployment story the whole method is for. Candidate pool:
**136,923 periodic MPtraj crystals**, 89 elements, **1–444 atoms**, 160 million
periodic image edges. Every other system in this project is one chemistry at one
fixed size.

Scanned in 3,436 s on one A100 (39.9 structures/s, float64).

**Scoring model chosen for contamination, not convenience.** The pool file is
`mp_traj_qbc_79th_not_selected.xyz` — the *complement* of the published QBC
selection — so the QBC model's 8,000 training structures are provably disjoint
from it (0 of 8,000 match by content hash). The alternatives are contaminated:
`random`'s training set has **7,498 of 8,000** structures inside the pool and
`max_mean_force` **6,236**. Scoring with either would have scored thousands of
structures the committee was fitted on.

**Limit recorded up front:** because the pool is the QBC selection's complement, a
head-to-head against the published QBC *acquisition* is impossible — their picks
are excluded by construction. That comparison is not attempted and not claimed.

## Q1 — PASS. Factor A transfers, and that is a real result

| signal | e_max | e_rmse | e_maxcomp | e_q95 |
|---|---|---|---|---|
| exact max-atom | **0.836** | 0.819 | 0.824 | 0.806 |
| exact max-component | 0.821 | 0.808 | 0.822 | 0.805 |
| exact global | 0.798 | 0.769 | 0.793 | 0.759 |
| leading-only (5 lanes) | 0.791 | 0.763 | 0.786 | 0.753 |
| **`‖f̄‖` (free)** | **0.943** | **0.933** | **0.942** | **0.930** |
| `energy_std` (free) | 0.481 | 0.447 | 0.483 | 0.450 |
| random null | 0.499 | 0.498 | 0.498 | 0.498 |

Committee disagreement ranks reference-force error at AUROC 0.76–0.84 across
136,923 crystals spanning 89 elements, with a frozen trunk it never trained. The
random null sits at 0.499, which is the check that the pipeline is sound at this
scale. **Factor A is not a property of small organic molecules.**

## Q3 — PASS, at twice the predicted magnitude. The global statistic is extensive

Predicted the global statistic's acquisitions would be ≥25% larger than
max-component's. They are **150% larger**:

| signal | median atoms acquired (budget 8000) | corr(signal, N) |
|---|---|---|
| **exact global** | **70** | **0.631** |
| `energy_std` | 68 | 0.577 |
| leading-only | 66 | 0.578 |
| exact max-atom | 34 | 0.365 |
| exact max-component | 28 | 0.285 |
| `‖f̄‖` | 30 | 0.173 |
| random | 20 | 0.002 |
| *(pool median)* | *20* | — |

The global statistic sums `v_d` over 3N coordinates, so it grows with system size
whether or not the model is uncertain. Acquiring by it means acquiring **the
biggest structures in the pool** — median 70 atoms from a pool whose median is 20.
At budget 1000 it is worse still: median 80.

**This is why Q2 fails.** Global loses to max-component on *all four* error
scores, every one significant:

| error score | global − max-component | 95% CI |
|---|---|---|
| e_max | −0.023 | [−0.027, −0.020] |
| e_rmse | −0.039 | [−0.043, −0.036] |
| e_maxcomp | −0.029 | [−0.033, −0.026] |
| e_q95 | −0.045 | [−0.049, −0.041] |

The project's central constructive recommendation — *prefer the global statistic*
— held 24 of 24 on molecules and 15 of 16 on naive committees, and **reverses on
variable-size systems**. That is not a contradiction: every earlier system had
fixed N, where an extensive statistic and an intensive one differ only by a
constant and the pathology is invisible **by construction**. The recommendation
was never wrong where it was measured; it was measured only where it could not
fail.

**The recommendation must therefore be restated with its scope: the global
statistic is preferable at fixed system size, and must not be used to rank
structures of differing size without normalisation.** Dividing by 3N is the
obvious repair and is untested here — it is the first thing to try next, not
something to assert.

## Q4 — FAIL. The selection-overlap bound does not hold

The plan proposed bounding the active-learning difference without retraining: if
the gate's acquired set overlaps exact MHC's at Jaccard ≥ 0.95 (protocol: ≥ 0.90),
the downstream difference is bounded. Measured:

| budget | acquisition rate | Jaccard(leading-only, exact global) |
|---|---|---|
| 8000 | 6.5% | **0.794** |
| 1000 | 0.8% | **0.718** |

So the bound is not available.

> **Corrected.** This said "roughly one in five acquired structures differs",
> which misreads the metric: 1 − Jaccard = 20.6% is the share of the *union* held
> by only one of the two sets. Of the 8,000 structures the gate actually acquires,
> **918 differ from the exact-global choice — 11.5%, about one in nine**
> (intersection 7,082, union 8,918). The bound still fails at the pre-registered
> 0.90 threshold; only the gloss was wrong.

Retraining would be required to say what that difference costs — which is exactly
the expensive experiment the bound was meant to avoid.

> **A methodological trap worth recording.** On a 20,000-structure partial scan
> this same measurement gave Jaccard 0.913 and Q4 "passed". That was an artifact:
> 8,000 of 20,000 is a **44% acquisition rate**, and any two ranking signals agree
> when you take nearly half the pool. At the real rate the number is 0.794. The
> partial was never quoted, and the lesson generalises — overlap metrics must be
> reported with their acquisition rate or they mean nothing.

## Q5 — FAIL, decisively, and this is the finding

The free signal beats the committee by a wide, significant margin:

**`‖f̄‖` 0.943 vs exact global 0.798** — gap −0.145, CI [−0.149, −0.140].

`‖f̄‖` is the max over atoms of the committee mean-force norm. It costs **nothing**
— it is already computed in the forward pass — and it is `max_mean_force`, one of
the *published acquisition baselines shipped with this very experiment*. Eight
backward passes buy a signal 0.145 AUROC worse than one that is free.

Two post-hoc diagnostics (not pre-registered, not scored) test whether this is an
artifact. Neither rescues the committee:

- **Size-stratified** — AUROC within atom-count quartiles, removing the
  extensivity confound: `‖f̄‖` 0.933, max-component 0.798, global 0.794. The free
  signal's win is *not* a size effect.
- **Force-stratified** — AUROC within force-magnitude quintiles, holding the
  confound itself roughly fixed: `‖f̄‖` 0.710, max-component 0.656, global 0.632,
  `energy_std` 0.511, random null 0.499.

The force-stratified row is the fair one, and it says something more balanced than
the headline: **the committee does carry genuine information beyond force
magnitude** (0.63–0.66 against a 0.499 null), but a free proxy is still at least as
good. Its limit: quintiles are wide, so force is not fully controlled.

A third diagnostic — AUROC for *relative* error `e_max/‖f̄‖` — puts every
**informative** signal below 0.5 (0.31–0.42; the random null sits at 0.502, as it
should). It is reported for completeness but should not be leaned
on: `‖f̄‖` sits in that ratio's denominator and is mechanically anti-correlated
with it.

`energy_std` deserves its own sentence: **0.481, below chance**, and extensive
(corr 0.577 with N). The one-forward-pass free signal is worthless here.

---

## What this does to the paper

1. **The gate's value proposition is unchanged and untouched.** Q1 shows exact
   committee UQ is informative at this scale; nothing here bears on whether a
   sketch reproduces it cheaply. But Q4 shows the *specific* cheap estimator we
   recommend does not reproduce exact acquisitions closely enough to skip
   retraining, at realistic acquisition rates.
2. **The global-statistic recommendation needs an explicit fixed-N scope.** This
   is the third system-regime to qualify it (water PIMD, now MPtraj), and the
   first with a clean mechanism: extensivity.
3. **The honest headline for practitioners is uncomfortable and should be
   stated anyway**: on a real foundation-model materials pool, the max mean-force
   norm outranks committee disagreement for finding high-error structures, at zero
   cost. Committee UQ should be justified against that baseline in every future
   claim, not against a random null.
4. **Extensivity is a general defect worth a section**, not a footnote. Any
   acquisition score that sums over atoms will preferentially acquire large
   structures; the published baselines already show it mildly (`max_mean_force`
   median 28, QBC 30, random 20) and our global statistic shows it severely (70).

## Reproduction and limits

- `experiments/j6b_foundation_scan.py` (scan) → `experiments/j6b_foundation_verdict.py`
  (score). Ragged structures make an `[S, A, 3, M]` cache impossible, so each
  structure is compressed on the fly to its 7×7 centred head-space Gram `A_s`;
  every global-statistic estimator is a function of `A_s` alone. The identity is
  asserted against the direct computation before the scan starts (1.8e-13 and
  4.4e-13). Max-component is not a function of `A_s` and is accumulated separately.
- **A silent-zero bug this caught.** ASE parses this file's `forces` column into a
  `SinglePointCalculator`, while MACE reads `arrays["REF_forces"]` and substitutes
  **zeros** when absent. The first smoke run had reference error equal to the
  predicted force exactly, making `e_max` identical to `‖f̄‖` and every AUROC a
  comparison of a signal against itself. There is now a guard that raises on
  identically-zero reference forces.
- One committee, one budget pair, one pool. No retraining was done, so no claim is
  made about downstream model quality — only about which structures get selected.
- The bootstrap needed chunking to run at this n: a 10,000 × 136,923 index tensor
  OOM-kills the process. `paired_difference` now chunks above a 40M-element budget;
  every previously computed interval used at most 21M and is unchanged.
