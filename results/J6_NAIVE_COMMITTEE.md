# J6 — naive committees: what survives without the shared trunk

**Pre-registered** in `protocols/j6_naive_committee.yaml`, sha256
`d84289909e4d1a9f…`, registered at 2026-08-16T01:47:27Z with **zero naive caches
on disk** (`results/records/j6_protocol_registration.json` records the empty
list). The scorer refuses to run if that file has changed since.

**Verdict: 3 of 5 predictions confirmed** (P2, P3, P4). Both failures taught more
than the passes did, and one of them undermines a number this project was
otherwise about to report with a straight face.

Every naive number comes from the *same* analysis scripts as the shared-trunk
numbers, pointed at a different cache directory (`--cache-dirs`). No new analysis
code computes any scored quantity — otherwise the comparison would be between two
implementations rather than between two committee constructions.

---

## What was built

Four naive committees, each **8 independently trained MACE models** from the
Zenodo archive, evaluated on the same test frames as their shared-trunk
counterparts:

| naive committee | construction | matched shared-trunk system |
|---|---|---|
| `3bpa-naive-same` | identical train/valid files, seeds 0–7 | `same` (heads see identical data) |
| `3bpa-naive` | `--valid_fraction=0.2` per seed → different 80% each | `overlapping` (heads see overlapping subsets) |
| `water-naive` MD | as above, at D = 576 and periodic | `water-overlapping` MD |
| `water-naive` PIMD | as above, under distribution shift | `water-overlapping` PIMD |

The pairs are matched on architecture, training data, and evaluation frames, so
**construction is the only variable**. rMD17's naive committees exist in the
archive but their `train_all.xyz` / `test_all.xyz` were never fetched, so they are
out of scope; that omission is recorded in the protocol rather than left silent.

## Stated in advance, because it is a derivation and not a result

**Acceleration does not transfer to naive committees, and this was written into
the protocol before any measurement.** With M independent models there is no
shared trunk, so `g_k = F w_k` needs all M columns of F — M separate
forward+backward passes. No sketch of the head space can be cheaper than the exact
quantity it approximates. Everything below is therefore about **decision
quality**, which is linear algebra on `F` and indifferent to how `F` was made.

---

## P4 — PASS, and by more than expected

Overconfidence ratio = committee force RMSE ÷ median per-atom disagreement.
Computed for both halves of every pair **by one function**, since a matched
comparison assembled from two definitions is worthless.

| system | naive | shared-trunk | |
|---|---|---|---|
| 3BPA, identical data | **2.58×** | 25.77× | 10.0× better calibrated |
| 3BPA, resampled data | **2.56×** | 13.52× | 5.3× |
| water MD | **2.28×** | 6.40× | 2.8× |
| water PIMD | **3.08×** | 24.08× | 7.8× |

The shared-trunk committee understates its own error by 6–26×; the naive ensemble
by 2.3–3.1×. Neither is calibrated, but they are not in the same league, and the
gap is widest exactly where heads share the most (identical training data).

## P3 — PASS: the global statistic generalises

Global beats max-component in **15 of 16** cells (threshold: 14/16), the single
exception being `water-naive` MD's `e_maxcomp` at −0.044. So the project's main
constructive recommendation — *use the global statistic, not the max-component
rule* — is a property of committees, not of multi-head architecture. That is a
strictly larger claim than the workshop paper made, and it now has an out-of-
architecture confirmation.

## P2 — PASS: the srank rule survives out of range

J2B's rule (leading-only beats the control variate iff stable rank > 2.5) was read
off ten shared-trunk systems. Naive committees sit outside that range, and the
rule was scored blind:

| system | srank | rule predicts | measured LO − CV | |
|---|---|---|---|---|
| 3BPA, identical data | 4.998 | + | **+0.018** [+0.010, +0.026] | ✓ |
| 3BPA, resampled | 4.514 | + | −0.005 [−0.012, +0.003] | ✗ |
| water MD | 3.888 | + | **+0.095** [+0.066, +0.125] | ✓ |
| water PIMD | 1.275 | ≤ 0 | −0.006 [−0.015, +0.000] | ✓ |

3 of 4, meeting the pre-registered bar. The miss is a **null, not a reversal** —
its interval straddles zero — so the honest reading is that the rule predicted a
win where there was no difference, not that it got the direction backwards. Worth
saying plainly: had the bar been 4 of 4 this would have failed.

## P1 — FAIL: the mechanism story is wrong as stated

Predicted with *high* confidence that removing the shared trunk flattens the head
space. It does, on three of four pairs — and then water PIMD goes the other way:

| pair | naive srank | shared-trunk srank | |
|---|---|---|---|
| 3BPA, identical data | 4.998 | 4.175 | ✓ |
| 3BPA, resampled | 4.514 | 1.212 | ✓ |
| water MD | 3.888 | 2.944 | ✓ |
| **water PIMD** | **1.275** | **2.032** | ✗ |

Under distribution shift the *independently trained* ensemble is the more
concentrated one — 78% of its head-space energy in a single direction. The
prediction required all four pairs and it does not get them. "The shared trunk
concentrates the head space" is a fair description of in-distribution behaviour
and is **false under shift**, where whatever the models disagree about collapses
onto one mode regardless of how they were trained.

## P5 — FAIL, and this is the finding that matters

Factor A was required to reach AUROC ≥ 0.70 on all four naive systems. It reaches
0.873 and 0.832 on 3BPA and **0.685 / 0.643** on water. Taken alone that reads as
"committee UQ works for molecules, not for water". The matched controls say
something quite different:

| system | naive | shared-trunk | difference |
|---|---|---|---|
| 3BPA, identical data | 0.873 | 0.848 | **+0.026** |
| 3BPA, resampled | 0.832 | 0.788 | **+0.044** |
| water MD | 0.685 | 0.647 | **+0.039** |
| water PIMD | 0.643 | 0.976 | **−0.332** |

On three of four systems the naive committee is the **better** error detector. The
0.70 bar was set from molecular systems, and water MD fails it for *every*
construction — that is a property of the system, not of the committee. P5 as
written asked naive committees to clear a bar their own matched controls also
miss, which is a defect in the prediction and is recorded as one.

Water PIMD is the real result, and it is a warning:

| water PIMD | naive | shared-trunk |
|---|---|---|
| committee force RMSE | **81.4 meV/Å** | 219.7 meV/Å |
| AUROC, exact global | 0.643 | 0.976 |
| AUROC, free `‖f̄‖` | 0.575 | **0.943** |
| AUROC, free `energy_std` | 0.692 | 0.885 |

**The shared-trunk committee's spectacular shift-regime Factor A is largely a
symptom of the model being badly wrong.** On PIMD it is 2.7× less accurate than
the naive ensemble, and its errors are so dominated by a single force-magnitude-
correlated failure mode that a *free* signal — the norm of the mean force,
requiring no committee at all — scores 0.943. When the naive ensemble removes most
of that failure (RMSE 220 → 81), error ranking becomes hard again and every signal
falls back toward chance.

So the sentence "committee force UQ is *most* valuable under distribution shift",
which the water PIMD number invited, does not survive its own control. The
defensible version is: **committee UQ is easiest to demonstrate where the model is
failing badly, and much of that apparent skill is available for free from the mean
force.** Any shift-regime claim must be reported next to the free-signal baseline
and the RMSE, or it flatters the method.

---

## Consequences

1. **The global-over-max-component recommendation generalises** beyond multi-head
   models (P3). Report it as a committee-level finding.
2. **Multi-head committees trade accuracy for speed**, and the trade is not small:
   naive ensembles are more accurate on all four systems (158 vs 187, 167 vs 177,
   30 vs 40, 81 vs 220 meV/Å). The paper's speed claims must sit beside this.
3. **Recheck the distribution-shift section.** The J5 water PIMD Factor A number
   should be reported with its free-signal baseline and the accuracy gap, and the
   "UQ shines under shift" framing dropped.
4. **The srank rule holds up** in a regime it was not fitted on, which upgrades it
   from an observation to a usable predictor — but state it with the null.
5. **P1's mechanism claim needs restating** as an in-distribution statement.

## Defects and limits recorded

- The registered protocol named two 3BPA systems by variant only
  (`3bpa-naive`) while caches carry the split too. The protocol is hash-frozen,
  so resolution happens in the scorer and is required to be unambiguous — a bare
  prefix match would have let `3bpa-naive` swallow `3bpa-naive-same`, scoring a
  prediction against a committee it did not name.
- `results/records/j5_spectrum.json` had **no generating script**. It does now
  (`experiments/j6_spectrum.py`), validated to reproduce all ten shared-trunk
  systems to 1.1e-14 before any naive number was computed from it. Recovering
  `effective_rank` took one wrong guess worth recording: it is the participation
  ratio `(Σμ)²/Σμ²`, not the entropy exponential, which reproduces nothing here.
- The naive AUROCs are on the **full evaluation set**, matching J2A's protocol —
  legitimate because nothing is fitted, but the gate comparison (P2) does use the
  three-role blocked splits, whose manifests were generated for the four new
  systems. All 20 pre-existing manifests re-generated **bit-identically**.
