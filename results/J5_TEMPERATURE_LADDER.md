# J5 — the 3BPA temperature ladder: a hypothesis falsified, and a better one in its place

Nine head-force caches (3 committees × 300/600/1200 K), 9m20s on an A100-80GB.
All three committees were trained on 300 K data, so temperature is a clean,
graded distance from the training distribution with **model, molecule, atom count
and D all held fixed**. The workshop used only the 1200 K set.

This run also closes a Stage-1 gap: the committed `overlapping` and `same` caches
predated the addition of head energies, so `energy_std` covered only four of ten
system-regimes. All nine new caches carry `E`.

---

## The result

| committee | T (K) | force RMSE | disagreement | overconfidence | **AUROC (global)** | 95% CI | force_norm (free) |
|---|---|---|---|---|---|---|---|
| disjoint | 300 | 50.9 | 20.3 | 2.5× | 0.751 | [0.690, 0.809] | 0.546 |
| disjoint | 600 | 94.0 | 25.0 | 3.8× | 0.714 | [0.655, 0.766] | 0.601 |
| disjoint | 1200 | 200.9 | 33.6 | 6.0× | 0.773 | [0.732, 0.817] | 0.637 |
| overlapping | 300 | 42.4 | 5.0 | 8.4× | 0.808 | [0.762, 0.858] | 0.558 |
| overlapping | 600 | 79.9 | 6.9 | 11.5× | 0.767 | [0.720, 0.815] | 0.612 |
| overlapping | 1200 | 177.2 | 13.1 | 13.5× | 0.788 | [0.750, 0.833] | 0.606 |
| same | 300 | 43.4 | 4.4 | 9.9× | 0.773 | [0.704, 0.825] | 0.552 |
| same | 600 | 82.5 | 5.4 | 15.4× | 0.833 | [0.790, 0.871] | 0.612 |
| same | 1200 | 187.1 | 7.3 | 25.8× | 0.848 | [0.812, 0.879] | 0.702 |

## 1. Retraction: Factor A does **not** track distribution shift

`J5_WATER.md` §3 raised the hypothesis that Factor A tracks degree of
distribution shift, on the strength of water MD (0.647–0.689) versus water PIMD
(0.970–0.976). The ladder was named there as its test. **It fails.**

Force error grows **4×** across the ladder (50.9 → 200.9 meV/Å for `disjoint`).
Factor A moves by at most 0.08 within any committee, non-monotonically for two of
three, with intervals overlapping throughout:

| committee | 300 → 600 → 1200 K | monotone? |
|---|---|---|
| disjoint | 0.751 → 0.714 → 0.773 | no |
| overlapping | 0.808 → 0.767 → 0.788 | no |
| same | 0.773 → 0.833 → 0.848 | yes |

**1 of 3.** The hypothesis is withdrawn. Water's MD→PIMD jump therefore needs a
different explanation — most plausibly that quantum delocalisation produces a
distinct, more separable error tail, rather than that the distribution moved. The
water write-up is annotated accordingly.

## 2. What replaces it, and it is cleaner

The ladder does show one thing monotonically, in **3 of 3** committees:

| committee | overconfidence, 300 → 600 → 1200 K |
|---|---|
| disjoint | 2.5× → 3.8× → **6.0×** |
| overlapping | 8.4× → 11.5× → **13.5×** |
| same | 9.9× → 15.4× → **25.8×** |

So the controlled experiment separates two properties the water result had
conflated:

> **Distribution shift degrades the committee's magnitude calibration
> systematically and substantially, while leaving its rank informativeness
> essentially unchanged.**

That is more useful than the hypothesis it replaces, and it has a direct
consequence for the method: a split-conformal gate that refits `c_α` per regime
is exactly the right construction, because it depends only on ranks and on a
recalibrated multiplier — while any use of committee σ as an *absolute* error
estimate degrades by 2.5–2.6× across this ladder alone.

## 3. Committee construction matters more than distribution shift

Within-committee variation across a 4× change in error is ≤ 0.08 AUROC. But
*between* committees at fixed temperature the spread is comparable or larger
(0.751 / 0.808 / 0.773 at 300 K; 0.773 / 0.788 / 0.848 at 1200 K), and the
ordering is stable — `same` ≥ `overlapping` ≥ `disjoint` at the extrapolative end.

This corroborates the inverted-diversity finding from J2a on independent data:
the *least* diverse committee (`same`, heads differing only by initialisation, and
by far the most overconfident at 25.8×) ranks reference error the *best*. Two
properties usually assumed to travel together — well-scaled uncertainty and
useful uncertainty ranking — are close to anti-correlated here.

## 4. An uncomfortable finding: the free signal gains on the committee under shift

Mean-force magnitude costs nothing and improves with temperature in all three
committees (0.546 → 0.637; 0.558 → 0.606; 0.552 → 0.702). The committee does not.
So the margin **narrows** exactly where acquisition operates:

| committee | gain at 300 K | gain at 1200 K |
|---|---|---|
| disjoint | +0.205 | +0.136 |
| overlapping | +0.250 | +0.182 |
| same | +0.221 | +0.146 |

The case for paying for backward passes is therefore *weakest* in the
extrapolative regime, which is the regime the method is intended for. Combined
with water — where in-distribution the committee did not significantly beat
`force_norm` at all — this is the sharpest constraint on the paper's
physical-relevance claim, and it belongs in the abstract rather than the
limitations.

## What this changes

1. `J5_WATER.md` §3's hypothesis is retracted and annotated.
2. The calibration-versus-ranking decomposition replaces it, backed 3/3 on a
   controlled ladder rather than inferred from two water regimes.
3. The `energy_std` gap is closed: all ten original system-regimes plus these nine
   now carry head energies.
4. Nineteen system-regimes now exist in total, spanning D = 27–576, three
   committee constructions, four temperatures, molecular and condensed-phase,
   periodic and non-periodic.
