# J5 — bulk liquid water: caches, and what they say

Four head-force caches built on the A100 in **4.5 minutes total**
(`experiments/j5_water_caches.py`), schema-identical to the frozen ones so every
downstream experiment reads water with no special-casing.

| cache | S | A | D=3N | force RMSE | median disagreement | **overconfidence** |
|---|---|---|---|---|---|---|
| water-disjoint, MD 300 K | 500 | 192 | 576 | 38.9 meV/Å | 5.7 meV/Å | 6.8× |
| water-disjoint, PIMD 300 K | 500 | 192 | 576 | **160.9** | 9.8 | **16.5×** |
| water-overlapping, MD 300 K | 500 | 192 | 576 | 40.5 | 6.3 | 6.4× |
| water-overlapping, PIMD 300 K | 500 | 192 | 576 | **219.7** | 9.1 | **24.1×** |

## Periodic correctness, asserted rather than assumed

Every failure mode here is silent — a missing `pbc` flag or a mismatched key
yields finite, wrong numbers with no exception. Checked before any force was
computed: all frames `pbc = (True, True, True)`; cell constant at 12.42 Å cubic;
minimum-image valid (12.42 > 2·r_max = 12.0, margin **0.42 Å**);
`energy_ref`/`forces_ref` present (water already uses the keys the pipeline wants,
unlike 3BPA — a trap, since the careful rename in `00_prepare_data.py` does not
apply and one might assume it does).

The decisive check: **94.8% of graph edges are periodic-image edges** (48,778 of
51,474 for three frames). The periodic path is genuinely exercised. After
building, the estimator stack was cross-checked against the direct per-head
variance at `< 1e-9` relative, as the frozen script does.

Water carries **~17,150 edges per structure against 3BPA's 520 — 33× the work**,
which is the axis the J8 smoke test identified as deciding whether lane reduction
pays.

## 1. MD versus PIMD is a severe, and severely under-detected, distribution shift

Same temperature, same system, same cell — only quantum nuclear effects differ.
Force error grows **4.1–5.4×** (38.9 → 160.9, 40.5 → 219.7 meV/Å). Committee
disagreement grows only **1.4–1.7×** (5.7 → 9.8, 6.3 → 9.1).

So the committee **under-responds to the shift by roughly a factor of three**, and
its overconfidence degrades from ~6.5× in-distribution to 16.5–24.1× under shift.
For a split-conformal gate this is exactly the exchangeability violation the
method's coverage guarantee excludes: calibrate on MD, deploy on PIMD, and `c_α`
is fitted to a ratio distribution that has moved by 3×.

The in-distribution ratio is worth noting: 6.4–6.8× on water against
**2.5–9.9×** on 3BPA at 300 K.

> **Corrected.** This compared water's in-distribution ratio against 3BPA's
> **1200 K** figure (206/33 = 6.2×) — the regime §3 of this same document labels
> *extrapolative* — and concluded the two systems showed "the same scale of
> overconfidence". The genuinely in-distribution 3BPA numbers are the 300 K rungs
> of the temperature ladder: 2.50× (disjoint),
> 8.42× (overlapping), 9.91× (same).
> Water's 6.4–6.8× sits inside that span but the tidy agreement was a coincidence
> between two different regimes, and the spread across 3BPA committees at one
> temperature (2.5× to 9.9×) is itself
> wider than the water range.

## 2. Factor A on water: the weakest and the strongest cases in the project

AUROC for top-5% `e_max`, with block-bootstrap intervals, alongside the **free**
mean-force-magnitude baseline:

| system | exact global | 95% CI | force_norm (free) | gap | 95% CI | sig? |
|---|---|---|---|---|---|---|
| water-disjoint MD | 0.689 | [0.582, 0.799] | 0.620 | +0.069 | [−0.082, +0.254] | **no** |
| water-overlapping MD | 0.647 | [0.540, 0.767] | 0.635 | +0.012 | [−0.128, +0.193] | **no** |
| water-disjoint PIMD | **0.970** | [0.955, 0.986] | 0.947 | +0.023 | [+0.007, +0.035] | yes |
| water-overlapping PIMD | **0.976** | [0.960, 0.989] | 0.943 | +0.033 | [+0.014, +0.054] | yes |

Two uncomfortable results that must be reported plainly.

**In-distribution water is the weakest Factor A in the project** (0.647–0.689), and
**the committee does not significantly beat a free signal there.** Mean-force
magnitude costs nothing — it is already computed for MD — and its interval
overlaps the committee's. On these systems, eight backward passes buy nothing
detectable. (`energy_std` is worse still, at 0.418–0.495 — at or below chance.)

**Under shift the committee is excellent — but so is the free signal.** At
AUROC 0.970 the committee is genuinely detecting the bad structures, yet
`force_norm` reaches 0.943–0.947 on its own. The advantage is significant but
only +0.023 to +0.033 for roughly 8× the compute.

This is materially harsher than the molecular systems, where the committee beat
the best free signal by **+0.067 to +0.182** (the +0.088–+0.230 span is the
*pairwise* comparison against each free signal separately, significant in 9 of 10).
It has to go in the paper as-is.

## 3. An emerging unifying pattern — ~~hypothesis~~ **RETRACTED, see J5_TEMPERATURE_LADDER.md**

> **This hypothesis was tested on the 3BPA temperature ladder and failed.** With
> model, molecule, atom count and D held fixed and only the evaluation temperature
> varied over a 4× range in force error, Factor A moved by ≤0.08 and
> non-monotonically in 2 of 3 committees. What *does* rise monotonically in 3 of 3
> is **overconfidence** (2.5→6.0×, 8.4→13.5×, 9.9→25.8×). The correct statement is
> that shift degrades *magnitude calibration* while leaving *rank informativeness*
> essentially unchanged — so water's MD→PIMD jump needs a different explanation,
> most plausibly a more separable error tail from quantum delocalisation rather
> than the distribution having moved. The original text is kept below for the
> record.

Ordering every system by Factor A:

| system | regime | AUROC |
|---|---|---|
| water MD | in-distribution | 0.647–0.689 |
| rMD17 (held-out frames, molecules seen in training) | in-distribution | 0.737–0.830 |
| 3BPA @ 1200 K (model trained at 300 K) | extrapolative | 0.773 |
| water PIMD | quantum-nuclear shift | 0.970–0.976 |

The extremes track **degree of distribution shift** rather than system size,
dimension or chemistry, which would make the scoping statement crisp:
*committee force uncertainty is informative precisely when you are extrapolating,
which is also when acquisition matters.* But rMD17 sits above water MD despite
both being in-distribution, so shift is not the only driver, and ten
system-regime combinations is too few to assert a law. Recorded as the hypothesis
the naive committee and the rMD17 leave-one-molecule-out committees are the
natural tests of.

## 4. The global-vs-max-component advantage does **not** extend to water

On the six molecular caches the global statistic beat max-component in 24/24
cells. On water it wins **11 of 16**, with individual deltas ranging −0.036 to
+0.122:

| system | e_max | e_maxcomp | e_rmse | e_q95 |
|---|---|---|---|---|
| water-disjoint MD | +0.023 | −0.036 | +0.122 | +0.082 |
| water-disjoint PIMD | −0.002 | −0.002 | +0.008 | +0.044 |
| water-overlapping MD | +0.033 | +0.014 | +0.049 | +0.097 |
| water-overlapping PIMD | −0.008 | −0.012 | +0.001 | +0.102 |

The pattern is interpretable: the advantage vanishes when the signal is weak
(water MD, where nothing works well) or saturated (water PIMD, where both reach
0.97). It survives on `e_rmse` and `e_q95` — the bulk-error scores — and is
neutral on the two max-based ones. **The Stage 1 recommendation therefore needs
qualifying**: prefer the global statistic, but the evidence for it is strong on
small molecules and neutral on condensed-phase water, rather than universal.

## 5. The D = 576 extreme-value prediction: right ordering, growing optimism

The pre-registered blind test of J3's law at a 21× larger dimension. Top-5%
max-component recall, measured versus predicted from the Beta law:

| system | K | measured | predicted (shared) | error |
|---|---|---|---|---|
| water-disjoint MD | 4 | 0.660 | 0.712 | +0.052 |
| water-disjoint PIMD | 4 | 0.740 | 0.832 | +0.092 |
| water-overlapping MD | 4 | 0.612 | 0.648 | +0.036 |
| water-overlapping PIMD | 4 | 0.692 | 0.796 | +0.104 |

**Partially confirmed, and I will not overclaim it.** The model gets the ordering
right — it predicts water MD to have *higher* K=4 recall than 3BPA (0.712 vs
0.600) and it does (0.660 vs 0.564), which is a genuine non-trivial success since
the naive expectation is that larger D means worse recall. But the model is
systematically **optimistic, and the optimism grows with D**: ≈ +0.03 at D = 81,
+0.036 to +0.104 at D = 576.

The cause is identifiable: the `shared` simulation models the head-space
directions `a_d` as isotropic while using the exact variance magnitudes. As D
grows relative to the fixed r = 7 subspace, real coordinate directions become
increasingly non-isotropic and the isotropic null becomes a worse approximation.
So the honest statement is that the law explains the *mechanism* and predicts the
*ordering*, while a quantitative prediction at large D needs the real per-structure
geometry rather than an isotropic null. That is a concrete, cheap follow-up.

## What this changes

1. The "force UQ earns its backward passes" claim (Stage 1, +0.088 to +0.230 on
   molecules) **does not hold on water** and must be scoped to system class.
2. The global-statistic recommendation weakens from universal to
   molecule-supported / water-neutral.
3. The distribution-shift result is the strongest new material here and is free:
   a same-temperature, same-system MD-vs-PIMD pair where error moves 4–5× and
   uncertainty moves 1.5×.

   > **Qualified by J6.** A matched naive committee (8 independently trained
   > models, same data, same frames) is **2.7× more accurate on PIMD** — 81.4
   > against 219.7 meV/Å — and its Factor A collapses to 0.643, with `force_norm`
   > falling from 0.943 to 0.575. So most of the shift regime's apparent
   > detectability is a symptom of *this model family failing badly* on PIMD, not
   > a general property of committee UQ under shift. The pair is still the
   > cleanest shift instrument in the project, but the claim must be stated as
   > "when the model fails this way, the failure is detectable — largely for
   > free" rather than "UQ shines under shift". See `J6_NAIVE_COMMITTEE.md`.
4. The systems prediction from J8 is still untested — water's 33× edges per
   structure should push the batched path into the lane-proportional regime at
   modest B. That benchmark is the obvious next GPU job.
