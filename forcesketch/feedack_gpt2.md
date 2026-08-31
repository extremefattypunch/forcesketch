# Revised verdict

**This version has crossed the threshold from “promising but vulnerable” to a credible SIMBIOCHEM II accept candidate.** My reviewer-style recommendation would now be **Weak Accept as submitted**, moving to **Accept after three targeted corrections**. The title and abstract correctly lead with the negative replacement result, then make the narrower positive screening claim, supported by cross-system recall, skipped-evaluation, and runtime numbers.

For clarity, this assessment concerns **poster acceptance at the NeurIPS SIMBIOCHEM II workshop**, not the NeurIPS main track. The workshop explicitly welcomes partial and negative results and evaluates submissions on novelty, impact, correctness, and clarity, making your “replacement fails, screening succeeds” story unusually well matched to the venue. It accepts 5–8-page double-blind papers and has no rebuttal phase. ([SIMBIOCHEM][1])

| Criterion              |                 Current assessment |
| ---------------------- | ---------------------------------: |
| Venue fit              |                            **5/5** |
| Novelty                |                            **4/5** |
| Impact                 |                          **3.5/5** |
| Correctness            | **3.5/5**, pending one split issue |
| Clarity                |                            **4/5** |
| Experimental care      |                          **4.5/5** |
| Overall recommendation |             **6/10 — Weak Accept** |
| After the fixes below  |                  **7/10 — Accept** |

I would describe poster acceptance as **more likely than rejection**, but spotlight selection remains unlikely without a downstream active-learning or reference-error result.

## Why the revised paper is substantially stronger

**The primary scientific decision is now defined correctly.** The paper explicitly identifies maximum componentwise force disagreement as the primary acquisition score, explains why the global statistic is easier, defines the exact top-tail decision, and clarifies that metrics are computed independently for each frozen sketch seed rather than averaging ten sketches into an effective $10K$ estimator. This resolves one of the largest ambiguities in the previous version.

**The negative result has become scientifically informative.** Under the primary maximum-component rule, Haar recall is only $0.484$–$0.570$ at $K=4$, even though ranking is considerably better. The measured upward bias of the maximum of $3N$ noisy component estimators gives a plausible mechanism: marginally unbiased component estimates do not produce an unbiased extreme-value statistic. That is a useful warning for anyone using randomized approximations in tail-based active learning.

**The screening claim now has the decisive baselines it previously lacked.** You compare the control-variate gate against free energy disagreement, exact-mean-assisted head subsampling, and an ordinary Haar sketch under identical calibration and threshold procedures. The control variate skips $0.836$–$0.872$ of exact evaluations while retaining $0.964$–$0.982$ of high-UQ structures, and Pareto-dominates its comparator in $11$ of $12$ gate-by-system comparisons. That table on page 7 is now the central positive result of the paper.

**The systems evaluation is now defensible.** The serial-versus-batched contradiction has been removed. You diagnose the TensorExpr failure, verify the corrected batched path against serial differentiation, distinguish incremental from total speedup, attempt compilation, and explain why eager batched reverse mode is the fastest correct implementation available on this stack.

**The spectral analysis is corrected and useful.** The paper now reports the actual stable rank,

$$
\operatorname{srank}(FQ)
========================

# \frac{|FQ|_F^2}{|FQ|_2^2}

\frac{\operatorname{tr}(A)}{\lambda_{\max}(A)},
$$

as $2.91$–$3.08$ rather than incorrectly calling an effective-rank statistic “stable rank.” The conclusion that the spectrum is moderately concentrated—not isotropic—is consistent with the leading direction carrying approximately one third of the disagreement and supports the control-variate motivation.

**The manuscript is now appropriately anonymized and formatted.** The PDF uses anonymous author placeholders, the substantive paper ends at page 8, and references, appendix, and checklist follow. This appears compatible with the workshop’s double-blind and page-limit requirements.  ([SIMBIOCHEM][1])

## The most important remaining correctness issue

**The control-variate training split and conformal calibration split currently contradict each other.** The method says that $Q_{r_0}$ contains eigenvectors of the **calibration-set average** force Gram matrix. A few lines later, it says the conformal calibration split is disjoint from the **design** and test splits. But no design split is otherwise defined.

This matters mathematically. Standard split-conformal validity requires the approximate score function—including the learned control-variate basis—to be frozen independently of the calibration observations. If $Q_{r_0}$ was learned using the same structures used to calculate the conformal ratios, the claimed finite-sample coverage is not generally justified.

There are two possibilities:

* If the implementation actually learned $Q_{r_0}$ on a separate design split, change “calibration-set average” to “design-set average” and give $n_{\mathrm{design}}$, $n_{\mathrm{cal}}$, and $n_{\mathrm{test}}$ for every system.
* If the same calibration structures were used for both, rerun the gate with three disjoint splits.

This is the one issue that could turn an otherwise positive review into a correctness-based rejection. Because there is no rebuttal, it must be unambiguous in the submitted PDF. ([SIMBIOCHEM][1])

## The gate runtime needs one explicit accounting sentence

**The reported screening speedups are not fully reproducible from the prose.** The control-variate gate starts with five reverse lanes, but the paper does not state whether a fallback recomputes the complete eight-lane exact calculation or reuses the already evaluated directions and computes only the missing residual directions. Figure 1 visually suggests reuse, while the $1.33$–$1.37\times$ values appear numerically consistent with it.  

State something like:

> On fallback, we orthogonally complete the residual Haar frame and evaluate only the $r-K$ missing centered directions; the mean, leading control-variate directions, and existing residual directions are reused.

Then give the screening cost explicitly, for example

$$
T_{\mathrm{screen}}
===================

T(1+K)
+
p_{\mathrm{fallback}},
T_{\mathrm{additional}}(r-K).
$$

Also state directly beside the screening table:

* the batch size;
* whether the baseline is eager-batched or serial;
* whether the forward graph is retained;
* whether the reported speedup is total force-plus-UQ speedup;
* and how the already-computed directions are reused.

Your paper repeatedly emphasizes that speedups must not be conflated, so leaving these details implicit would be especially noticeable.

## The acquisition terminology is still slightly inconsistent

**The paper says there are “two defensible readings” but Equation 4 presents three scores:** maximum component, maximum atom, and global disagreement. It then reports maximum component as primary and global disagreement as secondary, while maximum atom largely disappears.

Choose one clean formulation:

> We evaluate the original extreme-value acquisition rule in two forms, maximum component and maximum atom; we additionally report global disagreement as an easier diagnostic statistic.

Alternatively, remove maximum atom from the main formulation if it is only in the supplement.

There is a second ambiguity. The paper says “top-$5%$” means the largest $5%$ of held-out structures, but also says the gate threshold $\tau$ is estimated from calibration structures. Those are not generally identical:

* Section 5.1 can evaluate overlap with the exact test-set top $5%$.
* Section 5.4 should define positives as test structures satisfying $S(x)\geq\tau_{\mathrm{cal}}$ and report the resulting test prevalence.

Do not say the two experiments use “the same threshold” unless they literally do.

## The main visual should show the primary statistic

**Both the principal fidelity plot in Figure 2 and the left side of Figure 3 use the easier global statistic $S$, even though maximum-component acquisition is now declared primary.** The maximum-component failure—the more scientifically interesting result—is presented only in prose.  

The left side of Figure 3 is mostly redundant with Figure 2. Replace it with one of:

* maximum-component recall versus $K$ for all four systems;
* global versus maximum-component recall on the same axes;
* or worst-seed/median-seed maximum-component screening performance.

That single visual change would make the paper’s primary claim immediately legible to a reviewer.

## Counterarguments a skeptical reviewer can still make

**The positive systems benefit is useful but narrow.** The gate is a net slowdown at $B=1$, the regime most relevant to ordinary single-structure MD, and achieves only about $1.33$–$1.37\times$ screening acceleration in the favorable reported setting. Results are from one eight-head architecture, one laptop GPU, and relatively small molecules. The paper acknowledges all of these limitations clearly, which makes them acceptable for a workshop but prevents a broad acceleration claim.

**The method preserves exact committee disagreement, not demonstrated physical utility.** There is no reference-force-error risk–coverage experiment and no actual active-learning loop. Consequently, the paper establishes that the gate preserves an existing uncertainty proxy, not that it preserves downstream model improvement or DFT-query efficiency. This is enough for a poster because the paper is carefully scoped, but one reference-error panel would materially improve impact and spotlight potential.

**The mathematical ingredients are mostly established.** Gaussian norm estimation, Haar projections, low-rank control variates, and split conformal calibration are not individually novel. The contribution is their formulation and empirical characterization for multi-head force uncertainty, particularly the tail failure and the screening crossover. The manuscript now acknowledges randomized linear algebra properly.

A related 2026 paper on unbiased approximate VJPs for efficient backpropagation should also be cited and distinguished. It approximates VJPs inside training, whereas ForceSketch compresses the output-head query space at inference to estimate an existing committee statistic. Adding that citation would reduce the chance that a reviewer interprets the paper as overlooking current VJP literature. ([arXiv][2])

**The learned basis may be system-specific.** The main paper does not state whether $Q_{r_0}$ is learned separately for each molecule, pooled across rMD17 systems, or transferred unchanged. Clarify this. A single basis that transfers across molecules would be a strong result; a per-system basis is still acceptable, but then the method should be described as requiring system-specific design and calibration data.

## Presentation polish

**The manuscript is readable, but several draft artifacts remain.** The rendered PDF contains conspicuous red boxes around internal references and green boxes around citations. Use hidden or color-only links in the submission build. The “Affiliation / Address / email” placeholders under “Anonymous Author(s)” can also be removed. Figure 3’s right panel would benefit from larger labels because the system names cluster tightly near the control-variate points.

These will not determine acceptance, but they affect the workshop’s clarity criterion.

## Final stance

**I would now recommend acceptance as a SIMBIOCHEM II poster, provided the design/calibration split is genuinely disjoint and is stated correctly.** The paper has a coherent and useful message:

$$
\text{small sketches fail for extreme-tail replacement}
$$

but

$$
\text{a learned control-variate sketch can safely screen exact force UQ},
$$

with a measured systems crossover that identifies where the procedure is and is not worthwhile.

The updated baseline table, explicit primary acquisition rule, corrected spectral analysis, and fair batched exact implementation address nearly all of the weaknesses in the previous draft. The remaining problems are mostly surgical rather than requiring a new research project. Fix the split contradiction, make fallback timing fully explicit, and put the primary maximum-component result in a main figure. After those changes, this is a **solid workshop accept candidate**, though still more likely to receive a poster than one of the six spotlight slots.

[1]: https://www.simbiochem.com/call-for-papers "Call for Papers · SIMBIOCHEM II @ NeurIPS 2026"
[2]: https://arxiv.org/abs/2602.14701?utm_source=chatgpt.com "Unbiased Approximate Vector-Jacobian Products for Efficient Backpropagation"
