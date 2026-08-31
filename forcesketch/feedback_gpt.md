
# Revised verdict

**The scientific contribution is sufficient for a realistic SIMBIOCHEM II workshop acceptance, but the current manuscript is not yet safe to submit.** After reading all seven pages and inspecting the figures, I would rate the revised scientific story as **lean accept for a poster**, rather than the more speculative assessment I gave from the project plan alone. The paper is not presently a spotlight-level submission, and several correctness and baseline issues could still turn it into a rejection.

SIMBIOCHEM II is a non-archival NeurIPS workshop rather than the NeurIPS main track. Crucially, its call explicitly welcomes partial and negative results, and evaluates submissions on novelty, impact, correctness, and clarity. Your paper’s central finding—small sketches fail as replacements but work as screens—is unusually well aligned with that remit. ([SIMBIOCHEM][1])

| Review criterion    | Current scientific draft |                                  After the critical fixes below |
| ------------------- | -----------------------: | --------------------------------------------------------------: |
| Novelty             |                  $3.5/5$ |                                                           $4/5$ |
| Impact              |                    $3/5$ |                                                     $3.5$–$4/5$ |
| Correctness         |                    $2/5$ |                                                           $4/5$ |
| Clarity             |                    $3/5$ |                                                           $4/5$ |
| Likely decision     | Borderline / weak reject |                                                 **Lean accept** |
| Spotlight potential |                      Low | Plausible only with decisive baselines and stronger calibration |

This assessment is for the workshop. **The paper is not presently sufficient for the NeurIPS main conference track**, which would normally require broader architectures, larger-scale systems, stronger theory, and more extensive downstream validation.

## Why the results are genuinely publication-worthy

**The negative result is useful rather than merely unsuccessful.** The paper shows that good global ranking does not imply reliable tail acquisition: at $K=4$, Haar reaches Spearman correlation $0.859$ while recovering only $0.619$ of the exact top-$5%$ set. Even the low-rank control variate reaches only $0.730$. That distinction between average ranking fidelity and rare-tail decision fidelity is scientifically important for active learning, because acquisition is governed by the tail rather than by average correlation.

**The screening result is the strongest contribution.** With $r_0=2$, $K=4$, and $\alpha=0.05$, the method skips between $76.5%$ and $86.0%$ of exact force-UQ evaluations while retaining between $97.1%$ and $98.8%$ of exact high-uncertainty configurations across 3BPA, ethanol, aspirin, and azobenzene. That cross-system consistency is much more convincing than a single favorable molecule.

**The paper distinguishes statistical savings from actual systems savings.** The result that the gate is $1.27\times$ faster at batch size $16$ but $0.93\times$ at batch size $1$ is not an embarrassment—it is an informative crossover. It shows that batched reverse mode and sketching attack the same redundancy, making them substitutes at small batch sizes. The conclusion that ForceSketch is appropriate for batched candidate sweeps but not single-structure MD is specific, falsifiable, and operationally useful.  

**Orthogonalization produces a substantive, measured improvement.** At $K=3$, Haar improves top-$5%$ recall over Gaussian probes by $0.225$ with a paired $95%$ interval of $[0.186,0.263]$, and the improvement is reported across all four systems. The control variate adds another $0.111$ at $K=4$. These are large enough effects to support a methodological conclusion, rather than merely demonstrating that several unbiased estimators exist.

**The systems work is unusually honest for a workshop paper.** You attempted batched VJPs, diagnosed the interaction between `vmap`, TorchScript, and the TensorExpr fuser, verified the corrected path against serial differentiation, and examined partial compilation rather than benchmarking only against a Python loop. That makes the performance conclusions more credible.

**The contribution is distinct from Beck et al.** The original MHC paper makes all head energies available together but states that force disagreement requires multiple reverse passes. ForceSketch addresses the unresolved inference-time cost of evaluating that established uncertainty model, rather than proposing another committee-training method. ([AIP Publishing][2])

## The strongest counterargument: the screening result lacks the decisive simple baselines

**A calibrated screening gate is a generic wrapper, not uniquely a ForceSketch contribution.** Any score correlated with exact force disagreement can be calibrated conservatively. Figure 3’s screening panel compares Haar and control-variate variants, but the main paper does not show whether the same gate built on simpler scores gives an equal or better skip–recall–latency frontier.

The two most important missing baselines are:

1. **Energy-disagreement screening.** All head energies are already produced by one forward pass, so their disagreement is nearly free. Even if energy disagreement is a worse general uncertainty estimator, it might still be adequate for ruling out clearly low-force-uncertainty structures. The original MHC work explicitly distinguishes the cheap simultaneous energy outputs from the expensive force standard deviation. ([AIP Publishing][2])

2. **Calibrated head-subsampling screening.** Apply exactly the same calibration and fallback rule to head subsampling at the same reverse-lane budget. Your raw comparison is close: Haar beats head subsampling by only $0.052$ when both include a mean-force lane, while head subsampling wins when the mean lane is omitted.

Because the exact mean force is already computed in the MD framing, the strongest head-subset estimator is not necessarily the ordinary sample variance among the selected heads. It should use the exact mean:

$$
\widehat v_d^{\mathrm{head}}
============================

\frac{M}{K(M-1)}
\sum_{i\in\mathcal S}
\left(F_{di}-\bar F_d\right)^2.
$$

This is unbiased under uniform head sampling and uses the same mean-force information available to ForceSketch. If the control-variate gate Pareto-dominates this estimator and the free energy-disagreement gate, your main positive claim becomes much stronger. If it does not, the paper can still be accepted, but should be reframed as a broader study of which cheap signals can screen exact MHC force uncertainty.

## The target acquisition statistic is currently ambiguous

**The paper does not clearly state what scalar defines the “exact top-$5%$ uncertainty set.”** It defines global disagreement $S$, atomwise RMS uncertainty, and the MHC component-standard-deviation score, but the gate is written in terms of $S$ and the primary results never explicitly identify the acquisition scalar. The limitations then state that extreme-value statistics converge more slowly than global statistics.  

This matters because Beck et al.’s active-learning workflow ranks structures using the **maximum disagreement of their force components**, not global Frobenius disagreement. ([AIP Publishing][2])

Therefore:

* If your top-$5%$ experiments use the original maximum-component or maximum-atom MHC rule, state that explicitly in Section 4 and every figure caption.
* If they use global $S$, the current manuscript does not yet establish preservation or screening of the motivating paper’s actual acquisition decision.

For acceptance, I would make the original MHC maximum-component statistic primary or co-primary. Global $S$ can remain the statistically easier secondary quantity. A result showing that global statistics screen well while maximum-component statistics require more directions would itself be a valuable scientific conclusion.

## There is a serious internal contradiction about the exact baseline

**Section 5.3 says batched reverse mode was repaired and used.** It reports that disabling the TensorExpr fuser produces $50/50$ successful batched calls agreeing with the serial path to $3\times10^{-15}$, and then reports performance against that batched baseline.

**Section 6 then says the exact baseline is serial and batched reverse mode is unusable.** The conclusion again says batched reverse mode was enabled by disabling the TensorExpr fuser. These three statements cannot all be true.

This is the most dangerous scientific-writing issue in the current draft. A reviewer may stop trusting every speed number after encountering it. The limitation paragraph appears to be stale text from before the fuser workaround was discovered.

You need one unambiguous account:

* batched eager exact baseline with the fuser disabled;
* compiled batched exact baseline, if available;
* serial exact baseline only where batched execution exceeds memory;
* exact batch sizes, dtypes, and hardware for each reported number.

The workshop has no rebuttal phase, so a reviewer’s confusion cannot be repaired after submission. ([SIMBIOCHEM][1])

## Several numerical claims currently contradict one another

**The abstract reports the best $K\leq4$ recall as $0.619$, but the control-variate estimator reaches $0.730$ at $K=4$.** The abstract should say “the best oblivious sketch reaches $0.619$; a learned control variate reaches $0.730$, still below replacement quality.”  

**The conclusion claims that $0.90$ recall is not reached below approximately $K=6$ of $7$, but the main results display $K\in{1,2,3,4,7}$.** Unless $K=5$ and $K=6$ results appear in the supplement, the main paper supports only the claim “not reached for $K\leq4$.” Either run and report $K=5,6$ or remove the interpolation.  

**The abstract’s wording implies the same $0.860$ skipped and $0.982$ recall on all four systems, whereas those are the 3BPA values.** The other systems range from $0.765$ to $0.849$ skipped and $0.971$ to $0.988$ recall. Report the ranges or say “on 3BPA, with comparable results across the other systems.”

These are individually easy to fix, but collectively they currently reduce the correctness score.

## The calibration claim is too strong in its present form

**“Conservative by construction” is not justified merely because an empirical calibration quantile covers $1-\alpha$ of the calibration structures.** That is an in-sample fact, not automatically a finite-sample test guarantee, and certainly not an OOD guarantee. The paper itself acknowledges that calibration may shift out of distribution.  

There are two defensible options:

* Call it an **empirically calibrated upper score** and make no formal coverage claim.
* Convert it into split-conformal or conformal-risk calibration.

For the latter, first freeze $Q_{r_0}$, $r_0$, $K$, and all method choices on a design split. Then use a separate calibration split for the ratios

$$
r_i=\frac{S_i}{\widehat S_i+\epsilon}
$$

and select the finite-sample order statistic corresponding to $1-\alpha$. Do not use the same structures both to learn the control-variate subspace and to claim split-conformal calibration. If the actual objective is high-UQ false-negative control rather than marginal upper-bound coverage, conformal risk control is an even closer match. ([ICLR Proceedings][3])

At minimum, the screening table needs confidence intervals for recall, skipped fraction, and speedup. With approximately $1000$ rMD17 structures, a nominal top-$5%$ subset contains only about $50$ positives, so recall estimates can have meaningful finite-sample uncertainty.

## The seed treatment must be stated more precisely

The paper says the confidence intervals are calculated on the “seed-averaged statistic,” while Figure 2 reports seed-to-seed standard deviations.

That phrase is potentially problematic:

* If you calculate the metric separately for each $K$-direction sketch and then average the metrics across ten seeds, the compute budget remains $K$.
* If you first average the ten approximate uncertainty scores and then calculate recall, the effective deployment budget is approximately $10K$, and the reported $K$-budget comparison is invalid.

State the exact procedure mathematically. For deployment-level results, one fixed random sketch should be used per deployed model; uncertainty across independently generated deployment sketches can then be reported separately.

## The head-spectrum terminology appears mathematically incorrect

The paper says the “stable rank” is $5.13$ while the leading direction contains $0.334$ of the energy.

For

$$
A=Q^\mathsf{T}F^\mathsf{T}FQ\succeq0,
$$

if

$$
\frac{\lambda_{\max}(A)}{\operatorname{tr}(A)}=0.334,
$$

then the stable rank of $FQ$ is

$$
\operatorname{srank}(FQ)
========================

# \frac{|FQ|_F^2}{|FQ|_2^2}

\frac{\operatorname{tr}(A)}{\lambda_{\max}(A)}
\approx
\frac{1}{0.334}
\approx 2.99,
$$

not $5.13$.

The value $5.13$ is likely the participation-ratio or effective rank

$$
r_{\mathrm{eff}}
================

\frac{\operatorname{tr}(A)^2}
{\operatorname{tr}(A^2)}.
$$

That should be renamed. The phrase “close to isotropic” should also be softened: a perfectly isotropic seven-dimensional spectrum has leading share $1/7\approx0.143$, considerably below $0.334$. “Moderately diffuse rather than strongly rank-concentrated” would be more defensible.

This is a small textual correction but an important correctness correction, particularly because randomized numerical linear algebra reviewers will notice it immediately.

## The systems baseline should use the fastest valid implementation

The paper reports that partial `torch.compile` gives a $1.04$–$1.10\times$ improvement, but then chooses eager execution as the primary baseline because compilation requires fallback handling.

A reviewer may reasonably argue that inconvenience is not a reason to avoid the fastest correct exact baseline. I would:

* make the fastest correct compiled-batched baseline primary;
* report eager-batched as the portable baseline;
* report serial only for OOM regimes;
* show that the qualitative crossover remains unchanged.

Also separate correctness precision from deployment precision. The paper performs estimator evaluation in float64 on an RTX 5070 Laptop GPU.  Numerical verification can remain float64, but performance should also be reported in the model’s normal deployment precision. A second GPU—ideally an A100, H100, L40S, or comparable research accelerator—would considerably strengthen the systems claim.

On fallback structures, investigate reusing the already-computed mean and $r_0$ exact control-variate directions. Recomputing the full $M$-lane exact path discards work that the gate has already paid for.

## The related-work section is currently too thin

The paper has only five references: Beck et al., MACE, the two datasets, and Hutchinson.

That is insufficient for a paper claiming randomized estimators, orthogonal probes, a low-rank control variate, calibrated screening, and approximate VJPs. Randomized trace estimation and low-rank variance reduction are established areas; Avron–Toledo, Hutch++, and related trace-estimation literature should be acknowledged. A recent 2026 paper also studies unbiased approximate VJPs for efficient backpropagation, which is adjacent even though it targets training rather than committee-force uncertainty. ([IBM Research][4])

The defensible novelty claim is not:

> We invent randomized norm or trace estimation.

It is:

> We identify force disagreement as a centered committee-output Jacobian statistic, characterize the accuracy of limited output-space VJP budgets for molecular-UQ decisions, demonstrate that small sketches fail for tail acquisition, and turn them into a calibrated screening policy whose utility depends on the automatic-differentiation regime.

That is a sufficient workshop contribution.

I would move most of the detailed TensorExpr debugging from the main text into an appendix, retaining one concise paragraph and a reproducibility note. Use the recovered space for related work, an explicit acquisition-score definition, and calibration details. Because the workshop instructs reviewers to judge the submitted PDF rather than search externally, those omissions matter. ([SIMBIOCHEM][1])

## What would most improve the acceptance probability before August 29

The submission deadline is August 29, 2026, and there is no rebuttal, so correctness and decisive baselines should take priority over additional kernel engineering. ([SIMBIOCHEM][1])

**Acceptance-critical changes**

1. Anonymize the paper completely. The uploaded PDF contains your name, Harvard affiliation, and email, while the call states that non-anonymized papers will be desk-rejected.  ([SIMBIOCHEM][1])
2. Reconcile the serial-versus-batched baseline contradiction and use the fastest correct exact implementation.
3. Explicitly define the primary uncertainty/acquisition score and evaluate the original Beck maximum-component rule.
4. Add calibrated screening baselines for free energy disagreement and head subsampling.
5. Correct the $0.619$ versus $0.730$ abstract claim, the $K\approx6$ statement, and the stable-rank terminology.
6. Clarify per-seed evaluation and add confidence intervals to all screening results.

**Highest-value strengthening**

1. Use separate design, calibration, and test splits; formalize or weaken the “conservative” claim.
2. Add one true-reference-force-error experiment, such as high-error recall or risk–coverage. This need not become a full active-learning campaign.
3. Report the screening Pareto for the exact-mean-assisted head estimator.
4. Add a second GPU or at least both FP32 and FP64 timing.
5. Release an anonymous supplement containing code, benchmark configurations, frozen seeds, and raw result records.

## Final stance

**The paper contains enough scientific substance for SIMBIOCHEM II, and I would submit it after the critical revisions.** Its strongest contribution is not a universally fast approximation to exact MHC uncertainty—the experiments correctly show that such a claim is false. Its publishable contribution is the combination of:

* a clear negative result for small-$K$ tail acquisition;
* a measured advantage from orthogonal and control-variate probes;
* a high-recall screening policy;
* and a systems crossover showing when reduced VJP dimension translates into actual GPU savings.

That is a coherent and useful workshop paper. However, the present draft is closer to **weak reject than weak accept** because the main screening claim lacks the simplest screening baselines, the target acquisition score is unclear, and the exact-baseline description contradicts itself. Resolve those issues and demonstrate that ForceSketch screening beats free energy disagreement and screened head subsampling; at that point, I would regard it as a **clear poster-level accept candidate**.

[1]: https://www.simbiochem.com/call-for-papers?utm_source=chatgpt.com "Call for Papers · SIMBIOCHEM II @ NeurIPS 2026"
[2]: https://pubs.aip.org/aip/jcp/article/163/23/234103/3374754/Multi-head-committees-enable-direct-uncertainty "aipp.silverchair-cdn.com"
[3]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html?utm_source=chatgpt.com "Conformal Risk Control"
[4]: https://research.ibm.com/publications/randomized-algorithms-for-estimating-the-trace-of-an-implicit-symmetric-positive-semi-definite-matrix?utm_source=chatgpt.com "Randomized algorithms for estimating the trace of an implicit symmetric positive semi-definite matrix for Journal of the ACM - IBM Research"
