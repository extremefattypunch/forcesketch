# ForceSketch: Complete Experimental and Paper Execution Plan

**Working title:**  
**ForceSketch: Randomized Head-Space VJPs for Fast Force Uncertainty in Multi-Head Molecular Potentials**

**Target:** NeurIPS 2026 SIMBIOCHEM II workshop  
**Submission deadline:** August 29, 2026, 11:59 PM UTC. The workshop accepts 5–8 page non-archival papers.

---

# 0. Governing Principle

This project is **not a Triton-kernel paper**.

The paper must remain scientifically coherent and submission-worthy if every custom Triton kernel is removed.

The central research question is:

> **Can exact multi-head force disagreement be replaced by a small number of randomized or orthogonal head-space VJPs while preserving the uncertainty decisions that matter in molecular simulation?**

The primary contribution is therefore:

1. an alternative algorithm for computing force uncertainty;
2. a mathematical characterization of its estimator;
3. a decision-level evaluation of whether it preserves uncertainty screening/acquisition;
4. a systems analysis of whether reducing the number of reverse-mode directions translates into real GPU wall-clock savings.

Triton is an **optional implementation optimization** applied only after the main algorithmic result is established.

The project should be terminated or reframed if its only positive result is:

> “A custom reduction kernel is faster than PyTorch.”

That alone is not the intended paper.

---

# 1. Core Research Question

Consider a shared-trunk molecular potential with \(M\) output energy heads.

Exact force disagreement requires differentiating the individual head energies, or equivalently evaluating the full \(M-1\)-dimensional centered head subspace.

The project asks whether only

\[
K \ll M-1
\]

carefully selected head-space directions are sufficient to estimate the uncertainty statistics used downstream.

The full scientific question is:

> **How many reverse-mode directions are actually needed to make the same uncertainty-selection decisions as exact multi-head force disagreement?**

The systems question is secondary but essential:

> **Does reducing the number of required VJP directions produce a meaningful wall-clock reduction when compared against a strong, vectorized exact baseline?**

---

# 2. Primary Hypotheses

## H1 — Statistical approximation

For a shared-trunk \(M\)-head model, a small number of randomized or orthogonal centered head-space VJPs can accurately estimate exact force disagreement.

Primary target:

\[
K \leq 4.
\]

---

## H2 — Decision preservation

The approximate uncertainty does not need to reproduce every variance component perfectly.

It must preserve downstream uncertainty decisions.

The main target is:

\[
\text{Recall of exact top-5\% uncertain configurations}
\geq 0.90.
\]

Strong target:

\[
\geq 0.95.
\]

---

## H3 — Better than simply evaluating fewer heads

At equal reverse-pass budget, structured head-space sketches should provide more reliable uncertainty estimates than randomly selecting a subset of ensemble heads.

The most important comparison is therefore:

\[
\text{ForceSketch}(K)
\quad\text{vs.}\quad
\text{head subsampling}(K).
\]

---

## H4 — GPU savings

Reducing exact centered uncertainty directions from

\[
M-1
\]

to

\[
K
\]

should reduce force-UQ latency through the shared model trunk.

Minimum target:

\[
\text{incremental UQ speedup}
\geq 1.5\times.
\]

Strong target:

\[
\geq 2\times.
\]

---

## H5 — Approximate screening can avoid exact calculations

A conservative two-stage policy can use ForceSketch first and compute exact force disagreement only for ambiguous/high-risk structures.

Strong practical target:

- skip at least 50% of exact force-UQ calculations;
- retain at least 95% of exact high-uncertainty configurations.

---

# 3. What the Paper Is Actually About

The paper should have four layers.

## Layer A — Mathematical reformulation

Express force disagreement as a quantity derived from the centered Jacobian of the multi-head energies with respect to atomic coordinates.

---

## Layer B — Approximate algorithm

Replace the complete \(M-1\)-dimensional centered head basis with \(K\) randomized/orthogonal directions.

---

## Layer C — Decision-level validation

Measure whether the approximation preserves:

- ranking;
- high-uncertainty acquisition sets;
- maximum-uncertainty atoms;
- error-detection behavior.

---

## Layer D — Systems evaluation

Measure whether the reduced number of reverse-mode directions actually lowers:

- incremental UQ latency;
- total force-plus-UQ latency;
- memory;
- GPU work.

---

# 4. What the Paper Is Not About

Do not frame the paper as:

- a new MACE architecture;
- a new molecular potential;
- a new uncertainty model requiring training;
- a Triton reduction kernel;
- a generic GPU fusion paper;
- a replacement for the mean molecular force;
- a claim that ensemble disagreement is Bayesian uncertainty;
- a claim that committee disagreement perfectly predicts physical error.

---

# 5. Primary Model and Systems

## Model

Use a **shared-trunk multi-head MACE committee**.

The critical property is:

```text
expensive shared representation
          |
    -----------------
    |   |   |   |   |
   E1  E2  E3 ...  EM
```

Multiple energy heads share the majority of their computation.

Force-UQ requires differentiating head-dependent energy combinations through this shared computation.

---

## Primary system

**3BPA**

Use for:

- implementation;
- exact uncertainty reproduction;
- sketch fidelity;
- performance;
- acquisition experiments;
- calibration.

---

## Generalization

Use at least three rMD17 systems spanning molecule size.

Choose:

- one small;
- one medium;
- one relatively large.

Do not optimize the method separately for each molecule.

---

## Optional systems

Only after the primary paper is complete:

- foundation-model multi-head MACE;
- periodic water;
- another OOD molecular dataset.

---

# 6. Mathematical Formulation

## 6.1 Head energies

For Cartesian coordinates

\[
x\in\mathbb{R}^{D},
\qquad
D=3N,
\]

define the \(M\) model energies

\[
e(x)
=

\begin{bmatrix}
E_1(x)\\
E_2(x)\\
\vdots\\
E_M(x)
\end{bmatrix}.
\]

---

## 6.2 Head forces

For head \(m\),

\[
f_m(x)
=

-\nabla_xE_m(x).
\]

Collect the head forces:

\[
F(x)
=

\begin{bmatrix}
f_1(x) &
f_2(x) &
\cdots &
f_M(x)
\end{bmatrix}
\in
\mathbb{R}^{D\times M}.
\]

---

## 6.3 Centered head space

Define

\[
P
=

I
-

\frac{1}{M}
\mathbf{1}\mathbf{1}^{\mathsf T}.
\]

Then

\[
FP
\]

contains the head forces after subtracting their mean.

The centered head space has dimension

\[
r=M-1.
\]

---

# 7. Exact Force-Uncertainty Quantities

## Component variance

For Cartesian coordinate \(d\),

\[
v_d
=

\frac{1}{M-1}
\sum_{m=1}^{M}
\left(
F_{dm}-\bar F_d
\right)^2.
\]

Equivalently,

\[
v_d
=

\frac{1}{M-1}
\left\|
e_d^{\mathsf T}FP
\right\|_2^2.
\]

---

## Global disagreement

\[
S
=

\sum_{d=1}^{D}v_d
=

\frac{1}{M-1}
\|FP\|_F^2.
\]

---

## Atomwise disagreement

For atom \(a\),

\[
S_a
=

\sum_{\alpha=1}^{3}
v_{a\alpha}.
\]

---

## RMS atom uncertainty

\[
u_a^{\mathrm{RMS}}
=

\sqrt{
\frac{1}{3}
\sum_{\alpha=1}^{3}
v_{a\alpha}
}.
\]

---

## Component-standard-deviation atom uncertainty

\[
u_a^{\mathrm{MHC}}
=

\frac{1}{3}
\sum_{\alpha=1}^{3}
\sqrt{v_{a\alpha}}.
\]

This should be the primary atom-level quantity if it corresponds to the existing MHC uncertainty implementation.

Both \(u^{\mathrm{RMS}}\) and \(u^{\mathrm{MHC}}\) should be retained because their approximation properties differ.

---

# 8. Exact Centered-Basis Computation

Construct

\[
Q
\in
\mathbb{R}^{M\times(M-1)}
\]

with

\[
Q^{\mathsf T}Q=I,
\]

\[
Q^{\mathsf T}\mathbf{1}=0,
\]

and

\[
QQ^{\mathsf T}=P.
\]

For column \(q_j\),

\[
g_j
=

-\nabla_x
\left(
q_j^{\mathsf T}e(x)
\right)
=

Fq_j.
\]

Then

\[
v_d
=

\frac{1}{M-1}
\sum_{j=1}^{M-1}
g_{j,d}^2.
\]

This formulation matters because the exact uncertainty calculation fundamentally requires only the centered \(M-1\)-dimensional head subspace.

The paper should therefore compare:

\[
M-1
\]

exact centered directions against

\[
K
\]

approximate directions.

---

# 9. Exact Mean Force

Define mean-energy seed

\[
s_0
=

\frac{1}{M}\mathbf{1}.
\]

Then

\[
-\nabla_x
\left(
s_0^{\mathsf T}e(x)
\right)
=

\bar f.
\]

For MD-like use, the computation therefore consists of:

```text
1 mean-force VJP
+
uncertainty VJPs
```

The paper should count the mean-force computation explicitly.

---

# 10. Method 1 — Gaussian ForceSketch

Draw

\[
z_k
\sim
\mathcal{N}(0,I_M).
\]

Center:

\[
w_k=Pz_k.
\]

Construct scalar objective

\[
L_k(x)
=

w_k^{\mathsf T}e(x).
\]

One VJP gives

\[
g_k
=

-\nabla_xL_k
=

Fw_k.
\]

Estimate coordinate variance:

\[
\widehat v_d
=

\frac{1}{K(M-1)}
\sum_{k=1}^{K}
g_{k,d}^2.
\]

Then

\[
\mathbb{E}
\left[
\widehat v_d
\right]
=

v_d.
\]

Global estimator:

\[
\widehat S
=

\sum_d
\widehat v_d.
\]

---

# 11. Finite-\(K\) Standard-Deviation Correction

For Gaussian probes,

\[
\frac{K\widehat v_d}{v_d}
\sim
\chi_K^2.
\]

The naïve quantity

\[
\sqrt{\widehat v_d}
\]

is downward biased.

Define

\[
c_K
=

\sqrt{\frac{2}{K}}
\frac{
\Gamma\left(
\frac{K+1}{2}
\right)
}{
\Gamma\left(
\frac{K}{2}
\right)
}.
\]

Use

\[
\widehat\sigma_d
=

\frac{
\sqrt{\widehat v_d}
}{
c_K
}.
\]

Then construct

\[
\widehat u_a^{\mathrm{MHC}}
=

\frac{1}{3}
\sum_{\alpha=1}^{3}
\widehat\sigma_{a\alpha}.
\]

This derivation should appear explicitly in the paper rather than treating the method as merely “Hutchinson estimation.”

---

# 12. Method 2 — Haar-Orthogonal ForceSketch

This should be treated as the preferred primary method if experiments support it.

Let

\[
r=M-1.
\]

Sample a random \(K\)-dimensional orthonormal subspace inside the centered head space:

\[
W\in\mathbb{R}^{M\times K}
\]

with

\[
W^{\mathsf T}W=I
\]

and

\[
W^{\mathsf T}\mathbf{1}=0.
\]

Compute

\[
G=FW.
\]

Then

\[
\widehat v_d^{\mathrm{Haar}}
=

\frac{1}{K}
\sum_{k=1}^{K}
G_{dk}^2.
\]

Because a random \(K\)-dimensional subspace captures \(K/r\) of squared norm in expectation,

\[
\mathbb{E}
\left[
\widehat v_d^{\mathrm{Haar}}
\right]
=

v_d.
\]

At

\[
K=M-1,
\]

the result becomes exact.

This is a particularly useful property to demonstrate experimentally.

---

# 13. Haar Standard-Deviation Correction

For \(K<r\), define

\[
c_{K,r}^{\mathrm{Haar}}
=

\sqrt{\frac{r}{K}}
\frac{
B\left(
\frac{K+1}{2},
\frac{r-K}{2}
\right)
}{
B\left(
\frac{K}{2},
\frac{r-K}{2}
\right)
}.
\]

At full rank define

\[
c_{r,r}^{\mathrm{Haar}}=1.
\]

Then estimate

\[
\widehat\sigma_d^{\mathrm{Haar}}
=

\frac{
\sqrt{
\widehat v_d^{\mathrm{Haar}}
}
}{
c_{K,r}^{\mathrm{Haar}}
}.
\]

The coding agent must test this formula against Monte Carlo simulations before using it in the paper.

---

# 14. Method 3 — Pairwise Head Differences

For randomly selected distinct heads \(i,j\),

\[
w=e_i-e_j.
\]

One VJP produces

\[
g=f_i-f_j.
\]

The identity

\[
\sum_{i<j}
\|f_i-f_j\|^2
=

M
\sum_i
\|f_i-\bar f\|^2
\]

implies, for a uniformly sampled unordered pair,

\[
\mathbb{E}
\left[
\frac{1}{2}
\|f_i-f_j\|^2
\right]
=

S.
\]

Coordinatewise,

\[
\mathbb{E}
\left[
\frac{1}{2}
(f_{i,d}-f_{j,d})^2
\right]
=

v_d.
\]

This gives a sparse-seed alternative to dense random projections.

It should be evaluated because GPU execution may behave differently for sparse versus dense head combinations.

---

# 15. Mandatory Baseline — Head Subsampling

For compute budget \(K\):

1. select \(K\) committee heads;
2. calculate their exact forces;
3. calculate the usual sample variance across those heads.

This baseline is essential.

The strongest algorithmic result would be:

> At the same number of reverse-mode directions, ForceSketch preserves exact acquisition decisions better than randomly evaluating \(K\) ensemble heads.

If ForceSketch does not beat head subsampling, the paper becomes substantially weaker.

---

# 16. Combined Batched VJP Execution

The mean-force direction and uncertainty directions should be computed together where supported.

For ForceSketch:

\[
S_{\mathrm{sketch}}
=

\begin{bmatrix}
s_0^{\mathsf T}\\
w_1^{\mathsf T}\\
\vdots\\
w_K^{\mathsf T}
\end{bmatrix}.
\]

For exact computation:

\[
S_{\mathrm{exact}}
=

\begin{bmatrix}
s_0^{\mathsf T}\\
q_1^{\mathsf T}\\
\vdots\\
q_{M-1}^{\mathsf T}
\end{bmatrix}.
\]

For \(M=8\):

| Computation | Reverse cotangent lanes |
|---|---:|
| Exact force + uncertainty | 8 |
| ForceSketch \(K=4\) | 5 |
| ForceSketch \(K=3\) | 4 |
| ForceSketch \(K=2\) | 3 |
| Mean force only | 1 |

The wall-clock question is whether this reduction in cotangent lanes translates into real savings.

---

# 17. Strong Exact Baselines

The project must not compare ForceSketch only against a Python loop.

Required exact implementations:

1. individual-head serial VJPs;
2. serial centered-basis VJPs;
3. exact centered-basis batched VJP;
4. `torch.func`/`vmap` implementation where applicable;
5. `torch.compile` version where beneficial;
6. combined mean-force + exact uncertainty batched computation.

The fastest correct exact method becomes the main performance baseline.

If PyTorch batched VJP already achieves near-optimal sharing, that result must be reported.

---

# 18. Reference PyTorch Implementation

The model adapter should expose:

```python
class MHCAdapter:
    def energies(self, batch):
        """
        Return:
            [B, M]
        """

    def exact_head_forces(self, batch):
        """
        Debug/reference path.
        Return:
            [B, M, A_max, 3]
        """

    def vjp_for_seeds(
        self,
        batch,
        seeds,
        *,
        batched=True,
        create_graph=False,
    ):
        """
        seeds:
            [L, B, M]

        output:
            [L, B, A_max, 3]
        """
```

Preferred VJP structure:

```python
energies = adapter.energies(batch)  # [B, M]

mean_seed = torch.full(
    (1, B, M),
    1.0 / M,
    dtype=energies.dtype,
    device=energies.device,
)

sketch_seeds = make_sketch_seeds(
    method=method,
    M=M,
    K=K,
    batch_size=B,
    seed=seed,
)

all_seeds = torch.cat(
    [mean_seed, sketch_seeds],
    dim=0,
)

grads = torch.autograd.grad(
    outputs=energies,
    inputs=batch.positions,
    grad_outputs=all_seeds,
    is_grads_batched=True,
    create_graph=False,
    retain_graph=False,
)[0]

forces = -grads

mean_force = forces[0]
sketch_forces = forces[1:]
```

Any `vmap` fallback warning must be recorded.

---

# 19. Development Order

The project must be implemented in the following order.

## Phase 1 — Exact correctness

Implement exact force disagreement.

## Phase 2 — Strong exact batched baseline

Establish the fastest correct exact implementation.

## Phase 3 — ForceSketch estimators

Implement Gaussian, Haar, pairwise, and head subsampling.

## Phase 4 — Accuracy–latency result

Determine whether \(K=2\), \(3\), or \(4\) is viable.

## Phase 5 — Exact-fallback screening

Turn the approximate estimator into a practical decision system.

## Phase 6 — Cross-system validation

Run rMD17/generalization experiments.

## Phase 7 — Triton

Only now optimize reduction/supporting operations.

Do not reverse this order.

---

# 20. Mandatory Synthetic Correctness Test

Use

\[
E_m(x)
=

a_m^{\mathsf T}x+b_m.
\]

Then

\[
f_m=-a_m
\]

is analytically known.

Test:

- exact member forces;
- centered force matrix;
- exact centered basis;
- global variance;
- coordinate variance;
- atom variance;
- Gaussian unbiasedness;
- Haar unbiasedness;
- pairwise normalization;
- head subsampling;
- finite-\(K\) standard-deviation correction.

No molecular experiments should begin until these pass.

---

# 21. Full-Rank Exactness Test

For Haar/orthogonal ForceSketch with

\[
K=M-1,
\]

require

\[
\widehat v_d=v_d
\]

within numerical tolerance.

FP32 target:

\[
\max_d
\frac{
|\widehat v_d-v_d|
}{
|v_d|+\epsilon
}
<
10^{-5}.
\]

A failure here indicates an implementation or normalization error.

---

# 22. Monte Carlo Estimator Tests

For synthetic fixed force matrices:

1. draw at least \(10^5\) sketches;
2. estimate each uncertainty statistic;
3. verify sample mean convergence;
4. verify variance behavior with \(K\);
5. verify standard-deviation correction;
6. compare Gaussian and Haar variance.

Produce an internal diagnostic plot of estimator error versus \(K\).

This may become an appendix figure.

---

# 23. Real-Model Exactness Test

On a small batch:

Compare:

1. explicit member-head forces;
2. centered-basis exact forces;
3. batched centered-basis VJP;
4. direct variance from all member forces.

Verify equivalence for:

- coordinate variance;
- global trace;
- atom RMS score;
- atom MHC score.

---

# 24. Padding and Variable-Size Tests

For batched molecular systems with different numbers of atoms:

- padded coordinates must not affect energies;
- padded force entries must be zero;
- padded uncertainty entries must be zero;
- padded atoms must not affect maximum uncertainty;
- global disagreement must use valid atoms only.

Add explicit unit tests.

---

# 25. Experiment 1 — Exact MHC Reproduction

## Objective

Verify the pretrained model and reproduce its exact force-UQ behavior.

## System

3BPA.

## Calculate

For each structure:

- \(E_m\);
- \(f_m\);
- exact \(v_d\);
- exact \(u_a^{\mathrm{RMS}}\);
- exact \(u_a^{\mathrm{MHC}}\);
- global \(S\);
- maximum atom uncertainty;
- maximum component uncertainty.

## Output

One concise reproduction figure.

Do not attempt to reproduce the complete original MHC study.

---

# 26. Experiment 2 — Exact Performance Baselines

For representative systems and batches, measure:

1. explicit per-head force loop;
2. exact serial centered basis;
3. exact batched centered basis;
4. compiled exact implementation;
5. mean-force-only baseline.

Break latency into:

\[
T_{\mathrm{total}}
=

T_{\mathrm{forward}}
+
T_{\mathrm{mean}}
+
T_{\mathrm{UQ}}
+
T_{\mathrm{reduce}}.
\]

Record:

- median latency;
- IQR;
- peak GPU memory;
- kernel launches;
- cotangent lanes.

The fastest exact implementation must be frozen before evaluating ForceSketch speedups.

---

# 27. Experiment 3 — ForceSketch Fidelity

Evaluate:

- Gaussian;
- Haar orthogonal;
- centered Rademacher;
- pairwise;
- head subsampling.

Use

\[
K\in\{1,2,3,4,M-1\}.
\]

Use ten fixed random seeds for stochastic methods.

---

# 28. Fidelity Metrics

## Numeric fidelity

For global score:

\[
\frac{
|\widehat S-S|
}{
S+\epsilon
}.
\]

Also report:

- normalized RMSE;
- median relative error;
- 90th percentile error.

---

## Ranking fidelity

Report:

- Spearman correlation;
- Kendall correlation.

---

## Acquisition fidelity

For exact top-\(p\)% sets with

\[
p\in\{1,5,10\},
\]

report:

- recall;
- precision;
- Jaccard overlap.

Top-5% recall is the primary metric.

---

## Atom localization

Report:

- fraction of structures where exact and approximate methods identify the same most-uncertain atom;
- distance in atom rank when they disagree.

---

# 29. Primary Figure — Accuracy–Latency Pareto

This is the most important figure in the paper.

Each point is a method and value of \(K\).

X-axis:

```text
incremental force-UQ latency
```

Y-axis:

```text
recall of exact top-5% uncertainty set
```

Include:

- exact batched VJP;
- Gaussian ForceSketch;
- Haar ForceSketch;
- pairwise;
- head subsampling.

The preferred method should lie toward the upper-left.

The paper is much stronger if ForceSketch Pareto-dominates simple head subsampling.

---

# 30. Secondary Pareto Figure

X-axis:

```text
total mean-force + UQ latency
```

Y-axis:

```text
Spearman correlation with exact uncertainty
```

This prevents a large incremental-UQ speedup from obscuring a negligible total workflow benefit.

---

# 31. Experiment 4 — Batch Scaling

Use

\[
B\in\{1,4,16,64\}
\]

subject to GPU memory.

Measure:

- incremental UQ latency;
- total force+UQ latency;
- structures/s;
- atoms/s;
- peak memory.

Determine where ForceSketch wins or loses.

Do not report only the best batch size.

---

# 32. Experiment 5 — Molecular-System Scaling

Use:

- 3BPA;
- small rMD17 molecule;
- medium rMD17 molecule;
- larger rMD17 molecule.

Ask:

> Does the speedup depend primarily on number of atoms, batch size, or reverse-lane count?

Plot:

\[
\text{ForceSketch speedup}
\]

against system size.

---

# 33. Experiment 6 — Uncertainty Screening Gate

This experiment converts ForceSketch from an approximation into a practical computational policy.

Split data into:

- calibration: 20%;
- validation: 20%;
- final test: 60%.

Do not tune the screening rule on the test data.

---

## Gate construction

Let exact score be

\[
S(x)
\]

and approximate score

\[
\widehat S(x).
\]

On calibration structures calculate

\[
r_i
=

\frac{
S(x_i)
}{
\widehat S(x_i)+\epsilon
}.
\]

Select a conservative quantile

\[
c_\alpha.
\]

Define

\[
U(x)
=

c_\alpha
\widehat S(x).
\]

For exact high-uncertainty threshold \(\tau\):

```text
if U(x) < tau:
    treat configuration as safely below threshold
    skip exact force-UQ
else:
    compute exact force-UQ
```

---

# 34. Gate Metrics

Report:

- high-UQ recall;
- false-negative rate;
- fraction of exact computations skipped;
- total screening latency;
- total screening speedup;
- precision;
- results under distribution shift.

Primary plot:

X-axis:

```text
fraction of exact UQ evaluations skipped
```

Y-axis:

```text
recall of exact high-UQ configurations
```

This may be the most application-relevant figure.

---

# 35. Experiment 7 — Reference Force Error

Treat this as secondary.

Test whether approximate and exact disagreement identify large reference-force errors similarly.

Report:

- AUROC;
- area under precision-recall curve;
- risk-coverage;
- mean force error among top uncertainty selections.

The main claim remains approximation of exact committee uncertainty, not universal uncertainty calibration.

---

# 36. Experiment 8 — Probe-Type Ablation

Compare at equal \(K\):

- Gaussian;
- Haar;
- Rademacher;
- pairwise;
- head subsampling.

Determine:

- estimator error;
- acquisition recall;
- seed-to-seed variance;
- GPU latency.

The ideal result is:

> Orthogonal head-space sketches reduce estimator variance and improve acquisition fidelity relative to independent random probes or member subsampling at the same VJP count.

---

# 37. Experiment 9 — \(K\) Ablation

For each method:

\[
K=1,2,3,4,\ldots,M-1.
\]

Plot:

- error versus \(K\);
- acquisition recall versus \(K\);
- latency versus \(K\).

This directly exposes the approximation-cost curve.

---

# 38. Optional Experiment — Head-Space Spectrum

Only run this after the primary paper works.

On a calibration set compute

\[
A_i
=

PF_i^{\mathsf T}F_iP.
\]

Average:

\[
\bar A
=

\frac{1}{n}
\sum_iA_i.
\]

Analyze its eigenvalues.

If the head-disagreement subspace is low rank, that may explain why small \(K\) works.

Plot cumulative spectrum:

\[
\frac{
\sum_{j=1}^{r'}
\lambda_j
}{
\sum_{j=1}^{M-1}
\lambda_j
}.
\]

This could provide useful mechanistic explanation.

---

# 39. Optional Method — Learned Control-Variate Subspace

Only implement if the spectrum is strongly concentrated.

Compute leading eigenvectors

\[
Q_{r_0}
\]

from calibration data.

Evaluate those directions exactly.

Sketch only the residual:

\[
R
=

P-Q_{r_0}Q_{r_0}^{\mathsf T}.
\]

Then approximate:

\[
S
\approx
S_{\mathrm{exact\ low-rank}}
+
S_{\mathrm{sketched\ residual}}.
\]

Do not include this if it complicates the story without significant gains.

---

# 40. Triton: Role in the Project

Do not start Triton implementation until:

- exact batched baseline works;
- ForceSketch accuracy works;
- the accuracy-latency Pareto is promising.

The purpose of Triton is then to ensure that inexpensive postprocessing does not become an unnecessary bottleneck.

---

# 41. Triton Kernel

Input:

```text
sketch_forces: [K, B, A_max, 3]
atom_mask:     [B, A_max]
```

Output:

```text
coord_var
coord_std
atom_mhc_score
atom_rms_score
global_trace
max_atom_score
```

Fuse:

1. square;
2. reduction across \(K\);
3. normalization;
4. finite-\(K\) correction;
5. Cartesian reduction;
6. padding mask;
7. global accumulation;
8. maximum reduction.

Because

\[
K\leq M-1
\]

and \(M\) is small, unroll \(K\) where beneficial.

---

# 42. Triton Baselines

Compare:

1. eager PyTorch;
2. `torch.compile`;
3. Triton.

Measure:

- microkernel latency;
- launch count;
- memory traffic;
- percentage of total ForceSketch latency.

The paper should explicitly state if Triton changes total runtime by only a few percent.

That result is acceptable.

---

# 43. Triton Decision Rule

Do not allocate more than one development day to Triton unless profiling shows:

\[
T_{\mathrm{postprocess}}
>
10\%
\]

of ForceSketch runtime.

If postprocessing is less than approximately 10%, prioritize experiments and paper quality instead.

---

# 44. Performance Benchmark Protocol

For every GPU benchmark:

1. preload data;
2. preallocate tensors;
3. compile/JIT before timing;
4. complete autotuning;
5. run at least 100 warmup iterations;
6. use at least 500 measured iterations for fast workloads;
7. use CUDA events;
8. synchronize only at timing boundaries;
9. report median;
10. report IQR;
11. record peak memory;
12. report compilation separately;
13. keep hardware state as stable as practical.

Never time Python data loading as part of model execution.

---

# 45. Required Timing Decomposition

Every primary performance result must separate:

\[
T_{\mathrm{forward}}
\]

\[
T_{\mathrm{mean-force}}
\]

\[
T_{\mathrm{uncertainty}}
\]

\[
T_{\mathrm{reduction}}
\]

and

\[
T_{\mathrm{total}}.
\]

Report both:

### Incremental UQ speedup

\[
\frac{
T_{\mathrm{UQ,exact}}
}{
T_{\mathrm{UQ,sketch}}
}.
\]

### Total workflow speedup

\[
\frac{
T_{\mathrm{total,exact}}
}{
T_{\mathrm{total,sketch}}
}.
\]

Never conflate these.

---

# 46. Profiler Analysis

For at least one representative configuration, use Nsight Systems or equivalent.

Determine:

- which kernels dominate exact VJPs;
- whether batched VJP shares computation;
- whether sketching reduces repeated trunk execution;
- whether GPU occupancy changes with lane count;
- memory overhead of batched cotangents;
- whether postprocessing matters.

Produce an internal profiler table.

Include it in the paper only if it materially explains performance behavior.

---

# 47. Statistical Protocol

## Fixed seeds

Choose ten ForceSketch random seeds before final testing.

Commit them to configuration.

Do not remove bad seeds.

---

## Paired comparisons

Every approximate uncertainty must be compared with exact uncertainty on the identical structure.

---

## Bootstrap

Use paired bootstrap over structures.

Default:

\[
1000
\]

resamples.

Report 95% intervals for:

- Spearman;
- top-5% recall;
- Jaccard;
- AUROC;
- other selection metrics.

---

## Multiple systems

Do not average away system-specific failures.

Report results per molecular system first.

Aggregate only as a secondary summary.

---

# 48. Main Success Criteria

A positive submission should ideally satisfy, on at least two molecular systems:

\[
K\leq4,
\]

\[
\text{top-5\% recall}
\geq0.90,
\]

and

\[
\text{incremental UQ speedup}
\geq1.5\times.
\]

The strongest paper would show:

\[
K=2\text{ or }3,
\]

\[
\text{top-5\% recall}
\geq0.95,
\]

\[
\text{incremental speedup}
\geq2\times,
\]

and meaningful total workflow improvement.

---

# 49. Critical Baseline Criterion

ForceSketch should preferably outperform **random head subsampling** at equal VJP count.

For example:

\[
K=3\ \text{Haar directions}
\]

versus

\[
3\ \text{random head forces}.
\]

If these perform almost identically, the methodological argument becomes weaker.

Investigate:

- rank correlations;
- tail recall;
- seed variance;
- OOD behavior.

---

# 50. Systems Kill Gate

After implementing exact batching and \(K=2,3\):

If

\[
\frac{
T_{\mathrm{exact}}
}{
T_{\mathrm{ForceSketch}}
}
<
1.2
\]

for incremental UQ across realistic regimes:

1. inspect `vmap` fallbacks;
2. verify timing;
3. profile the graph;
4. verify whether shared trunk work is actually repeated.

If the result remains:

- reframe as a performance-crossover study;
- or terminate the project.

Do not try to rescue it with reduction kernels.

---

# 51. Statistical Kill Gate

If every method with

\[
K\leq4
\]

has both:

\[
\text{top-5\% recall}<0.85
\]

and

\[
\rho_{\mathrm{Spearman}}<0.85,
\]

try:

1. Haar probes;
2. RMS/global uncertainty instead of maximum component;
3. calibrated exact fallback;
4. low-rank head-space control variate.

If none works, terminate the primary method.

---

# 52. Paper Go/No-Go Gate

Before spending significant time on final writing, require:

- strongest exact batched baseline complete;
- mathematical tests passing;
- one complete accuracy–latency Pareto curve;
- top-5% recall \(\geq0.90\) for at least one useful \(K\);
- incremental UQ speedup \(\geq1.5\times\);
- evidence that head-space sketching is at least competitive with head subsampling.

---

# 53. Required Main Figures

## Figure 1 — Method schematic

```text
             shared molecular trunk
                      |
             M scalar energy heads
                      |
        +-------------+-------------+
        |                           |
      EXACT                    FORCESKETCH
        |                           |
   mean direction              mean direction
        +                           +
   M-1 centered                K centered
    directions                  directions
        |                           |
 exact force-UQ            approximate force-UQ
```

Label:

\[
M
\quad\text{vs.}\quad
K+1
\]

total VJP lanes.

---

## Figure 2 — Primary accuracy–latency Pareto

X-axis:

```text
incremental force-UQ latency
```

Y-axis:

```text
top-5% exact-acquisition recall
```

This is the main figure.

---

## Figure 3 — \(K\) tradeoff

X-axis:

\[
K.
\]

Y-axes in separate plots:

- top-5% recall;
- Spearman;
- latency.

Compare methods.

---

## Figure 4 — Scaling

Plot:

\[
\text{speedup over exact batched VJP}
\]

against batch size for multiple molecular systems.

---

## Figure 5 — Screening gate

X-axis:

```text
fraction exact calculations skipped
```

Y-axis:

```text
high-UQ recall
```

This provides the clearest practical interpretation.

---

# 54. Required Tables

## Table 1 — Primary result

Columns:

```text
method
K
Spearman
top-5% recall
top-5% Jaccard
incremental UQ latency
incremental speedup
total latency
total speedup
peak memory
```

---

## Table 2 — Scaling

```text
system
number of atoms
batch size
exact UQ latency
ForceSketch latency
incremental speedup
total speedup
```

---

## Table 3 — Screening policy

```text
method
K
exact evaluations skipped
high-UQ recall
false-negative rate
screening speedup
```

---

# 55. Paper Structure

Target approximately 7 pages of main text.

## Abstract

Approximately 150–200 words.

Structure:

1. problem;
2. exact computational bottleneck;
3. ForceSketch idea;
4. strongest fidelity result;
5. strongest speed result;
6. screening result;
7. limitation/crossover.

Do not write the final abstract until experiments are frozen.

---

# 56. Introduction

Approximate budget:

\[
0.75-1.0\text{ page}.
\]

Narrative:

1. uncertainty is important for learned molecular simulation;
2. multi-head models amortize ensemble forward computation;
3. force disagreement still requires multiple reverse-mode evaluations;
4. downstream decisions usually require uncertainty statistics rather than every member force;
5. this motivates directly approximating the relevant force-variance statistic;
6. introduce ForceSketch;
7. state experimental question: fidelity versus VJP count versus wall-clock latency.

---

# 57. Contributions Paragraph

Write only after experiments are known.

Intended structure:

> We make four contributions. First, we formulate multi-head force disagreement as a statistic of the centered energy Jacobian with respect to atomic coordinates. Second, we introduce randomized and orthogonal head-space VJP estimators that replace the complete \(M-1\)-direction centered calculation with \(K\) directions. Third, we derive and evaluate finite-\(K\) estimators for componentwise force standard deviations and characterize their effect on uncertainty ranking and acquisition. Fourth, we evaluate the resulting accuracy–latency tradeoff against exact batched VJPs and head subsampling, including a conservative screening policy that selectively falls back to exact force disagreement.

Modify based on actual results.

---

# 58. Problem Formulation Section

Approximate budget:

\[
0.5-0.75\text{ page}.
\]

Include:

- \(e(x)\);
- \(F(x)\);
- \(P\);
- exact variance;
- atom uncertainty;
- exact centered basis;
- mean-force direction.

End with:

> Exact force disagreement requires resolving an \(M-1\)-dimensional centered output space. Our question is whether downstream uncertainty decisions require all \(M-1\) directions.

---

# 59. Method Section

Approximate budget:

\[
1.0-1.25\text{ pages}.
\]

Primary method order:

1. Gaussian ForceSketch;
2. Haar-orthogonal ForceSketch;
3. finite-\(K\) std correction;
4. combined mean+sketch execution.

Pairwise and head subsampling can be shorter subsections or experimental baselines.

Avoid filling the main method section with GPU implementation details.

---

# 60. Experimental Setup Section

Approximate budget:

\[
0.75\text{ page}.
\]

Include:

- model;
- systems;
- uncertainty target;
- exact batched VJP;
- sketch methods;
- \(K\);
- random seeds;
- GPU;
- timing procedure;
- statistical metrics;
- calibration split.

Mention Triton only if used.

---

# 61. Results Section

Approximate budget:

\[
2.25-2.75\text{ pages}.
\]

Recommended order:

## 5.1 Can force disagreement be sketched?

Show approximation/ranking versus \(K\).

## 5.2 Does sketching beat simply using fewer heads?

Direct head-subsampling comparison.

## 5.3 Does fewer VJP directions reduce GPU time?

Show exact batched baseline versus ForceSketch.

## 5.4 What is the accuracy–latency tradeoff?

Show main Pareto plot.

## 5.5 Can approximate screening safely avoid exact computation?

Show exact-fallback gate.

This sequence tells one coherent story.

---

# 62. Triton Placement in the Paper

If the Triton kernel is useful, include no more than:

- one implementation paragraph;
- one ablation row or appendix table.

Example:

> We additionally fuse sketch-force reduction and uncertainty-score construction in Triton. This removes several small elementwise and reduction launches but accounts for only \(X\%\) of the total speedup; the dominant improvement arises from reducing reverse-mode directions.

This framing directly addresses the collaborator's concern.

---

# 63. Limitations Section

Discuss:

1. the estimator is approximate for \(K<M-1\);
2. extreme maximum-component statistics may require larger \(K\);
3. the method depends on a shared-trunk multi-head architecture for the strongest systems benefit;
4. exact batched VJPs can reduce the apparent speed advantage;
5. committee disagreement is only an uncertainty proxy;
6. calibration may shift OOD;
7. small committees limit the maximum possible speedup;
8. GPU gains depend on system size and batch size.

Do not hide failed regimes.

---

# 64. Results Interpretation: Strong Outcome

If \(K=2\) or \(3\) gives both:

- \(\geq95\%\) top-5% recall;
- \(\geq2\times\) incremental speedup;

lead the abstract and paper with:

> A small number of orthogonal head-space VJPs preserves uncertainty acquisition while materially reducing force-UQ cost.

The screening gate becomes the application demonstration.

---

# 65. Results Interpretation: Moderate Outcome

If \(K=4\) or \(5\) is needed:

Focus on:

- measured accuracy-cost continuum;
- orthogonal probes outperforming head subsampling;
- calibrated fallback;
- regime-specific performance.

Do not describe it as a dramatic acceleration.

---

# 66. Results Interpretation: Tail Statistics Fail

If global/RMS uncertainty works but maximum-atom uncertainty is noisy:

This is still scientifically useful.

Show that:

- quadratic/global statistics sketch efficiently;
- extreme-value statistics converge more slowly;
- exact fallback resolves the tail.

Frame the method as **screening**, not complete replacement.

---

# 67. Results Interpretation: Exact Batching Is Extremely Strong

If exact batched VJPs nearly eliminate the performance difference:

The paper should report a crossover analysis:

> Reducing cotangent directions provides little benefit for \(B=\ldots\) but becomes beneficial for \(\ldots\).

This is preferable to manufacturing a kernel result.

---

# 68. Agent Output Discipline

All experiment outputs must be machine-readable.

Recommended result schema:

```text
experiment_id
git_commit
dataset
system
split
checkpoint_hash
method
K
sketch_seed
batch_size
num_atoms
num_heads
precision
gpu
forward_ms
mean_force_ms
uq_ms
reduction_ms
total_ms
peak_memory_bytes
exact_score
approx_score
spearman
kendall
top1_recall
top5_recall
top10_recall
top5_jaccard
```

Do not manually transcribe results into tables.

---

# 69. Repository Layout

```text
forcesketch/
├── README.md
├── pyproject.toml
├── environment/
│   ├── lockfile.txt
│   └── system_info.json
│
├── configs/
│   ├── 3bpa.yaml
│   ├── rmd17.yaml
│   ├── benchmark.yaml
│   └── seeds.yaml
│
├── src/forcesketch/
│   ├── adapters/
│   │   └── mace_mhc.py
│   │
│   ├── exact/
│   │   ├── member_forces.py
│   │   ├── centered_basis.py
│   │   └── batched_vjp.py
│   │
│   ├── sketches/
│   │   ├── gaussian.py
│   │   ├── haar.py
│   │   ├── rademacher.py
│   │   ├── pairwise.py
│   │   └── head_subsample.py
│   │
│   ├── estimators/
│   │   ├── variance.py
│   │   ├── std_correction.py
│   │   └── scores.py
│   │
│   ├── screening/
│   │   └── fallback_gate.py
│   │
│   ├── kernels/
│   │   └── reduce_scores.py
│   │
│   ├── benchmark/
│   │   ├── timing.py
│   │   ├── memory.py
│   │   └── profiler.py
│   │
│   └── utils/
│       ├── reproducibility.py
│       └── logging.py
│
├── tests/
│   ├── test_linear_model.py
│   ├── test_exact_centered_basis.py
│   ├── test_gaussian.py
│   ├── test_haar.py
│   ├── test_pairwise.py
│   ├── test_std_correction.py
│   ├── test_batched_vjp.py
│   ├── test_padding.py
│   └── test_triton.py
│
├── scripts/
│   ├── 00_environment.py
│   ├── 01_exact_reproduction.py
│   ├── 02_exact_benchmarks.py
│   ├── 03_sketch_fidelity.py
│   ├── 04_pareto.py
│   ├── 05_scaling.py
│   ├── 06_screening.py
│   ├── 07_rmd17.py
│   ├── 08_profile.py
│   └── 09_figures.py
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── manifests/
│
├── analysis/
│   ├── metrics.py
│   ├── bootstrap.py
│   ├── tables.py
│   └── figures.py
│
└── paper/
    ├── main.tex
    ├── references.bib
    └── figures/
```

---

# 70. Agent Roles

## Coding Agent

Responsible for:

- environment;
- model integration;
- exact baselines;
- sketch implementations;
- correctness tests;
- timing;
- Triton if justified;
- profiling;
- reproducible experiment scripts.

The coding agent must not write performance conclusions.

It outputs raw numbers and diagnostics.

---

## Analysis Agent

Responsible for:

- metric computation;
- paired comparisons;
- bootstrap intervals;
- statistical plots;
- acquisition sets;
- Pareto construction;
- gate calibration;
- checking for cherry-picking.

The analysis agent must not discard failed configurations.

---

## Writing Agent

Responsible for:

- maintaining manuscript structure;
- turning frozen results into prose;
- ensuring every quantitative claim maps to a result file;
- distinguishing incremental and total speedup;
- describing limitations;
- keeping the algorithm—not kernels—as the paper's main contribution.

The writing agent must not invent missing results.

Use placeholders such as:

```text
[RESULT: top-5 recall K=3 Haar]
```

until values are frozen.

---

# 71. Daily Execution Schedule

## August 14 — Exact setup

### Coding

- freeze environment;
- obtain pretrained model/data;
- hash model;
- obtain head energies;
- obtain individual forces;
- implement exact variance metrics.

### Analysis

- inspect uncertainty distribution;
- define exact acquisition sets.

### Writing

- create manuscript skeleton;
- write notation/problem statement.

---

## August 15 — Strong exact baseline

Implement and benchmark:

- serial member forces;
- serial centered VJP;
- batched centered VJP;
- combined exact mean+UQ.

Determine strongest exact baseline.

Do not implement Triton.

---

## August 16 — Sketch implementation

Implement:

- Gaussian;
- Haar;
- pairwise;
- Rademacher;
- head subsampling.

Complete all synthetic tests.

---

## August 17 — Real-model correctness

Verify:

- full-rank reconstruction;
- batched equivalence;
- padding;
- component variance;
- atom scores.

Run first small 3BPA sketch sweep.

---

## August 18 — Statistical feasibility

Run:

\[
K=1,2,3,4,M-1.
\]

Calculate:

- Spearman;
- top-5% recall;
- Jaccard;
- head-subsampling comparison.

Apply statistical kill gate.

---

## August 19 — Accuracy–latency Pareto

Run full timing suite.

Produce first:

- primary Pareto plot;
- exact-versus-sketch table.

Apply systems kill gate.

Only continue if the central result is viable.

---

## August 20 — Primary 3BPA results

Freeze:

- main accuracy table;
- main performance table;
- \(K\) ablation;
- probe-method ablation.

Begin Results section.

---

## August 21 — rMD17

Run selected best methods on three molecular systems.

Do not rerun every weak baseline if compute is limited.

Retain:

- exact;
- best ForceSketch;
- head subsampling;
- optionally Gaussian versus Haar.

---

## August 22 — Screening gate

Implement calibration and exact fallback.

Produce:

- skipped exact calculations;
- false-negative rate;
- wall-clock saving;
- acquisition recall.

This should be prioritized over Triton.

---

## August 23 — Profiling

Profile:

- exact;
- \(K=2\);
- \(K=3\);
- \(K=4\).

Understand where time is actually saved.

Only now decide whether Triton is worthwhile.

---

## August 24 — Optional Triton or head-spectrum experiment

If postprocessing is material:

- implement Triton reduction.

Otherwise:

- analyze head-space spectrum;
- or improve screening analysis.

Do not implement kernels just because they were originally planned.

---

## August 25 — Freeze experiments

Rerun final configurations.

Freeze:

- random seeds;
- metrics;
- confidence intervals;
- plots;
- tables.

No new methodological ideas after this point.

---

## August 26 — Full results draft

Complete:

- Method;
- Experiments;
- Results;
- Limitations.

Generate publication-quality figures.

---

## August 27 — Full manuscript

Complete full 5–8 page paper.

Review central narrative:

> fewer head-space VJPs  
> \(\rightarrow\) preserved acquisition decisions  
> \(\rightarrow\) reduced uncertainty-computation time.

Anything not supporting this narrative should be moved to the appendix or removed.

---

## August 28 — Revision

Check:

- every numerical claim;
- exact baseline fairness;
- no cherry-picking;
- anonymization;
- figures;
- page count;
- supplementary material.

---

## August 29 — Submission

Perform only:

- correctness spot checks;
- formatting fixes;
- metadata checks;
- submission.

Do not begin new experiments.

---

# 72. Required Internal Questions Before Writing the Abstract

The agents must be able to answer:

### Q1

What is the best useful \(K\)?

### Q2

At that \(K\), what fraction of the exact top-5% uncertainty set is recovered?

### Q3

Does it beat head subsampling at equal computational budget?

### Q4

What is the incremental UQ speedup against exact batched VJP?

### Q5

What is the total mean-force-plus-UQ speedup?

### Q6

For which batch/system sizes does the benefit disappear?

### Q7

How many exact evaluations can the screening gate skip?

### Q8

How much high-uncertainty recall is retained?

### Q9

Does Triton materially affect total runtime?

If these questions cannot be answered, the paper is not ready.

---

# 73. Central Result Template

The ideal final result has the form:

> With an eight-head shared-trunk potential, exact force disagreement requires seven centered uncertainty directions in addition to the mean-force direction. Using \(K=\mathbf{[X]}\) orthogonal ForceSketch directions, we recover \(\mathbf{[Y]\%}\) of the exact top-5% uncertainty set while reducing incremental uncertainty latency by \(\mathbf{[Z]\times}\) and total force-plus-UQ latency by \(\mathbf{[W]\%}\). A conservative screening policy avoids \(\mathbf{[A]\%}\) of exact uncertainty evaluations while retaining \(\mathbf{[B]\%}\) of high-uncertainty configurations.

Every value must come directly from frozen experiment output.

---

# 74. Final Framing Rule

When deciding whether any experiment, kernel, ablation, or paragraph belongs in the paper, ask:

> **Does this help answer whether fewer head-space VJPs can preserve force-uncertainty decisions while reducing real computational cost?**

If the answer is no, it is secondary.

In particular:

- Triton fusion is secondary;
- kernel-count reductions are secondary;
- low-level reduction optimization is secondary.

The paper's core sequence is:

\[
\boxed{
\text{Exact multi-head force uncertainty}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Reformulate in centered head space}
}
\]

\[
\downarrow
\]

\[
\boxed{
M-1\text{ exact directions}
\rightarrow
K\text{ sketch directions}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Preserve acquisition decisions}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Reduce measured GPU UQ cost}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Use exact fallback only when necessary}
}
\]

That is the paper.
