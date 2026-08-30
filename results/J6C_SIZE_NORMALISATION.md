# J6C / J6D — repairing the extensive statistic, and what the committee is worth

Two follow-ups to J6B, which found that on a variable-size pool the global
statistic is extensive, loses to max-component, and is itself beaten by a free
signal. J6C attempts the repair the extensivity diagnosis implies. J6D asks the
question the free-signal result forces.

**J6C is a post-hoc repair with pre-specified criteria** — the estimator family,
the α grid, the selection procedure and the six success criteria were fixed in
`protocols/j6c_size_normalisation.yaml` (sha256 `3979912824746501…`) before any
normalised quantity was computed, but the protocol was written *after* J6B exposed
the defect. That is a weaker guarantee than J6B's own pre-registration and is
labelled as such. **J6D is exploratory**: not pre-registered, nothing scored.

---

# J6C — size normalisation

## The estimator

`S_α(x) = (Σ_d v_d(x)) / (3N_x)^α`, with α = 0 the current global statistic and
α = 1 the mean per-coordinate variance. Intermediate α is admitted because the
*target* is itself mildly size-dependent — `e_max` is a maximum over N atoms, so it
grows with N even for a perfect model. Measured:

| target | corr with N |
|---|---|
| e_max | +0.185 |
| e_maxcomp | +0.175 |
| e_rmse | +0.123 |
| e_q95 | +0.112 |

So a fully intensive statistic should *over*-correct, and the grid must be allowed
to land between the two extremes. (Any strictly monotone transform of `S_α` — its
square root, say — gives an identical ranking and is not a separate hypothesis.)

## N1 — free at fixed N, but the check as first written was worthless

**Max |AUROC change| = 0.000e+00 across 20 distinct fixed-N systems** (23 cache
files; three are the same system under two naming conventions), for every α on
the grid.

> **Corrected after an adversarial audit.** The original version of this check
> was a **tautology**. It synthesised its own constant atom-count vector, so the
> normalisation reduced to dividing by one positive scalar, and AUROC is
> invariant under that. I verified the auditor's claim by monkey-patching the
> shipped function with three deliberately broken implementations — an
> operator-precedence bug, N taken from the wrong structure, and **no
> normalisation at all** — and every one returned 0.000e+00 and "passed". The
> check protected nothing.
>
> It is now split into what each part is worth: the fixed-N invariance is
> reported as a **derivation that cannot fail** (kept because the claim is
> load-bearing, not because it is evidence), alongside a **real** check —
> agreement with an independently written reference on *variable* N, max relative
> error 3.9e-16 — and a **mutation test** proving that check rejects all three
> broken implementations. The system count was also wrong: it counted files, not
> systems.

## α* = 0.75, chosen on a design split that is used for nothing else

| α | design AUROC (e_max) |
|---|---|
| 0.00 | 0.7997 |
| 0.25 | 0.8125 |
| 0.50 | 0.8236 |
| **0.75** | **0.8295** ← α* |
| 1.00 | 0.8264 |
| 1.25 | 0.8120 |
| 1.50 | 0.7871 |

## Held-out results (123,231 structures)

| estimator | e_max | e_rmse | e_maxcomp | e_q95 | corr with N |
|---|---|---|---|---|---|
| S₀ (the current global statistic) | 0.797 | 0.768 | 0.792 | 0.759 | 0.632 |
| S₀.₅ | 0.818 | 0.803 | 0.812 | 0.794 | 0.424 |
| **S₀.₇₅ (α\*)** | **0.821** | 0.817 | 0.815 | 0.808 | **0.275** |
| S₁.₀ | 0.815 | **0.822** | 0.809 | **0.814** | 0.115 |
| exact max-component | 0.820 | 0.807 | 0.821 | 0.804 | 0.286 |
| exact max-atom | **0.835** | 0.818 | **0.823** | 0.806 | 0.366 |
| `‖f̄‖` (free) | **0.942** | **0.933** | **0.942** | **0.930** | 0.172 |

**The α that wins tracks how size-dependent the target is.** `e_rmse` is intensive
(an RMS over 3N) and peaks at α = 1.0; `e_max` is a maximum over N atoms, mildly
extensive, and peaks at α = 0.75. That is a coherent mechanism rather than a fitted
curiosity, and it is why a single universal α should not be claimed.

## Scoring

| | criterion | result |
|---|---|---|
| N1 | invariance at fixed N | **PASS**, but the original check was vacuous — see below |
| N2 | α\* > 0 | **PASS** (0.75) |
| N3 | beats S₀ on all four error scores | **PASS** |
| N4 | beats max-component on e_max | **PASS** on the letter only; 2 wins / 1 tie / 1 significant loss across the panel |
| N5 | acquisition median atoms < 40 | **PASS** (32) |
| N6 | gate overlap restored to ≥ 0.90 | **FAIL** at both budgets (0.751 / 0.741) |

**5 of 6 by the registered criteria** — but N1's check could not fail and N4's was too loose to mean what it says, so the count overstates the result. The corrected reading is in the two subsections below.

### N4 passed the letter of the criterion and not its spirit — and only on one score

| comparison (held-out, e_max) | Δ AUROC | 95% CI | significant |
|---|---|---|---|
| S₀.₇₅ vs S₀ | **+0.0240** | [+0.0216, +0.0266] | yes |
| S₀.₇₅ vs max-component | +0.0010 | [−0.0013, +0.0032] | **no** |
| S₀.₇₅ vs `‖f̄‖` (free) | **−0.1213** | [−0.1257, −0.1169] | yes |

The criterion was written as "> 0" and the point estimate is +0.0010, so it
passes. But the interval straddles zero at n = 123,231, so the defensible claim is
**parity, not superiority**: normalisation removes the deficit J6B found (−0.023,
significant) and does not convert it into an advantage. Reported this way rather
than as a clean pass, because a criterion that a +0.001 non-significant difference
can satisfy was a criterion written slightly too loosely.

Note also that **max-atom (0.835) still beats S₀.₇₅ (0.821)** on e_max. The best
*exact committee* statistic on this pool is neither the global one nor the
max-component one.

> **The bigger correction: "the deficit is repaired" was generalised from the
> single score α was tuned on.** The audit asked for the full panel, and it does
> not support the claim:
>
> | error score | S₀.₇₅ − max-component | 95% CI | |
> |---|---|---|---|
> | e_max | +0.0010 | [−0.0013, +0.0032] | tie |
> | e_rmse | **+0.0100** | [+0.0076, +0.0123] | significant win |
> | e_q95 | **+0.0043** | [+0.0019, +0.0067] | significant win |
> | **e_maxcomp** | **−0.0058** | [−0.0085, −0.0036] | **significant loss** |
>
> So the panel is **2 significant wins, 1 tie, 1 significant loss** — not "the
> deficit is repaired". The surviving `e_maxcomp` deficit is about six times the
> magnitude of the e_max gap the parity claim rests on, and points the other way.
> J6B established its *failure* on all four scores with every one significant;
> reporting J6C's *success* from one score would have been an asymmetric
> evidential standard. The registered N4 criterion (a bare `> 0` point
> comparison) is left as registered and still reads PASS — retro-tightening it
> would be exactly the post-hoc tuning the protocol exists to prevent — but the
> record now carries the panel and a note, and **"5/6" should not be quoted
> without them**.

## Acquisition — the size bias is much reduced, not eliminated

| estimator | median atoms, top-1000 | median atoms, top-8000 |
|---|---|---|
| S₀ | 80 | 70 |
| **S₀.₇₅** | **36** | **32** |
| max-component | 32 | 28 |
| `‖f̄‖` | 28 | 30 |
| *(pool)* | *20* | *20* |

Pearson correlation with atom count falls 0.632 → 0.275, apparently matching
max-component's 0.286.

> **Corrected.** Acquisition is `topk` — a pure **rank** operation — so Spearman
> is the correlate that governs which structures get picked, and it tells a
> different story:
>
> | estimator | Pearson | **Spearman** |
> |---|---|---|
> | S₀ | +0.632 | +0.816 |
> | **S₀.₇₅** | +0.275 | **+0.555** |
> | max-component | +0.286 | **+0.457** |
> | max-atom | +0.366 | +0.566 |
> | `‖f̄‖` | +0.173 | +0.542 |
>
> On ranks S₀.₇₅ remains **~20% more size-biased than max-component**, reversing
> the ordering the Pearson pair implies. The honest statement is that
> normalisation removes most of the size bias (0.816 → 0.555 on ranks) but does
> **not** bring it to the max-component baseline.

## N6 — the gate-overlap bound is still not available, as predicted

| | budget 1000 | budget 8000 |
|---|---|---|
| unnormalised | 0.718 | 0.794 |
| normalised at α\* | **0.751** | 0.741 |

N6 fails at both budgets, so *"the overlap bound is not restored"* stands.

> **The mechanism I attached to it was wrong, and I withdraw it.** I wrote that
> the failure shows the gate's disagreement is "about the head-space subspace,
> not about system size". Three checks, all of which I reproduced:
>
> 1. **The comparison cannot test that.** Both sides are divided by the *same*
>    per-structure constant, so the per-structure discrepancy ratio `lo/g` is
>    unchanged by construction — I measured max\|lo₀/g₀ − lo\*/g\*\| = **2.2e-16**.
>    A change in Jaccard therefore cannot be attributed to subspace-versus-size.
> 2. **I quoted the one budget where it worsens.** Both budgets are in my own
>    record: normalisation *improves* overlap at 1000 (0.718 → 0.751) and worsens
>    it at 8000. Quoting only the latter and generalising the sign was a
>    cherry-pick.
> 3. **The gate's fidelity *is* size-dependent**, which directly contradicts the
>    claim. The spread of `log(lo/g)` falls monotonically with system size —
>    0.340 (N ≤ 8), 0.306, 0.220, 0.198, **0.172** (N ≥ 81) — i.e. the gate
>    reproduces the exact statistic markedly more faithfully on large structures,
>    by self-averaging over more coordinates.
>
> What survives is only the scored result: **the bound is unavailable at both
> budgets.** No mechanism is claimed.

---

# J6D — is the committee worth anything on top of the free signal?

J6B's uncomfortable result was that `‖f̄‖` beats every committee statistic. But a
practitioner already *has* `‖f̄‖` — it falls out of the forward pass. So the real
decision is not "committee or free signal", it is "free signal, or free signal plus
eight backward passes". Nested logistic models, fitted on design splits, scored on
held out.

## On the MPtraj pool: nothing

| model | held-out AUROC | marginal over `‖f̄‖` alone | significant |
|---|---|---|---|
| `‖f̄‖` alone | 0.9425 | — | — |
| committee alone | 0.8212 | −0.1213 | yes |
| N alone | 0.6576 | −0.2849 | yes |
| `‖f̄‖` + N | 0.9425 | +0.0000 | no |
| **`‖f̄‖` + committee** | **0.9427** | **+0.0002** | **no** |
| `‖f̄‖` + committee + N | 0.9427 | +0.0002 | no |

**The committee's information is, here, essentially a subset of the free signal's.**
Eight backward passes add 0.0002 AUROC to something already computed.

## On fixed-N systems: a great deal

Same method, using the project's own blocked splits (design role fits, test role
scores):

| system | `‖f̄‖` | committee | both | marginal | sig |
|---|---|---|---|---|---|
| rMD17 azobenzene | 0.580 | 0.836 | 0.823 | **+0.243** | yes |
| 3BPA same | 0.701 | 0.847 | 0.862 | **+0.161** | yes |
| 3BPA disjoint | 0.610 | 0.767 | 0.756 | **+0.146** | yes |
| 3BPA naive-same | 0.721 | 0.869 | 0.860 | **+0.139** | yes |
| 3BPA naive | 0.750 | 0.856 | 0.877 | **+0.127** | yes |
| rMD17 aspirin | 0.685 | 0.742 | 0.783 | +0.099 | no |
| rMD17 ethanol | 0.702 | 0.760 | 0.786 | **+0.084** | yes |
| 3BPA overlapping | 0.650 | 0.784 | 0.719 | +0.068 | no |
| water-overlapping PIMD | 0.937 | 0.968 | 0.969 | **+0.033** | yes |
| water-disjoint PIMD | 0.954 | 0.974 | 0.976 | **+0.022** | yes |

**Significant on 8 of 10, marginals +0.022 to +0.243.** Four water MD systems are
excluded because their design split contains only 4 positives at the top-5%
threshold (design n ≈ 91) — a power limit, recorded rather than hidden.

## The statement this supports

**The committee is not redundant in general. It is redundant *here*.** On
single-composition systems it carries substantial information the free signal
lacks — up to +0.24 AUROC. On the chemically diverse MPtraj pool, under a frozen
foundation trunk, it adds nothing measurable.

That is a **system-class** scoping claim, and it is far more useful than either
"committee UQ works" or "committee UQ is dominated". It also suggests the
mechanism: where the free signal is weak (0.58–0.75, the molecular systems) the
committee has room to contribute; where the free signal is already near-saturated
(0.94 on MPtraj, 0.94–0.95 on water PIMD) the marginal collapses. Note that the two
water PIMD rows sit at the saturated end — and J6 showed their high numbers are
largely an artifact of the model failing badly there.

---

## Consequences for the paper

1. **Adopt `S_α` with the fixed-N invariance stated.** It costs nothing on every
   system already measured (exactly nothing, 0.000e+00) and removes a real defect
   on variable-N pools. Report α as target-dependent, not universal.
2. **State the repair honestly**: normalisation restores *parity* with
   max-component on MPtraj, not superiority, and max-atom still beats both.
3. **Report the free-signal baseline everywhere.** J6D makes the criterion
   concrete: the question a referee will ask is not "does committee UQ beat
   random?" but "does it beat `‖f̄‖`, which is free?" — and the answer is
   system-class dependent, with numbers.
4. **The gate-overlap bound stays unavailable** at both budgets. No mechanism is
   claimed for why — the comparison that looked like a diagnosis turned out to be
   incapable of being one.
5. **Do not quote α = 0.75 as a constant.** The audit showed it is stable to
   resampling *this* pool (0.75 in 20/20 alternative design splits) but moves
   from 0.30 to 1.25 under changes in the pool's size composition, and two of the
   four registered error scores would have selected α = 1.0 — which loses to
   max-component significantly. α must be refitted per pool and per target, and
   reported with that dependence.
