# ForceSketch — §52 Paper Go/No-Go assessment

Execution stopped here per the agreed scope (plan Task 7.2). Every number below
traces to a file in `results/`; none was transcribed by hand.

**Systems:** 3BPA@1200K (2139 structures, disjoint/overlapping/same committees) and
rMD17 ethanol/aspirin/azobenzene (1000 structures each, joint 10-molecule committee).
M = 8 heads throughout. All estimator evaluation in float64; timing in float32.

---

## Verdict

**The §52 gate is NOT met as written.** Five of its six requirements pass; the
recall requirement fails, and it fails by a wide margin rather than marginally.

| §52 requirement | status | evidence |
|---|---|---|
| strongest exact batched baseline complete | **now complete** | `is_grads_batched` RESOLVED — disable the TensorExpr fuser before the first forward (50/50 calls, 3e-15 vs serial). It is the strongest baseline at B≤16. `torch.compile` gives 1.04–1.10×. `torch.func` remains unavailable. |
| mathematical tests passing | **pass** | 41/41, incl. §20, §21, §22, §23, §24 |
| one complete accuracy–latency Pareto curve | **pass** | `paper/figures/fig2_pareto_*.png` |
| top-5% recall ≥ 0.90 for at least one useful K | **FAIL** | best is 0.743 (control variate r0=2, K=4, 3BPA); 0.52–0.56 on rMD17 |
| incremental UQ speedup ≥ 1.5× | **pass** | **2.19–2.51×** at K=3; **3.11–4.03×** at K=2 (total workflow: 1.77–1.87× and 2.17–2.39×) |
| sketching competitive with head subsampling | **pass, conditional** | wins at equal total lanes *when an exact mean force is required* |

**§48 success criteria** (K≤4 **and** recall ≥0.90 **and** speedup ≥1.5× on ≥2 systems):
**not met** — the recall term fails on all four systems.

**§51 statistical kill gate:** does not trigger on 3BPA (Haar K=4 reaches ρ=0.859
against the 0.85 threshold — a narrow escape), but **triggers on all three rMD17
molecules**. Escalation rungs 3 (calibrated exact fallback) and 4 (low-rank control
variate) were both implemented in response, per the spec's ordering.

**§50 systems kill gate** (threshold 1.2× on incremental UQ): cleared comfortably and
uniformly — 2.19–2.51× at K=3 across every system and batch size tested.

---

## The result that does hold

Replacement fails; **screening succeeds, on every system tested**. Control variate
r0=2, K=4, α=0.05, evaluated on held-out 60% test splits with c_α and τ fitted on
the calibration split only:

| system | high-UQ recall | exact evaluations skipped | FNR | screening speedup |
|---|---|---|---|---|
| 3BPA@1200K | 0.982 | 86.0% | 0.018 | 1.25× |
| rMD17 ethanol | 0.973 | 79.3% | 0.027 | 1.16× |
| rMD17 aspirin | 0.988 | 76.5% | 0.013 | 1.12× |
| rMD17 azobenzene | 0.971 | 84.9% | 0.029 | 1.23× |

H5's targets (skip ≥50%, retain ≥95%) are met on all four with margin. This is
precisely the outcome §66 prescribes a framing for: **screening, not replacement.**

Screening speedups use the *measured* cost model T(L) = 10.38 + 11.78·L ms (least-squares
fit over the full lane scan), not lane counts. Lane count is not lane cost — the forward
pass is paid once regardless of L — and the lane-count model overstates the gate by
~0.06×. Both are recorded.

---

## §72 questions

**Q1 — best useful K?** K = 4 with an r0 = 2 control variate (5 total reverse lanes
including the mean-force lane). For plain sketching, Haar at K = 4.

**Q2 — top-5% recall at that K?** 0.743 on 3BPA; 0.52–0.56 on rMD17. **Below the
0.90 target on every system.** Reaching 0.90 requires K ≈ 6 of 7, a marginal saving.

**Q3 — beats head subsampling at equal budget?** Yes at equal *total* lanes when the
exact mean force is required: Haar K=3 (0.529) vs head-subsample K=3+mean (0.477) on
3BPA. No if the mean force is not required: head-subsample K=4 without a mean lane
scores 0.588. Both framings are in `results/raw/03_sketch_fidelity_*.jsonl`; the
distinction matters because MD needs the exact mean force.

**Q4 — incremental UQ speedup vs exact?** §45 defines this as
[T(L=8)−T(L=1)]/[T(L=1+K)−T(L=1)]. Against the **strongest available** baseline
(batched where it wins): **1.63× at B=1** and **3.18× at B=16**. Against a
serial-only baseline it was 2.19–2.51×. Note the statistic is unstable at small
batch because batching makes T(L=1+K) ≈ T(L=1) there.

**Q5 — total mean-force + UQ speedup?** T(L=8)/T(L=1+K). Against the strongest
baseline: **1.11× at B=1**, **1.85× at B=16**, **1.86× at B=64**. Against
serial-only it was a uniform 1.77–1.87×. **This is the headline correction of the
project**: the apparent batch-independence was an artifact of comparing to a serial
loop.

**Q6 — where does the benefit disappear?** **At B=1 and B=4.** Batched reverse mode
is nearly flat in lane count at small batch (25.5 ms at L=1 vs 30.5 ms at L=8), so
extra cotangents are almost free and ForceSketch buys 1.11×. The benefit returns at
B=16 (1.85×) and B=64 (1.86×, where batched exhausts the 8 GB card and serial wins).
Single-structure MD is the regime where this method does not pay for itself here.

**Q7 — exact evaluations the gate can skip?** 76.5–86.0% at α = 0.05.

**Q8 — high-uncertainty recall retained?** 97.1–98.8%.

**Q9 — does Triton materially affect total runtime?** **No, measured.** On the real
committee at B=16, the full §41 postprocessing chain costs 0.16 ms against a 46.2 ms
ForceSketch(K=3) step — **0.3% of runtime**, thirty times below §43's 10% threshold.
The profile also shows why: the shared forward is 9.4 ms of a 100.6 ms exact step, so
the reverse pass dominates and postprocessing is negligible against it. §43's rule
therefore says do not implement the Triton kernel, and §42 explicitly permits
reporting that.

---

## Caveats that must survive into any write-up

1. **RESOLVED — batched reverse mode now works, and it changed the conclusion.**
   Root cause: `is_grads_batched` runs under `torch._vmap_internals`, so cotangents
   are BatchedTensors with no storage; TorchScript's profiling executor emits an
   optimized plan after two warm-ups, and the TensorExpr fuser fuses the *reverse*
   graph into a `prim::TensorExprGroup` whose kernel needs a raw `data_ptr()`. The
   TorchScript on the path is e3nn's `_spherical_harmonics`, a module-level
   `@torch.jit.script` **free function** — not a module, which is why walking the
   module tree finds nothing and why a fully rebuilt 0-ScriptModule model still
   fails. Fix: disable the TensorExpr fuser **before the first forward** (it is dead
   afterwards — the plan is cached per graph). Verified 50/50 calls at 3e-15 vs
   serial, costing ~4% on the serial path from lost fusion.
   **Consequence: the speedups above are the corrected, regime-dependent ones.**

2. **cuEquivariance is unavailable** with the PR #800 readout, so all baselines are
   e3nn-only.

3. **§38's stated hypothesis is not supported.** The spectrum is fairly flat (stable
   rank 4.6–5.1 of 7 across systems), which is the regime that *favours* random
   sketching — so spectral concentration is not what limits recall here. The limit is
   simply estimator variance at small K.

4. **Timing is single-GPU, single-run.** §44's full protocol (≥500 iterations,
   counterbalanced ordering, subprocess isolation) is implemented for the lane scan
   only; medians and IQRs are recorded throughout. §46's phase profiling is done
   (Q9 above); of §26's six implementations the batched and `torch.func` variants are
   unavailable on this stack, and the compiled-forward variant is now measured
   (1.04–1.10×, eroding ForceSketch speedups by 2–6%); §31's batch sweep is covered
   by the lane scan at B in {1,4,16,64}.

6. **Nsight Systems is not usable on this stack; kernel attribution came from
   `torch.profiler` instead.** Two independent obstacles: (a) e3nn ships 18 compiled
   `ScriptModule`s and TorchScript rejects forward hooks on them, so per-module NVTX
   ranges stop at the tensor-product boundary; (b) more fundamentally, nsys traces
   the whole process, and e3nn's TorchScript codegen at model load is so slow under
   it that an 8-minute capture never reached a single CUDA kernel (the resulting
   trace contains no kernel data). A separate hang was diagnosed to nsys waiting on
   re-parented children and is fixed by `--wait=primary`. `torch.profiler` attaches
   only around the region of interest and answers §46's question directly.

7. **Kernel-level finding (§46).** The GPU-time composition is invariant in lane
   count -- elementwise 42.2% at L=7 vs 41.5-41.9% at L=2,3,4; tensor products 21.3%
   vs 20.0-20.9%; GEMM 16.2% vs 16.3-16.4%; the same 134 distinct kernels throughout.
   Only the repetition count changes (10,161 launches per exact step vs 4,796 at
   K=3). So sketching reduces repeated trunk execution linearly, which is the
   cleanest possible mechanism and means no kernel-level optimization is being
   left on the table.

5. **rMD17 spans 9–24 atoms**, so molecule size is a weak axis; the size story rests
   on atoms-per-batch.

---

## Recommendation

The work supports a **§66-framed paper**: exact multi-head force uncertainty can be
screened, not replaced, by a small number of head-space VJPs — skipping ~80% of exact
evaluations while retaining ~98% of high-uncertainty structures, across four systems,
with a low-rank control variate as the enabling ingredient.

It does **not** support the headline the spec hoped for (K ≤ 4 replacing exact UQ at
≥0.90 recall). Presenting it as such would require dropping the rMD17 results, which
§47 and §70 both forbid.

The decision this needs from a human: whether a screening-framed result is worth
submitting, or whether to spend the remaining time on the §51 rungs not yet tried —
notably a *learned* (rather than eigen-) subspace, or per-atom rather than per-
structure gating.
