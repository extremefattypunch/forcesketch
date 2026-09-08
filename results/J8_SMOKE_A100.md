> ## ⚠ TIMING NUMBERS IN THIS DOCUMENT ARE SUSPENDED
>
> An adversarial review found two defects in `experiments/j8_lane_bench.py` that
> affect every timing number here. Both are fixed in code; the measurements need
> re-running before any number below is quoted.
>
> **1. The `L=8` cell ran 7 lanes.** `exact_seed_bundle(M=8).seeds` holds only the
> `r = M-1 = 7` centred directions, so `seeds[:8]` silently returned 7. Every cell
> labelled `L=8` measured 7 lanes, and — worse — the least-squares fits regressed
> that point against `x=8`, biasing every reported **batched slope low by 15–18%**.
> Reported speedups were `T(7)/T(4)`, not `T(8)/T(4)`.
>
> **2. A filename collision destroyed a record.** The water sbatch hard-coded
> `--out …_water.jsonl` regardless of device, so the water-on-Blackwell run
> overwrote the water-on-A100 run. The surviving file is Blackwell (188 SMs); the
> A100 water numbers quoted here are no longer backed by any record, which
> violates rule R3. The file is renamed and the sbatch now derives the device.
>
> Qualitative conclusions (which regime is launch- vs compute-bound, the ordering
> of GPUs, the ~10⁴-edge threshold) are unaffected in direction, since both defects
> are monotone and affect all cells alike. The specific milliseconds, slopes and
> speedups are not to be quoted until the re-run lands.

# J8 smoke test — the acceleration kill gate, answered

A100-SXM4-80GB, MIG disabled, `--exclusive` whole node, 108 SMs.
torch 2.13.0+cu129, fp32, CUDA events, 100 warmup + 200 timed iterations per cell,
median and IQR. CUDA-event overhead floor measured at **0.0076 ms**, so every
number below is ~3000× the measurement floor.
Records: `results/records/contaminated/j8_lane_bench_NVIDIA_A100-SXM4-80GB_cotenant_gpu_requeue.jsonl (NOTE: the file at the original path was later overwritten by a co-tenanted gpu_requeue re-run and moved to contaminated/; this document's numbers come from the exclusive-node job 39408005, whose stdout in slurm/logs/ is now their only source)` (40 cells).

Batch construction verified independently — B = 64 is genuinely 1728 atoms and
33,138 edges, 64× the work of B = 1.

---

## 1. The serial path is launch-bound, and its cost model transfers across hardware

| impl | B | intercept (ms) | slope (ms/lane) | R² |
|---|---|---|---|---|
| serial | 1 | 14.44 | 10.92 | 0.994 |
| serial | 4 | 14.39 | 10.93 | 0.994 |
| serial | 16 | 14.16 | 11.07 | 0.994 |
| serial | 64 | 14.30 | 10.99 | 0.993 |
| **laptop RTX 5070 (workshop)** | 1 | **10.38** | **11.78** | — |

Two things stand out. **Serial time is independent of batch size** — identical to
within 1% from 27 atoms to 1728 atoms. And the **slope agrees with the laptop's to
within 8%** despite roughly a 10× difference in GPU class.

The first — independence from batch size — is robust and holds on every device
tested: MACE's graph batching issues the same number of kernel launches
regardless of B, and at ≤33k edges an A100 is nowhere near compute-saturated.

> **The second was wrong, and the sm_120 Blackwell control falsified it.** I
> concluded here that the workshop's `T(L) = 10.38 + 11.78·L` "was never really a
> property of the RTX 5070" but of the launch pattern, on the strength of the
> A100 matching its slope to within 8%. On the RTX PRO 6000 Blackwell the serial
> slope is **5.10 ms/lane** — 2.3× lower (corrected: the 4.58 figure came from the pre-fix 7-lane run; see the banner in `J8_SM120_CONTROL.md`). The laptop/A100 agreement was a
> coincidence. What survives is that the serial slope is *batch-independent*
> (work-independent at these sizes), not that it is *hardware-independent*: the
> per-lane cost is fixed GPU work at low occupancy and scales with GPU speed. See
> `J8_SM120_CONTROL.md`.

## 2. The batched path stays flat in lane count much further up in B

| impl | B | intercept (ms) | slope (ms/lane) |
|---|---|---|---|
| batched | 1 | 25.46 | **0.52** |
| batched | 4 | 25.50 | **0.51** |
| batched | 16 | 25.50 | **0.57** |
| batched | 64 | 17.29 | **8.08** |

On the 8 GB laptop the batched path was already strongly lane-proportional at
B = 16 (26.0 → 73.2 ms from L=1 to L=8). On the A100 at B = 16 it is still
essentially **flat** (25.85 → 29.93 ms), and only at B = 64 does it become
lane-proportional.

## 3. The answer to the kill gate

Total force+UQ speedup against the fastest correct baseline selected independently
at every cell (rule R5 — batched wins everywhere except L = 1):

| B | exact (L=8) | K=3 (L=4) | speedup | 5-lane gate | speedup |
|---|---|---|---|---|---|
| 1 | 29.60 ms | 27.61 ms | **1.07×** | 28.11 ms | 1.05× |
| 4 | 29.48 ms | 27.72 ms | **1.06×** | 28.16 ms | 1.05× |
| 16 | 29.93 ms | 27.95 ms | **1.07×** | 28.45 ms | 1.05× |
| 64 | 81.39 ms | 51.47 ms | **1.58×** | 58.95 ms | 1.38× |

The workshop reported **1.85× at B = 16** on the laptop. On an A100 the same
configuration gives **1.07×**.

**The acceleration is weaker on datacenter hardware, not stronger.** That is the
opposite of the naive expectation and it is the honest headline of this
experiment.

### The mechanism, as transferable design knowledge

Reducing cotangent lanes only saves time when the backward pass is
**compute-bound**. While it is **launch-bound**, extra lanes ride along nearly
free and there is nothing to reclaim. A larger GPU pushes the compute-bound
threshold to larger batches, so:

> The batch size at which lane reduction begins to pay **scales with GPU
> capability**. On an 8 GB laptop it is B ≈ 16; on an 80 GB A100 it is B ≈ 64; on
> an H200 it will be higher still.

This is a rule another group can apply to their own stack without re-running our
benchmark, and it is worth more than any single speedup number.

### Consequences for the paper

1. **Acceleration must not be load-bearing.** At B ≤ 16 the benefit is 1.05–1.07×,
   which is indistinguishable from nothing in practice. This vindicates the
   decision already taken in the plan: lead with **82–94% of exact evaluations
   avoided** — a hardware-free count, invariant to GPU, autodiff backend and
   batch size — and present wall-clock as a scoping section with a measured
   boundary.
2. **The crossover claim from the workshop needs restating.** "Batched loses to
   serial at B = 64" was an 8 GB memory artifact; on 80 GB batched still wins at
   B = 64 (81.4 vs 100.4 ms). The workshop's statement should be cited as
   hardware-specific, which it was careful to do.
3. **Larger systems become necessary for the systems story, not just the
   statistical one.** Water is 192 atoms, 7.1× the atoms per structure of 3BPA, so
   B = 16 water is comparable in work to B ≈ 114 of 3BPA — well inside the
   lane-proportional regime. **Concrete prediction to test:** water at B = 16 on
   this same A100 should show a batched slope well above 1 ms/lane and a total
   speedup near the B = 64 3BPA figure (~1.5×). If it does, the deployment claim
   becomes "screening pays for condensed-phase and large-molecule pools at modest
   batch, and for small molecules only at large batch" — which is a defensible,
   useful, and honest statement.

## Caveats

- **One asymmetry remains open.** The TensorExpr fuser is disabled process-wide
  (it must be, for batched to work at all), and that costs the *serial* path some
  fusion — the workshop measured ~4%. Both implementations were timed in the same
  process here, so the serial baseline is mildly handicapped. Since batched wins
  almost everywhere, this does not change any conclusion above, but a published
  R5 comparison needs process-per-implementation. Recorded as
  `texpr_fuser_enabled: false` in every cell.
- **L = 5 is interpolated** between the measured L = 4 and L = 8, not measured.
- 27-atom 3BPA only. The N axis is exactly what item 3 above proposes to fix.
- Single node, single run. IQRs are 0.02–0.58 ms, i.e. 0.1–2% of the medians, so
  the ordering is not in doubt, but the protocol's counterbalanced ordering and
  subprocess isolation are not yet applied.
