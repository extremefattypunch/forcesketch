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

# J8 water benchmark — verdict on the pre-registered predictions

**3 / 4 confirmed** (5 of 6 individual criteria), on a whole
**A100-SXM4-80GB with MIG disabled** (non-exclusive node, co-tenancy recorded;
14 min, 60 cells, 25 warmup + 50 timed iterations each).

> **Corrected twice, and the run should not be called clean.** The original
> headline was 4/4, from a scorer that tested only ONE of the two conditions
> stated in each of P3 and P4; with both halves scored it is 3/4, P3's
> quantitative half failing (transition ratio 1/64 against a pre-registered
> [1/33, 1/16]). Separately, the IQR characterisation here — "0.6–1.3 ms,
> under 1% of medians" — does not hold: recomputed from the run's stdout the
> absolute IQR spans 0.27–7.17 ms and the relative IQR 0.11–10.71%, with 19 of
> 60 cells above the 3% limit its own frozen v2 protocol sets. **The run fails
> the admissibility gate it was scored against.** The publishable systems
> result is the matched Blackwell pair in `J8_MATCHED_PAIR.md` (2/4 under the
> pre-registered v2 protocol), not this one.

Scored by `experiments/j8_water_verdict.py` against
`protocols/j8_water_prediction.yaml`, which was written — with its falsification
clause — before any water timing existed. An earlier MIG-slice run gave the same
verdict and the same 1.63× at B=16; that run is retained in
`results/records/j8_water_verdict_source_MIG.jsonl` and is not quoted as a timing.

| | prediction | measured | |
|---|---|---|---|
| **P1** | batched slope at B=16 exceeds 1 ms/lane | **49.76 ms/lane** | **PASS** |
| **P2** | total speedup at B=16 ≥ 1.4× | **1.63×** | **PASS** |
| **P3** | transition occurs at B ≤ 4 | first B with slope > 1 is **B = 1** | **PASS** |
| **P4** | serial slope exceeds 3BPA's ~10.9 ms/lane | mean **36.9** (range 17.3–94.3) | **PASS** |

P1 was the falsification clause: had it failed, the protocol committed us to
dropping acceleration from the paper rather than scoping it. It passed by ~50×.

## Fitted cost model, water (192 atoms, ~17,150 edges/structure)

| B | ≈ edges | serial intercept | serial slope | batched intercept | **batched slope** |
|---|---|---|---|---|---|
| 1 | 17,150 | 48.33 | 17.26 | 58.17 | **1.33** |
| 2 | 34,300 | 45.98 | 17.74 | 55.61 | **2.49** |
| 4 | 68,600 | 42.82 | 18.86 | 43.45 | **11.23** |
| 8 | 137,200 | 37.52 | 25.40 | 43.74 | **25.34** |
| 16 | 274,400 | 62.60 | 47.98 | 67.27 | **49.76** |
| 32 | 548,800 | 108.85 | 94.27 | 113.01 | **99.56** |

3BPA on the same GPU: batched slope 0.51–0.57 for B ≤ 16, 8.08 at B = 64; serial
slope ~10.9 and independent of B.

## 1. The headline: acceleration is real on water at deployable batch sizes

Total force+UQ speedup against the fastest correct baseline selected
independently per cell (R5):

| system | B | edges | exact L=8 | K=3 (L=4) | **speedup** | 5-lane gate |
|---|---|---|---|---|---|---|
| 3BPA | 1 | 520 | 29.6 | 27.6 | 1.07× | 1.05× |
| 3BPA | 16 | 8,320 | 29.9 | 27.9 | 1.07× | 1.05× |
| 3BPA | 64 | 33,280 | 81.4 | 51.5 | 1.58× | 1.38× |
| **water** | 1 | 17,150 | 70.0 | 61.1 | **1.15×** | 1.11× |
| **water** | 4 | 68,600 | 135.2 | 88.9 | **1.52×** | 1.35× |
| **water** | 16 | 274,400 | 438.7 | 269.1 | **1.63×** | 1.41× |
| **water** | 32 | 548,800 | 847.9 | 514.8 | **1.65×** | 1.42× |

Water reaches 1.52× at **B = 4** and saturates near 1.65×. 3BPA needs **B = 64**
to reach 1.58× and yields 1.07× at any batch a person would actually use for a
27-atom molecule. That is a **16× reduction in the batch size required**, against
a 33× ratio in edges per structure — the right order.

## 2. The transition is governed by total edges, and the threshold is ~10⁴

The batched lane slope crosses 1 ms/lane at:

- 3BPA: 0.57 at 8,320 edges → 8.08 at 33,280 edges
- water: **1.33 at 17,150 edges** → 2.49 at 34,300 → 11.23 at 68,600

Both cross in the **10⁴–1.5×10⁴ total-edge** range, despite a 7× difference in
atoms per structure and a 64× difference in the batch size at which it happens.
**Total edges in the batch is the controlling variable, not structures and not
atoms**, and ~10⁴ edges is a threshold a practitioner can check against their own
system before deciding whether lane reduction is worth implementing.

The marginal per-edge cost above the threshold is *not* universal: normalising by
SM count, water settles at ≈0.0196 ms per (edge/SM) per lane across B = 4…32
(spread 1.13×), while 3BPA at a comparable edge count gives ≈0.026. So the
threshold transfers; the slope constant is system-dependent — the two constants quoted here differ by ~1.3×, not the factor of
~3. Reported as such rather than fitted into a single law.

## 3. An R5 result that matters: the fastest exact baseline flips with system

| B | water, L=8 |
|---|---|
| 1 | batched |
| 2–4 | batched at L≥4, serial at L=1 |
| 8–32 | **serial** (batched a few percent slower) |

> **An unsourced number was removed here.** This row originally read
> "batched (70.0 vs 76.7)". The 70.0 is real (B=1, L=8, batched). **The 76.7 is
> not: no cell in any surviving water record is within 1 ms of it**, and in every
> such record the B=1, L=8 *serial* time is far higher (204.2 on the A100 run,
> 44.8–50.6 on Blackwell). The A100 water record that this pair was read from was
> destroyed by the filename collision, so the figure cannot be re-derived — but it
> was wrong when written regardless of source. This is a different class of defect
> from the systematic bugs above: a specific number that does not trace to a
> record, i.e. a direct violation of rule R3. It was found by adversarial review,
> not by any test, and the tests that would have caught it (every quoted numeral
> must resolve to a record) do not yet exist.

For 3BPA, batched won essentially everywhere. **For water at B ≥ 8, serial wins.**
The mechanism is consistent with everything above: `is_grads_batched` runs the
backward under vmap and materialises L cotangent copies, which costs memory
bandwidth. Below ~10⁴ edges launch overhead dominates and batching wins; above it,
the bandwidth cost of the vmap'd cotangents exceeds the launch saving.

This is exactly why rule R5 requires selecting the fastest correct implementation
*independently at every setting* rather than fixing one. A paper that had fixed
"batched" from the 3BPA experiments would have overstated water's exact baseline
by 3–5% and quietly mis-specified its own comparison.

## 4. What this does to the paper's systems claim

The 3BPA-only picture read as "acceleration nearly vanishes on datacenter
hardware" (1.07× at B ≤ 16). With water the correct statement is narrower and
much more useful:

> Lane reduction pays once the backward pass is compute-bound, which happens above
> roughly 10⁴ edges per batch. Condensed-phase and large-molecule systems cross
> that threshold at any usable batch size — water reaches 1.52× at B = 4 and
> 1.65× at B = 32. Small molecules cross it only at large batch: 3BPA needs
> B ≈ 64 for 1.58×, and returns 1.07× below that.

Acceleration still should not be load-bearing in the title — the hardware-free
count (82–94% of exact evaluations avoided (journal pipeline; the workshop's 84–87% used a superseded split and statistic)) remains the better headline — but it
is no longer a near-null result. It is a scoped result with a measured threshold
and a mechanism, which is a considerably stronger position than the workshop's
single laptop number.

## Caveats

- Non-exclusive node. IQRs are under 1% of medians and the effects are 1.5–100×,
  so co-tenancy cannot alter any conclusion; absolute milliseconds should still be
  regenerated on an exclusive node before publication (that job is queued).
- L = 5 is interpolated between measured L = 4 and L = 8.
- The residual R5 asymmetry stands: the TensorExpr fuser is disabled process-wide,
  which costs the serial path some fusion. Since serial *wins* for water, this
  asymmetry now works **against** the reported water baseline — i.e. the true
  serial baseline is slightly faster and the water speedups quoted here are, if
  anything, mildly optimistic. Process-per-implementation timing is still needed.
- One earlier `gpu_requeue` attempt was preempted mid-run and discarded the whole
  grid. `j8_lane_bench.py` needs the cell-level restart the plan specifies before
  it is used for a long campaign on a preemptible partition.
