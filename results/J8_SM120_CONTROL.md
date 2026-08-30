> ## ⚠ TIMING NUMBERS IN THIS DOCUMENT ARE SUSPENDED

> ## ⚠ SUPERSEDED NUMBERS — read this before quoting anything below
>
> Every timing in this document comes from the **pre-fix 7-lane run** (job
> 39413479), taken before the lane-count defect was found: `exact_seed_bundle`
> returns only the r = M−1 = 7 centred directions, so cells labelled L=8 executed
> 7 lanes. The corrected 8-lane record is
> `results/records/j8_lane_bench_NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.jsonl`
> and it moves the numbers materially:
>
> | quantity | this document (7 lanes) | corrected (8 lanes) |
> |---|---|---|
> | 3BPA serial slope, B=1 | 4.58 ms/lane | **5.104** |
> | serial slope across B | 4.58 / 4.57 / 4.61 / 4.92 | **5.10 / 5.39 / 5.41 / 5.78** |
> | total speedup by B | 1.07 / 1.07 / 1.28 / 1.62 | **1.09 / 1.10 / 1.45 / 1.83** |
> | per-lane cost vs laptop | 2.6× lower | **2.3× lower** |
> | guard fp64 agreement, Blackwell | 4.03e−15 | **2.77e−15** |
> | guard fp64 agreement, A100-80GB | 3.20e−15 | **1.92e−15** |
> | guard device coverage | four devices, two capabilities | **five devices, three** (8.0, 9.0, 12.0) |
>
> **The document's arguments survive; its numbers do not.** Blackwell is still far
> faster per lane than the laptop, the serial slope is still batch-independent
> (now within 13% rather than 8%), and 1.45× at B=16 is still well below the
> workshop laptop's 1.85×, so the conclusion that the workshop speedup was a
> property of GPU *size* rather than of sm_120 is unaffected — and the corrected
> B=16 figure strengthens it slightly. One claim does **not** survive: "batched
> wins at L ≥ 2 on all three GPUs" is false at B=64, where serial wins at every
> lane count on Blackwell and on the laptop.
>
> Quote `results/J8_MATCHED_PAIR.md` instead, which is built on the corrected
> record. This document is retained for the guard-reproduction narrative and as
> the record of what was believed before the lane fix.

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

# J8 — the sm_120 architecture-match control

The RTX PRO 6000 Blackwell Server Edition has **compute capability 12.0 — the same
as the RTX 5070 Laptop every workshop number came from**. It is the datacenter
sibling of the original device: 188 SMs and 102 GB against the laptop's ~36 SMs
and 8 GB. That makes it the one experiment that can separate *"the result changed
because the architecture changed"* from *"because the GPU got bigger"*.

Run non-exclusively on `gpu_requeue`, 4m49s, 3BPA, torch 2.13.0+cu129,
CUDA-event floor 0.0045 ms.

## 1. The batched-VJP guard now reproduces on three devices

| device | capability | fuser ON | fuser OFF | vs serial (fp64) |
|---|---|---|---|---|
| RTX 5070 Laptop (workshop, cu130) | 12.0 | fails at call 2 | 50/50 | 3e-15 |
| **RTX PRO 6000 Blackwell** | **12.0** | **fails at call 2** | **12/12** | **4.03e-15** |
| A100-SXM4-80GB | 8.0 | fails at call 2 | 12/12 | 3.20e-15 |
| A100 MIG 3g.20gb | 8.0 | fails at call 2 | 12/12 | 3.94e-15 |

Identical failure index, identical remedy, agreement with the serial path at the
same 3–4 × 10⁻¹⁵ on every device. The TensorExpr diagnosis is confirmed across
**two CUDA builds, two compute capabilities, four devices, and both a MIG slice
and whole GPUs**. It is a property of the TorchScript JIT layer, exactly as the
workshop's appendix argued, and that appendix is now much better supported than
it was on one laptop.

## 2. Correction: the serial cost model does **not** transfer across hardware

In `J8_SMOKE_A100.md` I wrote that the workshop's `T(L) = 10.38 + 11.78·L` "was
never really a property of the RTX 5070; it is a property of the launch pattern",
because the A100 reproduced its slope to within 8%. **The Blackwell control
falsifies that.**

Fitted serial slope, ms per lane:

| device | SMs | serial slope | serial intercept |
|---|---|---|---|
| RTX 5070 Laptop | ~36 | 11.78 | 10.38 |
| A100-SXM4-80GB | 108 | 10.92 | 14.44 |
| **RTX PRO 6000 Blackwell** | **188** | **4.58** | **6.26** |

The laptop/A100 agreement was a **coincidence**. On Blackwell the per-lane cost is
2.6× lower. What survives is the weaker and still useful statement: the serial
slope is **independent of batch size** on every device (Blackwell: 4.58, 4.57,
4.61, 4.92 across B = 1…64), i.e. the serial path is *work-independent* at these
sizes — but its per-lane cost is fixed GPU work at low occupancy, and that scales
with how fast the GPU is. It is not CPU launch latency, which would have been
hardware-invariant.

This matters for the paper because the workshop quotes that cost model, and the
J8 chapter would otherwise have generalised a coincidence into a law.

## 3. The workshop's 1.85× is a property of GPU *size*, not architecture

3BPA total speedup (exact L=8 versus K=3 at L=4), fastest correct baseline per cell:

| device | B=1 | B=4 | B=16 | B=64 |
|---|---|---|---|---|
| RTX 5070 Laptop (workshop) | 1.11× | — | **1.85×** | — |
| RTX PRO 6000 Blackwell (sm_120) | 1.07× | 1.07× | **1.28×** | 1.62× |
| A100-SXM4-80GB (sm_80) | 1.07× | 1.06× | **1.07×** | 1.58× |

**Same architecture as the laptop, and the 1.85× does not reproduce** — Blackwell
gives 1.28× at B = 16. So the workshop's headline speedup was not an sm_120
effect. It is a consequence of the laptop being a *small* GPU: with ~36 SMs it is
already compute-bound at B = 16, where a 108- or 188-SM part is still
launch-bound and extra cotangent lanes ride along nearly free.

This is the cleanest available statement of the systems boundary, and it could
only have come from this control:

> Lane reduction pays when the backward pass is compute-bound. Whether it is
> depends on work per batch relative to GPU size — not on architecture or
> generation. A small GPU saturates sooner and therefore benefits *more* at a
> given batch size, which is why the workshop's laptop showed the largest speedup
> in this study.

Blackwell sits between the laptop and the A100 exactly as that rule predicts: it
has more SMs than the laptop (so less benefit at B = 16: 1.28× vs 1.85×) and
fewer than... no — it has *more* SMs than the A100 (188 vs 108) yet shows *more*
benefit at B = 16 (1.28× vs 1.07×). SM count alone therefore does not order the
devices; memory bandwidth and per-SM throughput enter too. Reported as an
observation, not fitted into a one-parameter law.

## 4. Structure is invariant even where magnitude is not

Every qualitative feature holds on all three GPUs:

- serial affine in L and flat in B
- batched nearly flat in L at small batch, becoming lane-proportional at large batch
- the 3BPA transition at B = 64 on both datacenter parts
- batched the faster exact baseline for 3BPA at L ≥ 2

So the *shape* of the model in the paper is architecture-independent; only its
coefficients are hardware-specific. That is the right level at which to state the
systems contribution, and it is now supported by three GPU classes spanning
36–188 SMs, 8–102 GB, and two compute capabilities.

## Caveats

- Non-exclusive node; IQRs were small and the effects are large, but absolute
  milliseconds should be regenerated exclusively before publication.
- The laptop row is quoted from the frozen `workshop-v1.0` records and was
  measured under cu130, not cu129. Architecture and code match; the CUDA build
  does not.
- 3BPA only. Water on Blackwell would test whether the ~10⁴-edge threshold is also
  hardware-invariant, which is the obvious next control.
