# J8 — matched pairs on three architectures

**All three devices now have a clean matched pair.** Water and 3BPA measured in
the **same SLURM job**, on the **same exclusive node**, at **identical settings**
(warmup 50, iters 100). Every run is admissible under both protocol versions; SM
clocks at maximum (A100 1410 MHz, H200 1980 MHz, Blackwell 2430 MHz).

> This replaces an earlier Blackwell-only table whose two halves came from
> separate jobs at different settings (3BPA warmup 100 / iters 200, water
> warmup 25 / iters 50), so "the same run conditions" was false — a
> cross-document audit caught it. **The matched re-run reproduces the earlier
> Blackwell verdict exactly (2/4 under v2), so nothing was riding on the
> mismatch.** That is worth stating: the defect was real and worth fixing, and
> fixing it changed no conclusion.

## Total speedup, water (exact L=8 vs K=3 at L=4)

| device | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|---|
| H200 | 1.090× | 1.452× | 1.702× | 1.801× | **1.852×** | 1.867× |
| A100-SXM4-80GB | 1.527× | 1.711× | 1.818× | 1.866× | **1.880×** | 1.882× |
| RTX PRO 6000 Blackwell | 1.828× | 1.848× | 1.842× | 1.882× | **1.894×** | 1.894× |

**At B = 16 the speedup is 1.85–1.89× on all three architectures** — sm_80,
sm_90 and sm_120, three GPU generations, agreeing to 2.3%. That is the most
hardware-robust number in the systems chapter. 3BPA in the same jobs reaches only
1.10× (A100), 1.09× (H200) and 1.27× (Blackwell) at B = 16.

## The ramp differs by device, exactly as the mechanism predicts

| device | water batched slope at B=1 (ms/lane) | transition batch | speedup at B=1 |
|---|---|---|---|
| H200 | 0.56 | 2 | 1.090× |
| A100-80GB | 2.46 | 1 | 1.527× |
| Blackwell | 2.18 | 1 | 1.828× |

Lane reduction pays only once the backward pass is compute-bound, so **the batch
size at which it starts paying scales UP with how fast the machine is.** The H200
is the fastest of the three and is still launch-bound at B = 1
(0.56 ms/lane, 1.090×); the A100 is partly there (1.527×); Blackwell pays
almost in full immediately (1.828×). All three converge by B = 16.

## Pre-registered verdicts, by device

| device | v2 (pre-registered) | v3 (post-hoc) | P1 | P2 | P3 | P4 | admissible |
|---|---|---|---|---|---|---|---|
| H200 | **4/4** | 4/4 | ✓ | ✓ | ✓ | ✓ | yes |
| A100-SXM4-80GB | **3/4** | 3/4 | ✓ | ✓ | ✗ | ✓ | yes |
| RTX PRO 6000 Blackwell | **2/4** | 3/4 | ✓ | ✓ | ✗ | ✗ | yes |

**P1 and P2 — the two predictions that carry the speedup claim — pass on every
device.** On A100 and H200 v2 and v3 agree, so the post-hoc protocol repair
changes nothing where the frozen thresholds happen to fit; it matters only on
Blackwell, and only for P4.

### Both failures are "stronger or faster than the frozen threshold assumed"

- **P3** requires water's launch-to-compute transition at 1/16–1/33 of 3BPA's
  batch size. H200 transitions at B = 2 → 1/32, inside the window. A100 and
  Blackwell transition at B = 1 → 1/64, **outside it because the effect is larger
  than predicted**. Direction and mechanism confirmed on all three.
- **P4** requires water's serial slope to exceed 3BPA's, threshold frozen at
  10.9 ms/lane — an *A100* number. A100 gives 13.25 ✓, H200 11.72 ✓,
  Blackwell 6.56 ✗ — failing only because Blackwell is roughly twice as fast
  per lane as the machine the threshold came from. This is precisely the defect v3
  fixes by deriving the reference from the matched 3BPA run.

## Provenance

Records `j8_lane_bench{,_water}_<device>_pair.jsonl`, jobs 39518641 (A100),
39518642 (H200), 40846458 (Blackwell). Verdicts
`j8_verdict_pair_<device>_v{2,3}.json`. One water cell on Blackwell
(B=1, L=1, serial, 14.5% relative IQR) is excluded as an outlier by both protocol
versions; it is not load-bearing and 59 of 60 cells sit under 3%.

**Queue cost of exclusivity, recorded so it is budgeted rather than
rediscovered.** The A100 and H200 jobs waited **~2h20m** on `(Resources)` before
starting, then ran 30 and 17 minutes. `--exclusive` drains a whole node: the
allocations granted were 64 CPUs + 4 GPUs (A100) and 112 CPUs (H200) for a job
that uses **one** GPU and 16 threads. It is not removable for a scored run —
`require_exclusive_node: true` is pre-registered, and a co-tenanted run once
doubled absolute times with the SM clock at 780 MHz — but for exploratory timing,
drop `--exclusive` and check `sm_clock_mhz` instead.
