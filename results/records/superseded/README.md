# Superseded records — kept, never quoted

## `j8_lane_bench_water_Blackwell.jsonl`

An earlier water lane-bench run on the RTX PRO 6000 Blackwell Server Edition,
written under a filename that did not encode the full device name. That is the
exact defect the adversarial review flagged: a second Blackwell part, or a rerun,
would have overwritten it with no way to tell the two apart afterwards.

It is a **genuinely different run** from the one that backs `J8_MATCHED_PAIR.md`
(`j8_lane_bench_water_NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.jsonl`), not a
copy: 12 of 60 cells differ by more than 2%, all of them at L = 8.

Which one is published matters, so it is stated plainly: **the published run is
the slower of the two.** At B = 1 this file gives serial 4.87 and batched 2.06
ms/lane against the published 5.76 and 2.51. The reported speedups are therefore
if anything conservative, and no quoted number comes from this file.

Retained because deleting a superseded measurement is how a project loses the
ability to answer "was it always like that?".

## `j2a_global_vs_maxcomp_ci.json`, `j2a_free_signal_ci.json`

The original copies of the two records behind J2a's most-quoted claims. They had
**no generating script and no recorded bootstrap seed**, and could not be
reproduced by the documented procedure — the deltas and block lengths matched
exactly, but the interval bounds did not, and no candidate seed recovered them.

Replaced by `experiments/j2a_confidence_intervals.py` at 100,000 resamples with
the seed recorded in the record's `meta` block. What changed:

* **Nothing in the headline**: still 24/24 positive, 19/24 significant.
* The free-signal count moved **8 of 10 → 9 of 10**: aspirin's `force_norm`
  interval resolves above zero (+0.112, CI [+0.008, +0.216]) where the old record
  had [−0.003, +0.206]. At 100,000 resamples the lower bound is +0.0089 ± 0.0006
  across seeds, so this is a resolved call rather than a coin flip.
* Section 1's per-system intervals shifted by 0.003–0.016 and changed no
  conclusion.

Kept because an unreproducible record is still evidence of what was reported at
the time, and because the direction of the error matters: this one *understated*
the result.
