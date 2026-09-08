# ForceSketch

Code, experimental records and manuscript source for:

> **When is committee force uncertainty worth computing in MD?
> Decision quality, cheap gates, and the baselines that must be beaten**
> Ian Poon, Department of Bioengineering, Imperial College London

Ensembles of machine-learned interatomic potentials are the standard way to choose
which structures to label next, and cheap approximations of their force
disagreement are normally validated by checking that they reproduce the exact
ensemble. This work argues that is the wrong target — the ensemble is itself only
a proxy for the model's error against the reference calculation — and asks the two
questions separately.

## The one thing worth knowing about this repository

**No number in the manuscript is typed.** Every value is generated from a record in
`results/records/` by `tools/paper_numbers.py`, and `tools/audit.py` fails if any
of them stops deriving. That is not a stylistic choice: an internal audit found
roughly 70 quantitative claims in the prose reports that a record contradicts,
almost all of them stale copies left by corrections that were never propagated —
and the macro-derived numbers were **not** among the failures. See
`results/CONSISTENCY_AUDIT.md`.

## Reproducing the results

```bash
python tools/reproduce.py --tier offline   # CPU, ~4 min, no downloads, no CUDA
python tools/audit.py                      # procedural audit; see below
python tools/paper_numbers.py --check      # every macro still derives
python -m pytest tests/ -q                 # 66 tests
```

`--tier offline` uses only the caches tracked here and re-derives the headline
statistical results, checking each against its committed value. `--tier full`
additionally needs a GPU and the data fetch; the gap is deliberate and documented
rather than implied.

`tools/audit.py` exits non-zero only on **procedural** failure — a hash that moved,
a manifest that no longer recomputes, a quoted number that no longer derives.
Whether a result came out favourably is none of its business. That distinction is
deliberate: an audit that fails when the science disappoints creates pressure to
make the science pass.

## Layout

| path | what it holds |
|---|---|
| `src/forcesketch_journal/` | estimators, split-conformal calibration, block bootstrap, split logic |
| `experiments/` | `j0`–`j9` experiment scripts, one per phase |
| `protocols/` | pre-registered protocols, hash-bound to registration records written *before* each experiment produced any artifact |
| `manifests/` | split manifests; every split content-hashed and recomputable from its seed |
| `results/records/` | 93 raw records — the evidence every documented number is checked against |
| `results/*.md` | per-phase reports, including the adversarial review and the consistency audit |
| `tools/` | number, figure and table generation; audit; reproduction; release packaging |
| `paper/` | manuscript source, shared body with per-venue wrappers |
| `slurm/` | cluster job scripts for the GPU phases |

## What is not here

Roughly 384 MB of regenerable `.pt` caches, excluded on purpose. Their rebuild
commands are in `.gitignore` and `results/STAGE2_ENVIRONMENT.md`, and the offline
reproduction tier does not need them.

The multi-head committee checkpoints and the 3BPA and liquid-water frames are **not
redistributed here**. They are the supporting data of Beck et al.,
[doi:10.5281/zenodo.17829635](https://doi.org/10.5281/zenodo.17829635); the
acquisition scripts fetch them and verify every file against a recorded hash, so a
silently substituted input fails rather than propagates. MPtraj structures are
those distributed with CHGNet, [doi:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3).

## Timing results and hardware

Only the timing experiments need a GPU. They were run on A100-SXM4-80GB, H200 and
RTX PRO 6000 Blackwell cards under one pinned toolchain, and the device, driver, SM
count and clock state of every run are recorded beside its measurements — the
wall-clock conclusions are properties of the hardware as much as of the method.

## Use of large language models

LLM-based coding agents were used for experiment and
analysis code, the adversarial review, and drafting. The verification here is
mechanical rather than a matter of inspection for exactly that reason — numbers are
derived rather than typed, and protocols are hash-bound to registrations written
before any artifact existed, so a hypothesis cannot move after the outcome is seen.

## Licence

Code under the MIT licence; experimental records under CC BY 4.0.
