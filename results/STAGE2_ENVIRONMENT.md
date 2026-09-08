# Stage 2 — environment bring-up and the J0/J8 verification

## 1. The environment, and a correction to the plan

The approved plan asserted the laptop's `torch 2.13.0+cu130` could be reused on
FASRC. **It cannot**: the driver is `575.57.08`, which caps at CUDA 12.9, and a
cu130 (CUDA 13) wheel needs r580+.

The resolution is better than the documented rung-B fallback. `torch 2.13.0+cu129`
exists on the stable index, keeping the torch **version** — and therefore the
entire TorchScript/NNC layer the fuser finding actually depends on — identical,
and changing only the CUDA toolkit. Rung B (torch 2.8 + cu128) would have changed
the torch minor version *and* the toolkit at once, leaving no way to attribute any
difference.

**Verified on the A100:** `torch.cuda.get_arch_list()` returns
`['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`. One wheel therefore covers
Blackwell (sm_120, the laptop's architecture and the 192 RTX PRO 6000 nodes),
A100 (sm_80) and H200 (sm_90). **Hardware is now the only variable across the
whole benchmark grid** — which is exactly what rule R5 and phase J8 need.

`fs-gpu` at `/n/holylabs/hekstra_lab/Everyone/ianpoon/envs/fs-gpu`:
torch 2.13.0+cu129, numpy 2.5.2, scipy 1.18.0, ase 3.29.0, e3nn 0.4.4,
mace-torch 0.3.10 editable from `beckobert/mace@86ad191`, matscipy 1.2.0.

Three practical notes for reproduction, none of which is in `rung.txt`:

- `pip install -e … --no-deps` is required (the fork's `setup.cfg` pins
  `numpy<2.0`, an inherited false pin) **but then the genuine dependencies must be
  installed by hand** — `torch_ema`, `opt_einsum`, `prettytable`, `h5py`,
  `torchmetrics`, `python-hostlist`, `configargparse`, `GitPython`, `tqdm`,
  `matscipy`. Omitting `torch_ema` alone makes every checkpoint load fail at
  unpickle time with a confusing `ModuleNotFoundError`.
- `matscipy` is installed, so the neighbour-list backend matches the laptop's.
  This matters: matscipy and ASE produce the same edge set in a different order,
  which changes floating-point reduction order and would show up as phantom
  ~1e-13 discrepancies in any exactness check.
- The `numpy<2.0` pip resolver warning is expected and harmless.

## 2. The fork patch, changed from a hardcode to a toggle

`environment/mace-fork.patch` flips `loss = "dpose"` → `"normal"` inside
`ScaleShiftMACE.forward`. Without it `output["heads"]` is always `None` and there
is no committee at all. But that function runs at **training** time too, and the
two branches compute `total_energy` differently (`dpose` takes a mean over heads).
The published checkpoints were trained under `dpose`; hardcoding `normal` would
silently change the training objective and make any committee we train
incomparable to theirs — contaminating the J4 M-sweep at its own control point.

The journal patch (`manifests/mace-fork-journal.patch`) therefore reads

```python
loss = os.environ.get("FORCESKETCH_MACE_LOSS", "dpose")
```

so the default reproduces upstream exactly, evaluation sets
`FORCESKETCH_MACE_LOSS=normal` explicitly, and the value is recorded per run.

## 3. Data and checkpoints, hash-verified

Staged to `/n/holylabs/hekstra_lab/Everyone/ianpoon/forcesketch-data`, symlinked
into the frozen tree as `data/` and `models/` so **none of the fifteen scripts'
hardcoded relative paths need editing** — which would have broken the J0
bit-for-bit reproduction claim.

- 3BPA (5 files) from `github.com/davkovacs/BOTNet-datasets`: **all five sha256
  match `freeze.json` exactly.** The cluster data is bit-identical to what
  produced the workshop results.
- Zenodo 17829635 (634 MB): **md5 `897cf20183ef0537851755a8e306c0c8` matches the
  API record.** 75 members extracted (219.8 MB) covering 3BPA trainset_100 (three
  multihead + three naive committees), rMD17 full_trainset, and all of water
  including the 192-atom periodic MD and PIMD trajectories.

## 4. J0 reproduction on the cluster: exact

Re-running `00_validate_checkpoints.py`'s evaluation on the A100 against the
published `checkpoints.json`:

| variant | metric | cluster | published | rel. diff |
|---|---|---|---|---|
| disjoint | force_rmse_committee (meV/Å) | 206.4814 | 206.4814 | 4.4e-08 |
| disjoint | head_rmse_spread_ratio | 1.0852 | 1.0852 | 3.9e-08 |
| disjoint | median_atom_disagreement | 33.4835 | 33.4833 | 4.4e-06 |
| overlapping | force_rmse_committee | 180.3375 | 180.3375 | 1.0e-07 |
| overlapping | median_atom_disagreement | 12.8093 | 12.8091 | 1.5e-05 |
| same | force_rmse_committee | 190.9938 | 190.9938 | 1.3e-08 |
| same | median_atom_disagreement | 7.1660 | 7.1660 | 3.7e-06 |

**All nine metrics reproduce to 1e-8–1.5e-5 — pure float32 rounding.** A different
GPU architecture, a different CUDA toolkit, and a freshly built environment give
the same numbers.

Two definition traps cost an hour and are worth recording, because both look like
scientific discrepancies:

- **`force_rmse_committee_mev_A` is the mean of the eight per-head RMSEs**, not the
  RMSE of the committee-mean force. The latter is a different (and smaller, 201.3
  vs 206.5) quantity, since averaging heads reduces error.
- **`median_atom_disagreement` is `F.std(dim=0).mean(dim=-1)`** — std over heads,
  then *mean* over xyz (i.e. `u_atom_mhc`). Using `v.sum(-1).sqrt()` instead
  inflates it by ≈√3.

### The checkpoint-hash mystery, closed

`checkpoint_hash` computed at fp64 on the cluster returns **`98e44023749c`,
exactly matching the committed cache**, while `checkpoints.json` records
`71ab2b8b07cc`. This confirms the diagnosis in the Stage 1 report: the function
hashes `str(t.dtype)` alongside the float64-cast bytes despite documenting
dtype-invariance, and `00_validate_checkpoints.py` loads at fp32 while
`01_exact_reproduction.py` loads at fp64. Same checkpoint file, two labels. No
science was affected; the provenance chain is now understood and the one-line fix
belongs in the journal package.

## 5. The batched-VJP guard: the workshop result stands, and is now stronger

The single biggest Stage-2 risk was that the TensorExpr-fuser fix — which is what
makes the *strongest* exact baseline valid at all — might be torch- or
hardware-dependent. `experiments/j0b_batched_vjp_guard.py` runs each fuser state
in a fresh subprocess (TorchScript caches an optimized plan per graph, so the
state cannot be changed after the first forward):

| probe | texpr enabled | successes | first failure | vs serial (fp64) |
|---|---|---|---|---|
| fuser ON (upstream default) | True | 2 / 12 | **call index 2** | — |
| fuser OFF (shipped fix) | False | **12 / 12** | none | **3.94e-15** |

The failure signature is exactly the one `configure_e3nn_for_batched_vjp()`'s
docstring predicts: calls 0 and 1 succeed, call 2 raises *"Cannot access data
pointer of Tensor that doesn't have storage"* — TorchScript's profiling executor
emits its optimized plan after two warm-ups, and the TensorExpr fuser then fuses
the reverse graph into a kernel that needs a raw `data_ptr()` a BatchedTensor
cannot provide.

**Verdict: `fuser_needed=True`, `fix_sufficient=True`.** The workshop's diagnosis
now holds across two CUDA builds (cu130 → cu129) and two compute capabilities
(sm_120 → sm_80). That is a strengthening, not a mere replication: it confirms the
cause is the TorchScript JIT layer rather than a hardware quirk, which is what
makes the appendix a transferable contribution to anyone building UQ on e3nn.

## 6. Timing: submitted, not yet measured

This session holds a `gpu_test` allocation, which is **entirely MIG
`a100_3g.20gb` slices** — 42 SMs of an A100 and a shared memory controller.
Correctness work is fine there; timings are not publishable, and
`j8_lane_bench.py` refuses to present them as such (it records `is_mig` and warns).

`slurm/j8_bench.sbatch` is submitted to the `gpu` partition (job 39407869,
`--exclusive`, full A100-SXM4-80GB, B ∈ {1,4,16,64} × L ∈ {1,2,3,4,8} ×
{serial, batched}). It re-runs the guard on the real device first.

The open question it answers is the kill gate: on the 8 GB laptop the batched path
was nearly flat in lane count at B=1 (25.5 → 30.5 ms from L=1 to L=8) and became
lane-proportional by B=16. With 80 GB the memory boundary moves by ~10×, so the
crossover must move too. **If the batched path is flat in L across the whole
feasible range, reducing lanes saves nothing and acceleration leaves the paper's
headline** — which is precisely why this ran early rather than at week 13.

One asymmetry to close before any published speedup: the fuser state is
process-global and disabling it costs the serial path some fusion, so a fair R5
comparison needs process-per-implementation. The smoke test records
`texpr_fuser_enabled` so the asymmetry is visible rather than silent.
