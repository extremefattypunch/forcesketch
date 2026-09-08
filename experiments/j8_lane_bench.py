#!/usr/bin/env python
"""J8 smoke -- serial vs batched reverse mode as a function of lane count and batch size.

The kill gate for the acceleration framing. On the workshop's 8 GB laptop, batched
reverse mode was nearly flat in lane count at small batch (25.5 ms at L=1 vs
30.5 ms at L=8), which is why ForceSketch bought only 1.11x at B=1; it became
strongly lane-proportional by B=16, where the speedup rose to 1.85x. On an 80 GB
A100 or a 141 GB H200 that crossover must move, and if the batched path is flat in
L across the whole feasible range then reducing lanes saves nothing and
acceleration has to leave the paper's headline.

Protocol, following the frozen scripts' conventions so the numbers are comparable:
one CUDA event pair per iteration, a single synchronize at the timing boundary,
median and IQR reported. Two additions the frozen harness lacks -- the CUDA event
overhead floor is measured and recorded (without it, "post-processing is 0.2% of
runtime" is not a defensible statement), and OOM is recorded as an outcome rather
than skipped, so the memory boundary appears in the data instead of as a gap.

Both implementations run in the SAME process here, which is deliberate for a smoke
test: the fuser state is process-global and disabling it costs the serial path
some fusion, so a fair R5 comparison eventually needs process-per-implementation.
That asymmetry is recorded in `texpr_fuser_enabled` and must be closed before any
published speedup.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch.adapters.mace_data import load_frames, make_loader  # noqa: E402
from forcesketch.adapters.mace_mhc import MaceMHCAdapter  # noqa: E402
from forcesketch.exact.centered_basis import exact_seed_bundle, mean_seed  # noqa: E402
from forcesketch.utils.reproducibility import pin_numerics  # noqa: E402
from forcesketch_journal.benchmark.lanes import lane_seeds, record_path  # noqa: E402


def timed(fn, *, iters: int, warmup: int) -> dict:
    """Median/IQR over `iters` CUDA-event-timed calls, after `warmup` untimed ones."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.reset_peak_memory_stats()
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    q1, q3 = ms[len(ms) // 4], ms[(3 * len(ms)) // 4]
    return {"median_ms": statistics.median(ms), "iqr_ms": q3 - q1,
            "min_ms": ms[0], "iters": iters, "warmup": warmup,
            "peak_alloc_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def event_overhead_ms() -> float:
    """Floor of the measurement itself, so small numbers stay defensible."""
    r = timed(lambda: None, iters=200, warmup=50)
    return r["median_ms"]


def gpu_fingerprint() -> dict:
    name = torch.cuda.get_device_name(0)
    p = torch.cuda.get_device_properties(0)
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version,clocks.sm,clocks.mem,power.draw,"
         "clocks_throttle_reasons.active", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    return {
        "device_name": name, "is_mig": "MIG" in name,
        "compute_capability": f"{p.major}.{p.minor}", "sm_count": p.multi_processor_count,
        "total_memory_bytes": p.total_memory, "nvidia_smi": smi,
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
        "arch_list": torch.cuda.get_arch_list(),
        "texpr_fuser_enabled": torch._C._jit_texpr_fuser_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "platform": platform.platform(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(FROZEN / "models/zenodo/3BPA/trainset_100/multihead-disjoint/multihead_committee_stagetwo.model"))
    ap.add_argument("--data", default=str(FROZEN / "data/3bpa/test_1200K_ref.xyz"))
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--lanes", type=int, nargs="+", default=[1, 2, 3, 4, 8])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pin_numerics()
    fp = gpu_fingerprint()
    fp["event_overhead_ms"] = event_overhead_ms()
    print(json.dumps({k: fp[k] for k in ("device_name", "compute_capability", "sm_count",
                                         "is_mig", "texpr_fuser_enabled",
                                         "event_overhead_ms")}, indent=1))
    if fp["is_mig"]:
        print("\n*** WARNING: MIG device. Timings are NOT publishable. ***\n")

    ad = MaceMHCAdapter.from_checkpoint(args.ckpt, device="cuda", dtype=torch.float32)
    M = ad.num_heads
    records = []
    for B in args.batch_sizes:
        frames = load_frames(args.data, limit=B)
        batch = ad.prepare(next(iter(make_loader(frames, ad.model, batch_size=B))).to_dict())
        for L in args.lanes:
            # exact_seed_bundle gives the r = M-1 CENTRED directions only; the
            # mean-force lane is a separate seed. Asking for L > r silently
            # returned r lanes, so every cell previously labelled L=8 ran 7.
            # Concatenate the mean seed so the lane count is what it says.
            eb = exact_seed_bundle(M, B, dtype=torch.float32, device=ad.device)
            ms = mean_seed(M, B, dtype=torch.float32, device=ad.device)
            seeds = lane_seeds(eb.seeds, ms, L)
            for impl in ("serial", "batched"):
                rec = {**fp, "batch_size": B, "lanes": L, "impl": impl,
                       "num_atoms": len(frames[0]), "M": M, "dtype": "float32",
                       "experiment_id": "j8_lane_bench"}
                try:
                    r = timed(lambda: ad.vjp_for_seeds(batch, seeds, batched=(impl == "batched")),
                              iters=args.iters, warmup=args.warmup)
                    rec.update(r); rec["status"] = "ok"
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    rec.update({"status": "oom", "failure_reason": str(e)[:200]})
                except Exception as e:
                    torch.cuda.empty_cache()
                    rec.update({"status": "failed", "failure_reason": f"{type(e).__name__}: {str(e)[:200]}"})
                records.append(rec)
                m = rec.get("median_ms")
                print(f"  B={B:3d} L={L} {impl:8s} "
                      f"{('%8.2f ms' % m) if m else rec['status']:>12s}"
                      f"{('  IQR %.2f' % rec['iqr_ms']) if m else ''}")

    out = record_path("j8_lane_bench", fp["device_name"], explicit=args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    print(f"\nwrote {len(records)} records to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
