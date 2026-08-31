#!/usr/bin/env python
"""Kernel-level breakdown from an nsys trace (plan Task 7.1; spec SS46).

Answers "which kernels dominate the exact VJP" by running `nsys stats` and
folding the raw kernel names into families (tensor product, scatter/gather,
elementwise, GEMM, reduction), which is the level at which the answer is
actionable -- a list of 200 mangled template instantiations is not.

Falls back to torch.profiler if no nsys trace is available; that path needs no
external tooling and answers the same question, at the cost of the CPU-side
timeline.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

FAMILIES = [
    ("tensor product / e3nn", r"tensor_product|TensorProduct|_tp_|einsum|opt_einsum|codegen"),
    ("GEMM / linear", r"gemm|Gemm|GEMM|cutlass|ampere|sm\d+_xmma|addmm|matmul"),
    ("scatter / gather / index", r"scatter|gather|index_|IndexKernel|embedding|Sort|sort"),
    ("reduction", r"reduce|Reduce|sum_kernel|norm|softmax|var_"),
    ("elementwise / activation", r"elementwise|Elementwise|vectorized_|silu|tanh|mul|add|copy|fill"),
    ("memory ops", r"Memcpy|Memset|memcpy|memset"),
]


def classify(name: str) -> str:
    for family, pattern in FAMILIES:
        if re.search(pattern, name):
            return family
    return "other"


def from_nsys(rep: Path) -> list[dict]:
    out = subprocess.run(
        ["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", str(rep)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    rows, header = [], None
    for line in out.stdout.splitlines():
        if line.startswith("Time (%)") or line.startswith('"Time'):
            header = [h.strip().strip('"') for h in line.split(",")]
            continue
        if header is None or not line.strip():
            continue
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < len(header):
            continue
        rec = dict(zip(header, parts))
        try:
            rows.append({"name": rec.get("Name", parts[-1]),
                         "time_pct": float(rec.get("Time (%)", 0) or 0),
                         "total_ns": float(rec.get("Total Time (ns)", 0) or 0),
                         "instances": int(float(rec.get("Instances", 0) or 0))})
        except ValueError:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="results/profile/timeline_phases.nsys-rep")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="results/processed/kernel_breakdown.json")
    args = ap.parse_args()

    rep = Path(args.rep)
    if not rep.exists():
        print(f"no trace at {rep}; run scripts/08_profile.py under nsys first")
        return 1

    rows = from_nsys(rep)
    if not rows:
        print("nsys stats returned no kernel rows")
        return 1
    total = sum(r["total_ns"] for r in rows)

    fam = defaultdict(lambda: {"ns": 0.0, "instances": 0, "kernels": 0})
    for r in rows:
        f = fam[classify(r["name"])]
        f["ns"] += r["total_ns"]
        f["instances"] += r["instances"]
        f["kernels"] += 1

    print(f"kernel families ({len(rows)} distinct kernels, {total/1e6:.1f} ms GPU total)\n")
    print(f"{'family':<28}{'GPU %':>8}{'ms':>10}{'launches':>10}{'kernels':>9}")
    print("-" * 65)
    for name, v in sorted(fam.items(), key=lambda kv: -kv[1]["ns"]):
        print(f"{name:<28}{100*v['ns']/total:>7.1f}%{v['ns']/1e6:>10.2f}"
              f"{v['instances']:>10}{v['kernels']:>9}")

    print(f"\ntop {args.top} individual kernels")
    print("-" * 65)
    for r in sorted(rows, key=lambda r: -r["total_ns"])[:args.top]:
        nm = r["name"][:52]
        print(f"  {100*r['total_ns']/total:5.1f}%  {r['total_ns']/1e6:8.2f} ms  "
              f"{r['instances']:>6}x  {nm}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"total_gpu_ns": total,
         "families": {k: dict(v) for k, v in fam.items()},
         "kernels": sorted(rows, key=lambda r: -r["total_ns"])[:50]}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
