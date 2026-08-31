#!/usr/bin/env python
"""Environment verification gate (plan Task 0.2).

Seven checks, run top to bottom; each must pass before the next is meaningful.

Checks 4 and 6 are the ones that matter and that a naive smoke test would miss.
An e3nn version that silently ignored `instructions=` would still run, still
return [N, 8], and give you EIGHT IDENTICAL HEADS -- a committee with zero
disagreement, which makes every ForceSketch number vacuously correct while
looking perfectly healthy in the loss curve. Do not skip them.

Usage:  python scripts/00_environment.py verify [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

RESULTS: dict = {"checks": {}, "passed": False}
_FAILED = False


def line(status: str, msg: str) -> None:
    colour = {"OK": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m"}.get(status, "")
    print(f"  {colour}[{status:4}]\033[0m {msg}")


def record(name: str, ok: bool, msg: str, value=None) -> None:
    global _FAILED
    RESULTS["checks"][name] = {"ok": bool(ok), "detail": msg, "value": value}
    line("OK" if ok else "FAIL", msg)
    if not ok:
        _FAILED = True


def stage0_env() -> None:
    print("\n[0] process environment")
    ok = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") == "1"
    record(
        "env_weights_only",
        ok,
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
        if ok
        else "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD is NOT set. e3nn 0.4.4 does a bare "
        "torch.load() at import time on a file containing `slice` objects, which "
        "torch>=2.6 rejects under weights_only=True. This must be exported BEFORE "
        "python starts; it cannot be set from Python. NOTE: the resulting "
        "UnpicklingError is NOT an e3nn/torch incompatibility -- do not change rung.",
    )
    if not ok:
        sys.exit(1)
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*deprecated.*")


def stage1_torch():
    print("\n[1] torch / GPU")
    import torch

    from forcesketch.utils.reproducibility import pin_numerics

    pin_numerics()
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    record("cuda_available", torch.cuda.is_available(), f"cuda available={torch.cuda.is_available()}")
    record("compute_capability", cc == (12, 0), f"compute capability {cc} (expect (12, 0))", str(cc))
    arch = torch.cuda.get_arch_list()
    record("sm120_in_wheel", "sm_120" in arch, f"wheel arch list contains sm_120: {arch}", arch)
    # Prove real kernels launch, rather than trusting get_arch_list().
    a = torch.randn(2048, 2048, device="cuda")
    record("kernels_execute", bool(torch.isfinite(a @ a).all()), "2048^2 matmul produces finite output")
    RESULTS["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "arch_list": arch}
    return torch


def stage2_e3nn(torch) -> None:
    print("\n[2] e3nn tensor products + the multi-head readout mechanism")
    import e3nn
    from e3nn import o3

    RESULTS["e3nn_version"] = e3nn.__version__
    dev = "cuda"
    # fp64 tolerance is 1e-8, not 1e-10: the check composes a random SO(3) matrix
    # (itself from an fp64 QR), Wigner D-matrices, and a tensor product, so ~1e-9
    # is the healthy floor. Measured 2.6e-9 on this stack.
    for dtype, tol in ((torch.float32, 1e-4), (torch.float64, 1e-8)):
        i1, i2, io = o3.Irreps("32x0e + 32x1o"), o3.Irreps("1x0e + 1x1o + 1x2e"), o3.Irreps("32x0e + 32x1o")
        tp = o3.FullyConnectedTensorProduct(i1, i2, io).to(dev, dtype)
        x1 = torch.randn(64, i1.dim, device=dev, dtype=dtype, requires_grad=True)
        x2 = torch.randn(64, i2.dim, device=dev, dtype=dtype)
        y = tp(x1, x2)
        (g,) = torch.autograd.grad(y.sum(), x1)
        tag = str(dtype).split(".")[-1]
        record(f"tp_finite_{tag}", bool(torch.isfinite(y).all() and torch.isfinite(g).all()),
               f"tensor product forward+backward finite ({tag})")
        record(f"tp_backward_nonzero_{tag}", g.abs().sum().item() > 0,
               f"tensor product backward is nonzero ({tag})")

        R = o3.rand_matrix(dtype=dtype, device=dev)
        lhs = tp(x1.detach() @ i1.D_from_matrix(R).T, x2 @ i2.D_from_matrix(R).T)
        rhs = y.detach() @ io.D_from_matrix(R).T
        err = ((lhs - rhs).abs().max() / rhs.abs().max()).item()
        record(f"tp_equivariance_{tag}", err < tol, f"equivariance rel-err {err:.3e} < {tol:g} ({tag})", err)

    # CHECK 4 -- the PR#800 mechanism. If instructions= is silently ignored, this
    # still runs, still returns [N, 8], and every head is identical.
    M, hid = 8, o3.Irreps("16x0e")
    lin1 = o3.Linear(o3.Irreps("32x0e + 32x1o"), M * hid, instructions=[(0, i) for i in range(M)]).to(dev)
    lin2 = o3.Linear(M * hid, M * o3.Irreps("0e"), instructions=[(i, i) for i in range(M)]).to(dev)
    z = torch.randn(64, 32 + 32 * 3, device=dev, requires_grad=True)
    h = lin2(torch.nn.functional.silu(lin1(z)))
    record("readout_shape", tuple(h.shape) == (64, M), f"multi-head readout shape {tuple(h.shape)} == (64, {M})")
    per_head = torch.stack(
        [torch.autograd.grad(h[:, m].sum(), z, retain_graph=True)[0] for m in range(M)]
    )
    spread = per_head.std(dim=0).abs().max().item()
    record("heads_distinct", spread > 0,
           f"the {M} readout heads have DISTINCT gradients (max std {spread:.3e}); "
           "zero here means instructions= was ignored and the committee is degenerate", spread)


def stage3_mace(torch):
    print("\n[3] MACE multi-head committee: forward, per-head forces, disagreement")
    import numpy as np
    import ase
    from e3nn import o3
    from mace import data, modules, tools
    from mace.tools import torch_geometric

    RESULTS["mace_version"] = getattr(__import__("mace"), "__version__", "?")
    dev = "cuda"
    torch.manual_seed(0)
    z_table = tools.AtomicNumberTable([1, 6, 7, 8])
    e0 = np.array([-13.587222780835477, -1029.4889999855063, -1484.9814568572233, -2041.9816003861047])
    heads = [f"committee-{i}" for i in range(8)]

    model = modules.ScaleShiftMACE(
        r_max=6.0, num_bessel=8, num_polynomial_cutoff=5, max_ell=3,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=len(z_table.zs),
        hidden_irreps=o3.Irreps("32x0e + 32x1o"), MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.stack([e0] * len(heads), axis=0),
        avg_num_neighbors=8.0, atomic_numbers=z_table.zs, correlation=3,
        gate=torch.nn.functional.silu, atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        heads=heads,
    ).to(dev)

    at = ase.Atoms("C2H6O", positions=np.random.RandomState(0).randn(9, 3) * 1.2)
    cfg = data.config_from_atoms(at)
    ad = data.AtomicData.from_config(cfg, z_table=z_table, cutoff=6.0, heads=heads)
    loader = torch_geometric.dataloader.DataLoader([ad, ad], batch_size=2, shuffle=False)
    batch = next(iter(loader)).to(dev).to_dict()
    batch["positions"].requires_grad_(True)

    committee = torch.arange(len(heads), device=dev)
    out = model(batch, training=False, compute_force=False, committee_heads=committee)
    E = out["heads"]["energy"]
    record("head_energy_shape", tuple(E.shape) == (2, 8), f"heads['energy'] shape {tuple(E.shape)} == (2, 8)")
    record("head_energy_finite", bool(torch.isfinite(E).all()), "head energies finite")
    record("head_energy_spread", E.std(dim=-1).min().item() > 0,
           f"head energies are non-degenerate (min std {E.std(dim=-1).min().item():.3e})")

    # CHECK 6 -- per-head forces must exist, be nonzero, and DISAGREE.
    Fh = torch.stack(
        [-torch.autograd.grad(E[:, m].sum(), batch["positions"], retain_graph=True)[0]
         for m in range(8)], dim=0)  # [M, N, 3]
    record("head_forces_finite", bool(torch.isfinite(Fh).all()), "per-head forces finite")
    record("head_forces_nonzero", Fh.abs().max().item() > 0, "per-head forces nonzero")
    spread = Fh.std(dim=0).abs().max().item()
    record("force_disagreement", spread > 0,
           f"per-head forces DISAGREE (max std {spread:.3e}); zero here means there is "
           "nothing to sketch and every result would be vacuously exact", spread)
    return model, batch, E


def stage4_finite_difference(torch, model, batch) -> None:
    """Central-difference check that force == -dE/dx, in FLOAT64.

    This must not run in float32. 3BPA-like energies are ~-4200 eV (isolated-atom
    references dominate), so with h=1e-4 the fp32 round-off in E(+h) - E(-h) is
    ~4200 * 1.2e-7 / 2e-4 ~ 2.5 eV/A -- roughly 50x a real force. A float32
    finite-difference check measures its own cancellation, not the gradient. This
    is the same mechanism that forces every exactness reference in this project to
    float64 (plan resolution R5).
    """
    print("\n[4] finite-difference check (float64): force == -dE/dx for head 0")
    model64 = model.double()
    pos = batch["positions"].detach().double().requires_grad_(True)
    b = dict(batch)
    b["positions"] = pos
    for k in ("node_attrs", "shifts", "cell", "unit_shifts"):
        if k in b and torch.is_floating_point(b[k]):
            b[k] = b[k].double()
    committee = torch.arange(8, device=pos.device)

    E = model64(b, training=False, compute_force=False,
                committee_heads=committee)["heads"]["energy"]
    (g,) = torch.autograd.grad(E[:, 0].sum(), pos, retain_graph=False)
    f_analytic = (-g)[3, 1].item()

    def energy_at(delta: float) -> float:
        p = pos.detach().clone()
        p[3, 1] += delta
        bb = dict(b)
        bb["positions"] = p.requires_grad_(True)
        return model64(bb, training=False, compute_force=False,
                       committee_heads=committee)["heads"]["energy"][:, 0].sum().item()

    h = 1e-5
    fd = -(energy_at(h) - energy_at(-h)) / (2 * h)
    # Normalize by the FORCE SCALE, not by this one component. A central
    # difference carries O(h^2 E''') truncation error, so the absolute agreement
    # is what is bounded; dividing by a component that happens to be near zero
    # reports a large "relative error" for a perfectly good gradient. This is the
    # same reasoning that replaces spec SS21's |v_hat-v|/(|v|+eps) with a
    # scale-normalized form.
    scale = (-g).abs().max().item()
    err = abs(fd - f_analytic) / scale
    record("finite_difference_fp64", err < 1e-6,
           f"analytic {f_analytic:+.9f} vs central-difference {fd:+.9f}; "
           f"|diff|/max|F| = {err:.2e} (max|F| = {scale:.4f})", err)
    model.float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="verify", choices=["verify"])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("=" * 72)
    print("ForceSketch environment gate (plan Task 0.2)")
    print("=" * 72)
    stage0_env()
    torch = stage1_torch()
    stage2_e3nn(torch)
    model, batch, _E = stage3_mace(torch)
    stage4_finite_difference(torch, model, batch)

    RESULTS["passed"] = not _FAILED
    print("\n" + "=" * 72)
    print("RESULT:", "\033[32mALL CHECKS PASSED\033[0m" if not _FAILED else "\033[31mFAILED\033[0m")
    print("=" * 72)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(RESULTS, fh, indent=2)
        print(f"wrote {args.json}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
