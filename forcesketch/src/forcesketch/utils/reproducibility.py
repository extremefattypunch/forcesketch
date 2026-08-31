"""Deterministic seed derivation and run fingerprinting (spec SS47, SS68)."""

from __future__ import annotations

import hashlib
import subprocess

import torch


def derive_seed(base_seed: int, *parts: object) -> int:
    """Stable (base_seed, method, K, split, ...) -> int64 seed.

    blake2b over a canonical key string, so the same configuration yields the same
    probes on any machine, in any execution order, under any level of test
    parallelism. Spec SS47 requires the ten sketch seeds be frozen before final
    testing; this makes "the same seed" mean something reproducible.
    """
    key = "forcesketch|v1|" + "|".join(str(p) for p in parts) + f"|base={base_seed}"
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**63 - 1)


def make_generator(
    base_seed: int | None, *parts: object, device: torch.device | str = "cpu"
) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(0 if base_seed is None else derive_seed(base_seed, *parts))
    return g


def set_correctness_determinism(warn_only: bool = True) -> None:
    """For tests and the SS25 reproduction run -- NOT for benchmarks.

    Deterministic kernels are a different workload from the one SS26/SS31 aim to
    characterise, so benchmarks deliberately do not enable this. Both settings are
    recorded in the run manifest.
    """
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False


def pin_numerics() -> None:
    """TF32 off, highest fp32 matmul precision.

    TF32 on Blackwell shifts forces at the 1e-3 level, which would silently
    contaminate the SS21/SS23 exactness tests while looking like a real
    discrepancy.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def git_commit(repo_dir: str = ".") -> tuple[str, bool]:
    """(commit sha, dirty flag) for the SS68 `git_commit` field."""
    try:
        sha = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", repo_dir, "status", "--porcelain"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        return sha, dirty
    except Exception:
        return "unknown", True


def checkpoint_param_sha256(model: torch.nn.Module) -> str:
    """Semantic hash of a checkpoint, invariant to serialization format.

    Tensors are cast to float64 before hashing so an fp32 checkpoint and its fp64
    copy share a hash -- which is correct, because spec SS68 carries `precision`
    as its own column. Contrast a raw file sha256, which changes on any re-save.
    """
    h = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        t = state[name].detach().to("cpu")
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.to(torch.float64).contiguous().numpy().tobytes())
    h.update(str(getattr(model, "heads", None)).encode())
    h.update(type(model).__name__.encode())
    return h.hexdigest()


def checkpoint_hash(model: torch.nn.Module) -> str:
    """The spec SS68 `checkpoint_hash` field: first 12 chars of the semantic hash."""
    return checkpoint_param_sha256(model)[:12]
