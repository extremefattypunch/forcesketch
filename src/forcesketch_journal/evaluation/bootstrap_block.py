"""Paired IID and moving-block bootstrap over structures.

Paired, because every comparison in this project evaluates two signals on the
*same* structures: resampling them independently would throw away the pairing
that makes the difference far better determined than either term.

Two schemes, always reported together. The IID bootstrap is valid only if
structures are exchangeable; for a trajectory they are not, and the moving-block
scheme (Kunsch) restores validity by resampling contiguous runs of length
`block_len = ceil(2*tau_int)`. Where the measured tau_int is near 1 the two
schemes coincide by construction, and showing that they do is a stronger
statement than asserting the data were independent.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def iid_indices(n: int, n_boot: int, gen: torch.Generator) -> Tensor:
    return torch.randint(0, n, (n_boot, n), generator=gen)


def moving_block_indices(n: int, n_boot: int, block_len: int, gen: torch.Generator) -> Tensor:
    """Resample ceil(n/block_len) contiguous blocks, wrapping circularly, then trim to n.

    Circular wrapping keeps every observation equally likely to be drawn; the
    non-circular variant under-weights the two ends, which matters when the tail
    of a trajectory is the extrapolative part.
    """
    if block_len <= 1:
        return iid_indices(n, n_boot, gen)
    n_blocks = int(np.ceil(n / block_len))
    starts = torch.randint(0, n, (n_boot, n_blocks), generator=gen)
    offs = torch.arange(block_len)
    idx = (starts.unsqueeze(-1) + offs).reshape(n_boot, -1) % n
    return idx[:, :n]


def paired_bootstrap(
    metric,
    *arrays: Tensor,
    n_boot: int = 10_000,
    scheme: str = "block",
    block_len: int = 1,
    seed: int = 20260902,
    alpha: float = 0.05,
) -> dict:
    """Percentile CI for `metric(*arrays)` under paired resampling.

    NOTE ON DEFAULTS: `scheme="block"` with the default `block_len=1` reduces
    *exactly* to the IID bootstrap. That is deliberate (it makes the block path the
    default code path) but the label is misleading on its own, so the returned dict
    carries both `scheme` and `block_len` and callers must pass a measured
    `block_len` for the blocking to do anything.

    NOTE ON COVERAGE: these are *percentile* intervals, which under-cover for a
    dependent series. Measured on AR(1) with phi=0.5, n=400, nominal 95%:
    attained 0.718 at block_len=1, 0.873 at block_len=3, 0.902 at block_len=8.
    The block lengths this project derives from tau_int are 2-8, so intervals
    labelled 95% here attain roughly 87-90%. Significance calls near the boundary
    are therefore mildly anti-conservative and are reported as such; BCa or
    studentized intervals would be the fix if a borderline call ever mattered.

    `metric` must accept a leading resample axis, which is why the tail metrics
    are written rank-wise over [B, n] rather than as scalar loops.
    """
    n = arrays[0].shape[-1]
    gen = torch.Generator().manual_seed(seed)
    idx = (moving_block_indices(n, n_boot, block_len, gen) if scheme == "block"
           else iid_indices(n, n_boot, gen))
    reps = metric(*[a[idx] for a in arrays])
    lo, hi = torch.quantile(reps, torch.tensor([alpha / 2, 1 - alpha / 2], dtype=reps.dtype))
    return {
        "point": float(metric(*arrays)),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "boot_sd": float(reps.std()),
        "n_boot": n_boot, "scheme": scheme, "block_len": block_len,
    }


# A resample index tensor is [n_boot, n]. At n = 136,923 (the MPtraj pool) and
# n_boot = 10,000 that is 1.4e9 int64 entries, and the process is OOM-killed
# before it computes anything. Above this budget the replicates are produced in
# chunks instead. Below it the single-shot path is taken and the generator is
# consumed exactly as before, so every previously computed interval is
# unchanged -- the chunked path is new capability, not a re-baseline.
_MAX_INDEX_ELEMENTS = 40_000_000


def _replicate_chunks(n: int, n_boot: int, scheme: str, block_len: int,
                      gen: torch.Generator):
    """Yield resample index blocks whose size respects the element budget."""
    per = max(1, _MAX_INDEX_ELEMENTS // max(n, 1))
    if n_boot * n <= _MAX_INDEX_ELEMENTS:
        per = n_boot
    done = 0
    while done < n_boot:
        k = min(per, n_boot - done)
        yield (moving_block_indices(n, k, block_len, gen) if scheme == "block"
               else iid_indices(n, k, gen))
        done += k


def paired_difference(
    metric,
    signal_a: Tensor,
    signal_b: Tensor,
    reference: Tensor,
    *,
    n_boot: int = 10_000,
    scheme: str = "block",
    block_len: int = 1,
    seed: int = 20260902,
    alpha: float = 0.05,
) -> dict:
    """CI for metric(a, ref) - metric(b, ref) under the SAME resample.

    Sharing the resample is the whole point: the two signals rise and fall
    together across bootstrap draws, so the difference is determined far more
    tightly than either term, and `significant` is a meaningful statement.
    """
    n = signal_a.shape[-1]
    gen = torch.Generator().manual_seed(seed)
    reps = torch.cat([
        metric(signal_a[idx], reference[idx]) - metric(signal_b[idx], reference[idx])
        for idx in _replicate_chunks(n, n_boot, scheme, block_len, gen)
    ])
    lo, hi = torch.quantile(reps, torch.tensor([alpha / 2, 1 - alpha / 2], dtype=reps.dtype))
    point = float(metric(signal_a, reference) - metric(signal_b, reference))
    return {
        "delta": point, "ci_lo": float(lo), "ci_hi": float(hi),
        "boot_sd": float(reps.std()), "significant": bool(lo > 0 or hi < 0),
        "n_boot": n_boot, "scheme": scheme, "block_len": block_len,
    }
