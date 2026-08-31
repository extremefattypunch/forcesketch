"""Lane construction and record naming for the timing benchmarks.

Both functions here exist because a review found the same class of defect twice:
a benchmark that *labelled* a cell one way and *ran* it another, with nothing in
the code able to notice.

  * `lane_seeds` -- `exact_seed_bundle` yields the r = M-1 centred directions
    only, so slicing `seeds[:L]` for L = M silently returned M-1 lanes and every
    cell labelled L=8 actually ran 7. The mean-force lane is a separate seed and
    has to be concatenated. Returning fewer lanes than requested is now an
    exception, not a shrug.
  * `record_path` -- two runs on different GPUs wrote to the same filename, so
    the second silently overwrote the first and the surviving file could not be
    attributed to either device. The device is now part of the path by
    construction, and an explicitly supplied path is *rejected* unless it also
    carries the device.

Both are pure and import no CUDA, so they are testable on a login node.
"""

from __future__ import annotations

import pathlib
import re

import torch


def lane_seeds(exact_seeds: torch.Tensor, mean_seed: torch.Tensor, lanes: int) -> torch.Tensor:
    """Concatenate the r centred lanes with the mean-force lane and take `lanes`.

    `exact_seeds` is [r, B, M] from `exact_seed_bundle`; `mean_seed` is [1, B, M].
    Returns exactly `lanes` seeds or raises -- never fewer, which is the whole
    point of the function.
    """
    if exact_seeds.ndim != 3 or mean_seed.ndim != 3:
        raise ValueError(f"expected [lanes, B, M] tensors, got {tuple(exact_seeds.shape)} "
                         f"and {tuple(mean_seed.shape)}")
    if exact_seeds.shape[1:] != mean_seed.shape[1:]:
        raise ValueError(f"seed shapes disagree past the lane axis: "
                         f"{tuple(exact_seeds.shape)} vs {tuple(mean_seed.shape)}")
    if lanes < 1:
        raise ValueError(f"lanes must be >= 1, got {lanes}")
    allseeds = torch.cat([exact_seeds, mean_seed], dim=0)
    if lanes > allseeds.shape[0]:
        raise ValueError(f"requested L={lanes} lanes but only {allseeds.shape[0]} exist "
                         f"({exact_seeds.shape[0]} centred + {mean_seed.shape[0]} mean)")
    out = allseeds[:lanes]
    assert out.shape[0] == lanes, "lane count must equal the request"  # invariant, not input check
    return out


def device_slug(device_name: str) -> str:
    """Filesystem-safe device tag. Non-empty, or the caller has no device."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", device_name.strip()).strip("_")
    if not slug:
        raise ValueError(f"device name {device_name!r} produced an empty slug")
    return slug


def record_path(prefix: str, device_name: str, *, explicit: str | pathlib.Path | None = None,
                root: str | pathlib.Path = "results/records") -> pathlib.Path:
    """Where a benchmark's records go. The device is always in the name.

    An `explicit` path is honoured only if it already carries the device slug;
    otherwise two devices could still collide on one filename, which is exactly
    the failure this function exists to prevent.
    """
    slug = device_slug(device_name)
    if explicit is not None:
        p = pathlib.Path(explicit)
        if slug not in p.name:
            raise ValueError(f"output path {p.name!r} does not encode the device {slug!r}; "
                             "a second device would overwrite these records")
        return p
    return pathlib.Path(root) / f"{prefix}_{slug}.jsonl"
