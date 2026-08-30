"""Method dispatch and the spec SS49 matched-budget enumeration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from forcesketch.exact.centered_basis import exact_seed_bundle
from forcesketch.sketches.generators import (
    gaussian_seeds,
    haar_seeds,
    head_subsample_exact_mean_seeds,
    head_subsample_seeds,
    pairwise_seeds,
    rademacher_seeds,
)
from forcesketch.types import LaneBudget, Method, SeedBundle

SEED_GENERATORS = {
    "gaussian": gaussian_seeds,
    "haar": haar_seeds,
    "rademacher": rademacher_seeds,
    "pairwise": pairwise_seeds,
    "head_subsample": head_subsample_seeds,
    "head_subsample_exact_mean": head_subsample_exact_mean_seeds,
}


def make_sketch_seeds(
    method: Method, *, M: int, K: int, batch_size: int, seed: int | None, **kwargs
) -> SeedBundle:
    """Matches the spec SS18 call site. `method='exact'` returns the Helmert bundle."""
    if method == "exact":
        return exact_seed_bundle(
            M=M,
            batch_size=batch_size,
            dtype=kwargs.get("dtype", None) or __import__("torch").float32,
            device=kwargs.get("device", "cpu"),
        )
    if method not in SEED_GENERATORS:
        raise ValueError(f"unknown method {method!r}; known: {sorted(SEED_GENERATORS)}")
    return SEED_GENERATORS[method](M=M, K=K, batch_size=batch_size, seed=seed, **kwargs)


@dataclass(frozen=True, slots=True)
class MethodSpec:
    method: Method
    K: int
    kwargs: dict
    budget: LaneBudget
    label: str


def matched_budget_configs(
    *, M: int, budget: int, mode: Literal["total", "uq"],
    methods: Sequence[Method] | None = None,
) -> list[MethodSpec]:
    """Every configuration spending exactly `budget` lanes (spec SS49).

    `mode='total'` counts mean + uncertainty lanes; `mode='uq'` counts uncertainty
    lanes only. Head subsampling appears in both, with and without its mean lane,
    so a one-sided comparison cannot be produced by accident. Spec SS15 calls this
    the paper's most important comparison, and "equal budget" is exactly the term
    that decides it.
    """
    methods = list(methods or ["gaussian", "haar", "rademacher", "pairwise", "head_subsample"])
    out: list[MethodSpec] = []

    for method in methods:
        if method == "head_subsample":
            for with_mean in (False, True):
                K = budget - int(with_mean) if mode == "total" else budget
                if K < 2 or K > M:
                    continue
                out.append(MethodSpec(
                    method, K, {"with_mean_lane": with_mean},
                    LaneBudget(K, int(with_mean), with_mean),
                    f"head_subsample(K={K}{', +mean' if with_mean else ''})",
                ))
        else:
            K = budget - 1 if mode == "total" else budget
            if K < 1 or (method == "haar" and K > M - 1):
                continue
            out.append(MethodSpec(
                method, K, {}, LaneBudget(K, 1, True), f"{method}(K={K})"
            ))

    if mode == "total" and budget == M:
        out.append(MethodSpec(
            "exact", M - 1, {}, LaneBudget(M - 1, 1, True), "exact centered basis"
        ))
    return out
