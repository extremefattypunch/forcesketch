"""Shared fixtures. Correctness runs in float64 unless a test is explicitly about
precision; the estimator noise being measured is ~1e-3, so a float32 reference
would be measuring its own round-off."""

from __future__ import annotations

import torch

from forcesketch.adapters.linear import LinearBatch, LinearMHCAdapter

REF_DTYPE = torch.float64
ATOL = 1e-12

# Deliberately disjoint from configs/seeds.yaml so a unit test can never consume
# an experiment seed.
TEST_SEED = 20260820
DEFAULT_ATOMS = (4, 7, 5)


def make_generator(offset: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(TEST_SEED + offset)
    return g


def projector(M: int, dtype: torch.dtype = REF_DTYPE) -> torch.Tensor:
    """P = I - (1/M) 11^T."""
    return torch.eye(M, dtype=dtype) - 1.0 / M


def make_linear_case(
    *, M: int = 8, atoms_per_structure=DEFAULT_ATOMS, dtype=REF_DTYPE,
    common_mode: float = 0.0, seed_offset: int = 0,
):
    """(adapter, batch, coeffs) for E_m(x) = a_m^T x + b_m.

    `common_mode` injects a head-independent force component of that magnitude.
    It is the rho knob: rho = |mean force| / sqrt(variance) sets the float32 error
    floor, and a real committee sits at rho in [10, 1000].
    """
    g = make_generator(seed_offset)
    counts = torch.tensor(atoms_per_structure, dtype=torch.long)
    n_total = int(counts.sum())
    batch_index = torch.repeat_interleave(torch.arange(len(counts)), counts)

    coeffs = torch.randn(M, n_total, 3, generator=g, dtype=REF_DTYPE)
    if common_mode:
        coeffs = coeffs + common_mode * torch.randn(1, n_total, 3, generator=g, dtype=REF_DTYPE)
    biases = torch.randn(M, generator=g, dtype=REF_DTYPE)
    positions = torch.randn(n_total, 3, generator=g, dtype=REF_DTYPE)

    adapter = LinearMHCAdapter(coeffs=coeffs.to(dtype), biases=biases.to(dtype))
    batch = LinearBatch(
        positions=positions.to(dtype).requires_grad_(True), batch_index=batch_index
    )
    return adapter, batch, coeffs.to(dtype)
