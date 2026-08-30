"""The exact-mean head-subsampling estimator must be unbiased (revision C3).

This is the baseline ForceSketch has to beat at equal budget. An earlier version
of the comparison charged head subsampling for a mean-force lane and then scored
it with the sample variance among the drawn heads, which ignores that lane --
flattering ForceSketch. This test pins the fair estimator.
"""

from __future__ import annotations

import torch

from forcesketch.sketches.registry import make_sketch_seeds


def _apply(F, bundle):
    G = torch.einsum("sadm,ksm->ksad", F, bundle.seeds.double())
    if bundle.estimator_kind == "quadratic":
        return bundle.variance_scale * (G**2).sum(dim=0)
    return G.var(dim=0, unbiased=True)


def test_unbiased_against_exact_variance():
    torch.manual_seed(0)
    M, S, A, K = 8, 24, 5, 3
    F = torch.randn(S, A, 3, M, dtype=torch.float64)
    exact = F.var(dim=-1, unbiased=True)

    acc = torch.zeros_like(exact)
    n = 4000
    for seed in range(n):
        b = make_sketch_seeds("head_subsample_exact_mean", M=M, K=K, batch_size=S,
                              seed=seed, dtype=torch.float64)
        acc += _apply(F, b)
    est = acc / n
    rel = ((est - exact).abs() / exact.clamp_min(1e-12)).mean().item()
    assert rel < 0.02, f"mean relative bias {rel:.4f} over {n} draws"


def test_exact_at_full_budget():
    """With every head drawn, the estimator IS the exact variance."""
    torch.manual_seed(1)
    M, S, A = 8, 6, 4
    F = torch.randn(S, A, 3, M, dtype=torch.float64)
    b = make_sketch_seeds("head_subsample_exact_mean", M=M, K=M, batch_size=S,
                          seed=0, dtype=torch.float64)
    torch.testing.assert_close(_apply(F, b), F.var(dim=-1, unbiased=True))
