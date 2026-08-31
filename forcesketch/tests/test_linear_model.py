"""Spec SS20 -- the mandatory synthetic correctness test.

E_m(x) = a_m^T x + b_m  =>  f_m = -a_m analytically, so every quantity in SS7-SS15
has a closed form. Nothing here touches MACE, e3nn, ASE, or CUDA, which is what
lets SS20's gate ("no molecular experiments should begin until these pass") be
cleared independently of the environment track.

One test per SS20 bullet: exact member forces / centered force matrix / exact
centered basis / global variance / coordinate variance / atom variance / Gaussian
unbiasedness / Haar unbiasedness / pairwise normalization / head subsampling /
finite-K std correction.
"""

from __future__ import annotations

import math

import pytest
import torch

from _mc import gaussian_rel_std, haar_rel_std, mc_tolerance, sqrt_rel_std
from conftest import ATOL, REF_DTYPE, make_generator, make_linear_case, projector

from forcesketch.adapters.linear import head_force_matrix
from forcesketch.estimators.scores import uncertainty_scores
from forcesketch.estimators.std_correction import gaussian_std_correction
from forcesketch.estimators.variance import coordinate_variance
from forcesketch.exact.centered_basis import (
    assert_valid_centered_basis,
    exact_seed_bundle,
    helmert_basis,
    mean_seed,
)
from forcesketch.sketches.registry import make_sketch_seeds
from forcesketch.utils.layout import segment_sum

ALL_M = [2, 3, 8]


# --------------------------------------------------------------------------
# SS20.1  exact member forces
# --------------------------------------------------------------------------
@pytest.mark.parametrize("M", ALL_M)
def test_exact_member_forces_match_analytic_solution(M):
    adapter, batch, coeffs = make_linear_case(M=M)
    forces = adapter.exact_head_forces(batch)
    assert forces.shape == coeffs.shape
    torch.testing.assert_close(forces, -coeffs, rtol=0, atol=ATOL)


def test_vjp_returns_forces_not_gradients():
    """The sign cancels for variance but NOT for the mean-force lane, so if
    someone drops the minus sign every variance test still passes and only the
    mean force is silently wrong. Pin it."""
    adapter, batch, coeffs = make_linear_case(M=8)
    B = adapter.num_structures(batch)
    s0 = mean_seed(8, B, dtype=REF_DTYPE)
    f_mean = adapter.vjp_for_seeds(batch, s0)[0]
    torch.testing.assert_close(f_mean, (-coeffs).mean(0), rtol=0, atol=ATOL)
    assert not torch.allclose(f_mean, coeffs.mean(0), atol=1e-6)


# --------------------------------------------------------------------------
# SS20.2  centered force matrix
# --------------------------------------------------------------------------
@pytest.mark.parametrize("M", ALL_M)
def test_centered_force_matrix_has_zero_head_mean(M):
    _, _, coeffs = make_linear_case(M=M)
    F = head_force_matrix(coeffs)
    P = projector(M)
    FP = F @ P
    torch.testing.assert_close(FP.sum(-1), torch.zeros_like(FP.sum(-1)), rtol=0, atol=ATOL)
    torch.testing.assert_close(P @ P, P, rtol=0, atol=ATOL)  # idempotent
    torch.testing.assert_close(F @ P @ P, FP, rtol=0, atol=ATOL)


# --------------------------------------------------------------------------
# SS20.3  exact centered basis (spec SS8)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("M", [2, 3, 4, 8, 16])
def test_helmert_basis_satisfies_spec_section_8(M):
    Q = helmert_basis(M, dtype=REF_DTYPE)
    assert Q.shape == (M, M - 1)
    torch.testing.assert_close(Q.T @ Q, torch.eye(M - 1, dtype=REF_DTYPE), rtol=0, atol=ATOL)
    # Q^T 1 = 0 is a sum over the HEAD axis (dim 0), not over columns.
    torch.testing.assert_close(Q.sum(dim=0), torch.zeros(M - 1, dtype=REF_DTYPE), rtol=0, atol=ATOL)
    torch.testing.assert_close(Q @ Q.T, projector(M), rtol=0, atol=ATOL)
    assert_valid_centered_basis(Q)


def test_helmert_basis_is_deterministic_and_seedless():
    a = helmert_basis(8, dtype=REF_DTYPE)
    torch.manual_seed(12345)
    b = helmert_basis(8, dtype=REF_DTYPE)
    assert torch.equal(a, b)
    torch.testing.assert_close(
        a[:, 0],
        torch.tensor([1.0, -1.0, 0, 0, 0, 0, 0, 0], dtype=REF_DTYPE) / math.sqrt(2),
        rtol=0, atol=ATOL,
    )
    torch.testing.assert_close(
        a[:, 1],
        torch.tensor([1.0, 1.0, -2.0, 0, 0, 0, 0, 0], dtype=REF_DTYPE) / math.sqrt(6),
        rtol=0, atol=ATOL,
    )


def test_assert_valid_centered_basis_rejects_a_bad_basis():
    Q = helmert_basis(8, dtype=REF_DTYPE).clone()
    Q[:, 0] += 0.1  # breaks both orthonormality and centering
    with pytest.raises(ValueError, match="invalid centered basis"):
        assert_valid_centered_basis(Q)


@pytest.mark.parametrize("M", ALL_M)
def test_exact_centered_basis_reproduces_direct_variance(M):
    """Spec SS8: v_d = (1/(M-1)) sum_j g_{j,d}^2 with g_j = F q_j."""
    adapter, batch, coeffs = make_linear_case(M=M)
    B = adapter.num_structures(batch)
    F = head_force_matrix(coeffs)
    v_direct = F.var(dim=-1, unbiased=True)

    bundle = exact_seed_bundle(M, B, dtype=REF_DTYPE)
    assert bundle.K == M - 1
    assert bundle.rng_seed is None
    assert bundle.lane_budget.total_lanes == M
    assert bundle.lane_budget.exact_mean_force

    lanes = adapter.vjp_for_seeds(batch, bundle.seeds)
    torch.testing.assert_close(coordinate_variance(lanes, bundle), v_direct, rtol=1e-12, atol=ATOL)


def test_variance_is_invariant_to_choice_of_centered_basis():
    """Any Q with QQ^T = P gives the same v_d, so the Helmert choice is safe."""
    import dataclasses

    adapter, batch, coeffs = make_linear_case(M=8)
    B = adapter.num_structures(batch)
    bundle = exact_seed_bundle(8, B, dtype=REF_DTYPE)
    v_helmert = coordinate_variance(adapter.vjp_for_seeds(batch, bundle.seeds), bundle)

    g = make_generator(99)
    O, _ = torch.linalg.qr(torch.randn(7, 7, generator=g, dtype=REF_DTYPE))
    rotated = torch.einsum("kbm,kj->jbm", bundle.seeds, O)
    rot_bundle = dataclasses.replace(bundle, seeds=rotated.contiguous())
    v_rot = coordinate_variance(adapter.vjp_for_seeds(batch, rot_bundle.seeds), rot_bundle)
    torch.testing.assert_close(v_rot, v_helmert, rtol=1e-11, atol=ATOL)


# --------------------------------------------------------------------------
# the plumbing test: seeds -> cotangent -> forces, incl. per-structure seeds
# --------------------------------------------------------------------------
def test_vjp_equals_analytic_head_force_matrix_product():
    """Most load-bearing structural test: proves the seed -> is_grads_batched ->
    flat [L, N_total, 3] path computes exactly F @ w, including per-structure seed
    indexing through batch_index."""
    adapter, batch, coeffs = make_linear_case(M=8)
    B, bidx = adapter.num_structures(batch), adapter.batch_index(batch)
    F = head_force_matrix(coeffs)
    bundle = make_sketch_seeds("gaussian", M=8, K=5, batch_size=B, seed=7,
                               dtype=REF_DTYPE, per_structure=True)
    assert bundle.seeds.shape == (5, B, 8)
    lanes = adapter.vjp_for_seeds(batch, bundle.seeds)
    ref = torch.einsum("iam,kim->kia", F, bundle.seeds[:, bidx, :])
    torch.testing.assert_close(lanes, ref, rtol=1e-12, atol=ATOL)


def test_serial_and_batched_vjp_agree():
    adapter, batch, _ = make_linear_case(M=8)
    B = adapter.num_structures(batch)
    bundle = make_sketch_seeds("haar", M=8, K=4, batch_size=B, seed=3, dtype=REF_DTYPE)
    a = adapter.vjp_for_seeds(batch, bundle.seeds, batched=True)
    b = adapter.vjp_for_seeds(batch, bundle.seeds, batched=False)
    torch.testing.assert_close(a, b, rtol=1e-12, atol=ATOL)


def test_no_vmap_fallback_on_the_batched_path():
    """Spec SS18: any vmap fallback warning must be recorded."""
    adapter, batch, _ = make_linear_case(M=8)
    B = adapter.num_structures(batch)
    bundle = exact_seed_bundle(8, B, dtype=REF_DTYPE)
    _, fallbacks = adapter.vjp_for_seeds(batch, bundle.seeds, record_fallbacks=True)
    assert fallbacks == (), f"unexpected vmap fallback: {fallbacks}"


# --------------------------------------------------------------------------
# SS20.4-SS20.6  global / coordinate / atom variance
# --------------------------------------------------------------------------
def test_global_coordinate_and_atom_statistics():
    adapter, batch, coeffs = make_linear_case(M=8)
    B, bidx = adapter.num_structures(batch), adapter.batch_index(batch)
    F = head_force_matrix(coeffs)
    v = F.var(dim=-1, unbiased=True)

    bundle = exact_seed_bundle(8, B, dtype=REF_DTYPE)
    lanes = adapter.vjp_for_seeds(batch, bundle.seeds)
    sc = uncertainty_scores(lanes, bundle, batch_index=bidx, n_structures=B)

    torch.testing.assert_close(sc.coord_var, v, rtol=1e-12, atol=ATOL)
    torch.testing.assert_close(sc.coord_std, v.sqrt(), rtol=1e-12, atol=ATOL)
    torch.testing.assert_close(sc.atom_sum, v.sum(-1), rtol=1e-12, atol=ATOL)
    torch.testing.assert_close(sc.atom_rms, v.mean(-1).sqrt(), rtol=1e-12, atol=ATOL)
    torch.testing.assert_close(sc.atom_mhc, v.sqrt().mean(-1), rtol=1e-12, atol=ATOL)

    # S per structure, two independent routes: sum of v_d, and ||FP||_F^2/(M-1)
    torch.testing.assert_close(sc.global_trace, segment_sum(v.sum(-1), bidx, B),
                               rtol=1e-12, atol=ATOL)
    fro = ((F @ projector(8)) ** 2).sum((-1, -2))
    torch.testing.assert_close(sc.global_trace, segment_sum(fro / 7.0, bidx, B),
                               rtol=1e-11, atol=ATOL)

    expected_max = torch.stack([sc.atom_mhc[bidx == b].max() for b in range(B)])
    torch.testing.assert_close(sc.max_atom_mhc, expected_max, rtol=0, atol=ATOL)


# --------------------------------------------------------------------------
# SS20.7-SS20.10  estimator unbiasedness (smoke; >=1e5-draw versions elsewhere)
# --------------------------------------------------------------------------
def _mc_variance_ratio(method: str, M: int, K: int, n_draws: int, **kw):
    """Drive n_draws independent sketches through the analytic F with no autograd,
    by treating the draw index as the batch axis. 2e4 draws in milliseconds."""
    g = make_generator(3)
    F = torch.randn(4, 3, M, generator=g, dtype=REF_DTYPE)
    v = F.var(dim=-1, unbiased=True)
    bundle = make_sketch_seeds(method, M=M, K=K, batch_size=n_draws, seed=11,
                               dtype=REF_DTYPE, **kw)
    lanes = torch.einsum("iam,kdm->kdia", F, bundle.seeds)
    v_hat = coordinate_variance(lanes, bundle)
    return v_hat.mean(0) / v, v_hat, v


@pytest.mark.parametrize("method,K,rel_std", [
    ("gaussian", 1, gaussian_rel_std(1)),
    ("gaussian", 3, gaussian_rel_std(3)),
    ("haar", 1, haar_rel_std(1, 7)),
    ("haar", 3, haar_rel_std(3, 7)),
    ("rademacher", 3, gaussian_rel_std(3)),
    ("pairwise", 3, 1.6),
    ("head_subsample", 3, 0.9),
])
def test_estimators_are_unbiased(method, K, rel_std):
    n = 40_000
    ratio, _, _ = _mc_variance_ratio(method, 8, K, n)
    tol = mc_tolerance(rel_std, n, n_sigma=5.0)
    assert (ratio - 1.0).abs().max().item() < tol, f"{method} K={K}: ratio={ratio}"


def test_pairwise_K1_equals_head_subsample_K2():
    """The ddof=1 sample variance of two heads IS (1/2)(f_i - f_j)^2 -- algebraically
    the same estimator, with the same distribution. But pairwise spends ONE reverse
    lane and head subsampling spends TWO. That is a zero-experiment demonstration
    of the spec SS15/SS49 claim, so it is pinned here.

    The identity is exact in exact arithmetic but NOT bitwise in floating point:
    torch.var takes a two-pass mean-then-deviation route while the pairwise form is
    direct. Measured over 2000 samples: max relative difference 9.6e-13 (float64)
    and 2.3e-7 (float32), bitwise-equal about half the time. Assert round-off, not
    bit equality.
    """
    g = make_generator(5)
    F = torch.randn(64, 3, 8, generator=g, dtype=REF_DTYPE)
    i, j = 2, 5
    head_sub = F[..., [i, j]].var(dim=-1, unbiased=True)
    pairwise = 0.5 * (F[..., i] - F[..., j]) ** 2
    torch.testing.assert_close(head_sub, pairwise, rtol=1e-11, atol=0)

    # The scientific point is the lane count, so assert that structurally too.
    hs = make_sketch_seeds("head_subsample", M=8, K=2, batch_size=1, seed=0, dtype=REF_DTYPE)
    pw = make_sketch_seeds("pairwise", M=8, K=1, batch_size=1, seed=0, dtype=REF_DTYPE)
    assert hs.lane_budget.uq_lanes == 2
    assert pw.lane_budget.uq_lanes == 1


def test_haar_is_exact_at_full_rank_while_head_subsample_is_not():
    """At K = M-1 = 7 the Haar estimator has zero variance; head subsampling at the
    same K still fluctuates. Head subsampling only becomes exact at K = M = 8, i.e.
    8 lanes against the exact centered basis's 7."""
    n = 5_000
    _, v_haar, _ = _mc_variance_ratio("haar", 8, 7, n)
    _, v_hs, _ = _mc_variance_ratio("head_subsample", 8, 7, n)
    assert v_haar.std(0).max().item() < 1e-12
    assert v_hs.std(0).max().item() > 1e-3


# --------------------------------------------------------------------------
# SS20.11  finite-K standard-deviation correction
# --------------------------------------------------------------------------
@pytest.mark.parametrize("K", [1, 2, 3, 5])
def test_finite_K_std_correction_removes_downward_bias(K):
    """Uncorrected sqrt(v_hat) is biased low by exactly c_K. Both directions are
    asserted, so a no-op correction fails the test."""
    n = 60_000
    g = make_generator(4)
    F = torch.randn(2, 3, 8, generator=g, dtype=REF_DTYPE)
    v = F.var(dim=-1, unbiased=True)
    bundle = make_sketch_seeds("gaussian", M=8, K=K, batch_size=n, seed=17, dtype=REF_DTYPE)
    lanes = torch.einsum("iam,kdm->kdia", F, bundle.seeds)
    v_hat = coordinate_variance(lanes, bundle)

    c_K = gaussian_std_correction(K)
    assert bundle.std_correction == pytest.approx(c_K, rel=1e-12)

    raw = v_hat.sqrt().mean(0) / v.sqrt()
    corrected = (v_hat.sqrt() / c_K).mean(0) / v.sqrt()
    tol = mc_tolerance(sqrt_rel_std(gaussian_rel_std(K)), n, n_sigma=5.0)

    torch.testing.assert_close(raw, torch.full_like(raw, c_K), rtol=0, atol=tol)
    torch.testing.assert_close(corrected, torch.ones_like(corrected), rtol=0, atol=tol)
    assert raw.max().item() < 1.0  # the bias is genuinely downward
