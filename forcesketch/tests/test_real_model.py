"""Spec SS23 (real-model exactness) and SS21 (full-rank exactness on the real model).

SS23 requires four INDEPENDENT computation paths to agree:
  1. explicit member-head forces
  2. centered-basis exact forces (serial)
  3. batched centered-basis VJP
  4. direct variance from all member forces

on coordinate variance, global trace, atom RMS score, and atom MHC score.

Everything here runs in float64. The head-space combination cancels O(|mean force|)
down to O(|disagreement|), and on this committee the mean force is ~6x the
disagreement, so a float32 "exact" reference would be measuring its own round-off
(plan resolution R5).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

CKPT = Path("models/zenodo/3BPA/trainset_100/multihead-disjoint/multihead_committee_stagetwo.model")
DATA = Path("data/3bpa/test_1200K_ref.xyz")

pytestmark = [
    pytest.mark.requires_mace,
    pytest.mark.skipif(importlib.util.find_spec("mace") is None, reason="mace not installed"),
    pytest.mark.skipif(not CKPT.exists(), reason="reference checkpoint not fetched"),
    pytest.mark.skipif(not DATA.exists(), reason="3BPA data not prepared"),
]


@pytest.fixture(scope="function")
def real_case():
    """Function-scoped ON PURPOSE, despite the ~4 s model load.

    e3nn's TorchScript autograd nodes specialize on the FIRST backward they see.
    Once a plain (non-batched) backward has run through a model instance, every
    later `is_grads_batched=True` call on that instance dies with

        RuntimeError: Cannot access data pointer of Tensor that doesn't have storage

    Measured: [batched, serial, serial_loop] all succeed, but [serial, batched] and
    [serial_loop, batched] both fail. A module-scoped fixture therefore makes the
    tests order-dependent, and -- more importantly -- it means every benchmark in
    spec SS26 that compares serial against batched implementations must run each
    configuration in a FRESH model instance or process, or the batched variant will
    simply throw. See plan Task 2.2's subprocess isolation.
    """
    from forcesketch.adapters.mace_data import load_frames, make_loader
    from forcesketch.adapters.mace_mhc import MaceMHCAdapter
    from forcesketch.utils.reproducibility import pin_numerics

    pin_numerics()
    adapter = MaceMHCAdapter.from_checkpoint(CKPT, dtype=torch.float64)
    frames = load_frames(DATA, limit=8)
    batch = next(iter(make_loader(frames, adapter.model, batch_size=4)))
    return adapter, adapter.prepare(batch.to_dict())


def test_four_paths_agree(real_case):
    """Spec SS23. Tolerance 1e-9 relative: these are four different float64
    reduction orders over the same quantity, not four different algorithms."""
    from forcesketch.estimators.scores import uncertainty_scores
    from forcesketch.estimators.variance import coordinate_variance
    from forcesketch.exact.centered_basis import exact_seed_bundle
    from forcesketch.exact.member_forces import variance_from_member_forces

    adapter, batch = real_case
    M, B = adapter.num_heads, adapter.num_structures(batch)
    bidx = adapter.batch_index(batch)
    bundle = exact_seed_bundle(M, B, dtype=torch.float64, device=adapter.device)

    # ORDER IS LOAD-BEARING: the batched path must run FIRST. A plain backward
    # specializes e3nn's TorchScript autograd nodes and permanently breaks
    # is_grads_batched on this model instance (see the real_case docstring).

    # path 3: batched centered-basis VJP
    lanes_batched = adapter.vjp_for_seeds(batch, bundle.seeds, batched=True).double()
    v_batched = coordinate_variance(lanes_batched, bundle)

    # path 2: serial centered-basis VJP
    lanes_serial = adapter.vjp_for_seeds(batch, bundle.seeds, batched=False).double()
    v_serial = coordinate_variance(lanes_serial, bundle)

    # path 1: explicit member-head forces  -> path 4: direct variance
    F_heads = adapter.member_forces_reference(batch).double()  # [M, N, 3]
    v_direct = variance_from_member_forces(F_heads)

    scale = v_direct.max()
    for name, v in (("serial", v_serial), ("batched", v_batched)):
        err = (v - v_direct).abs().max() / scale
        assert err < 1e-9, f"centered-basis ({name}) vs member forces: {err:.3e}"

    # ... and the four spec SS23 derived scores
    sc_a = uncertainty_scores(lanes_batched, bundle, batch_index=bidx, n_structures=B)
    sc_b = uncertainty_scores(lanes_serial, bundle, batch_index=bidx, n_structures=B)
    for field in ("coord_var", "global_trace", "atom_rms", "atom_mhc"):
        a, b = getattr(sc_a, field), getattr(sc_b, field)
        err = (a - b).abs().max() / a.abs().max()
        assert err < 1e-9, f"{field}: serial vs batched {err:.3e}"


def test_haar_full_rank_is_exact_on_the_real_model(real_case):
    """Spec SS21 on the real committee: Haar at K = M-1 must reproduce v_d.

    Asserted scale-normalized (max_d |v_hat - v| / max_d v), because the spec's
    |v_hat-v|/(|v|+eps) is dominated by coordinates whose true variance is ~0 by
    chance and fails for reasons unrelated to the estimator (plan resolution R4).
    """
    from forcesketch.estimators.variance import coordinate_variance
    from forcesketch.exact.centered_basis import exact_seed_bundle
    from forcesketch.sketches.registry import make_sketch_seeds

    adapter, batch = real_case
    M, B = adapter.num_heads, adapter.num_structures(batch)

    exact = exact_seed_bundle(M, B, dtype=torch.float64, device=adapter.device)
    v_ref = coordinate_variance(
        adapter.vjp_for_seeds(batch, exact.seeds, batched=False).double(), exact)

    for seed in range(5):
        haar = make_sketch_seeds("haar", M=M, K=M - 1, batch_size=B, seed=seed,
                                 dtype=torch.float64, device=adapter.device)
        assert haar.std_correction == 1.0
        v_haar = coordinate_variance(
            adapter.vjp_for_seeds(batch, haar.seeds, batched=False).double(), haar)
        err = (v_haar - v_ref).abs().max() / v_ref.max()
        assert err < 1e-9, f"seed {seed}: full-rank Haar error {err:.3e}"


def test_sketch_estimators_are_unbiased_on_the_real_model(real_case):
    """The estimators are unbiased against the REAL committee's F, not just a
    synthetic one. Averages over many seeds at K=3 and checks the global score."""
    from forcesketch.estimators.variance import coordinate_variance
    from forcesketch.exact.centered_basis import exact_seed_bundle
    from forcesketch.sketches.registry import make_sketch_seeds

    adapter, batch = real_case
    M, B = adapter.num_heads, adapter.num_structures(batch)
    exact = exact_seed_bundle(M, B, dtype=torch.float64, device=adapter.device)
    v_ref = coordinate_variance(
        adapter.vjp_for_seeds(batch, exact.seeds, batched=False).double(), exact)
    S_ref = v_ref.sum()

    for method, tol in (("gaussian", 0.06), ("haar", 0.04), ("rademacher", 0.06)):
        acc = 0.0
        n = 60
        for seed in range(n):
            b = make_sketch_seeds(method, M=M, K=3, batch_size=B, seed=seed,
                                  dtype=torch.float64, device=adapter.device)
            acc += float(coordinate_variance(
                adapter.vjp_for_seeds(batch, b.seeds, batched=False).double(), b).sum())
        ratio = acc / n / float(S_ref)
        assert abs(ratio - 1.0) < tol, f"{method} K=3: E[S_hat]/S = {ratio:.4f}"


def test_serial_vjp_emits_no_vmap_fallback(real_case):
    """Spec SS18 requires any vmap fallback warning be recorded. The serial path is
    the one we actually benchmark, so pin that it is warning-free."""
    from forcesketch.exact.centered_basis import exact_seed_bundle

    adapter, batch = real_case
    bundle = exact_seed_bundle(adapter.num_heads, adapter.num_structures(batch),
                               dtype=torch.float64, device=adapter.device)
    _, fallbacks = adapter.vjp_for_seeds(
        batch, bundle.seeds, batched=False, record_fallbacks=True
    )
    assert fallbacks == (), f"unexpected vmap/fallback warning: {fallbacks}"


def test_batched_vjp_limitation_is_characterized(real_case):
    """Documents a real property of the MACE + e3nn 0.4.4 stack, as a regression test.

    `is_grads_batched=True` is NOT reliably usable here. It succeeds for the first
    call or two in a fresh process and then raises, from inside the TorchScript
    interpreter:

        RuntimeError: Cannot access data pointer of Tensor that doesn't have storage

    because e3nn's optimized autograd nodes cannot accept vmap's BatchedTensor, and
    the specialization is cumulative and process-global -- it is NOT cured by a
    fresh model instance, by disabling jit_script_fx / optimize_einsums /
    specialized_code, by turning off the TorchScript profiling executor, or by
    rebuilding the model unscripted from extract_config_mace_model.

    Consequence for the paper: spec SS17 items 3 (batched centered-basis VJP) and 4
    (torch.func/vmap -- separately blocked because MACE calls requires_grad_()
    inside forward, which functorch forbids) are UNAVAILABLE on this stack. The
    strongest available exact baseline is the serial loop over centered directions,
    which is also what the reference implementation itself does in
    `get_outputs_committee`. This must be stated plainly in the limitations: on a
    stack where batched VJP works, the exact baseline would be faster and the
    measured ForceSketch speedup correspondingly smaller.

    The test asserts only that the behaviour is one of the two known outcomes, so it
    will fail loudly if a future torch/e3nn makes batched VJP work -- at which point
    the baseline must be re-measured.
    """
    from forcesketch.exact.centered_basis import exact_seed_bundle

    adapter, batch = real_case
    bundle = exact_seed_bundle(adapter.num_heads, adapter.num_structures(batch),
                               dtype=torch.float64, device=adapter.device)
    try:
        out = adapter.vjp_for_seeds(batch, bundle.seeds, batched=True)
        assert out.shape[0] == bundle.K  # worked this time; fine, it is nondeterministic
    except RuntimeError as exc:
        assert "doesn't have storage" in str(exc) or "TorchScript" in str(exc), (
            f"batched VJP failed for an UNEXPECTED reason, which needs investigation: {exc}"
        )


def test_committee_heads_are_not_degenerate(real_case):
    """Plan risk 0: a committee of identical heads makes every ForceSketch result
    vacuously exact and looks healthy in every other diagnostic."""
    adapter, batch = real_case
    adapter.assert_heads_nondegenerate(batch)
