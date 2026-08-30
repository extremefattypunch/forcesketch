"""J1 acceptance tests: split disjointness, guard-band separation, and the
proof that a learned basis depends only on design data.

The basis test is the important one. Asserting "we passed design indices" only
checks the call site; poisoning every non-design row with NaN and demanding a
finite, hash-identical basis proves the *function* cannot have read them.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch_journal.data.splits import (  # noqa: E402
    alpha_feasible, contiguous_block_split, coverage_law, random_frame_split,
)


@pytest.mark.parametrize("n", [500, 1000, 2139])
@pytest.mark.parametrize("guard", [0, 3, 8])
def test_roles_partition_and_are_disjoint(n, guard):
    s = contiguous_block_split(n, system="t", seed=1, tau_int=2.0, guard=guard)
    s.assert_valid()
    assert len(s.design) + len(s.cal) + len(s.test) + len(s.dropped) == n


@pytest.mark.parametrize("guard", [1, 3, 8])
def test_guard_band_separates_roles_in_time(guard):
    """No two indices in different roles may be within `guard` of each other."""
    s = contiguous_block_split(2139, system="t", seed=7, tau_int=4.0, guard=guard)
    role = np.empty(s.n, dtype=object)
    for r in ("design", "cal", "test"):
        role[s.indices(r)] = r
    assigned = np.array([i for i in range(s.n) if role[i] is not None])
    lab = role[assigned]
    for a, b in zip(range(len(assigned) - 1), range(1, len(assigned))):
        if lab[a] != lab[b]:
            assert assigned[b] - assigned[a] > guard, (
                f"roles {lab[a]}/{lab[b]} only {assigned[b]-assigned[a]} apart, guard={guard}")


def test_manifest_hash_is_deterministic_and_assignment_sensitive():
    a = contiguous_block_split(1000, system="s", seed=3, tau_int=2.0, guard=3)
    b = contiguous_block_split(1000, system="s", seed=3, tau_int=2.0, guard=3)
    c = contiguous_block_split(1000, system="s", seed=4, tau_int=2.0, guard=3)
    assert a.sha256 == b.sha256, "same inputs must hash identically"
    assert a.sha256 != c.sha256, "different assignment must hash differently"
    # tau_int is metadata, not assignment: it must not enter the content hash
    d = contiguous_block_split(1000, system="s", seed=3, tau_int=99.0, guard=3)
    assert a.sha256 == d.sha256


def test_random_frame_split_matches_workshop_proportions():
    s = random_frame_split(2139, system="3bpa", seed=20260901)
    s.assert_valid()
    assert (len(s.design), len(s.cal), len(s.test)) == (428, 428, 1283)


def test_alpha_feasibility_thresholds():
    # ceil((n+1)(1-a)) <= n  =>  a=0.10 needs n>=9, a=0.05 needs n>=19, a=0.01 needs n>=99
    for alpha, n_min in ((0.10, 9), (0.05, 19), (0.01, 99)):
        assert not alpha_feasible(n_min - 1, alpha)[0]
        assert alpha_feasible(n_min, alpha)[0]


def test_coverage_law_brackets_nominal():
    law = coverage_law(200, 0.05)
    assert law["feasible"] and law["q05"] < 1 - 0.05 < law["q95"]
    # a bigger calibration set must give a tighter coverage distribution
    assert coverage_law(2000, 0.05)["sd"] < law["sd"]


def _leading_basis(F: torch.Tensor, idx: np.ndarray, r0: int) -> torch.Tensor:
    from forcesketch.sketches.control_variate import leading_head_directions
    return leading_head_directions(F[torch.as_tensor(idx.copy())], r0)


@pytest.mark.parametrize("r0", [1, 2, 3])
def test_basis_uses_design_only(r0):
    """Poison every non-design structure with NaN.

    If `leading_head_directions` touched any of them the eigendecomposition
    would return NaN. A finite basis with an unchanged hash is proof the
    function read design rows and nothing else.
    """
    torch.manual_seed(0)
    S, A, M = 300, 9, 8
    F = torch.randn(S, A, 3, M, dtype=torch.float64)
    split = contiguous_block_split(S, system="t", seed=11, tau_int=1.0, guard=2)

    clean = _leading_basis(F, split.design, r0)

    poisoned = F.clone()
    mask = torch.ones(S, dtype=torch.bool)
    mask[torch.as_tensor(split.design.copy())] = False
    poisoned[mask] = float("nan")

    got = _leading_basis(poisoned, split.design, r0)
    assert torch.isfinite(got).all(), "basis saw NaN -> it read non-design rows"
    assert torch.allclose(clean, got, atol=0, rtol=0), "basis changed under poisoning"


BANNED = {"split_indices", "matched_budget_configs"}


def test_banned_frozen_helpers_are_never_imported_or_called():
    """`fallback_gate.split_indices` is one of the two divergent implementations,
    and `registry.matched_budget_configs` is dead code in the frozen tree.

    Checked by AST rather than by grep, so that prose explaining *why* they are
    banned does not trip the test -- only a real import or call does.
    """
    import ast

    offenders = []
    for root in (ROOT / "src", ROOT / "experiments"):
        for p in root.rglob("*.py"):
            tree = ast.parse(p.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for a in node.names:
                        if a.name in BANNED:
                            offenders.append(f"{p.name}: imports {a.name}")
                elif isinstance(node, ast.Call):
                    fn = node.func
                    name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                    if name in BANNED:
                        offenders.append(f"{p.name}:{node.lineno}: calls {name}")
    assert not offenders, f"banned frozen helpers used: {offenders}"
