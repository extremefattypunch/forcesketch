"""The correctness tests the plan lists as "write these first", finally written.

Three things carry published guarantees and had no test until now:

  * the split-conformal constant -- the paper's central safety claim;
  * the control variate's unbiasedness and its full-rank completion -- the
    property that makes the sketch an estimator rather than a heuristic;
  * the test-access ledger.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT.parents[0] / "forcesketch"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FROZEN / "src"))

from forcesketch_journal.calibration.conformal import (  # noqa: E402
    conformal_c, coverage_simulation,
)
# aliased: pytest would otherwise collect the imported context manager as a test
from forcesketch_journal.data.ledger import audit_ledger  # noqa: E402
from forcesketch_journal.data.ledger import test_indices as open_test_indices  # noqa: E402


# ----------------------------------------------------------------- conformal
def test_conformal_c_is_the_stated_order_statistic():
    """Exactly the ceil((n+1)(1-alpha))-th smallest ratio, not a quantile guess."""
    g = torch.Generator().manual_seed(3)
    for n in (19, 99, 100, 412):
        for alpha in (0.10, 0.05):
            s = torch.rand(n, generator=g, dtype=torch.float64) * 10
            c = conformal_c(s, torch.ones(n, dtype=torch.float64), alpha)
            k = math.ceil((n + 1) * (1 - alpha))
            assert c == pytest.approx(float(s.sort().values[k - 1]), rel=1e-12)


@pytest.mark.parametrize("n_cal,alpha", [(19, 0.05), (99, 0.01), (94, 0.01), (50, 0.01)])
def test_conformal_c_returns_inf_exactly_when_alpha_is_infeasible(n_cal, alpha):
    """Infeasible must mean +inf (skip nothing), never the max ratio.

    Clamping to the maximum produced an anti-conservative gate wearing the
    requested alpha's label. Water's blocked splits (n_cal 84-94) at alpha=0.01
    are the real case.
    """
    s = torch.rand(n_cal, dtype=torch.float64) + 0.5
    c = conformal_c(s, torch.ones(n_cal, dtype=torch.float64), alpha)
    feasible = math.ceil((n_cal + 1) * (1 - alpha)) <= n_cal
    assert (c != float("inf")) == feasible
    if not feasible:
        assert c == float("inf")


@pytest.mark.parametrize("n_cal,alpha", [(99, 0.10), (199, 0.05), (412, 0.05)])
def test_conformal_coverage_matches_the_finite_sample_law(n_cal, alpha):
    """Realised coverage must track Beta(k, n+1-k)'s mean, k/(n+1).

    This is the claim itself, checked by simulation rather than asserted from the
    literature. Tolerance is 3 Monte-Carlo standard errors.
    """
    r = coverage_simulation(n_cal, alpha, n_trials=4000, seed=1)
    se = math.sqrt(r["beta_mean"] * (1 - r["beta_mean"]) / r["n_trials"])
    assert abs(r["realised"] - r["beta_mean"]) < 3 * se + 0.005, r
    assert r["realised"] >= 1 - alpha - 3 * se, "under-covers the nominal level"


# ---------------------------------------------------------- control variate
def _setup(S=40, A=6, M=8, r0=2, seed=5):
    """Leading directions from the pooled Gram, as the gate actually builds them."""
    from forcesketch.exact.centered_basis import helmert_basis

    g = torch.Generator().manual_seed(seed)
    F = torch.randn(S, A, 3, M, generator=g, dtype=torch.float64)
    Qc = helmert_basis(M, dtype=torch.float64)                      # [M, r]
    X = torch.einsum("sadm,mr->sadr", F, Qc).flatten(0, 2)
    lead = torch.linalg.eigh(X.T @ X)[1].flip(1)[:, :r0]            # [r, r0]
    return F, Qc, Qc @ lead                                         # Q_lead [M, r0]


def test_control_variate_is_unbiased_for_the_exact_variance():
    """E[v_hat] = v, averaged over Haar draws of the residual directions.

    Uses the SHIPPED `control_variate_seeds`, which builds the residual subspace
    inside the orthogonal complement of the leading directions. A first version
    of this test drew residual directions from the whole centred space, so they
    were not orthogonal to the leading block, the leading subspace was counted
    twice, and the estimator looked 18% biased. The bug was the test's.
    """
    from forcesketch.sketches.control_variate import (
        control_variate_seeds, control_variate_variance,
    )

    S, A, M, r0, K = 40, 6, 8, 2, 5
    F, _, Q_lead = _setup(S, A, M, r0)
    v_exact = F.var(dim=-1, unbiased=True)
    acc, n_draw = torch.zeros_like(v_exact), 300
    for i in range(n_draw):
        bundle, got_r0 = control_variate_seeds(Q_lead, M=M, K=K, batch_size=S,
                                               seed=1000 + i, dtype=torch.float64)
        assert got_r0 == r0
        G = torch.einsum("sadm,ksm->ksad", F, bundle.seeds)
        acc += control_variate_variance(G, r0=r0, M=M)
    rel = float(((acc / n_draw - v_exact).abs().mean() / v_exact.abs().mean()))
    assert rel < 0.05, f"control variate is biased: mean relative error {rel:.4f}"


def test_control_variate_is_exact_when_the_basis_is_complete():
    """With K = r the estimator must return v exactly, for any leading block.

    Full-rank completion is the boundary condition: a sketch that does not become
    exact when it stops being a sketch is wrong somewhere in its scaling.
    """
    from forcesketch.sketches.control_variate import (
        control_variate_seeds, control_variate_variance,
    )

    S, A, M = 12, 5, 8
    r = M - 1
    for r0 in (1, 2, 4, 6):
        F, _, Q_lead = _setup(S, A, M, r0, seed=6 + r0)
        v_exact = F.var(dim=-1, unbiased=True)
        bundle, _ = control_variate_seeds(Q_lead, M=M, K=r, batch_size=S,
                                          seed=7, dtype=torch.float64)
        G = torch.einsum("sadm,ksm->ksad", F, bundle.seeds)
        v_hat = control_variate_variance(G, r0=r0, M=M)
        assert torch.allclose(v_hat, v_exact, rtol=1e-9, atol=1e-12), f"r0={r0}"


# ----------------------------------------------------------------- ledger
def test_ledger_records_every_access_and_requires_a_purpose(tmp_path):
    led = tmp_path / "ledger.jsonl"
    man = sorted((ROOT / "manifests/splits").glob("*__contiguous_block.json"))
    if not man:
        pytest.skip("no split manifests")
    system = json.loads(man[0].read_text())["system"]

    with pytest.raises(ValueError):
        with open_test_indices(system, experiment_id="x", purpose="", ledger=led):
            pass

    with open_test_indices(system, experiment_id="demo", purpose="unit test",
                           ledger=led) as idx:
        assert len(idx) > 0
    rows = [json.loads(l) for l in led.open()]
    assert len(rows) == 1 and rows[0]["experiment_id"] == "demo"
    assert rows[0]["split_sha256"] and rows[0]["purpose"] == "unit test"


def test_ledger_audit_flags_a_split_changing_under_one_experiment(tmp_path):
    """The condition that actually matters: one experiment, two different splits."""
    led = tmp_path / "ledger.jsonl"
    with led.open("w") as fh:
        for h in ("aaaa", "aaaa"):
            fh.write(json.dumps({"utc": "t", "experiment_id": "e", "system": "s",
                                 "scheme": "c", "purpose": "p",
                                 "split_sha256": h, "n_test": 1}) + "\n")
    assert audit_ledger(led)["ok"], "a repeated identical access is not a failure"
    with led.open("a") as fh:
        fh.write(json.dumps({"utc": "t", "experiment_id": "e", "system": "s",
                             "scheme": "c", "purpose": "p",
                             "split_sha256": "bbbb", "n_test": 1}) + "\n")
    out = audit_ledger(led)
    assert not out["ok"] and out["conflicts"][0]["n_distinct_splits"] == 2
