"""There must be exactly ONE lane-cost model (revision F2b / verification item 4).

The manuscript once carried four mutually inconsistent fits of T(L) = a + bL --
in the prose, in a table, in the screening library and in the figure code -- so a
reader who refitted the printed table got a slope 6% off the text. This test makes
the library's cached constants and the least-squares fit the paper prints agree,
so they cannot drift apart again silently.
"""

from __future__ import annotations

import pytest

from analysis.tables import DEFAULT_COST, fit_cost_model
from forcesketch.screening.fallback_gate import LaneCostModel


def test_library_defaults_match_the_fitted_model():
    a, b = fit_cost_model()
    if (a, b) == DEFAULT_COST:
        pytest.skip("lane_scaling records absent; fit fell back to the cached value")
    m = LaneCostModel()
    assert m.intercept_ms == pytest.approx(a, abs=0.01)
    assert m.slope_ms_per_lane == pytest.approx(b, abs=0.01)


def test_figures_use_the_same_model():
    from analysis.figures import INTERCEPT, SLOPE

    m = LaneCostModel()
    assert (INTERCEPT, SLOPE) == (m.intercept_ms, m.slope_ms_per_lane)
