import numpy as np
import pytest

from overlap_diag import LEARNERS, ols_linear, plr

import dgp


def test_exogenous_price_recovers_theta():
    y, d, X, _ = dgp.reactive(4000, kappa=0.0, sigma_v=1.0, seed=1)
    r = plr(y, d, X, seed=1)
    assert abs(r.theta - dgp.THETA) < 4 * r.se
    lo, hi = r.ci
    assert lo < r.theta < hi
    assert 0 < r.overlap < 1


def test_reactive_bias_matches_closed_form_and_logging_removes_it():
    y, d, X, F = dgp.reactive(8000, kappa=1.0, sigma_v=0.5, seed=2)
    unlogged = plr(y, d, X, seed=2)
    logged = plr(y, d, np.column_stack([X, F]), seed=2)
    expected = dgp.THETA + dgp.reactive_bias(1.0, 0.5)
    assert abs(unlogged.theta - expected) < 0.1
    assert abs(logged.theta - dgp.THETA) < 4 * logged.se
    # high overlap does not protect the unlogged fit
    assert unlogged.overlap > logged.overlap


def test_repeated_splits_report_the_median():
    y, d, X, _ = dgp.reactive(2000, kappa=0.0, sigma_v=1.0, seed=3)
    r = plr(y, d, X, n_rep=3, seed=3)
    assert len(r.rep_thetas) == 3
    assert r.theta == pytest.approx(np.median(r.rep_thetas))
    assert np.isfinite(r.split_dispersion)


def test_learner_names_and_knots():
    y, d, X, _ = dgp.reactive(1000, kappa=0.0, sigma_v=1.0, seed=4)
    assert {"spline", "boosting", "boosting_cat", "random_forest"} <= set(LEARNERS)
    r8 = plr(y, d, X, learner="spline:8")
    r12 = plr(y, d, X, learner="spline")
    assert r8.theta != r12.theta
    with pytest.raises(KeyError):
        plr(y, d, X, learner="no_such_learner")


def test_ols_linear_on_a_linear_model():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(500, 3))
    d = X @ [0.5, -0.2, 0.1] + rng.normal(size=500)
    y = 2.0 * d + X @ [1.0, 0.0, -1.0] + 0.01 * rng.normal(size=500)
    assert ols_linear(y, d, X) == pytest.approx(2.0, abs=1e-2)
