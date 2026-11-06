import numpy as np
import pytest

from overlap_diag import benchmark, overlap_inflation, paired_z, plr, tipping_point

import dgp


def fit_pair(kappa, seed=7):
    """Coarse: X. Fine: X and the forecast F. F carries hidden information only if kappa > 0."""
    y, d, X, F = dgp.reactive(8000, kappa, 1.0, seed=seed)
    coarse = plr(y, d, X, seed=seed)
    fine = plr(y, d, np.column_stack([X, F]), seed=seed)
    return coarse, fine


def test_flags_a_confounded_refinement():
    check = overlap_inflation(*fit_pair(kappa=1.0))
    assert check.absorbed > 0.05
    assert check.flag and check.flag_paired


def test_does_not_flag_a_harmless_refinement():
    check = overlap_inflation(*fit_pair(kappa=0.0))
    assert not check.flag
    assert not check.flag_paired


def test_paired_z_is_antisymmetric():
    coarse, fine = fit_pair(kappa=1.0)
    assert paired_z(coarse, fine) == pytest.approx(-paired_z(fine, coarse))


def test_tipping_point_is_theta_times_residual_sd():
    y, d, X, _ = dgp.reactive(2000, 0.0, 1.0, seed=8)
    r = plr(y, d, X, seed=8)
    assert tipping_point(r, d) == pytest.approx(abs(r.theta) * np.sqrt(r.overlap * np.var(d)))


def test_benchmark_rows():
    y, d, X = dgp.seasonal(2000, dgp.sigma_v_for_overlap(0.4), seed=9)
    rows = benchmark(y, d, X, {"calendar": [0, 1, 2], "X1": [3]}, seed=9)
    assert [r["group"] for r in rows] == ["calendar", "X1"]
    for r in rows:
        assert r["k"] >= 0 and r["t"] >= 0 and r["tau"] > 0
        assert r["ratio"] == pytest.approx(r["tau"] / r["k"])
