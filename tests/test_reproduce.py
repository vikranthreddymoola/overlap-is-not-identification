"""Re-runs single replications and compares them with the stored results.

Exact agreement needs the versions in requirements.txt; other versions of
scikit-learn or numpy can change the last digits.
"""

import os

import pandas as pd
import pytest

import diagnostic
import run_sims

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
COLS = ["theta", "se", "overlap"]


def stored(name, **keys):
    df = pd.read_csv(os.path.join(RES, f"{name}.csv"))
    for k, v in keys.items():
        df = df[df[k] == v]
    return df.reset_index(drop=True)


def test_study1_replication():
    new = pd.DataFrame(run_sims.s1_task(1.0, 0.5, 0))
    old = stored("s1", kappa=1.0, sigma_v=0.5, rep=0)
    assert list(new.controls) == list(old.controls)
    pd.testing.assert_frame_equal(new[COLS], old[COLS], rtol=1e-8)


def test_study4_replication():
    new = run_sims.s4_task(0.5, "spline", 0)
    old = stored("s4", sigma_v=0.5, learner="spline", rep=0).iloc[0]
    for c in COLS:
        assert new[c] == pytest.approx(old[c], rel=1e-8)


def test_diagnostic_replication():
    new = diagnostic.task("forecast_hidden_info", 1.0, 0)
    old = stored("diagnostic", case="forecast_hidden_info", level=1.0, rep=0).iloc[0]
    for c in ("theta_coarse", "theta_fine", "absorbed", "z", "z_paired"):
        assert new[c] == pytest.approx(old[c], rel=1e-8)
    assert new["flag_prespec"] == old["flag_prespec"]
