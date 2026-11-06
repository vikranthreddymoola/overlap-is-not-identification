"""Evaluates the overlap-inflation check (Sec. 8).

For each (coarse, fine) adjustment pair, fitted on the same rows and folds:
    absorbed = (omega_c - omega_f) / omega_c
    z        = shift / sqrt(se_c^2 + se_f^2)   (conservative)
    z_paired = shift / SE of the difference (from the influence values)
Pre-specified rule (primary): absorbed > 5% and |z| > 2. Paired test
(higher-power variant): |z_paired| > 2. The het_* cases have heterogeneous
effects and no confounding.

    python experiments/diagnostic.py
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from overlap_diag import diagnostics, dml  # noqa: E402

import dgp  # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(BASE, "results")
REPS = 200
JOBS = int(os.environ.get("N_JOBS", "-2"))  # joblib: -2 = all cores but one


def fourier(t: np.ndarray, k: int) -> np.ndarray:
    return np.column_stack(
        [f(2 * np.pi * j * t) for j in range(1, k + 1) for f in (np.sin, np.cos)]
    )


def case_data(case: str, sigma_v: float, seed: int):
    """Returns y, d, X_coarse, X_fine, learner_coarse, learner_fine, confounded."""
    if case in ("month_to_exact", "exact_to_finer", "drop_price_only"):
        y, d, X = dgp.seasonal(8000, sigma_v, seed=seed)
        t = X[:, 0]
        if case == "month_to_exact":
            Xm = dgp.seasonal(8000, sigma_v, seed=seed, calendar="month")[2]
            return y, d, Xm, X, "spline", "spline", True
        if case == "exact_to_finer":
            return y, d, X, np.column_stack([X, fourier(t, 3)]), "spline", "spline", False
        # X1 only moves price, so leaving it out isn't confounding
        return y, d, np.delete(X, 3, axis=1), X, "spline", "spline", False
    if case == "learner_interaction":
        y, d, X, _ = dgp.reactive(8000, 0.0, sigma_v, seed=seed, interaction=True)
        return y, d, X, X, "spline", "spline_interaction", True
    kappa = 1.0 if case == "forecast_hidden_info" else 0.0
    y, d, X, F = dgp.reactive(8000, kappa, sigma_v, seed=seed)
    return y, d, X, np.column_stack([X, F]), "spline", "spline", kappa > 0


HET_SIGMA = dgp.sigma_v_for_overlap(0.1)


def het_data(seed: int, delta: float):
    """Seasonal design with theta_i = 0.5 + delta * (1{X2 > 0} - 0.5) and a
    price-only shifter Z whose effect on price is 0.8 * 1{X2 > 0}. No confounding.
    Returns y, d, X (calendar and X1..X6), Z."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(size=8000)
    Xo = rng.normal(size=(8000, 6))
    z = rng.normal(size=8000)
    s = dgp.season(t)
    hi = (Xo[:, 1] > 0).astype(float)
    d = s + 0.3 * Xo[:, 0] + HET_SIGMA * rng.normal(size=8000) + 0.8 * hi * z
    theta_i = dgp.THETA + delta * (hi - 0.5)
    g = np.sin(2 * Xo[:, 1]) + 0.5 * Xo[:, 2] ** 2
    y = theta_i * d + g + dgp.GAMMA * s + rng.normal(size=8000)
    cal = np.column_stack([t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    return y, d, np.column_stack([cal, Xo]), z


CASES = (
    "month_to_exact",
    "learner_interaction",
    "forecast_hidden_info",
    "exact_to_finer",
    "drop_price_only",
    "forecast_no_hidden_info",
    "het_price_only",
    "het_exact_to_finer",
)
# cases on the seasonal design, where the level is the target overlap rather than sigma_v
SEASONAL_CASES = {"month_to_exact", "exact_to_finer", "drop_price_only"}


def task(case: str, level: float, rep: int) -> dict:
    if case.startswith("het_"):
        y, d, X, z = het_data(60_000 + rep, level)
        if case == "het_price_only":
            Xc, Xf = X, np.column_stack([X, z])
        else:
            Xc = np.column_stack([X, z])
            Xf = np.column_stack([Xc, fourier(X[:, 0], 3)])
        lc = lf = "spline"
        confounded = False
    else:
        sigma_v = dgp.sigma_v_for_overlap(level) if case in SEASONAL_CASES else level
        y, d, Xc, Xf, lc, lf, confounded = case_data(case, sigma_v, seed=60_000 + rep)
    rc = dml.plr(y, d, Xc, seed=rep, learner=lc)
    rf = dml.plr(y, d, Xf, seed=rep, learner=lf)
    return pair_row(case, level, rep, rc, rf, confounded)


def pair_row(case, level, rep, rc, rf, confounded) -> dict:
    c = diagnostics.overlap_inflation(rc, rf)
    return dict(
        case=case,
        level=level,
        rep=rep,
        theta_coarse=rc.theta,
        theta_fine=rf.theta,
        omega_coarse=rc.overlap,
        omega_fine=rf.overlap,
        absorbed=c.absorbed,
        z=c.z,
        z_paired=c.z_paired,
        confounded=confounded,
        flag_prespec=c.flag,
        flag=c.flag_paired,
    )


def rideshare_pair(draw: int) -> dict:
    import application as A

    g = A.load()
    y = A.outcome(g, A.B_MAIN, seed=70_000 + draw)
    d = g.pay_per_mile.to_numpy()
    rc = dml.plr(y, d, A.design(g, "zone + day part + demand"), seed=draw, learner="boosting_cat")
    rf = dml.plr(
        y, d, A.design(g, "zone + hour of week + demand"), seed=draw, learner="boosting_cat"
    )
    return pair_row("rideshare_daypart_to_hour", 0, draw, rc, rf, True)


def delivery_pair(kappa: float, draw: int) -> dict:
    import delivery as D

    df = D.panel(seed=95_000 + draw, kappa=kappa)
    y, d = df.filled.to_numpy(), df.pay.to_numpy()
    cats, nums = D.ADJUSTMENTS["+ logged forecast inputs"]
    rc = dml.plr(y, d, df[cats + nums].to_numpy(dtype=float), seed=draw, learner="boosting_cat3")
    cats, nums = D.ADJUSTMENTS["+ logged forecast"]
    rf = dml.plr(y, d, df[cats + nums].to_numpy(dtype=float), seed=draw, learner="boosting_cat3")
    return pair_row(f"delivery_inputs_to_forecast_k{int(kappa)}", 0, draw, rc, rf, kappa > 0)


def main() -> None:
    levels = {
        "month_to_exact": (0.02, 0.1, 0.4),
        "exact_to_finer": (0.02, 0.1, 0.4),
        "drop_price_only": (0.02, 0.1, 0.4),
        "learner_interaction": (0.5, 1.0, 2.0),
        "forecast_hidden_info": (0.5, 1.0, 2.0),
        "forecast_no_hidden_info": (0.5, 1.0, 2.0),
        "het_price_only": (0.0, 0.1, 0.3),
        "het_exact_to_finer": (0.0, 0.1, 0.3),
    }
    jobs = [delayed(task)(c, lv, r) for c in CASES for lv in levels[c] for r in range(REPS)]
    jobs += [delayed(rideshare_pair)(k) for k in range(100)]
    jobs += [delayed(delivery_pair)(kp, k) for kp in (0.0, 1.0) for k in range(100)]
    rows = Parallel(n_jobs=JOBS)(jobs)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "diagnostic.csv"), index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
