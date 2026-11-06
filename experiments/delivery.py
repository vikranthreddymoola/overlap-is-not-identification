"""Delivery-style reactive-pricing study, calibrated to NYC DCWP delivery-app data.

Zones x hours over 12 weeks. Pay reacts to a shortfall forecast built from
logged inputs (zone, hour of week, demand at the same hour last week, last
week's utilization) and, when kappa = 1, also from real-time supply the analyst
never sees. From DCWP: the weekly utilization series and the pay level.
Assumed: demand profile, pay dispersion, outcome.

    python experiments/delivery.py
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

from overlap_diag import dml  # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DCWP = os.path.join(BASE, "data", "dcwp")
OUT = os.path.join(BASE, "results")

THETA = 0.5  # pp of orders filled per $1/hr
PAY_LEVEL = 21.63  # DCWP: average pay per hour, restaurant apps, Q2 2026
ZONES, WEEKS = 60, 13  # the first week only provides lags
RHO = 0.6  # week-to-week persistence of demand shocks
ALPHA, SIGMA_V = 2.0, 1.0  # $/hr: reaction to the forecast, exogenous noise
BETA, SIGMA_EPS = 2.0, 2.0  # outcome: loss per unit of shortfall, noise sd
DRAWS = 100
JOBS = int(os.environ.get("N_JOBS", "-2"))  # joblib: -2 = all cores but one


def weekly_utilization() -> np.ndarray:
    """Weekly utilization (%), from prepare_data.py --dcwp."""
    return pd.read_csv(os.path.join(DCWP, "weekly_utilization.csv")).utilization_pct.to_numpy()


def demand_profile() -> np.ndarray:
    """Assumed hour-of-week log-demand shape: lunch and dinner peaks, busier Fri-Sun evenings."""
    h = np.arange(168) % 24
    day = np.arange(168) // 24
    lunch = np.exp(-0.5 * ((h - 12.5) / 1.5) ** 2)
    dinner = np.exp(-0.5 * ((h - 19.0) / 1.8) ** 2) * np.where(day >= 4, 1.3, 1.0)
    night = np.where((h < 7), -1.5, 0.0)
    return 0.8 * lunch + 1.1 * dinner + night


def panel(seed: int, kappa: float) -> pd.DataFrame:
    """One simulated panel; week 0 is dropped (it only supplies lags)."""
    rng = np.random.default_rng(seed)
    util = weekly_utilization()[:WEEKS]
    tight = (util - util.mean()) / util.std()  # real week-to-week supply tightness
    prof = demand_profile()
    n_hours = WEEKS * 168
    zone_scale = rng.normal(0.0, 0.5, ZONES)
    zone_supply = rng.normal(0.0, 0.3, ZONES)
    # AR(1) across weeks at the same zone and hour
    eta = np.zeros((ZONES, n_hours))
    eta[:, :168] = rng.normal(0.0, 0.5, (ZONES, 168))
    for w in range(1, WEEKS):
        prev = eta[:, (w - 1) * 168 : w * 168]
        eta[:, w * 168 : (w + 1) * 168] = RHO * prev + rng.normal(0.0, 0.4, (ZONES, 168))
    z = np.repeat(np.arange(ZONES), n_hours)
    t = np.tile(np.arange(n_hours), ZONES)
    week, how = t // 168, t % 168
    e = eta.ravel()
    e_lag = np.where(week > 0, np.roll(eta, 168, axis=1).ravel(), np.nan)
    u = rng.normal(0.0, 1.0, z.size)  # real-time supply shock: platform sees it, analyst doesn't
    structural = 0.6 * prof[how] - zone_supply[z] + 0.5 * zone_scale[z]
    shortfall = structural + e + 0.5 * tight[week] + u
    prev_tight = np.where(week > 0, tight[np.maximum(week - 1, 0)], np.nan)
    forecast = structural + RHO * e_lag + 0.5 * prev_tight + kappa * u
    fz = (forecast - np.nanmean(forecast)) / np.nanstd(forecast)
    pay = PAY_LEVEL + ALPHA * fz + SIGMA_V * rng.normal(size=z.size)
    log_orders = zone_scale[z] + prof[how] + e
    log_orders_lag = zone_scale[z] + prof[how] + e_lag
    filled = (
        85.0 + THETA * (pay - PAY_LEVEL) - BETA * shortfall + SIGMA_EPS * rng.normal(size=z.size)
    )
    df = pd.DataFrame(
        dict(
            zone=z,
            how=how,
            week=week,
            pay=pay,
            filled=filled,
            log_orders_lag=log_orders_lag,
            prev_tight=prev_tight,
            forecast=forecast,
            log_orders=log_orders,
        )
    )
    return df[df.week > 0].reset_index(drop=True)


ADJUSTMENTS = {
    "zone + hour of week + week": (["zone", "how", "week"], []),
    "+ logged forecast inputs": (["zone", "how", "week"], ["log_orders_lag"]),
    "+ logged forecast": (["zone", "how", "week"], ["log_orders_lag", "forecast"]),
}


def task(kappa: float, name: str, draw: int) -> dict:
    df = panel(seed=95_000 + draw, kappa=kappa)
    cats, nums = ADJUSTMENTS[name]
    X = df[cats + nums].to_numpy(dtype=float)
    r = dml.plr(df.filled.to_numpy(), df.pay.to_numpy(), X, seed=draw, learner="boosting_cat3")
    # closed-form bias (Prop. 1); pay loads on F with slope ALPHA / sd(F)
    a = ALPHA / df.forecast.std()
    var_e = 0.25  # stationary var of the demand shock
    if name == "zone + hour of week + week":
        cov_dw = a * (RHO**2 * var_e + kappa)  # Cov(D~, e + U)
        var_d = a**2 * (RHO**2 * var_e + kappa**2) + SIGMA_V**2
    elif name == "+ logged forecast inputs":
        cov_dw = a * kappa
        var_d = a**2 * kappa**2 + SIGMA_V**2
    else:
        cov_dw, var_d = 0.0, SIGMA_V**2
    predicted = -BETA * cov_dw / var_d
    return dict(
        kappa=kappa,
        adjustment=name,
        draw=draw,
        n=len(df),
        theta=r.theta,
        se=r.se,
        covers=bool(abs(r.theta - THETA) <= 1.96 * r.se),
        overlap=r.overlap,
        predicted_bias=predicted,
        exogenous_share=SIGMA_V**2 / float(df.pay.var()),
        pay_sd=float(df.pay.std()),
        filled_mean=float(df.filled.mean()),
    )


def main() -> None:
    jobs = [(k, n, d) for k in (0.0, 1.0) for n in ADJUSTMENTS for d in range(DRAWS)]
    rows = Parallel(n_jobs=JOBS)(delayed(task)(*j) for j in jobs)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "delivery.csv"), index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
