"""Semi-synthetic study on the NYC FHV trip records (Mar 2023).

Real pay, demand, zone and calendar at the zone-hour level; simulated outcome,
so the truth is known.

    python experiments/application.py [--tuning | --subsample]
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
from sklearn.model_selection import KFold  # noqa: E402

from overlap_diag import dml  # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(BASE, "data", "zone_hour_2023-03.parquet")
OUT = os.path.join(BASE, "results")
THETA, GAMMA, SIGMA_EPS = 0.5, 1.0, 0.5
B_GRID = (-2.0, -1.0, -0.5, 0.5, 1.0)
B_MAIN = -1.0
DRAWS = 100
JOBS = int(os.environ.get("N_JOBS", "-2"))  # joblib: -2 = all cores but one


def demean(v: pd.Series, keys: list) -> pd.Series:
    """Fixed-effect residual by alternating projections."""
    r = v.astype(float).copy()
    for _ in range(10):
        for k in keys:
            r = r - r.groupby(k).transform("mean")
    return r


def load() -> pd.DataFrame:
    g = pd.read_parquet(DATA)
    g = g[g.trips >= 10].copy()
    lo, hi = g.pay_per_mile.quantile([0.01, 0.99])
    g = g[g.pay_per_mile.between(lo, hi)].reset_index(drop=True)
    g["zone"] = g.PULocationID.astype("category").cat.codes
    g["how"] = g.hour.dt.dayofweek * 24 + g.hour.dt.hour  # hour of week, 0-167
    g["weekend"] = (g.hour.dt.dayofweek >= 5).astype(int)
    g["daypart"] = g.hour.dt.hour // 4 + 6 * g.weekend  # six 4-hour blocks x weekend
    g["logq"] = np.log(g.trips)
    # outcome seasonality = real hour-of-week demand profile
    prof = g.groupby("how").logq.mean()
    g["season"] = ((prof - prof.mean()) / prof.std()).reindex(g.how).to_numpy()
    # demand surprise, net of zone and hour-of-week effects
    g["surprise"] = demean(g.logq, [g.zone, g.how])
    g["zone_effect"] = g.groupby("zone").logq.transform("mean")
    return g


def outcome(g: pd.DataFrame, b: float, seed: int) -> np.ndarray:
    eps = np.random.default_rng(seed).normal(scale=SIGMA_EPS, size=len(g))
    zone_part = (g.zone_effect - g.zone_effect.mean()).to_numpy()
    return (
        THETA * g.pay_per_mile.to_numpy()
        + 0.3 * zone_part
        + GAMMA * g.season.to_numpy()
        + b * g.surprise.to_numpy()
        + eps
    )


ADJUSTMENTS = {
    # name: (categorical columns, numeric columns)
    "zone + hour of week": (["zone", "how"], []),
    "zone + hour of week + demand": (["zone", "how"], ["logq"]),
    "zone + day part + demand": (["zone", "daypart"], ["logq"]),
}


def design(g: pd.DataFrame, name: str) -> np.ndarray:
    cats, nums = ADJUSTMENTS[name]
    return g[cats + nums].to_numpy(dtype=float)


def predicted_bias(g: pd.DataFrame, name: str) -> dict:
    """Plug-in Prop. 1 from the estimator's own cross-fitted residuals.
    Returns the slopes c_w, c_s; bias = b * c_w + GAMMA * c_s.
    """
    X = design(g, name)
    folds = list(KFold(5, shuffle=True, random_state=0).split(X))
    lrn = dml.LEARNERS["boosting_cat"](0)
    d = g.pay_per_mile.to_numpy()
    d_res = d - dml.crossfit(lrn, X, d, folds)
    out = {}
    for col, key in (("surprise", "c_w"), ("season", "c_s")):
        v = g[col].to_numpy()
        v_res = v - dml.crossfit(lrn, X, v, folds)
        out[key] = float(np.sum(d_res * v_res) / np.sum(d_res**2))
    out["overlap"] = float(np.var(d_res) / np.var(d))
    out["sd_dres"] = float(np.std(d_res))
    return out


def task(g: pd.DataFrame, name: str, b: float, draw: int) -> dict:
    y = outcome(g, b, seed=70_000 + draw)
    r = dml.plr(y, g.pay_per_mile.to_numpy(), design(g, name), seed=draw, learner="boosting_cat")
    return dict(
        adjustment=name,
        b=b,
        draw=draw,
        theta=r.theta,
        se=r.se,
        covers=bool(abs(r.theta - THETA) <= 1.96 * r.se),
        overlap=r.overlap,
        tipping_point=abs(r.theta) * np.sqrt(r.overlap * np.var(g.pay_per_mile)),
    )


def tuning_task(g: pd.DataFrame, learner: str, draw: int) -> dict:
    name = "zone + hour of week + demand"
    y = outcome(g, B_MAIN, seed=80_000 + draw)
    r = dml.plr(y, g.pay_per_mile.to_numpy(), design(g, name), seed=draw, learner=learner)
    return dict(
        adjustment=name,
        learner=learner,
        draw=draw,
        theta=r.theta,
        se=r.se,
        covers=bool(abs(r.theta - THETA) <= 1.96 * r.se),
        overlap=r.overlap,
    )


def tuning() -> None:
    """Coverage check for the demand-adjusted fit: default vs tuned boosting."""
    g = load()
    jobs = [(lrn, k) for lrn in ("boosting_cat", "boosting_cat_tuned") for k in range(200)]
    rows = Parallel(n_jobs=JOBS)(delayed(tuning_task)(g, *j) for j in jobs)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "application_tuning.csv"), index=False)
    print("tuning done", flush=True)


def subsample_task(g: pd.DataFrame, frac: float, draw: int) -> dict:
    name = "zone + hour of week + demand"
    sub = g.sample(frac=frac, random_state=90_000 + draw).reset_index(drop=True) if frac < 1 else g
    y = outcome(sub, B_MAIN, seed=91_000 + draw)
    r = dml.plr(
        y, sub.pay_per_mile.to_numpy(), design(sub, name), seed=draw, learner="boosting_cat"
    )
    return dict(
        frac=frac,
        n=len(sub),
        draw=draw,
        theta=r.theta,
        se=r.se,
        covers=bool(abs(r.theta - THETA) <= 1.96 * r.se),
    )


def subsample() -> None:
    """Same estimate on 25/50/100% subsamples of the cells."""
    g = load()
    jobs = [(f, k) for f in (0.25, 0.5, 1.0) for k in range(200)]
    rows = Parallel(n_jobs=JOBS)(delayed(subsample_task)(g, *j) for j in jobs)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "application_subsample.csv"), index=False)
    print("subsample done", flush=True)


def main() -> None:
    g = load()
    print(f"cells: {len(g):,}, zones: {g.zone.nunique()}, hours: {g.hour.nunique()}", flush=True)
    summary = pd.DataFrame(
        {
            "pay_sd": [g.pay_per_mile.std()],
            "pay_mean": [g.pay_per_mile.mean()],
            "surprise_sd": [g.surprise.std()],
            "slope_pay_on_surprise": [
                np.polyfit(g.surprise, demean(g.pay_per_mile, [g.zone, g.how]), 1)[0]
            ],
            "n": [len(g)],
            "zones": [g.zone.nunique()],
        }
    )
    summary.to_csv(os.path.join(OUT, "application_data_summary.csv"), index=False)
    pred = pd.DataFrame([dict(adjustment=n, **predicted_bias(g, n)) for n in ADJUSTMENTS])
    pred.to_csv(os.path.join(OUT, "application_predicted.csv"), index=False)
    print(pred.to_string(index=False), flush=True)
    jobs = [(n, B_MAIN, k) for n in ADJUSTMENTS for k in range(DRAWS)]
    jobs += [
        (n, b, k) for n in list(ADJUSTMENTS)[:1] for b in B_GRID if b != B_MAIN for k in range(20)
    ]
    rows = Parallel(n_jobs=JOBS)(delayed(task)(g, *j) for j in jobs)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "application.csv"), index=False)
    print("done", flush=True)


if __name__ == "__main__":
    import sys

    if "--tuning" in sys.argv:
        tuning()
    elif "--subsample" in sys.argv:
        subsample()
    else:
        main()
