"""Monte Carlo studies (Sec. 5). Output goes to results/*.csv.

python experiments/run_sims.py               # Studies 1-4
python experiments/run_sims.py --only s4checks,s4tuned
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from overlap_diag import dml  # noqa: E402

import dgp  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
JOBS = int(os.environ.get("N_JOBS", "-2"))  # joblib: -2 = all cores but one
REPS = 500


def covers(r: dml.PLRResult, theta: float = dgp.THETA) -> bool:
    lo, hi = r.ci
    return bool(lo <= theta <= hi)


# Study 1: reactive pricing
def s1_task(kappa, sigma_v, rep, n=8000):
    y, d, X, F = dgp.reactive(n, kappa, sigma_v, seed=10_000 + rep)
    out = []
    for adj, Xc in (("X", X), ("X+F", np.column_stack([X, F]))):
        r = dml.plr(y, d, Xc, seed=rep)
        out.append(
            dict(
                kappa=kappa,
                sigma_v=sigma_v,
                rep=rep,
                controls=adj,
                theta=r.theta,
                se=r.se,
                covers=covers(r),
                overlap=r.overlap,
                ols=dml.ols_linear(y, d, Xc),
            )
        )
    return out


def study1():
    grid = [
        (k, s, rep)
        for k in (0.0, 0.25, 0.5, 1.0)
        for s in (0.25, 0.5, 1.0, 2.0)
        for rep in range(REPS)
    ]
    rows = Parallel(n_jobs=JOBS)(delayed(s1_task)(*g) for g in grid)
    return pd.DataFrame([r for rr in rows for r in rr])


# Study 2: coarse vs exact calendar
def s2_task(target, calendar, learner, rep, n=8000):
    sv = dgp.sigma_v_for_overlap(target)
    y, d, X = dgp.seasonal(n, sv, seed=20_000 + rep, calendar=calendar)
    r = dml.plr(y, d, X, n_rep=5, seed=rep, learner=learner)
    return dict(
        target_overlap=target,
        sigma_v=sv,
        calendar=calendar,
        learner=learner,
        rep=rep,
        theta=r.theta,
        se=r.se,
        covers=covers(r),
        overlap=r.overlap,
        split_dispersion=r.split_dispersion,
    )


def study2():
    grid = [
        (o, cal, lrn, rep)
        for o in (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
        for cal in ("exact", "month")
        for lrn in ("spline", "boosting")
        for rep in range(REPS)
    ]
    return pd.DataFrame(Parallel(n_jobs=JOBS)(delayed(s2_task)(*g) for g in grid))


# Study 3: overlap and the usual checks across scenarios
SCENARIOS = (
    "identified",
    "weak_overlap",
    "coarse_calendar",
    "reactive_unlogged",
    "reactive_forecast_logged",
)


def scenario_data(scenario, n, seed):
    """Data and population bias for one Study 3 scenario."""
    if scenario in ("identified", "weak_overlap", "coarse_calendar"):
        target = 0.4 if scenario == "identified" else 0.02
        sv = dgp.sigma_v_for_overlap(target)
        cal = "month" if scenario == "coarse_calendar" else "exact"
        y, d, X = dgp.seasonal(n, sv, seed=seed, calendar=cal)
        bias = dgp.coarse_calendar_bias(sv) if cal == "month" else 0.0
        return y, d, X, bias
    y, d, X, F = dgp.reactive(n, 1.0, 0.5, seed=seed)
    if scenario == "reactive_forecast_logged":
        return y, d, np.column_stack([X, F]), 0.0
    return y, d, X, dgp.reactive_bias(1.0, 0.5)


def s3_task(scenario, rep, n=8000):
    seed = 30_000 + rep
    y, d, X, bias = scenario_data(scenario, n, seed)
    r = dml.plr(y, d, X, n_rep=5, seed=rep)
    rb = dml.plr(y, d, X, seed=rep, learner="boosting")
    d_perm = np.random.default_rng(seed).permutation(d)
    return dict(
        scenario=scenario,
        rep=rep,
        population_bias=bias,
        theta=r.theta,
        se=r.se,
        covers=covers(r),
        pval=r.pval,
        overlap=r.overlap,
        split_dispersion=r.split_dispersion,
        learner_gap_in_se=(rb.theta - r.theta) / r.se,
        placebo_p=dml.plr(y, d_perm, X, seed=rep).pval,
    )


def study3():
    grid = [(s, rep) for s in SCENARIOS for rep in range(REPS)]
    return pd.DataFrame(Parallel(n_jobs=JOBS)(delayed(s3_task)(*g) for g in grid))


# Study 4: harder h(X), no confounding
def s4_task(sigma_v, learner, rep, n=8000):
    y, d, X, _ = dgp.reactive(n, 0.0, sigma_v, seed=40_000 + rep, interaction=True)
    r = dml.plr(y, d, X, seed=rep, learner=learner)
    return dict(
        sigma_v=sigma_v,
        learner=learner,
        rep=rep,
        theta=r.theta,
        se=r.se,
        covers=covers(r),
        overlap=r.overlap,
    )


def study4():
    grid = [
        (s, lrn, rep)
        for s in (0.5, 1.0, 2.0)
        for lrn in ("spline", "boosting", "boosting_tuned")
        for rep in range(REPS)
    ]
    return pd.DataFrame(Parallel(n_jobs=JOBS)(delayed(s4_task)(*g) for g in grid))


def s4_task_n(sigma_v, learner, n, rep):
    return dict(s4_task(sigma_v, learner, rep, n=n), n=n)


def study4_checks():
    """Study 4 checks: learners that can / can't fit X3*X4, and boosting at larger n."""
    grid = [(s, "spline_interaction", 8000, rep) for s in (0.5, 1.0, 2.0) for rep in range(REPS)]
    grid += [(0.5, "random_forest", 8000, rep) for rep in range(200)]
    grid += [(0.5, "boosting", n, rep) for n in (32_000, 128_000) for rep in range(200)]
    return pd.DataFrame(Parallel(n_jobs=JOBS)(delayed(s4_task_n)(*g) for g in grid))


def study4_tuned_n():
    """Tuned boosting at n = 32k and 128k."""
    grid = [(0.5, "boosting_tuned", 32_000, rep) for rep in range(100)]
    grid += [(0.5, "boosting_tuned", 128_000, rep) for rep in range(60)]
    return pd.DataFrame(Parallel(n_jobs=JOBS)(delayed(s4_task_n)(*g) for g in grid))


STUDIES = {
    "s1": study1,
    "s2": study2,
    "s3": study3,
    "s4": study4,
    "s4checks": study4_checks,
    "s4tuned": study4_tuned_n,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name, fn in STUDIES.items():
        if (args.only and name not in args.only.split(",")) or (
            not args.only and name in ("s4checks", "s4tuned")
        ):
            continue
        t0 = time.time()
        print(f"== {name} started", flush=True)
        df = fn()
        df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(f"== {name} done: {len(df)} rows in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
