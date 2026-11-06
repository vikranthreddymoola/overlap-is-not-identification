"""Re-runs key designs through DoubleML (0.10.1) as a check on overlap_diag.dml.

Same data, learners and seeds. Seeding numpy's global RNG gives DoubleML the
same folds as ours, so the estimates should match exactly (they do).

    python experiments/doubleml_check.py      (needs doubleml, see README)
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import doubleml  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from overlap_diag import dml  # noqa: E402

import dgp  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
REPS = 200
JOBS = int(os.environ.get("N_JOBS", "-2"))  # joblib: -2 = all cores but one


def doubleml_plr(y, d, X, learner: str, seed: int) -> tuple[float, float]:
    np.random.seed(seed)  # DoubleML takes its folds from numpy's global RNG
    data = doubleml.DoubleMLData.from_arrays(X, y, d)
    model = doubleml.DoubleMLPLR(
        data,
        ml_l=dml.LEARNERS[learner](seed),
        ml_m=dml.LEARNERS[learner](seed),
        n_folds=5,
        score="partialling out",
    )
    model.fit()
    return float(model.coef[0]), float(model.se[0])


def sim_task(study: str, cell: str, learner: str, rep: int) -> dict:
    if study == "study1":
        kappa, controls = {
            "kappa1_unlogged": (1.0, "X"),
            "kappa1_logged": (1.0, "X+F"),
            "kappa0": (0.0, "X"),
        }[cell]
        y, d, X, F = dgp.reactive(8000, kappa, 0.5, seed=10_000 + rep)
        if controls == "X+F":
            X = np.column_stack([X, F])
        truth_bias = dgp.reactive_bias(kappa, 0.5) if controls == "X" else 0.0
    else:
        y, d, X, _ = dgp.reactive(8000, 0.0, 0.5, seed=40_000 + rep, interaction=True)
        truth_bias = 0.0
    ours = dml.plr(y, d, X, seed=rep, learner=learner)
    theta, se = doubleml_plr(y, d, X, learner, seed=rep)
    return dict(
        study=study,
        cell=cell,
        learner=learner,
        rep=rep,
        population_bias=truth_bias,
        ours_theta=ours.theta,
        ours_se=ours.se,
        doubleml_theta=theta,
        doubleml_se=se,
    )


def app_task(name: str, draw: int) -> dict:
    import application as A

    g = A.load()
    y = A.outcome(g, A.B_MAIN, seed=70_000 + draw)
    X = A.design(g, name)
    d = g.pay_per_mile.to_numpy()
    ours = dml.plr(y, d, X, seed=draw, learner="boosting_cat")
    theta, se = doubleml_plr(y, d, X, "boosting_cat", seed=draw)
    return dict(
        study="nyc",
        cell=name,
        learner="boosting_cat",
        rep=draw,
        population_bias=np.nan,
        ours_theta=ours.theta,
        ours_se=ours.se,
        doubleml_theta=theta,
        doubleml_se=se,
    )


def main() -> None:
    jobs = [
        ("study1", c, "spline", r)
        for c in ("kappa1_unlogged", "kappa1_logged", "kappa0")
        for r in range(REPS)
    ]
    jobs += [
        ("study4", "interaction", lrn, r)
        for lrn in ("spline", "spline_interaction", "boosting")
        for r in range(REPS)
    ]
    rows = Parallel(n_jobs=JOBS)(delayed(sim_task)(*j) for j in jobs)
    import application as A

    rows += Parallel(n_jobs=JOBS)(delayed(app_task)(n, k) for n in A.ADJUSTMENTS for k in range(50))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "doubleml_check.csv"), index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
