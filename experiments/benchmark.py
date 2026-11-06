"""Benchmarks for the tipping point (Sec. 9), in the style of Cinelli & Hazlett.

For each group G of controls: k_G (outcome side), t_G (price side), the
one-sided ratio tau / k_G, the two-sided multiple c* from Prop. 1, and the
actual change in the estimate when G is dropped. The computation itself is
overlap_diag.diagnostics.benchmark.

    python experiments/benchmark.py
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from overlap_diag import diagnostics  # noqa: E402

import dgp  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
JOBS = int(os.environ.get("N_JOBS", "-2"))  # joblib: -2 = all cores but one


def nyc_task(draw: int) -> list[dict]:
    import application as A

    g = A.load()
    y = A.outcome(g, A.B_MAIN, seed=70_000 + draw)
    X = A.design(g, "zone + hour of week + demand")
    groups = {"zone": [0], "hour of week": [1], "demand": [2]}
    rows = diagnostics.benchmark(
        y, g.pay_per_mile.to_numpy(), X, groups, [True, True, False], seed=draw
    )
    return [dict(r, study="nyc", draw=draw) for r in rows]


def delivery_task(draw: int) -> list[dict]:
    import delivery as D

    df = D.panel(seed=95_000 + draw, kappa=1.0)
    cols = ["zone", "how", "week", "log_orders_lag", "forecast"]
    X = df[cols].to_numpy(dtype=float)
    groups = {"zone": [0], "hour of week": [1], "week": [2], "lagged demand": [3], "forecast": [4]}
    rows = diagnostics.benchmark(
        df.filled.to_numpy(),
        df.pay.to_numpy(),
        X,
        groups,
        [True, True, True, False, False],
        seed=draw,
    )
    return [dict(r, study="delivery", draw=draw) for r in rows]


def study3_task(rep: int) -> list[dict]:
    y, d, X = dgp.seasonal(8000, dgp.sigma_v_for_overlap(0.4), seed=30_000 + rep)
    groups = {
        "calendar": [0, 1, 2],
        "X1 (price only)": [3],
        "X2 (outcome only)": [4],
        "X3 (outcome only)": [5],
    }
    rows = diagnostics.benchmark(y, d, X, groups, None, seed=rep)
    return [dict(r, study="study3_identified", draw=rep) for r in rows]


def main() -> None:
    tasks = (
        [delayed(nyc_task)(k) for k in range(20)]
        + [delayed(delivery_task)(k) for k in range(20)]
        + [delayed(study3_task)(k) for k in range(100)]
    )
    rows = Parallel(n_jobs=JOBS)(tasks)
    pd.DataFrame([r for rr in rows for r in rr]).to_csv(
        os.path.join(OUT, "benchmark.csv"), index=False
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
