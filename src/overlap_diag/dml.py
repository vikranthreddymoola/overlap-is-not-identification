"""Partially linear DML (Robinson 1988; Chernozhukov et al. 2018).

Y = theta * D + g(X) + eps,  D = m(X) + V.

Cross-fitted residuals of Y and D on X; theta is the slope of one on the other.
With n_rep > 1 the folds are redrawn each time and the median is reported.
Learners are named in LEARNERS; "spline:8" means the spline learner with 8 knots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer

Z95 = stats.norm.ppf(0.975)


def spline_learner(seed: int = 0, n_knots: int = 12):
    """Cubic B-splines per column + RidgeCV (additive)."""
    return make_pipeline(
        SplineTransformer(n_knots=n_knots, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13))
    )


def boosting_learner(seed: int = 0):
    """scikit-learn defaults."""
    return HistGradientBoostingRegressor(random_state=seed)


def boosting_categorical_learner(seed: int = 0, n_categorical: int = 2):
    """First n_categorical columns treated as categorical (zone / hour codes)."""
    return HistGradientBoostingRegressor(
        categorical_features=list(range(n_categorical)), max_iter=300, random_state=seed
    )


def boosting_tuned_learner(seed: int = 0):
    """lr 0.05, up to 2000 trees, 63 leaves, early stopping."""
    return HistGradientBoostingRegressor(
        max_iter=2000,
        learning_rate=0.05,
        max_leaf_nodes=63,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=seed,
    )


def boosting_categorical_tuned_learner(seed: int = 0, n_categorical: int = 2):
    """Tuned settings with the first n_categorical columns as categorical."""
    return HistGradientBoostingRegressor(
        categorical_features=list(range(n_categorical)),
        max_iter=2000,
        learning_rate=0.05,
        max_leaf_nodes=63,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=seed,
    )


def _x3x4(X):
    return (X[:, 2] * X[:, 3]).reshape(-1, 1)


def spline_interaction_learner(seed: int = 0):
    """Spline learner plus an X3*X4 column, so it can fit Study 4's h(X)."""
    features = FeatureUnion(
        [
            ("splines", SplineTransformer(n_knots=12, degree=3)),
            ("x3x4", FunctionTransformer(_x3x4)),
        ]
    )
    return make_pipeline(features, RidgeCV(alphas=np.logspace(-3, 3, 13)))


def random_forest_learner(seed: int = 0):
    """300 trees, min leaf 5, half the columns per split."""
    return RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=1, random_state=seed
    )


def boosting_three_categorical_learner(seed: int = 0):
    """zone, hour of week, week as categoricals."""
    return boosting_categorical_learner(seed, n_categorical=3)


LEARNERS = {
    "spline": spline_learner,
    "spline_interaction": spline_interaction_learner,
    "boosting": boosting_learner,
    "boosting_tuned": boosting_tuned_learner,
    "boosting_cat": boosting_categorical_learner,
    "boosting_cat_tuned": boosting_categorical_tuned_learner,
    "boosting_cat3": boosting_three_categorical_learner,
    "random_forest": random_forest_learner,
}


@dataclass
class PLRResult:
    theta: float
    se: float
    overlap: float  # Var(D - m_hat(X)) / Var(D), out of fold
    rep_thetas: np.ndarray
    rep_ses: np.ndarray
    influence: np.ndarray | None = None  # psi_i / J from the first split

    @property
    def ci(self) -> tuple[float, float]:
        return self.theta - Z95 * self.se, self.theta + Z95 * self.se

    @property
    def pval(self) -> float:
        return float(2 * stats.norm.sf(abs(self.theta / self.se)))

    @property
    def split_dispersion(self) -> float:
        """sd of the estimate across splits / median SE."""
        if len(self.rep_thetas) < 2:
            return float("nan")
        return float(np.std(self.rep_thetas, ddof=1) / np.median(self.rep_ses))


def crossfit(learner, X: np.ndarray, t: np.ndarray, folds) -> np.ndarray:
    """Out-of-fold predictions of t from X."""
    out = np.empty(len(t))
    for tr, te in folds:
        out[te] = clone(learner).fit(X[tr], t[tr]).predict(X[te])
    return out


def plr(
    y: np.ndarray,
    d: np.ndarray,
    X: np.ndarray,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    learner: str = "spline",
) -> PLRResult:
    """Cross-fitted PLR estimate of theta with a heteroskedasticity-robust SE.

    y, d: outcome and treatment (price), shape (n,). X: controls, shape (n, p).
    learner: a key of LEARNERS, optionally with ":knots" for the spline learner.
    The same learner is used for E[Y|X] and E[D|X]. For n_rep > 1 the estimate
    is the median over splits and the variance is median(var_r + (theta_r - theta)^2).
    """
    thetas, ses, overlaps, infl = [], [], [], None
    for r in range(n_rep):
        name, _, knots = learner.partition(":")
        lrn = LEARNERS[name](seed + r, int(knots)) if knots else LEARNERS[name](seed + r)
        folds = list(KFold(n_folds, shuffle=True, random_state=seed + r).split(X))
        y_res = y - crossfit(lrn, X, y, folds)
        d_res = d - crossfit(lrn, X, d, folds)
        j = np.mean(d_res**2)
        th = np.mean(d_res * y_res) / j
        psi = d_res * (y_res - th * d_res)
        thetas.append(th)
        ses.append(np.sqrt(np.mean(psi**2) / j**2 / len(y)))
        overlaps.append(np.var(d_res) / np.var(d))
        if r == 0:
            infl = psi / j
    thetas, ses = np.array(thetas), np.array(ses)
    theta = float(np.median(thetas))
    se = float(np.sqrt(np.median(ses**2 + (thetas - theta) ** 2)))
    return PLRResult(theta, se, float(np.median(overlaps)), thetas, ses, infl)


def ols_linear(y: np.ndarray, d: np.ndarray, X: np.ndarray) -> float:
    """OLS of y on (1, d, X); returns the d coefficient."""
    A = np.column_stack([np.ones(len(y)), d, X])
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])
