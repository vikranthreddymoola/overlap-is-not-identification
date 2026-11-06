"""Diagnostics from the paper: the overlap-inflation check, the tipping point
and benchmarks for it.

overlap_inflation(): compares a coarse and a fine adjustment fitted on the same
rows and folds. If the finer controls absorb much of the price variation and
the estimate moves, the coarse fit was relying on variation that was not
exogenous.

tipping_point(): tau = |theta| * sd(D - m(X)), the smallest outcome-scale
effect |b| s_W of an unadjusted component that could explain the estimate
away (Corollary 1).

benchmark(): for each group of observed controls, how large its outcome and
price parts are relative to tau, in the style of Cinelli and Hazlett (2020).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from . import dml

ABSORB_MIN = 0.05  # share of residual price variance absorbed by the finer controls
Z_MIN = 2.0


def paired_z(coarse: dml.PLRResult, fine: dml.PLRResult) -> float:
    """Shift between two single-split fits on the same rows and folds, in units
    of the SE of the difference (from the difference of the influence values)."""
    diff = coarse.influence - fine.influence
    se = np.sqrt(np.mean(diff**2) / len(diff))
    return float((coarse.theta - fine.theta) / se)


@dataclass
class OverlapCheck:
    absorbed: float  # (omega_coarse - omega_fine) / omega_coarse
    shift: float  # theta_coarse - theta_fine
    z: float  # shift / sqrt(se_c^2 + se_f^2), conservative
    z_paired: float  # shift / SE of the difference
    flag: bool  # pre-specified rule: absorbed > absorb_min and |z| > z_min
    flag_paired: bool  # higher-power variant: |z_paired| > z_min


def overlap_inflation(
    coarse: dml.PLRResult, fine: dml.PLRResult, absorb_min: float = ABSORB_MIN, z_min: float = Z_MIN
) -> OverlapCheck:
    """Overlap-inflation check for a (coarse, fine) pair of fits.

    Both fits must use the same rows, folds and seed (n_rep = 1) so that the
    paired test is valid; see the README for an example.
    """
    absorbed = (coarse.overlap - fine.overlap) / coarse.overlap
    shift = coarse.theta - fine.theta
    z = shift / np.sqrt(coarse.se**2 + fine.se**2)
    z_paired = paired_z(coarse, fine)
    return OverlapCheck(
        float(absorbed),
        float(shift),
        float(z),
        z_paired,
        bool(absorbed > absorb_min and abs(z) > z_min),
        bool(abs(z_paired) > z_min),
    )


def tipping_point(result: dml.PLRResult, d: np.ndarray) -> float:
    """|theta| * sd of the cross-fitted price residual."""
    return float(abs(result.theta) * np.sqrt(result.overlap * np.var(d)))


def _boosting(categorical: list[bool], seed: int):
    return HistGradientBoostingRegressor(
        categorical_features=categorical or None, max_iter=300, random_state=seed
    )


def benchmark(
    y: np.ndarray,
    d: np.ndarray,
    X: np.ndarray,
    groups: dict[str, list[int]],
    categorical: list[bool] | None = None,
    seed: int = 0,
) -> list[dict]:
    """One row per group G of columns of X.

    k: sd of the part of the outcome nuisance that only G explains.
    t: sd of the part of the price nuisance that only G explains.
    ratio: tau / k, the one-sided worst case. c_star: sqrt(tau * sd(D~) / (k t)),
    the multiple of G, on both sides, that would explain the estimate away.
    theta_without: the estimate with G dropped; shift_in_se: the change in SEs.
    categorical=None uses the spline learner; otherwise boosting with the
    given columns treated as categorical.
    """

    def learner(cols):
        if categorical is None:
            return dml.spline_learner(seed)
        return _boosting([categorical[c] for c in cols], seed)

    all_cols = list(range(X.shape[1]))
    folds = list(KFold(5, shuffle=True, random_state=seed).split(X))
    l_hat = dml.crossfit(learner(all_cols), X, y, folds)
    m_hat = dml.crossfit(learner(all_cols), X, d, folds)
    d_res, y_res = d - m_hat, y - l_hat
    theta = float(np.sum(d_res * y_res) / np.sum(d_res**2))
    se = float(
        np.sqrt(np.mean((d_res * (y_res - theta * d_res)) ** 2) / np.mean(d_res**2) ** 2 / len(y))
    )
    tau = abs(theta) * float(np.std(d_res))
    g_hat = l_hat - theta * m_hat
    rows = []
    for name, cols in groups.items():
        keep = [c for c in all_cols if c not in cols]
        Xk = X[:, keep]
        g_rest = dml.crossfit(learner(keep), Xk, g_hat, folds)
        k = float(np.std(g_hat - g_rest))
        m_rest = dml.crossfit(learner(keep), Xk, m_hat, folds)
        t = float(np.std(m_hat - m_rest))
        c_star = float(np.sqrt(tau * np.std(d_res) / (k * t))) if k * t > 0 else np.inf
        l_k = dml.crossfit(learner(keep), Xk, y, folds)
        m_k = dml.crossfit(learner(keep), Xk, d, folds)
        dk, yk = d - m_k, y - l_k
        theta_k = float(np.sum(dk * yk) / np.sum(dk**2))
        rows.append(
            dict(
                group=name,
                theta=theta,
                se=se,
                tau=tau,
                sd_dres=float(np.std(d_res)),
                k=k,
                t=t,
                ratio=tau / k,
                c_star=c_star,
                theta_without=theta_k,
                shift_in_se=(theta_k - theta) / se,
            )
        )
    return rows
