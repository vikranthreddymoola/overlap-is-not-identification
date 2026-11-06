"""DGPs used in the paper.

reactive():  Z = h(X) + U, F = h(X) + kappa*U, D = a*F + sigma_v*V,
             Y = theta*D - b*Z + eps                       (Studies 1, 4)
seasonal():  D = s(t) + 0.3*X1 + sigma_v*V,
             Y = theta*D + g(X) + gamma*s(t) + eps          (Studies 2, 3)
             calendar adjusted exactly or by month
"""

from __future__ import annotations

import numpy as np

THETA = 0.5


def h(X: np.ndarray) -> np.ndarray:
    return np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.5 * X[:, 2]


def h_interaction(X: np.ndarray) -> np.ndarray:
    """h(X) with an X3*X4 term (Study 4)."""
    return np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + X[:, 2] * X[:, 3]


def reactive(n, kappa, sigma_v, a=1.0, b=1.0, s_u=1.0, p=8, seed=0, theta=THETA, interaction=False):
    """Returns y, d, X, F."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    U = s_u * rng.normal(size=n)
    hx = h_interaction(X) if interaction else h(X)
    F = hx + kappa * U
    d = a * F + sigma_v * rng.normal(size=n)
    y = theta * d - b * (hx + U) + rng.normal(size=n)
    return y, d, X, F


def reactive_bias(kappa, sigma_v, a=1.0, b=1.0, s_u=1.0) -> float:
    """Closed-form bias when F is left out (Prop. 1)."""
    return -b * a * kappa * s_u**2 / (a**2 * kappa**2 * s_u**2 + sigma_v**2)


VAR_H = 0.5 * (1 - np.exp(-8)) + 0.5 + 0.25  # Var h(X) for X ~ N(0, I)


def reactive_overlap(kappa, sigma_v, a=1.0, s_u=1.0, with_forecast=False) -> float:
    """Population overlap, with or without F in the controls."""
    var_d = a**2 * (VAR_H + kappa**2 * s_u**2) + sigma_v**2
    resid = sigma_v**2 if with_forecast else a**2 * kappa**2 * s_u**2 + sigma_v**2
    return resid / var_d


def season(t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * t) + 0.5 * np.cos(4 * np.pi * t)


VAR_SEASON_PART = 0.625 + 0.09  # Var(s(t) + 0.3 X1)


def sigma_v_for_overlap(target: float) -> float:
    """sigma_v that gives the target overlap under exact calendar adjustment."""
    return float(np.sqrt(target * VAR_SEASON_PART / (1 - target)))


GAMMA = 2.0


def seasonal(n, sigma_v, gamma=GAMMA, p=6, seed=0, theta=THETA, calendar="exact"):
    """Returns y, d, X. First three columns of X are the calendar: (t, sin, cos)
    for calendar='exact', 12 month dummies for calendar='month'.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(size=n)
    Xo = rng.normal(size=(n, p))
    s = season(t)
    d = s + 0.3 * Xo[:, 0] + sigma_v * rng.normal(size=n)
    g = np.sin(2 * Xo[:, 1]) + 0.5 * Xo[:, 2] ** 2
    y = theta * d + g + gamma * s + rng.normal(size=n)
    if calendar == "exact":
        cal = np.column_stack([t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    else:
        cal = np.eye(12)[np.minimum((t * 12).astype(int), 11)]
    return y, d, np.column_stack([cal, Xo])


CALENDAR_COLS = [0, 1, 2]
VAR_WITHIN_MONTH = 0.02232  # Var(s(t) - E[s(t) | month]), by simulation


def coarse_calendar_bias(sigma_v, gamma=GAMMA, var_w=VAR_WITHIN_MONTH) -> float:
    """Prop. 1 for month dummies: W = s(t) - E[s(t) | month], a = 1, b = gamma."""
    return gamma * var_w / (var_w + sigma_v**2)
