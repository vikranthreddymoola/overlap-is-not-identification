"""A pricing rule reacts to a forecast of low supply. The forecast uses
information the analyst does not log, so overlap looks healthy but the
estimate is biased. Logging the forecast and comparing the two fits shows it.

    python examples/quickstart.py
"""

import warnings

import numpy as np

from overlap_diag import overlap_inflation, plr, tipping_point

# numpy 2.0 with Apple's Accelerate BLAS warns spuriously in matmul; results are unaffected.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

rng = np.random.default_rng(0)
n = 8000
X = rng.normal(size=(n, 8))  # logged controls
U = rng.normal(size=n)  # supply shock the analyst does not observe
Z = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.5 * X[:, 2] + U  # shortfall
F = Z.copy()  # the platform's forecast of the shortfall
D = F + 1.0 * rng.normal(size=n)  # pay reacts to the forecast, plus exogenous noise
Y = 0.5 * D - Z + rng.normal(size=n)  # true effect of pay: 0.5

coarse = plr(Y, D, X, seed=0)
fine = plr(Y, D, np.column_stack([X, F]), seed=0)

for name, r in (("controls X", coarse), ("controls X + forecast", fine)):
    lo, hi = r.ci
    ci = f"[{lo:6.3f}, {hi:6.3f}]"
    print(f"{name:24s} theta = {r.theta:6.3f}  95% CI {ci}  overlap = {r.overlap:.2f}")

check = overlap_inflation(coarse, fine)
print(f"\nabsorbed = {check.absorbed:.1%}, z = {check.z:.1f}, flag = {check.flag}")
print(f"tipping point of the coarse fit: {tipping_point(coarse, D):.3f}")
