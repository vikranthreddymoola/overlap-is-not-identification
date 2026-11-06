"""Overlap diagnostics for partially linear double machine learning."""

from .diagnostics import OverlapCheck, benchmark, overlap_inflation, paired_z, tipping_point
from .dml import LEARNERS, PLRResult, crossfit, ols_linear, plr

__version__ = "1.0.0"

__all__ = [
    "LEARNERS",
    "OverlapCheck",
    "PLRResult",
    "benchmark",
    "crossfit",
    "ols_linear",
    "overlap_inflation",
    "paired_z",
    "plr",
    "tipping_point",
]
