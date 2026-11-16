"""Pre-registered statistics for the pilot gate.

Effect sizes are reported in **percentage points (pp)**; the gate threshold is
>=10pp at p<0.01 (see CLAUDE.md > Falsification Conditions). Multiple testing
across position bins uses Holm-Bonferroni; paired comparisons use a bootstrap.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def amplitude_pp(values: Sequence[float]) -> float:
    """Position-bias amplitude U = (max - min) over bins, in percentage points.

    `values` are metric values (e.g. accuracy) as fractions in [0, 1].
    """
    arr = np.asarray(values, dtype=float)
    return float(arr.max() - arr.min()) * 100.0


def pp_difference(a: float, b: float) -> float:
    """Signed difference (a - b) expressed in percentage points."""
    return (float(a) - float(b)) * 100.0


def holm_bonferroni(
    pvalues: Sequence[float], alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down correction.

    Returns (reject, adjusted) as boolean / float arrays in the ORIGINAL order.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]

    # adjusted_(i) = cummax over ascending ranks of (m - i) * p_(i), clipped to 1
    factors = m - np.arange(m)
    adj_sorted = np.clip(np.maximum.accumulate(factors * p_sorted), 0.0, 1.0)
    reject_sorted = adj_sorted <= alpha

    adjusted = np.empty(m, dtype=float)
    reject = np.empty(m, dtype=bool)
    adjusted[order] = adj_sorted
    reject[order] = reject_sorted
    return reject, adjusted


def paired_bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Paired bootstrap of the mean difference mean(x) - mean(y).

    Resamples paired indices with replacement. Returns observed diff, percentile
    CI, and a two-sided bootstrap p-value (fraction of resamples crossing zero).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")

    diff = x - y
    observed = float(diff.mean())
    n = diff.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)

    lo = (1.0 - ci) / 2.0 * 100.0
    hi = (1.0 + ci) / 2.0 * 100.0
    ci_low, ci_high = np.percentile(boot, [lo, hi])

    p_left = float(np.mean(boot <= 0.0))
    p_right = float(np.mean(boot >= 0.0))
    p_value = float(min(1.0, 2.0 * min(p_left, p_right)))

    return {
        "diff": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
    }
