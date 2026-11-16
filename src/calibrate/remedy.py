"""Training-free multi-crop aggregation for the window-edge effect (T2.A).

The window-edge recovery loss (C1) is one-sided context-loss (T1.B): a residue scored
from a window edge has truncated context on the near side. When a residue can be scored
from several overlapping windows (e.g. tiling a long protein), aggregating those
predictions recovers part of the loss *without retraining the backbone*.

Strategies characterized in `scripts/phase2/t2a_remedy*.py`:
  - **centering** — score each residue from the window where it is most central (rel≈0.5);
    the upper bound, but needs controllable placement (re-tiling per residue).
  - **uniform multi-crop** — average the per-placement distributions equally (the cheap,
    always-available ensemble).
  - **centrality-weighted multi-crop** — average them weighted by the residue's centrality
    in each window (``min(r, 1-r)``), trusting balanced-context placements more.

This module holds the pure aggregation logic (no model); backbones feed in per-placement
probability distributions.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def centrality_weights(placements: Sequence[float]) -> np.ndarray:
    """Normalized weights ∝ ``min(r, 1-r)`` — the target's distance to the nearer window
    edge (its balanced-context budget). Peaks at the center placement, ~0 at the edges."""
    w = np.array([min(r, 1.0 - r) for r in placements], dtype=float)
    total = w.sum()
    if total <= 0:
        return np.full(len(w), 1.0 / len(w))
    return w / total


def aggregate_distributions(dists, weights: Sequence[float] | None = None) -> np.ndarray:
    """Weighted average of per-placement probability distributions.

    Args:
        dists: (n_placements, vocab) array of probability distributions.
        weights: optional per-placement weights (need not be pre-normalized);
            uniform if None.
    """
    D = np.asarray(dists, dtype=float)
    if weights is None:
        w = np.full(D.shape[0], 1.0 / D.shape[0])
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
    return (D * w[:, None]).sum(axis=0)


def recovered_fraction(strategy: float, edge: float, center: float) -> float:
    """Fraction of the center−edge gap that a strategy recovers: 0 at edge, 1 at center.

    Guarded against a degenerate (near-zero) gap so it stays finite.
    """
    gap = center - edge
    if abs(gap) < 1e-9:
        return 0.0
    return float((strategy - edge) / gap)


def off_center_recovery(uniform, edge, center):
    """Mean recovery at the two placements next to the center (rel 0.3 and 0.7 of the five placements
    {0.1, 0.3, 0.5, 0.7, 0.9}), recovered exactly from the stored aggregates by linearity:
    5 * uniform = 2 * edge + center + 2 * off_center. Because the most central of K = 5 overlapping
    windows is never more than 0.1 W off center, this value (0.2 W off center) bounds its recovery
    from below (Table 5)."""
    uniform, edge, center = (np.asarray(x, dtype=float) for x in (uniform, edge, center))
    return (5 * uniform - 2 * edge - center) / 2
