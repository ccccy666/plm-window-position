"""Benchmark bias metrics for crop/window-position behavior (T1.C).

All operate on a 2D array ``M`` of shape (n_targets, n_placements): rows are probe
targets (residue / mutation / contact pair), columns are the metric value at each
window placement (e.g., recovery, fidelity, score). ``placements`` lists the rel-pos
of each column. Report these alongside the raw task metric (they describe *stability*,
not accuracy). Pair with `stats.paired_bootstrap_ci` for CIs on the gap.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _col(placements: Sequence[float], rel: float) -> int:
    return min(range(len(placements)), key=lambda i: abs(placements[i] - rel))


def position_gap(
    M: np.ndarray, placements: Sequence[float], center: float = 0.5,
    edges: tuple[float, float] | None = None,
) -> float:
    """Mean over targets of (value at center placement − mean value at edge placements).

    Positive ⇒ center is favored (the bidirectional inverted-U). `edges` defaults to the
    extreme placements.
    """
    M = np.asarray(M, dtype=float)
    if edges is None:
        edges = (placements[0], placements[-1])
    c = M[:, _col(placements, center)]
    e = M[:, [_col(placements, edges[0]), _col(placements, edges[1])]].mean(axis=1)
    return float(np.mean(c - e))


def shift_sensitivity(M: np.ndarray) -> float:
    """Mean over targets of the std of the value across placements (placement instability)."""
    return float(np.mean(np.asarray(M, dtype=float).std(axis=1)))


def robust_accuracy(M: np.ndarray) -> float:
    """Mean over targets of the worst-placement (row-min) value — conservative aggregate."""
    return float(np.mean(np.asarray(M, dtype=float).min(axis=1)))
