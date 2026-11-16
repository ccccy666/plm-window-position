"""Crop-position benchmark harness (T1.C): backbone-agnostic evaluation.

Add a backbone by passing an ``embed_fn(seq) -> (L, hidden)`` per-residue embedding
function; everything else (fixed probe snapshot, window-relocation, bias metrics) is
shared. See `data/processed/benchmark/` for the versioned snapshot + manifest.
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from calibrate import metrics
from calibrate.stats import paired_bootstrap_ci

BENCH = (_ROOT / "data/processed/benchmark")
DEFAULT_REL = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_W = 256


def load_probes(split: str | None = None, path: Path = BENCH / "probes.jsonl") -> list[dict]:
    probes = [json.loads(l) for l in path.read_text().splitlines()]
    return [p for p in probes if split is None or p.get("split") == split]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def fidelity_matrix(
    embed_fn: Callable[[str], np.ndarray], probes: list[dict],
    W: int = DEFAULT_W, rel_pos: Sequence[float] = DEFAULT_REL,
) -> np.ndarray:
    """(n_probes, n_placements) cosine of each target's windowed vs full-context embedding."""
    rows = []
    for p in probes:
        seq, t = p["seq"], p["target"]
        full = embed_fn(seq)[t]
        row = []
        for r in rel_pos:
            s = t - round(r * W)
            row.append(_cos(embed_fn(seq[s : s + W])[t - s], full))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def report(M: np.ndarray, rel_pos: Sequence[float] = DEFAULT_REL) -> dict:
    """Bias-metric report for a fidelity matrix, with a bootstrap CI on the center-edge gap."""
    center_col = min(range(len(rel_pos)), key=lambda i: abs(rel_pos[i] - 0.5))
    edge = M[:, [0, -1]].mean(axis=1)
    gap = paired_bootstrap_ci(M[:, center_col], edge, seed=42)
    return dict(
        n=M.shape[0],
        curve={float(r): float(M[:, i].mean()) for i, r in enumerate(rel_pos)},
        position_gap_pp=metrics.position_gap(M, rel_pos) * 100,
        gap_ci_pp=(gap["ci_low"] * 100, gap["ci_high"] * 100),
        gap_p=gap["p_value"],
        shift_sensitivity=metrics.shift_sensitivity(M),
        robust_accuracy=metrics.robust_accuracy(M),
    )
