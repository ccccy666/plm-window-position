"""Tests for calibrate.metrics — benchmark bias metrics (T1.C)."""
import numpy as np
import pytest

from calibrate.metrics import position_gap, robust_accuracy, shift_sensitivity

# rows = targets, cols = placements at these rel positions
PLACEMENTS = [0.1, 0.3, 0.5, 0.7, 0.9]
M = np.array(
    [[0.9, 0.8, 1.0, 0.85, 0.7],
     [0.8, 0.7, 0.9, 0.75, 0.6]]
)


def test_position_gap_center_minus_edges():
    # t0: 1.0 - (0.9+0.7)/2 = 0.2 ; t1: 0.9 - (0.8+0.6)/2 = 0.2 ; mean 0.2
    g = position_gap(M, PLACEMENTS, center=0.5, edges=(0.1, 0.9))
    assert g == pytest.approx(0.2)


def test_shift_sensitivity_is_mean_rowwise_std():
    assert shift_sensitivity(M) == pytest.approx(np.mean(M.std(axis=1)))


def test_robust_accuracy_is_mean_rowwise_min():
    # t0 min 0.7, t1 min 0.6 -> 0.65
    assert robust_accuracy(M) == pytest.approx(0.65)


def test_position_gap_zero_when_flat():
    flat = np.ones((3, 5)) * 0.8
    assert position_gap(flat, PLACEMENTS, center=0.5, edges=(0.1, 0.9)) == pytest.approx(0.0)
