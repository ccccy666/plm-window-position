"""Tests for calibrate.stats — effect size + multiple testing + bootstrap (P0.0.3)."""
import numpy as np
import pytest

from calibrate.stats import (
    amplitude_pp,
    holm_bonferroni,
    paired_bootstrap_ci,
    pp_difference,
)


# --- effect sizes (percentage points) ---

def test_amplitude_pp_is_max_minus_min_in_pp():
    # accuracies as fractions in [0,1]; U = (max - min) * 100
    assert amplitude_pp([0.80, 0.60, 0.75]) == pytest.approx(20.0)


def test_amplitude_pp_constant_is_zero():
    assert amplitude_pp([0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_pp_difference_is_signed_and_in_pp():
    assert pp_difference(0.82, 0.70) == pytest.approx(12.0)
    assert pp_difference(0.70, 0.82) == pytest.approx(-12.0)


# --- Holm-Bonferroni ---

def test_holm_bonferroni_rejects_correct_set():
    # m=4, alpha=0.05; sorted asc: 0.005(*4=.02), 0.01(*3=.03), 0.03(*2=.06 STOP)
    pvals = [0.01, 0.04, 0.03, 0.005]
    reject, _ = holm_bonferroni(pvals, alpha=0.05)
    assert np.asarray(reject).tolist() == [True, False, False, True]


def test_holm_bonferroni_adjusted_pvalues_known_and_bounded():
    pvals = [0.01, 0.04, 0.03, 0.005]
    _, adj = holm_bonferroni(pvals, alpha=0.05)
    adj = np.asarray(adj)
    assert np.all((adj >= 0.0) & (adj <= 1.0))
    smallest = np.argsort(pvals)[0]  # the 0.005 entry
    assert adj[smallest] == pytest.approx(0.02)


def test_holm_all_significant():
    reject, _ = holm_bonferroni([1e-4, 2e-4, 3e-4], alpha=0.05)
    assert all(reject)


def test_holm_none_significant():
    reject, _ = holm_bonferroni([0.9, 0.8, 0.95], alpha=0.05)
    assert not any(reject)


# --- paired bootstrap ---

def test_paired_bootstrap_diff_equals_observed_mean_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 0.1, 200)
    y = rng.normal(0.0, 0.1, 200)
    res = paired_bootstrap_ci(x, y, n_boot=2000, seed=42)
    assert res["diff"] == pytest.approx(float(np.mean(x) - np.mean(y)))


def test_paired_bootstrap_clear_effect_excludes_zero():
    rng = np.random.default_rng(1)
    x = rng.normal(1.0, 0.1, 200)
    y = rng.normal(0.0, 0.1, 200)
    res = paired_bootstrap_ci(x, y, n_boot=2000, seed=42)
    assert res["ci_low"] > 0.0
    assert res["p_value"] < 0.01


def test_paired_bootstrap_no_effect_includes_zero():
    rng = np.random.default_rng(2)
    z = rng.normal(0.0, 1.0, 300)
    res = paired_bootstrap_ci(z, z.copy(), n_boot=2000, seed=42)
    assert res["diff"] == pytest.approx(0.0)
    assert res["ci_low"] <= 0.0 <= res["ci_high"]


def test_paired_bootstrap_is_reproducible_under_same_seed():
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 0.5, 50)
    a = paired_bootstrap_ci(x, y, n_boot=1000, seed=7)
    b = paired_bootstrap_ci(x, y, n_boot=1000, seed=7)
    assert a == b


def test_paired_bootstrap_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1, 2, 3], [1, 2], n_boot=10)
