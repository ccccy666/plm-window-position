"""Tests for calibrate.remedy — training-free multi-crop aggregation (T2.A)."""
import numpy as np
import pytest

from calibrate.remedy import aggregate_distributions, centrality_weights, recovered_fraction


def test_centrality_weights_normalized_and_symmetric():
    w = centrality_weights([0.1, 0.3, 0.5, 0.7, 0.9])
    assert np.isclose(w.sum(), 1.0)
    # symmetric placements get symmetric weights
    assert np.isclose(w[0], w[-1])
    assert np.isclose(w[1], w[-2])


def test_centrality_weights_peak_at_center():
    w = centrality_weights([0.1, 0.3, 0.5, 0.7, 0.9])
    assert w.argmax() == 2          # center (0.5) gets the most weight
    assert w[0] < w[1] < w[2]       # rises toward the center


def test_aggregate_uniform_is_mean():
    dists = [[0.2, 0.8], [0.6, 0.4], [0.4, 0.6]]
    got = aggregate_distributions(dists)              # default uniform
    assert np.allclose(got, np.mean(dists, axis=0))


def test_aggregate_weighted_matches_manual():
    dists = np.array([[1.0, 0.0], [0.0, 1.0]])
    w = np.array([0.25, 0.75])
    got = aggregate_distributions(dists, w)
    assert np.allclose(got, [0.25, 0.75])


def test_aggregate_weights_need_not_be_prenormalized():
    dists = np.array([[1.0, 0.0], [0.0, 1.0]])
    got = aggregate_distributions(dists, [1.0, 3.0])  # -> normalized to [.25,.75]
    assert np.allclose(got, [0.25, 0.75])


def test_recovered_fraction_endpoints():
    assert np.isclose(recovered_fraction(0.8, edge=0.8, center=0.9), 0.0)   # edge -> 0%
    assert np.isclose(recovered_fraction(0.9, edge=0.8, center=0.9), 1.0)   # center -> 100%
    assert np.isclose(recovered_fraction(0.85, edge=0.8, center=0.9), 0.5)  # halfway -> 50%


def test_recovered_fraction_degenerate_gap_is_safe():
    # no edge loss to recover -> defined, finite (not a divide-by-zero blowup)
    val = recovered_fraction(0.8, edge=0.8, center=0.8)
    assert np.isfinite(val)


def test_off_center_recovery_from_linearity():
    from calibrate.remedy import off_center_recovery
    r = np.array([[0.2, 0.5, 0.9, 0.6, 0.1], [0.3, 0.4, 0.8, 0.5, 0.2]])   # recovery at rel 0.1 ... 0.9
    uniform, edge, center = r.mean(1), r[:, [0, 4]].mean(1), r[:, 2]
    assert off_center_recovery(uniform, edge, center) == pytest.approx(r[:, [1, 3]].mean(1))


def test_most_central_bound_reproduces_table5():
    from pathlib import Path

    import pandas as pd

    from calibrate.remedy import off_center_recovery
    root = Path(__file__).resolve().parents[1] / "experiments/phase2_remedy"
    got = {}
    for b in ("esm2", "esmc"):
        d = pd.read_parquet(root / f"{b}_remedy.parquet")
        for w, g in d.groupby("W"):
            m = off_center_recovery(g.multicrop_uniform.values, g.single_edge.values, g.single_center.values)
            got[(b, w)] = round(recovered_fraction(m.mean(), g.single_edge.mean(), g.single_center.mean()), 2)
    assert got == {("esm2", 256): 0.64, ("esm2", 384): 0.91, ("esmc", 256): 0.87, ("esmc", 384): 0.85}
