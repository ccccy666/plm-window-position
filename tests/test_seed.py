"""Tests for calibrate.seed — reproducibility helpers (P0.0.3)."""
import numpy as np

from calibrate.seed import DEFAULT_SEED, VARIANCE_SEEDS, set_global_seed


def test_set_global_seed_makes_numpy_reproducible():
    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_different_seeds_produce_different_draws():
    set_global_seed(1)
    a = np.random.rand(5)
    set_global_seed(2)
    b = np.random.rand(5)
    assert not np.allclose(a, b)


def test_default_seed_is_42():
    assert DEFAULT_SEED == 42


def test_variance_seeds_are_five_unique_starting_at_42():
    assert tuple(VARIANCE_SEEDS) == (42, 43, 44, 45, 46)
    assert len(set(VARIANCE_SEEDS)) == 5


def test_set_global_seed_returns_the_seed():
    assert set_global_seed(7) == 7


def test_seeds_torch_when_available():
    import torch

    set_global_seed(99)
    a = torch.randn(4)
    set_global_seed(99)
    b = torch.randn(4)
    assert torch.allclose(a, b)
