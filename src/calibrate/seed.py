"""Reproducibility helpers. See CLAUDE.md > Conventions (seed policy)."""
from __future__ import annotations

import random

import numpy as np

DEFAULT_SEED = 42
# For variance / error bars: report mean +/- std across these seeds.
VARIANCE_SEEDS = (42, 43, 44, 45, 46)


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed python `random`, numpy, and torch (cpu+cuda if available).

    Returns the seed so callers can log it into a run config.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
