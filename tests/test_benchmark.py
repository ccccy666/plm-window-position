"""Tests for calibrate.benchmark harness (T1.C)."""
import numpy as np

from calibrate.benchmark import fidelity_matrix


def _ctx_free_embed(seq):
    # context-free: each residue's embedding depends only on its AA -> windowed == full
    return np.array([[float(ord(c)), 1.0, 2.0, 3.0] for c in seq])


def test_fidelity_matrix_shape_and_locates_target():
    # one probe; long enough to place target at all rel positions in W
    W = 64
    probes = [{"seq": "ACDEFGHIKLMNPQRSTVWY" * 20, "target": 200}]
    rel = [0.1, 0.5, 0.9]
    M = fidelity_matrix(_ctx_free_embed, probes, W=W, rel_pos=rel)
    assert M.shape == (1, 3)
    # context-free embedding: the SAME target residue is compared -> cosine ~1 at every placement
    assert np.allclose(M, 1.0, atol=1e-6)
