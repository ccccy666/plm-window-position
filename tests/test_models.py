"""Tests for calibrate.models — the ESM-2 inference API (P0.0.2).

Loads the real 650M checkpoint once per module (no mocks: TDD wants real code).
These are GPU tests; sequences are short so attention L x L stays tiny.
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def esm2():
    import os
    from pathlib import Path

    home = Path(os.environ.get("TORCH_HOME", Path(__file__).resolve().parents[1] / "data/cache/models/torch"))
    if not (home / "hub/checkpoints/esm2_t33_650M_UR50D.pt").exists():
        pytest.skip("ESM-2 650M weights not downloaded (see README, 'Environments and models')")
    from calibrate.models import load_esm2

    return load_esm2()


def test_reports_architecture(esm2):
    assert esm2.num_layers == 33
    assert esm2.num_heads == 20
    assert esm2.embed_dim == 1280


def test_embed_returns_per_residue_vectors_without_special_tokens(esm2):
    embs = esm2.embed(["MKTAYIA", "MKT"], layer=33)
    assert len(embs) == 2
    assert embs[0].shape == (7, 1280)   # no BOS/EOS in the returned slice
    assert embs[1].shape == (3, 1280)
    assert np.isfinite(embs[0]).all()


def test_recovery_prob_returns_probabilities(esm2):
    probs = esm2.recovery_prob("MKTAYIAKQRQISFVKSHF", positions=[0, 5, 10])
    assert probs.shape == (3,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_recovery_prob_varies_by_position(esm2):
    probs = esm2.recovery_prob("MKTAYIAKQRQISFVKSHFSRQLEER", positions=[1, 2, 3])
    assert probs.shape == (3,)
    assert np.unique(np.round(probs, 6)).size > 1  # not all identical


def test_attention_into_shape_and_nonnegative(esm2):
    att = esm2.attention_into("MKTAYIAKQRQISFVKSHF", target_positions=[4, 9], layers=[1, 33])
    assert att.shape == (2, 20, 2)  # (n_layers, n_heads, n_targets)
    assert np.all(att >= 0.0)
    assert np.isfinite(att).all()


def test_out_of_range_position_raises(esm2):
    with pytest.raises((ValueError, IndexError)):
        esm2.recovery_prob("MKT", positions=[5])


def test_position_logprobs_shape_and_normalized(esm2):
    seq = "MKTAYIAKQRQISFVKSHF"
    lp = esm2.position_logprobs(seq)
    assert lp.shape[0] == len(seq)
    p = np.exp(lp)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-3)
    assert np.isfinite(lp).all()


def test_aa_token_index_distinct(esm2):
    idx = [esm2.aa_token_index(a) for a in "ACDEFG"]
    assert len(set(idx)) == 6


def test_masked_distribution_shape_and_normalized(esm2):
    d = esm2.masked_distribution("MKTAYIAKQR", [2, 5])
    assert d.shape[0] == 2
    assert np.allclose(d.sum(axis=1), 1.0, atol=1e-3)
    assert np.all(d >= 0.0)


def test_masked_distribution_matches_recovery_prob(esm2):
    seq = "MKTAYIAKQRQISFVKSHF"
    d = esm2.masked_distribution(seq, [3, 7])
    rec = esm2.recovery_prob(seq, [3, 7])
    true = [d[0, esm2.aa_token_index(seq[3])], d[1, esm2.aa_token_index(seq[7])]]
    assert np.allclose(true, rec, atol=1e-4)


def test_contacts_shape_symmetric_in_unit_interval(esm2):
    seq = "MKTAYIAKQRQISFVKSHFSRQLEER"
    c = esm2.contacts(seq)
    assert c.shape == (len(seq), len(seq))
    assert np.all((c >= 0.0) & (c <= 1.0))
    assert np.allclose(c, c.T, atol=1e-4)  # contact map is symmetric
