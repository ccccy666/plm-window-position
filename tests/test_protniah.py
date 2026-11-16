"""Tests for calibrate.probes.protniah — needle insertion + carrier ops (P1.0)."""
import re
from collections import Counter

import numpy as np

from calibrate.probes.protniah import insert_needle, make_nxst_motif, shuffle_sequence


def test_insert_needle_at_start():
    seq, idx = insert_needle("AAAA", "NES", 0.0)
    assert seq == "NESAAAA"
    assert idx == [0, 1, 2]


def test_insert_needle_at_end():
    seq, idx = insert_needle("AAAA", "NES", 1.0)
    assert seq == "AAAANES"
    assert idx == [4, 5, 6]


def test_insert_needle_middle_indices_and_length():
    seq, idx = insert_needle("AAAA", "NES", 0.5)
    assert idx[0] == 2
    assert seq[2:5] == "NES"
    assert len(seq) == len("AAAA") + len("NES")


def test_inserted_indices_recover_the_motif():
    seq, idx = insert_needle("MKTAYIAKQR", "NES", 0.3)
    assert "".join(seq[i] for i in idx) == "NES"


def test_make_nxst_motif_matches_n_x_st_pattern():
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = make_nxst_motif(rng)
        assert re.fullmatch(r"N[^P][ST]", m), m


def test_shuffle_preserves_composition_and_length():
    rng = np.random.default_rng(0)
    s = "MKTAYIAKQRQISFVKSHF"
    sh = shuffle_sequence(s, rng)
    assert Counter(sh) == Counter(s)
    assert len(sh) == len(s)


def test_shuffle_is_reproducible_under_same_seed():
    s = "MKTAYIAKQRQISFVKSHF"
    a = shuffle_sequence(s, np.random.default_rng(1))
    b = shuffle_sequence(s, np.random.default_rng(1))
    assert a == b
