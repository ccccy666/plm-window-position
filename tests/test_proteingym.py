"""Tests for calibrate.data.proteingym mutant parsing (P2)."""
import pytest

from calibrate.data.proteingym import parse_mutant


def test_parse_single_mutant():
    assert parse_mutant("I291A") == ("I", 291, "A")


def test_parse_mutant_multidigit_position():
    assert parse_mutant("M1042V") == ("M", 1042, "V")


def test_parse_mutant_rejects_multi_mutant():
    with pytest.raises(ValueError):
        parse_mutant("I291A:K300R")


def test_parse_mutant_rejects_garbage():
    with pytest.raises(ValueError):
        parse_mutant("not_a_mutant")
