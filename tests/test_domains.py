"""Characterization tests for calibrate.domains — the C1 domain-offset design and analysis.

The functions were moved out of scripts/phase3/c1_domain_offsets.py so other backbones can be run
through the same analysis; the stored ESM-2 run must come out identical.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from calibrate import domains as dm

ROOT = Path(__file__).resolve().parents[1]
C1 = ROOT / "experiments/phase3/c1_domain_offsets"


def test_build_residues_reproduces_the_c1_design():
    res = dm.build_residues()
    assert len(res) == 527
    assert res["acc"].nunique() == 60
    assert res.groupby(["acc", "d0"]).ngroups == 122


def test_summarize_domain_reproduces_the_stored_esm2_summary():
    stored = json.loads((C1 / "summary.json").read_text())
    s = dm.summarize_domain(pd.read_parquet(C1 / "residues.parquet"))
    for pol in ("dom_center", "dom_start", "dom_end"):
        assert s["Q1"][pol]["rf_off_midpoint"] == pytest.approx(stored["Q1"][pol]["rf_off_midpoint"])
        assert s["Q1"][pol]["ci"] == pytest.approx(stored["Q1"][pol]["ci"])
    for k in ("n_domains", "n_prot", "did_pp", "p"):
        assert s["Q2"][k] == pytest.approx(stored["Q2"][k])
    assert s["center_minus_edge"]["0.1"]["gap_pp"] == pytest.approx(
        stored["center_minus_edge"]["0.1"]["gap_pp"])
