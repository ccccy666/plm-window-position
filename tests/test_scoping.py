"""Tests for calibrate.scoping — the ProteinGym crop-shift (VEP) analysis.

The first version ranked positions instead of mutations: it averaged the model scores of all sampled mutants at
a position but kept the DMS value of only one of them. These tests pin the per-mutant analysis.
"""
from pathlib import Path

import pandas as pd
import pytest

from calibrate import scoping as sc

ROOT = Path(__file__).resolve().parents[1]


def test_crop_shift_ranks_mutants_not_positions():
    rows = []
    for pos, dms, score in [(10, 1.0, 1.0), (10, -1.0, -1.0), (20, 0.5, 0.5), (30, -0.5, -0.5), (40, 0.0, 0.0)]:
        for rp in (0.1, 0.5, 0.9):          # one row per placement, mutant by mutant (as the scoring loop writes)
            rows.append(dict(DMS_id="A", pos=pos, dms=dms, rel_pos=rp, score=score))
    t = sc.crop_shift_assays(pd.DataFrame(rows))
    assert t.loc["A", "n"] == 5             # two mutants at position 10 stay separate
    assert t.loc["A", "rho_center"] == pytest.approx(1.0)


def test_crop_shift_stored_run_per_mutant():
    t = sc.crop_shift_assays(pd.read_parquet(ROOT / "experiments/phase0_pilot/p2b/crop_shift.parquet"))
    s = sc.delta_rho_summary(t)
    assert s["n"] == 27
    assert s["delta_rho"] == pytest.approx(0.019, abs=0.0005)
    assert s["p"] > 0.05
    assert s["shift"] == pytest.approx(0.80, abs=0.005)
