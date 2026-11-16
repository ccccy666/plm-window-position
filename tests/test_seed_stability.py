"""The appendix seed-stability numbers (Appendix A.2) come from the stored multi-seed run
(scripts/phase3/b4_multiseed.py -> experiments/phase3/b4_multiseed/)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from calibrate import metrics

OUT = Path(__file__).resolve().parents[1] / "experiments/phase3/b4_multiseed"
REL = [0.1, 0.3, 0.5, 0.7, 0.9]


def test_seed_stability_matches_appendix():
    s = json.loads((OUT / "seeds.json").read_text())
    assert s["seeds"] == [42, 43, 44, 45, 46]
    assert s["position_gap_pp"]["mean"] == pytest.approx(3.51, abs=0.005)
    assert s["position_gap_pp"]["std"] == pytest.approx(0.46, abs=0.005)
    assert s["shift_sensitivity"]["mean"] == pytest.approx(0.028, abs=0.0005)
    assert s["shift_sensitivity"]["std"] == pytest.approx(0.001, abs=0.0005)
    assert s["robust_accuracy"]["mean"] == pytest.approx(0.910, abs=0.0005)
    assert s["robust_accuracy"]["std"] == pytest.approx(0.004, abs=0.0005)


def test_seed_metrics_recompute_from_stored_fidelities():
    s = json.loads((OUT / "seeds.json").read_text())
    d = pd.read_parquet(OUT / "fidelity.parquet")
    assert sorted(d.seed.unique()) == s["seeds"]
    for row in s["per_seed"]:
        M = d[d.seed == row["seed"]].pivot(index="target", columns="rel", values="fidelity")[REL].to_numpy()
        assert M.shape == (120, 5)
        assert metrics.position_gap(M, REL) * 100 == pytest.approx(row["position_gap_pp"])
        assert metrics.shift_sensitivity(M) == pytest.approx(row["shift_sensitivity"])
        assert metrics.robust_accuracy(M) == pytest.approx(row["robust_accuracy"])
