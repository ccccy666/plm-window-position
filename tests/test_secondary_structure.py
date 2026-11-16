"""Tests for calibrate.data.secondary_structure.ss_labels (Phase 1 D4)."""
from calibrate.data.secondary_structure import ss_labels


def _entry():
    return {
        "primaryAccession": "TEST1",
        "sequence": {"value": "AAAAAAAAAA"},  # length 10
        "features": [
            {"type": "Helix", "location": {"start": {"value": 2}, "end": {"value": 4}}},
            {"type": "Beta strand", "location": {"start": {"value": 7}, "end": {"value": 8}}},
            {"type": "Turn", "location": {"start": {"value": 9}, "end": {"value": 9}}},
        ],
    }


def test_ss_labels_maps_helix_strand_and_off_by_one():
    seq, lab = ss_labels(_entry())
    assert seq == "AAAAAAAAAA"
    # 1-based [2,4] helix -> idx 1,2,3 ; [7,8] strand -> idx 6,7 ; Turn ignored -> C
    assert lab == ["C", "H", "H", "H", "C", "C", "E", "E", "C", "C"]


def test_ss_labels_all_coil_when_no_features():
    seq, lab = ss_labels({"sequence": {"value": "MKT"}, "features": []})
    assert lab == ["C", "C", "C"]
