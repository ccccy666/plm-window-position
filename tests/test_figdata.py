"""Tests for calibrate.figdata — summary values behind the camera-ready composite figures.

Every expected value below is a number quoted in the paper (spec: docs/camera_ready_revision_plan.md §4.4).
If an artifact or a summary function drifts, the figure and the prose would disagree; these tests catch it.
"""
from pathlib import Path

import pytest

from calibrate import figdata as fd

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments/phase0_pilot/p1b/position_sweep.parquet"
FLANK = ROOT / "experiments/phase1_diagnosis/t1b_mechanism/mechanism.parquet"
CLIFF = ROOT / "experiments/phase3/b2_cliff/cliff.parquet"
RANDINIT = ROOT / "rebuttal/reviewer2/r2_randinit_randseq.json"
ATTN = ROOT / "experiments/phase3/a_mechgen/esm2/attention.parquet"   # all 120 targets (no length cap)
DOMAIN = ROOT / "experiments/phase3/c1_domain_offsets/residues.parquet"
PAIRS = ROOT / "experiments/phase3/b1_contacts_partner_out/pairs.parquet"
MULTIW = ROOT / "rebuttal/r1_partner_out_multiW.json"
TILING = ROOT / "rebuttal/r1_tiling.json"


def test_pilot_edge_gaps_per_stratum():
    gaps = fd.pilot_cassette_edge_gaps(PILOT)
    assert round(gaps[("real", 256)], 2) == -3.44
    assert round(gaps[("real", 512)], 2) == -1.86
    assert round(gaps[("shuffled", 512)], 2) == -2.28
    assert round(gaps[("shuffled", 256)], 2) == -0.33


def test_pilot_profile_is_zero_at_center_and_positive_at_ends():
    prof = fd.pilot_cassette_profile(PILOT)
    assert list(prof.index) == [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
    assert prof.loc[0.5, "mean_pp"] == pytest.approx(0.0, abs=1e-9)
    assert prof.loc[0.05, "mean_pp"] > 1.0 and prof.loc[0.95, "mean_pp"] > 1.0
    assert abs(prof.loc[0.35, "mean_pp"]) < 1.0


def test_flank_sweep_gaps_and_levels():
    curves, gaps = fd.flank_sweep(FLANK)
    assert {f: round(g, 2) for f, g in gaps.items()} == {10: -1.24, 20: -4.46, 40: 2.86, 80: -0.53}
    assert round(curves.loc[10, 0.5], 3) == 0.101
    assert round(curves.loc[80, 0.5], 3) == 0.595


def test_cliff_in_vs_out_of_crop():
    by_pos, in_crop, out_crop = fd.cliff_profile(CLIFF)
    assert round(in_crop, 3) == 0.348
    assert round(out_crop, 3) == 0.345
    assert list(by_pos.index) == [300, 600, 900, 1100, 1400, 1700, 1900]


def test_randinit_two_by_two_gaps():
    g = fd.randinit_gaps(RANDINIT)
    assert {k: round(v["gap_pp"], 2) for k, v in g.items()} == {
        "trained/natural": 3.29, "trained/random_seq": 0.42,
        "random_init/natural": 0.19, "random_init/random_seq": 0.06}


def test_attention_masses_dropped():
    a = fd.attention_masses(ATTN)
    assert tuple(round(x, 3) for x in a["trained"]) == (0.207, 0.304)
    assert tuple(round(x, 3) for x in a["random_init"]) == (0.619, 0.619)


def test_domain_side_asymmetry_flips_sign():
    a = fd.domain_asymmetry_pp(DOMAIN)
    assert {u: round(v, 2) for u, v in a.items()} == {
        0.1: 8.13, 0.25: 4.05, 0.5: 0.21, 0.75: -4.97, 0.9: -7.58}


def test_domain_recovered_fraction_off_midpoint():
    rf = fd.domain_recovered_fraction(DOMAIN)
    assert round(rf["dom_center"]["off_midpoint"] * 100, 1) == 97.8
    assert round(rf["dom_start"]["off_midpoint"] * 100, 1) == 64.4
    assert round(rf["dom_end"]["off_midpoint"] * 100, 1) == 62.4
    assert round(rf["dom_center"][0.5] * 100, 0) == 100


def test_partner_out_means_w256():
    m = fd.partner_out_means(PAIRS)
    assert {k: round(v, 2) for k, v in m.items()} == {
        "full": 0.84, "center": 0.78, "partner_edge": 0.58, "partner_out": 0.0}


def test_partner_multiW_gaps():
    m = fd.partner_multiW(MULTIW)
    assert {w: round(v["gap_pp"], 2) for w, v in m.items()} == {128: 24.03, 256: 20.28, 384: 11.01}


def test_tiling_split_rates():
    t = fd.tiling_split_rates(TILING)
    assert {w: tuple(round(x, 1) for x in v) for w, v in t.items()} == {
        128: (41.0, 11.7, 7.0), 256: (19.1, 0.8, 0.0), 384: (10.4, 0.0, 0.0)}


def test_split_by_separation():
    s = fd.split_by_separation(TILING)
    assert [b for b, _, _ in s] == ["30-60", "60-90", "90-120", "120-150", "150-200"]
    assert [round(p, 1) for _, _, p in s] == [14.7, 21.8, 32.4, 52.8, 41.9]


TAXO = ROOT / "experiments/phase1_diagnosis/t1a_fidelity"
B1 = ROOT / "experiments/phase3/b1_pe_axis"


def test_taxonomy_rows_match_table2():
    t = fd.taxonomy_rows(TAXO, B1)
    gaps = {m: round(t.loc[m, "gap_pp"], 2) for m in ["esm2", "esm1b", "esmc", "saprot", "prott5", "ankh", "protbert"]}
    assert gaps == {"esm2": 3.29, "esm1b": 1.55, "esmc": 3.04, "saprot": 2.82, "prott5": 5.49, "ankh": 7.44,
                    "protbert": 8.68}
    assert t.loc["esm1b", "n"] == 98 and t.loc["progen2", "n"] == 117 and t.loc["ankh", "n"] == 120
    assert t.loc["progen2", "curve"][0] < t.loc["progen2", "curve"][-1]   # causal: monotonic rise
