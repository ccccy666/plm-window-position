"""Tests for calibrate.mechgen — protocol helpers for the cross-backbone mechanism experiments.

The helpers must reproduce the protocol of the runs already in the paper (taxonomy, ESM-2 2x2, ESM-2
attention locality), so the new backbones are measured exactly the same way.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrate import mechgen as mg

ROOT = Path(__file__).resolve().parents[1]
TAXO = ROOT / "experiments/phase1_diagnosis/t1a_fidelity"


def test_window_start_matches_taxonomy_protocol():
    assert mg.window_start(300, 0.5) == 300 - 128
    assert mg.window_start(300, 0.1) == 300 - 26
    assert mg.window_start(300, 0.9) == 300 - 230


def test_dropped_mass_sums_attention_outside_the_window():
    row = np.full(1000, 1 / 1000)
    assert mg.dropped_mass(row, 100) == pytest.approx(1 - 256 / 1000)
    row2 = np.zeros(10)
    row2[[0, 5, 9]] = [0.2, 0.5, 0.3]
    assert mg.dropped_mass(row2, 3, w=4) == pytest.approx(0.5)  # keeps 3..6, drops 0 and 9


def test_residue_row_drops_special_tokens_and_renormalizes():
    tok_row = np.array([0.5, 0.1, 0.3, 0.1, 0.0])  # BOS, 3 residues, EOS
    out = mg.residue_row(tok_row, offset=1, length=3)
    assert out.sum() == pytest.approx(1.0)
    assert out == pytest.approx([0.2, 0.6, 0.2])


def test_preserved_position_ids_keep_every_token_at_its_full_context_index():
    ids = mg.preserved_position_ids(length=1000, start=300)
    assert len(ids) == 256 + 2
    assert ids[0] == 0 and ids[-1] == 1001          # [CLS] and [SEP] as in the full protein
    assert list(ids[1:4]) == [301, 302, 303]          # residue j sits at j + 1


def test_preserved_position_ids_offset_for_esm_style_embeddings():
    ids = mg.preserved_position_ids(length=1000, start=300, offset=2)  # HF ESM: <cls> sits at padding_idx + 1
    assert ids[0] == 2 and ids[1] == 303 and ids[-1] == 1003


def test_random_carriers_reuse_the_esm2_2x2_draw():
    seqs = ["A" * 500, "C" * 700]
    rc = mg.random_carriers(seqs)
    assert [len(s) for s in rc] == [500, 700]
    rs = np.random.default_rng(43)  # rebuttal/reviewer2/r2_randinit_randseq.py
    assert rc == ["".join(rs.choice(list("ACDEFGHIKLMNPQRSTVWY"), len(s))) for s in seqs]


def test_benchmark_targets_are_the_taxonomy_set():
    tg = mg.benchmark_targets()
    assert len(tg) == 120
    assert all(mg.MARGIN <= t < len(p) - mg.MARGIN for p, t in tg)
    lengths = sorted(len(p) for p, _ in tg)
    assert lengths[0] == 461 and lengths[-1] == 2303  # 3 proteins > 2000 aa (ProGen2 n = 117)
    assert sum(len(p) > 2000 for p, _ in tg) == 3


def test_gap_summary_reproduces_the_table2_esmc_gap():
    df = pd.read_parquet(TAXO / "fidelity_esmc.parquet")
    g = df.pivot_table(index="target", columns="rel_pos", values="fidelity")
    s = mg.gap_summary(g[0.5].values, g[[0.1, 0.9]].mean(axis=1).values)
    assert s["n"] == 120
    assert s["gap_pp"] == pytest.approx(3.04, abs=0.005)
    assert s["p"] < 0.001 and s["ci_lo"] > 0


def test_attention_summary_reports_edge_minus_center_and_correlation():
    dc = np.array([0.10, 0.20, 0.15, 0.12])
    de = dc + np.array([0.05, 0.10, 0.08, 0.02])
    fc = np.array([0.97, 0.96, 0.98, 0.97])
    fe = fc - np.array([0.02, 0.04, 0.03, 0.01])
    s = mg.attention_summary(dc, de, fc, fe)
    assert s["drop_edge_minus_center_pp"] == pytest.approx(6.25)
    assert s["fidelity_gap_pp"] == pytest.approx(2.5)
    assert s["corr_asym_vs_gap"] == pytest.approx(np.corrcoef(de - dc, fc - fe)[0, 1])


def test_attention_summary_correlation_permutation_p():
    rng = np.random.default_rng(0)
    asym = rng.normal(size=120)
    linked = mg.attention_summary(np.zeros(120), asym, 0.9 + 0.01 * asym + 0.002 * rng.normal(size=120),
                                  np.full(120, 0.9))
    noise = mg.attention_summary(np.zeros(120), asym, 0.9 + 0.01 * rng.normal(size=120), np.full(120, 0.9))
    assert linked["corr_p"] < 0.01
    assert noise["corr_p"] > 0.05
