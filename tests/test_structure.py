"""Tests for calibrate.structure — AlphaFold-structure checks of the contact results (group B2)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrate import structure as st

ROOT = Path(__file__).resolve().parents[1]


def atom(serial, name, resname, resseq, x, b, chain="A"):
    """One fixed-column PDB ATOM record (y = z = 0)."""
    return (f"ATOM  {serial:5d} {name:<4s} {resname:3s} {chain:1s}{resseq:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{b:6.2f}           {name[0]:>1s}")


PDB = "\n".join([
    atom(1, "N", "ALA", 1, 9.0, 91.5), atom(2, "CA", "ALA", 1, 9.5, 91.5), atom(3, "CB", "ALA", 1, 10.0, 91.5),
    atom(4, "N", "GLY", 2, 13.0, 65.0), atom(5, "CA", "GLY", 2, 14.0, 65.0),
    atom(6, "N", "SER", 3, 15.0, 88.0), atom(7, "CA", "SER", 3, 15.5, 88.0), atom(8, "CB", "SER", 3, 16.0, 88.0),
    atom(9, "CA", "ALA", 4, 39.0, 90.0), atom(10, "CB", "ALA", 4, 40.0, 90.0),
    "TER", "END",
])


def test_parse_pdb_uses_cb_and_ca_for_glycine_and_reads_plddt():
    s = st.parse_pdb_cb(PDB)
    assert s["seq"] == "AGSA"
    assert s["coords"][:, 0] == pytest.approx([10.0, 14.0, 16.0, 40.0])
    assert s["plddt"] == pytest.approx([91.5, 65.0, 88.0, 90.0])


def test_label_pairs_contact_noncontact_unverifiable():
    s = st.parse_pdb_cb(PDB)
    lab = st.label_pairs(s["coords"], s["plddt"], np.array([0, 0, 0]), np.array([2, 1, 3]))
    assert list(lab) == ["contact", "unverifiable", "non-contact"]  # 6 A; Gly pLDDT 65; 30 A


def test_tiling_split_reproduces_the_r1_split_rates():
    pairs = pd.read_parquet(ROOT / "experiments/phase3/b1_contacts_partner_out/pairs.parquet")
    stored = json.loads((ROOT / "rebuttal/r1_tiling.json").read_text())["overlap_grid"]
    for w, name, frac in [(256, "non-overlap (s=W)", 1.0), (256, "50% overlap (s=W/2)", 0.5),
                          (128, "75% overlap (s=W/4)", 0.25)]:
        stride = max(1, int(round(w * frac)))
        rate = st.tiling_split(pairs.i.to_numpy(), pairs.j.to_numpy(), w, stride).mean()
        assert rate == pytest.approx(stored[str(w)][name]["miss_all"])
