"""Structure checks for the contact results (camera-ready group B2): AlphaFold DB models.

Env-agnostic (numpy only). A contact is structure-verified when the C-beta distance (C-alpha for
glycine) is below 8 A and both residues are confidently modelled (pLDDT >= 70).
"""
from __future__ import annotations

import numpy as np

AA3 = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I",
           LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")


def parse_pdb_cb(text: str, chain: str = "A") -> dict:
    """Sequence, C-beta (C-alpha for Gly) coordinates and per-residue pLDDT (B-factor) of one chain."""
    res = {}
    for line in text.splitlines():
        if not line.startswith("ATOM") or line[21] != chain:
            continue
        name, resname, key = line[12:16].strip(), line[17:20], int(line[22:26])
        want = "CA" if resname == "GLY" else "CB"
        rec = res.setdefault(key, dict(aa=AA3.get(resname, "X"), xyz=None, b=float(line[60:66])))
        if name == want:
            rec["xyz"] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    keys = sorted(res)
    coords = np.array([res[k]["xyz"] if res[k]["xyz"] is not None else (np.nan,) * 3 for k in keys])
    return dict(seq="".join(res[k]["aa"] for k in keys), coords=coords,
                plddt=np.array([res[k]["b"] for k in keys]))


def label_pairs(coords, plddt, i, j, cutoff: float = 8.0, min_plddt: float = 70.0) -> np.ndarray:
    """'contact' / 'non-contact' when both residues have pLDDT >= min_plddt, else 'unverifiable'."""
    i, j = np.asarray(i), np.asarray(j)
    d = np.linalg.norm(coords[i] - coords[j], axis=1)
    ok = (plddt[i] >= min_plddt) & (plddt[j] >= min_plddt) & np.isfinite(d)
    return np.where(~ok, "unverifiable", np.where(d < cutoff, "contact", "non-contact"))


def tiling_split(i, j, w: int, stride: int) -> np.ndarray:
    """True where a stride-`stride`, width-`w` tiling puts the pair in no common tile (R1 rule)."""
    i, j = np.asarray(i), np.asarray(j)
    return (i // stride) * stride + w - 1 < j
