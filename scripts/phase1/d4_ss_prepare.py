"""D4-SS data prep: per-residue secondary structure from UniProt (Phase 1).

Fetches reviewed human proteins with experimental 3D structure, parses Q3 (H/E/C)
labels, splits into train (probe fit) / test (length>=480 for window-relocation eval).

Leakage note: SS labels come from PDB structures; ESM-2 may have seen these sequences.
This affects *absolute* probe accuracy but NOT the within-residue position-gap comparison
(same residue, different window position) — that is internally controlled. Family-aware
MMseqs2 split is the rigorous Phase-1 refinement; here we use a protein-level split.

Usage: python scripts/phase1/d4_ss_prepare.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import json
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.data.secondary_structure import ss_labels
from calibrate.seed import set_global_seed

OUT = (_ROOT / "data/processed/d4_ss")
AA = set("ACDEFGHIKLMNPQRSTVWY")
TEST_MIN_LEN = 480   # need room to place targets at window edges (W=256)
N_TEST = 100


def main() -> None:
    set_global_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    url = ("https://rest.uniprot.org/uniprotkb/search?"
           "query=(reviewed:true)+AND+(organism_id:9606)+AND+(structure_3d:true)+AND+(length:[150 TO 900])"
           "&format=json&fields=accession,sequence,ft_helix,ft_strand,ft_turn&size=500")
    print("fetching UniProt SS proteins ...")
    r = requests.get(url, timeout=180); r.raise_for_status()
    results = r.json()["results"]
    print(f"  {len(results)} raw entries")

    prots = []
    for e in results:
        seq, lab = ss_labels(e)
        if set(seq) <= AA and (lab.count("H") + lab.count("E")) >= 0.15 * len(seq):
            prots.append(dict(acc=e["primaryAccession"], seq=seq, labels="".join(lab)))
    print(f"  {len(prots)} clean proteins with >=15% H/E")

    rng = np.random.default_rng(42)
    rng.shuffle(prots)
    test = [p for p in prots if len(p["seq"]) >= TEST_MIN_LEN][:N_TEST]
    test_accs = {p["acc"] for p in test}
    train = [p for p in prots if p["acc"] not in test_accs]

    for name, split in [("train", train), ("test", test)]:
        with (OUT / f"{name}.jsonl").open("w") as f:
            for p in split:
                f.write(json.dumps(p) + "\n")
    # label distribution sanity
    allchars = "".join(p["labels"] for p in train)
    dist = {c: round(allchars.count(c) / len(allchars), 3) for c in "HEC"}
    print(f"train={len(train)} proteins, test={len(test)} proteins (len>={TEST_MIN_LEN})")
    print(f"train Q3 distribution: {dist}")


if __name__ == "__main__":
    main()
