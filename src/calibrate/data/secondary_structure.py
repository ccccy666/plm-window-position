"""Per-residue secondary-structure labels from UniProt features (Phase 1 D4).

UniProt entries with experimental 3D structure carry "Helix" / "Beta strand" /
"Turn" features (1-based begin/end). We map to 3-state Q3: H (helix), E (strand),
C (everything else, incl. Turn). Used for the residue-level crop-position D4 probe.
"""
from __future__ import annotations

_TYPE_TO_Q3 = {"Helix": "H", "Beta strand": "E"}


def ss_labels(entry: dict) -> tuple[str, list[str]]:
    """(sequence, per-residue Q3 labels) from a UniProt JSON entry. Turn/other -> C."""
    seq = entry["sequence"]["value"]
    lab = ["C"] * len(seq)
    for f in entry.get("features", []):
        q3 = _TYPE_TO_Q3.get(f.get("type"))
        if q3 is None:
            continue
        loc = f["location"]
        b, e = loc["start"]["value"], loc["end"]["value"]
        if b is None or e is None:
            continue
        for i in range(b - 1, e):  # 1-based inclusive -> 0-based
            if 0 <= i < len(seq):
                lab[i] = q3
    return seq, lab
