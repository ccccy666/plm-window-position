"""ProtNIAH needle construction: insert a motif into a carrier at a relative
position, plus carrier/motif helpers (P1.0 / P1).

Positions are 0-based residue indices into the constructed sequence.
"""
from __future__ import annotations

import numpy as np

# 20 standard amino acids
AA = "ACDEFGHIKLMNPQRSTVWY"


def insert_needle(carrier: str, motif: str, rel_pos: float) -> tuple[str, list[int]]:
    """Insert `motif` into `carrier` at relative position `rel_pos` in [0, 1].

    Returns (new_sequence, inserted_indices) where inserted_indices are the
    0-based positions of the motif residues in the new sequence.
    """
    if not 0.0 <= rel_pos <= 1.0:
        raise ValueError(f"rel_pos must be in [0, 1], got {rel_pos}")
    insert_at = int(round(rel_pos * len(carrier)))
    new_seq = carrier[:insert_at] + motif + carrier[insert_at:]
    indices = list(range(insert_at, insert_at + len(motif)))
    return new_seq, indices


def make_nxst_motif(rng: np.random.Generator) -> str:
    """Sample an N-linked glycosylation sequon: N - X - [S/T], X != Proline."""
    x_choices = [a for a in AA if a != "P"]
    x = rng.choice(list(x_choices))
    st = rng.choice(["S", "T"])
    return f"N{x}{st}"


def shuffle_sequence(seq: str, rng: np.random.Generator) -> str:
    """Return a composition-preserving random permutation of `seq`."""
    chars = list(seq)
    perm = rng.permutation(len(chars))
    return "".join(chars[i] for i in perm)
