"""ProteinGym substitution scoring helpers (P2).

ESM-2 zero-shot **wt-marginals**: one forward on the (full) WT sequence, then
score(mut) = logP(mut | pos) - logP(wt | pos). For the >1022 truncation test we
feed the FULL sequence and let RoPE extrapolate. A windowed variant (1022-window
centered on the mutation) is the Risk-A contrast.
"""
from __future__ import annotations

import re

import numpy as np

_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

# ESM-2's max trained crop (positions beyond this were never seen in training).
TRAIN_CROP = 1022


def parse_mutant(mutant: str) -> tuple[str, int, str]:
    """Parse a single substitution like 'I291A' -> ('I', 291, 'A'); pos is 1-based.

    Raises ValueError on multi-mutants (containing ':') or malformed strings.
    """
    m = _MUT_RE.match(mutant.strip())
    if not m:
        raise ValueError(f"not a single substitution: {mutant!r}")
    return m.group(1), int(m.group(2)), m.group(3)


def score_full_sequence(model, seq: str, mutants: list[str]) -> list[float]:
    """wt-marginal scores from a single full-sequence forward (RoPE extrapolation)."""
    lp = model.position_logprobs(seq)  # (L, vocab)
    out = []
    for mut in mutants:
        wt, pos, alt = parse_mutant(mut)
        i = pos - 1
        out.append(float(lp[i, model.aa_token_index(alt)] - lp[i, model.aa_token_index(wt)]))
    return out


def score_windowed(model, seq: str, mutants: list[str], window: int = TRAIN_CROP) -> list[float]:
    """wt-marginal scores using a `window`-residue window centered on each mutation
    (Risk-A contrast: gives every mutation local context, expected to erase the cliff).
    Forwards are cached per window-start to avoid recompute.
    """
    cache: dict[int, np.ndarray] = {}
    out = []
    for mut in mutants:
        wt, pos, alt = parse_mutant(mut)
        i = pos - 1
        start = max(0, min(i - window // 2, len(seq) - window))
        start = max(start, 0)
        if start not in cache:
            cache[start] = model.position_logprobs(seq[start : start + window])
        lp = cache[start]
        j = i - start
        out.append(float(lp[j, model.aa_token_index(alt)] - lp[j, model.aa_token_index(wt)]))
    return out
