"""Protocol helpers for the cross-backbone mechanism experiments (camera-ready group A).

Env-agnostic (numpy only), so the `esmc` and `progen` envs can import it as well. The constants and
the target selection mirror the taxonomy runners (scripts/phase1/t1a_fidelity*.py) and the ESM-2
2x2 / attention-locality runs (rebuttal/reviewer2/r2_*.py), so every backbone is measured the same way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from calibrate.stats import paired_bootstrap_ci

ROOT = Path(__file__).resolve().parents[2]
POOL_FASTA = ROOT / "data/processed/p1_0/swissprot_sample.fasta"
W = 256
REL = (0.1, 0.3, 0.5, 0.7, 0.9)
MARGIN = round(0.9 * W)
N_TARGETS = 120
AA = "ACDEFGHIKLMNPQRSTVWY"


def parse_fasta(text: str) -> list[str]:
    seqs, cur = [], []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        elif line.strip():
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def benchmark_targets(path: Path = POOL_FASTA, n: int = N_TARGETS, seed: int = 42) -> list[tuple[str, int]]:
    """The taxonomy target set: (protein, target index), non-terminal by MARGIN on both sides."""
    rng = np.random.default_rng(seed)
    pool = [s for s in parse_fasta(Path(path).read_text()) if set(s) <= set(AA)]
    rng.shuffle(pool)
    targets = []
    for prot in pool:
        if len(targets) >= n:
            break
        if len(prot) >= 2 * MARGIN + 1:
            targets.append((prot, int(rng.integers(MARGIN, len(prot) - MARGIN))))
    return targets


def random_carriers(seqs: list[str], seed: int = 43) -> list[str]:
    """Uniform i.i.d. amino-acid sequences of matched lengths (the ESM-2 2x2 draw)."""
    rs = np.random.default_rng(seed)
    return ["".join(rs.choice(list(AA), len(s))) for s in seqs]


def window_start(t: int, rel: float, w: int = W) -> int:
    """Start of the width-w window that puts residue t at relative position rel."""
    return t - round(rel * w)


def dropped_mass(row: np.ndarray, start: int, w: int = W) -> float:
    """Attention mass (over residue keys) on the residues the window [start, start + w) drops."""
    keep = np.zeros(len(row), dtype=bool)
    keep[start:start + w] = True
    return float(np.asarray(row)[~keep].sum())


def residue_row(token_row: np.ndarray, offset: int, length: int) -> np.ndarray:
    """Keep the residue keys of a token-level attention row and renormalize them to sum to 1."""
    row = np.asarray(token_row, dtype=float)[offset:offset + length]
    return row / (row.sum() + 1e-12)


def preserved_position_ids(length: int, start: int, w: int = W, offset: int = 0) -> np.ndarray:
    """Ids for [CLS] + window + [SEP] that keep each token's full-context index (+ offset, e.g. the
    padding_idx + 1 that HF ESM adds to every position)."""
    return np.concatenate([[0], np.arange(start + 1, start + w + 1), [length + 1]]) + offset


def gap_summary(center, edge) -> dict:
    """Center - edge paired-bootstrap summary in pp (fields as in r2_randinit_randseq.json)."""
    center, edge = np.asarray(center, float), np.asarray(edge, float)
    r = paired_bootstrap_ci(center, edge, seed=42)
    return dict(n=len(center), center=float(center.mean()), edge=float(edge.mean()),
                gap_pp=r["diff"] * 100, ci_lo=r["ci_low"] * 100, ci_hi=r["ci_high"] * 100,
                p=r["p_value"])


def attention_summary(drop_center, drop_edge, fid_center, fid_edge) -> dict:
    """Dropped attention mass (edge - center) and its per-target link to the fidelity gap."""
    dc, de = np.asarray(drop_center, float), np.asarray(drop_edge, float)
    fc, fe = np.asarray(fid_center, float), np.asarray(fid_edge, float)
    md = paired_bootstrap_ci(de, dc, seed=42)
    mf = paired_bootstrap_ci(fc, fe, seed=42)
    asym, gap = de - dc, fc - fe
    corr = float(np.corrcoef(asym, gap)[0, 1]) if len(dc) > 2 else float("nan")
    rng = np.random.default_rng(42)  # two-sided permutation test of the correlation
    perm = np.array([np.corrcoef(asym, rng.permutation(gap))[0, 1] for _ in range(10_000)])
    corr_p = float((np.sum(np.abs(perm) >= abs(corr)) + 1) / (len(perm) + 1))
    return dict(n=len(dc), drop_center=float(dc.mean()), drop_edge=float(de.mean()),
                drop_edge_minus_center_pp=md["diff"] * 100,
                drop_ci=[md["ci_low"] * 100, md["ci_high"] * 100], drop_p=md["p_value"],
                fidelity_gap_pp=mf["diff"] * 100, fidelity_p=mf["p_value"],
                corr_asym_vs_gap=corr, corr_p=corr_p)
