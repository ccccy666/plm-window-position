"""B-tier (B4) — multi-seed stability of the Position Gap (ESM-2 anchor), for the appendix.

The main-text gaps carry within-seed bootstrap CIs; this adds the between-seed (target-sampling)
variance by re-sampling the n=120 target set under seeds {42,43,44,45,46} and recomputing the
embedding-fidelity Position Gap each time. Reports mean ± std across seeds and stores the fidelities
and metrics in experiments/phase3/b4_multiseed/ (fidelity.parquet, seeds.json).

Usage: HF_HUB_OFFLINE=1 python scripts/phase3/b4_multiseed.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import sys
from pathlib import Path

import json

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate import metrics
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed

SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
W, REL, MARGIN, N = 256, [0.1, 0.3, 0.5, 0.7, 0.9], round(0.9 * 256), 120
SEEDS = [42, 43, 44, 45, 46]
OUT = Path(__file__).resolve().parents[2] / "experiments/phase3/b4_multiseed"
AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_fasta(text):
    s, c = [], []
    for l in text.splitlines():
        if l.startswith(">"):
            if c:
                s.append("".join(c)); c = []
        elif l.strip():
            c.append(l.strip())
    if c:
        s.append("".join(c))
    return s


def main() -> None:
    model = load_esm2()
    embed = lambda seq: model.embed([seq])[0]
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    pool0 = [s for s in parse_fasta(SP.read_text()) if set(s) <= AA]

    print(f"{'seed':>6}{'PositionGap(pp)':>18}{'ShiftSens':>12}{'RobustAcc':>12}")
    gaps, shifts, robs, fid = [], [], [], []
    for seed in SEEDS:
        set_global_seed(seed); rng = np.random.default_rng(seed)
        pool = list(pool0); rng.shuffle(pool)
        targets = []
        for prot in pool:
            if len(targets) >= N:
                break
            if len(prot) >= 2 * MARGIN + 1:
                targets.append((prot, int(rng.integers(MARGIN, len(prot) - MARGIN))))
        rows = []
        for prot, t in tqdm(targets, desc=f"seed{seed}", leave=False):
            full = embed(prot)[t]
            rows.append([cos(embed(prot[t - round(r * W): t - round(r * W) + W])[round(r * W)], full)
                         for r in REL])
        M = np.asarray(rows)
        fid += [dict(seed=seed, target=i, rel=r, fidelity=float(M[i, j]))
                for i in range(len(M)) for j, r in enumerate(REL)]
        g = metrics.position_gap(M, REL) * 100
        s = metrics.shift_sensitivity(M); rb = metrics.robust_accuracy(M)
        gaps.append(g); shifts.append(s); robs.append(rb)
        print(f"{seed:>6}{g:>18.2f}{s:>12.4f}{rb:>12.4f}")

    print(f"\nPosition Gap over seeds {SEEDS}: mean={np.mean(gaps):.2f}pp  std={np.std(gaps, ddof=1):.2f}pp")
    print(f"Shift Sensitivity: mean={np.mean(shifts):.4f}  std={np.std(shifts, ddof=1):.4f}")
    print(f"Robust Accuracy:   mean={np.mean(robs):.4f}  std={np.std(robs, ddof=1):.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fid).to_parquet(OUT / "fidelity.parquet")
    stat = lambda v: dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)))
    (OUT / "seeds.json").write_text(json.dumps(dict(
        seeds=SEEDS, n_targets=N, W=W, rel=REL,
        per_seed=[dict(seed=s, position_gap_pp=float(g), shift_sensitivity=float(h), robust_accuracy=float(b))
                  for s, g, h, b in zip(SEEDS, gaps, shifts, robs)],
        position_gap_pp=stat(gaps), shift_sensitivity=stat(shifts), robust_accuracy=stat(robs)), indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
