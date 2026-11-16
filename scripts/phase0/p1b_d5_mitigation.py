"""D5 — inference-time mitigation of crop-position brittleness (no retraining).

On the controlled window-relocation probe, compare per-residue recovery under:
  - single_edge   : target at window edge (rel 0.1/0.9) — the bad case a residue hits.
  - single_center : target centered (rel 0.5) — the "center this residue" fix (upper bound).
  - multicrop     : symmetric multi-crop ENSEMBLE — average the masked softmax distribution
                    over placements {0.1..0.9}; the deployable fix (every residue gets ~center
                    treatment without centering each one individually).
Claim (D5): multicrop ~ center >> edge -> the ~10pp edge loss is recovered for free.

Usage: python scripts/phase0/p1b_d5_mitigation.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
GFP = (_ROOT / "data/processed/p1_0/gfp.fasta")
OUT = (_ROOT / "experiments/phase0_pilot/p1b")
FIG = (_ROOT / "results/figures/p1b_d5_mitigation.png")

WINDOWS = [256, 384]
PLACEMENTS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_TARGETS = 80
REC_MIN = 0.30
AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_fasta(text):
    seqs, cur = [], []
    for ln in text.splitlines():
        if ln.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur = []
        elif ln.strip():
            cur.append(ln.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def main() -> None:
    set_global_seed(42)
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_esm2()

    pool = [s for s in parse_fasta(SP.read_text()) + parse_fasta(GFP.read_text()) if set(s) <= AA]
    rng.shuffle(pool)
    margin = round(0.9 * max(WINDOWS))

    rows, n = [], 0
    for prot in tqdm(pool, desc="D5"):
        if n >= N_TARGETS:
            break
        if len(prot) < 2 * margin + 1:
            continue
        t = int(rng.integers(margin, len(prot) - margin))
        c0 = prot[t - max(WINDOWS) // 2 : t - max(WINDOWS) // 2 + max(WINDOWS)]
        if float(model.recovery_prob(c0, [max(WINDOWS) // 2])[0]) < REC_MIN:
            continue
        true_aa = prot[t]
        ti = model.aa_token_index(true_aa)
        for W in WINDOWS:
            dists = {}
            for r in PLACEMENTS:
                s = t - round(r * W)
                crop = prot[s : s + W]
                j = t - s
                dists[r] = model.masked_distribution(crop, [j])[0]  # (vocab,)
            edge = float(np.mean([dists[0.1][ti], dists[0.9][ti]]))
            center = float(dists[0.5][ti])
            ens = np.mean([dists[r] for r in PLACEMENTS], axis=0)  # ensemble distribution
            multicrop = float(ens[ti])
            rows.append(dict(target=n, W=W, single_edge=edge, single_center=center,
                             multicrop=multicrop))
        n += 1
    print(f"used {n} targets")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "d5_mitigation.parquet")

    print("\n=== D5 mitigation (mean recovery; paired vs single_edge baseline) ===")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    width = 0.25
    strategies = ["single_edge", "single_center", "multicrop"]
    for k, W in enumerate(WINDOWS):
        g = df[df.W == W]
        means = {s: g[s].mean() for s in strategies}
        bc = paired_bootstrap_ci(g["multicrop"].values, g["single_edge"].values, seed=42)
        rec_frac = (means["multicrop"] - means["single_edge"]) / (means["single_center"] - means["single_edge"] + 1e-9)
        print(f"  W={W}: edge={means['single_edge']:.3f} center={means['single_center']:.3f} "
              f"multicrop={means['multicrop']:.3f} | multicrop-edge={bc['diff']*100:+.2f}pp "
              f"(p={bc['p_value']:.3f}) | recovered_fraction={rec_frac:.2f}")
        ax.bar(np.arange(len(strategies)) + k * width, [means[s] for s in strategies],
               width, label=f"W={W}")
    ax.set_xticks(np.arange(len(strategies)) + width / 2)
    ax.set_xticklabels(strategies)
    ax.set(title="D5: symmetric multi-crop recovers the edge loss (no retraining)",
           ylabel="mean masked recovery")
    ax.legend()
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"\nfigure -> {FIG}")


if __name__ == "__main__":
    main()
