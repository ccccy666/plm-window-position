"""Combine per-backbone embedding-fidelity parquets into the cross-architecture taxonomy
figure + summary (T1.A). Shape: bidirectional = inverted-U (center-edge gap > 0, non-monotonic);
causal = monotonic (Spearman(rel, fidelity) ~ +1)."""
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.stats import paired_bootstrap_ci

OUT = (_ROOT / "experiments/phase1_diagnosis/t1a_fidelity")
FIG = (_ROOT / "results/figures/t1a_taxonomy.png")
REL = [0.1, 0.3, 0.5, 0.7, 0.9]
ORDER = ["esm2", "esmc", "saprot", "prott5", "protbert", "progen2"]
KIND = {"esm2": "bidir RoPE", "esmc": "bidir RoPE", "saprot": "bidir RoPE (struct)",
        "prott5": "bidir relative-PE", "protbert": "bidir learned-PE", "progen2": "causal RoPE"}
# publication labels / distinct markers + colorblind-friendly colors (A4/A5)
NAME = {"esm2": "ESM-2 650M (RoPE)", "esmc": "ESM C 300M (RoPE)", "saprot": "SaProt 650M (RoPE, struct)",
        "prott5": "ProtT5 (relative PE)", "protbert": "ProtBERT (learned PE)", "progen2": "ProGen2 (causal)"}
MARKER = {"esm2": "o", "esmc": "s", "saprot": "^", "prott5": "D", "protbert": "v", "progen2": "X"}
COLOR = {"esm2": "#0072B2", "esmc": "#56B4E9", "saprot": "#009E73", "prott5": "#E69F00",
         "protbert": "#CC79A7", "progen2": "#D55E00"}


def main():
    parts = []
    for f in ["fidelity_both.parquet", "fidelity_esmc.parquet", "fidelity_progen.parquet",
              "fidelity_protbert.parquet", "fidelity_prott5.parquet"]:
        p = OUT / f
        if p.exists():
            parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    print("=== cross-architecture taxonomy (embedding-fidelity vs window position) ===")
    print(f"{'backbone':<10}{'kind':<22}{'center-edge(pp)':>16}{'p':>8}{'Spearman(rel,fid)':>20}")
    for b in ORDER:
        sub = df[df.backbone == b]
        if sub.empty:
            continue
        g = sub.pivot_table(index="target", columns="rel_pos", values="fidelity")
        res = paired_bootstrap_ci(g[0.5].values, g[[0.1, 0.9]].mean(axis=1).values, seed=42)
        means = g.mean(axis=0)
        rho = spearmanr(REL, [means[r] for r in REL]).statistic
        print(f"{b:<10}{KIND[b]:<22}{res['diff']*100:>+15.2f}{res['p_value']:>8.3f}{rho:>+20.2f}")
        ax.plot(REL, [means[r] for r in REL], marker=MARKER[b], color=COLOR[b], ms=6,
                lw=1.8, label=NAME[b])
    ax.set(xlabel="target relative position in window",
           ylabel="cosine to full-context embedding")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=200)
    print(f"\nfigure -> {FIG}")
    print("read: bidirectional = inverted-U (gap>0, Spearman~0); causal = monotonic (Spearman~+1)")


if __name__ == "__main__":
    main()
