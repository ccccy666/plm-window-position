"""T2.A — combine the per-backbone remedy summaries into one figure + table.

Reads `experiments/phase2_remedy/{esm2,esmc}_remedy.json` (recovery-level, produced in
their own envs) and emits a cross-backbone recovered-fraction figure and a CSV.

Usage: python scripts/phase2/t2a_combine.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

EXP = (_ROOT / "experiments/phase2_remedy")
FIG = (_ROOT / "results/figures/t2a_remedy_combined.png")
TAB = (_ROOT / "results/tables/t2a_remedy.csv")
BACKBONES = ["esm2", "esmc"]
LABEL = {"esm2_650M": "ESM-2 650M", "esmc_300m": "ESM C 300M"}


def main() -> None:
    summaries = {}
    for b in BACKBONES:
        f = EXP / f"{b}_remedy.json"
        if f.exists():
            s = json.loads(f.read_text())
            summaries[s["backbone"]] = s
    if not summaries:
        print("no per-backbone summaries found; run t2a_remedy*.py first"); return

    rows = []
    for bb, s in summaries.items():
        for W, d in s["by_W"].items():
            for strat in ["multicrop_uniform", "multicrop_weighted"]:
                r = d["strategies"][strat]
                rows.append(dict(backbone=bb, W=int(W), strategy=strat.replace("multicrop_", ""),
                                 n=s["n_targets"], edge=round(d["edge"], 4),
                                 center=round(d["center"], 4), mean=round(r["mean"], 4),
                                 gain_pp=round(r["gain_pp"], 3), p=round(r["p"], 4),
                                 recovered_pct=round(r["recovered_fraction"] * 100, 1)))
    tab = pd.DataFrame(rows)
    TAB.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(TAB, index=False)
    print(tab.to_string(index=False))

    # figure: recovered fraction, grouped by (backbone, W), uniform vs weighted
    groups = [(bb, W) for bb in summaries for W in (256, 384)]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for j, strat in enumerate(["uniform", "weighted"]):
        vals = [tab[(tab.backbone == bb) & (tab.W == W) & (tab.strategy == strat)]
                ["recovered_pct"].iloc[0] for bb, W in groups]
        ax.bar(x + j * 0.4, vals, 0.4, label=f"multi-crop {strat}")
    ax.axhline(100, ls="--", c="k", lw=1, label="centering (upper bound)")
    ax.set_xticks(x + 0.2)
    ax.set_xticklabels([f"{LABEL.get(bb, bb)}\nW={W}" for bb, W in groups], fontsize=8)
    ax.set(ylabel="% of center−edge gap recovered")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    print(f"\ntable  -> {TAB}\nfigure -> {FIG}")


if __name__ == "__main__":
    main()
