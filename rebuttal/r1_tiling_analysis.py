"""Reviewer-1 RC1 (no GPU): how often does a fixed tiling SPLIT a long-range contact,
as a function of tiling overlap, contact separation, and protein length?

A long-range contact (i,j) is recoverable only if some tile contains BOTH endpoints. For a
tiling of width W and stride s (s=W non-overlapping; s<W overlapping), the tile that best covers
j while containing i starts at floor(i/s)*s, so the contact is recoverable iff
    floor(i/s)*s + W - 1 >= j.
We report the miss rate (= 1 - recoverable) over the n=1178 confident long-range contacts from
the partner-out experiment, for several strides and window sizes, and stratified by separation
and by protein length.

Pure combinatorics on the existing pairs.parquet — no model, no GPU.
Usage: python rebuttal/r1_tiling_analysis.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[1]

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = _ROOT
PAIRS = ROOT / "experiments/phase3/b1_contacts_partner_out/pairs.parquet"
TEST = ROOT / "data/processed/d4_ss/test.jsonl"
OUTFIG = ROOT / "rebuttal/figures/r1_tiling.png"
OUTJSON = ROOT / "rebuttal/r1_tiling.json"


def miss_rate(i, j, W, s):
    """Fraction of contacts a stride-s, width-W tiling fails to co-tile (partners split)."""
    best_start = (i // s) * s                 # largest stride multiple <= i (best tile covering i)
    recoverable = best_start + W - 1 >= j     # does that tile also reach j?
    return not bool(recoverable)              # True == contact is split (missed)


def main() -> None:
    df = pd.read_parquet(PAIRS).copy()
    i = df["i"].to_numpy(); j = df["j"].to_numpy(); sep = df["sep"].to_numpy()
    # protein lengths joined by accession
    lens = {}
    for line in TEST.read_text().splitlines():
        r = json.loads(line); lens[r["acc"]] = len(r["seq"])
    df["L"] = df["prot"].map(lens)

    out = {"n_contacts": int(len(df)), "sep_range": [int(sep.min()), int(sep.max())]}

    # ---- (1) miss rate vs tiling overlap, for a few window sizes ----
    print("=== miss rate (contact split by a fixed tiling) vs stride/overlap ===")
    Ws = [128, 256, 384]
    strides = {"non-overlap (s=W)": 1.0, "50% overlap (s=W/2)": 0.5, "75% overlap (s=W/4)": 0.25}
    grid = {}
    for W in Ws:
        grid[W] = {}
        placeable = sep < W          # both endpoints can co-occur only if sep < W
        for name, frac in strides.items():
            s = max(1, int(round(W * frac)))
            m = np.array([miss_rate(int(a), int(b), W, s) for a, b in zip(i, j)])
            # among all confident long-range contacts (contacts with sep>=W can never co-tile)
            rate_all = float(m.mean())
            rate_placeable = float(m[placeable].mean()) if placeable.any() else float("nan")
            grid[W][name] = {"stride": s, "miss_all": rate_all, "miss_sep_lt_W": rate_placeable}
            print(f"  W={W:4d}  {name:22s} stride={s:4d}  miss(all)={rate_all*100:5.1f}%  "
                  f"miss(sep<W)={rate_placeable*100:5.1f}%")
    out["overlap_grid"] = grid

    # ---- (2) miss rate vs contact separation (non-overlap W=256) ----
    W = 256; s = 256
    m256 = np.array([miss_rate(int(a), int(b), W, s) for a, b in zip(i, j)])
    bins = [30, 60, 90, 120, 150, 200]
    sep_tab = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (sep >= lo) & (sep < hi)
        if sel.any():
            sep_tab.append({"sep_bin": f"{lo}-{hi}", "n": int(sel.sum()),
                            "miss_pct": round(float(m256[sel].mean()) * 100, 1)})
    out["miss_by_separation_W256_nonoverlap"] = sep_tab
    print("\n=== miss rate vs separation (non-overlap W=256) ===")
    for r in sep_tab:
        print(f"  sep {r['sep_bin']:8s} n={r['n']:4d}  miss={r['miss_pct']:5.1f}%")

    # ---- (3) miss rate vs protein length quartile (non-overlap W=256) ----
    q = pd.qcut(df["L"], 4, duplicates="drop")
    len_tab = []
    for lab, gidx in df.groupby(q, observed=True).groups.items():
        idx = df.index.get_indexer(gidx)
        len_tab.append({"len_bin": str(lab), "n": len(idx),
                        "miss_pct": round(float(m256[idx].mean()) * 100, 1)})
    out["miss_by_length_W256_nonoverlap"] = len_tab
    print("\n=== miss rate vs protein length quartile (non-overlap W=256) ===")
    for r in len_tab:
        print(f"  len {r['len_bin']:16s} n={r['n']:4d}  miss={r['miss_pct']:5.1f}%")

    OUTJSON.write_text(json.dumps(out, indent=2))

    # ---- figure ----
    OUTFIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    xlabels = list(strides.keys())
    x = np.arange(len(xlabels))
    cols = {128: "#56B4E9", 256: "#0072B2", 384: "#009E73"}
    wdt = 0.25
    for k, W in enumerate(Ws):
        vals = [grid[W][nm]["miss_all"] * 100 for nm in xlabels]
        ax.bar(x + (k - 1) * wdt, vals, wdt, color=cols[W], label=f"W={W}")
    ax.set_xticks(x); ax.set_xticklabels(["non-\noverlap", "50%\noverlap", "75%\noverlap"], fontsize=9)
    ax.set_ylabel("long-range contacts split by tiling (%)", fontsize=10)
    ax.set_title("Overlap fixes most tiling splits", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, title="window"); ax.grid(True, axis="y", alpha=0.25, lw=0.5)

    ax2 = axes[1]
    xs = [r["sep_bin"] for r in sep_tab]; ys = [r["miss_pct"] for r in sep_tab]
    ax2.plot(range(len(xs)), ys, "o-", color="#D55E00", lw=1.8, ms=7)
    ax2.set_xticks(range(len(xs))); ax2.set_xticklabels(xs, fontsize=9)
    ax2.set_xlabel("contact separation $|i-j|$ (residues)", fontsize=10)
    ax2.set_ylabel("split by non-overlap W=256 tiling (%)", fontsize=10)
    ax2.set_title("Longer-range contacts split more often", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(OUTFIG, dpi=200)
    print(f"\nfigure -> {OUTFIG}\njson   -> {OUTJSON}")


if __name__ == "__main__":
    main()
