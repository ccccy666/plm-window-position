"""Fig 1 - conceptual schematic of the crop-position probe (paper front matter).

Redesigned for a clean, aligned layout (all annotation text sits OUTSIDE the sequence
bars, in fixed columns): (a) full-context reference; (b) window-relocation placing the
SAME target at rel-pos {0.1,0.5,0.9} with one-sided context truncation; (c) the
embedding-fidelity signal; (d) previewed taxonomy (bidirectional U vs causal monotonic).

Pure schematic (no data). Output: results/figures/fig1_schematic.png
Usage: python scripts/phase3/fig1_schematic.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FIG = (_ROOT / "results/figures/fig1_schematic.png")

L = 100.0          # abstract protein length
W = 42.0           # abstract window width
pT = 56.0          # target residue position
BIDIR = "#0072B2"  # bidirectional (blue)
CAUSAL = "#D55E00" # causal (orange)
TARGET = "#D62728" # target residue (red)
WIN = "#9ecae1"    # window fill
SEQ = "#e9e9e9"    # full sequence (light)
NOTE = "#666666"


def seq_bar(ax, y, h=0.36, fc=SEQ):
    ax.add_patch(Rectangle((0, y), L, h, facecolor=fc, edgecolor="#bdbdbd", lw=0.8, zorder=1))


def target(ax, y, h=0.36):
    ax.add_patch(Rectangle((pT - 0.8, y - 0.12), 1.6, h + 0.24, facecolor=TARGET, ec="none", zorder=4))


def main() -> None:
    fig = plt.figure(figsize=(11, 4.7))
    gs = GridSpec(2, 2, width_ratios=[1.6, 1.0], height_ratios=[1, 1],
                  wspace=0.18, hspace=0.45, figure=fig)

    # ---------------- left: windowing schematic (spans both rows) ----------------
    axL = fig.add_subplot(gs[:, 0]); axL.axis("off")
    axL.set_xlim(-34, L + 40); axL.set_ylim(0.2, 6.2)
    NOTE_X = L + 2.5                                   # fixed column for right-side notes

    # (a) full-context reference
    ya = 5.4
    axL.text(0, ya + 0.55, r"(a) full context $\rightarrow$ reference embedding $h_{\mathrm{full}}(T)$",
             fontsize=10, fontweight="bold")
    seq_bar(axL, ya); target(axL, ya)
    axL.text(pT, ya - 0.42, "target $T$", ha="center", color=TARGET, fontsize=9)

    # (b) window-relocation
    axL.text(0, 4.35, r"(b) window-relocation: place $T$ at relative position $r$",
             fontsize=10, fontweight="bold")
    rows = [(0.1, 3.45, "$r=0.1$ (edge)", "left context cut"),
            (0.5, 2.35, "$r=0.5$ (center)", "both sides kept"),
            (0.9, 1.25, "$r=0.9$ (edge)", "right context cut")]
    for r, y, lab, note in rows:
        s = pT - r * W
        seq_bar(axL, y)                                                  # full sequence (light)
        x0 = max(0.0, s); w = min(L, s + W) - x0
        axL.add_patch(Rectangle((x0, y), w, 0.36, facecolor=WIN, edgecolor=BIDIR,
                                lw=1.6, zorder=2))                       # window
        target(axL, y)
        axL.text(-18, y + 0.18, lab, fontsize=8.5, ha="left", va="center")
        axL.text(NOTE_X, y + 0.18, note, fontsize=8, color=NOTE, va="center", style="italic")

    # slide arrow in a dedicated far-left column, clear of the row labels (which start at x=-18)
    axL.add_patch(FancyArrowPatch((-29, 3.55), (-29, 1.45), arrowstyle="-|>",
                                  mutation_scale=13, color="#555555", lw=1.4))
    axL.text(-32, 2.5, "slide", fontsize=8.5, rotation=90, va="center", ha="center", color="#555555")

    # ---------------- right top: (c) per-residue signal ----------------
    axc = fig.add_subplot(gs[0, 1]); axc.axis("off")
    axc.set_xlim(0, 1); axc.set_ylim(0, 1)
    axc.add_patch(FancyBboxPatch((0.03, 0.12), 0.94, 0.78, boxstyle="round,pad=0.02",
                                 facecolor="#f4f6f8", edgecolor="#cccccc", lw=1.0))
    axc.text(0.5, 0.74, "(c) per-residue signal", ha="center", fontsize=10, fontweight="bold")
    axc.text(0.5, 0.52, r"fidelity $= \cos(\,h_{\mathrm{window}}(T),\ h_{\mathrm{full}}(T)\,)$",
             ha="center", fontsize=10.5)
    axc.text(0.5, 0.28, "same residue and local biology;\nonly the window position changes",
             ha="center", fontsize=8.2, color=NOTE)

    # ---------------- right bottom: (d) previewed taxonomy ----------------
    axd = fig.add_subplot(gs[1, 1])
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    u = 1 - 2.6 * (x - 0.5) ** 2          # inverted-U (bidirectional)
    m = 0.55 + 0.5 * x                    # monotonic (causal)
    axd.plot(x, u, "-o", color=BIDIR, ms=5, lw=1.8, label="bidirectional (inverted-U)")
    axd.plot(x, m, "-s", color=CAUSAL, ms=5, lw=1.8, label="causal (monotonic)")
    axd.set_xlabel("relative position in window", fontsize=9)
    axd.set_ylabel("signal", fontsize=9)
    axd.set_title("(d) previewed taxonomy", fontsize=10, fontweight="bold")
    axd.tick_params(labelsize=7.5); axd.set_yticks([]); axd.set_xticks([0.1, 0.5, 0.9])
    axd.legend(fontsize=7.5, loc="lower center"); axd.margins(y=0.2)
    axd.grid(True, alpha=0.2, lw=0.5)

    fig.suptitle("Does a residue's position within the inference window change a protein LM's behavior?",
                 fontsize=12, y=0.99)
    fig.savefig(FIG, dpi=200, bbox_inches="tight")
    print(f"figure -> {FIG}")


if __name__ == "__main__":
    main()
