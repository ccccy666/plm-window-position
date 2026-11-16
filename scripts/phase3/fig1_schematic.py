"""Fig 1 - schematic of the two probes, drawn at print size with residues as beads (no data).

(a) Window relocation: the full protein (the full-context reference) above three windowed copies that place the
target T at relative positions r = 0.1, 0.5, 0.9; residues outside the window are hollow, so the cut context is
visible and aligned with the reference. (b) Sliding cassette: T and its native flank move as one unit inside a
fixed input of shuffled carrier residues, so the flank travels with T and only its position changes. Styled
with calibrate.figstyle like Figures 2-4 (6.5 in text width, 7-8 pt type); colors of T, flank and window frame
checked with the dataviz palette validator.

Output: results/figures/fig1_schematic.png
Usage: python scripts/phase3/fig1_schematic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from calibrate import figstyle as st  # noqa: E402

FIG = ROOT / "results/figures/fig1_schematic.png"

FRAME = "#0072B2"        # model input (window) outline
FRAME_FILL = "#EAF3FB"
RES = "#7F9BBD"          # native residue inside the input (neutral)
RES_EDGE = "#5E7DA3"
GHOST = "#B9C3CE"        # residue outside the input (cut): hollow outline
CARRIER = "#DCE1E7"      # shuffled carrier residue (neutral)
CARRIER_EDGE = "#BCC5CF"
FLANK = "#E69F00"        # native flank of the cassette (Okabe-Ito orange)
FLANK_EDGE = "#A97400"
TARGET = "#D62728"       # target residue T

R = 0.38                 # bead radius (bead spacing = 1)
N, T, WB = 22, 11, 9     # protein beads, target index, window width (odd: r = 0.5 is exact)
WC, F = 13, 2            # cassette-probe input width, flank half-width (beads)
XB = 30.0                # x of the first bead of panel (b)
ROWS = (-2.1, -3.95, -5.8)
STYLE = {
    "target": dict(fc=TARGET, ec="white", lw=0.6),
    "res": dict(fc=RES, ec=RES_EDGE, lw=0.35),
    "flank": dict(fc=FLANK, ec=FLANK_EDGE, lw=0.35),
    "carrier": dict(fc=CARRIER, ec=CARRIER_EDGE, lw=0.35),
    "ghost": dict(fc="white", ec=GHOST, lw=0.5),
}


def bead(ax, x, y, kind):
    ax.add_patch(Circle((x, y), R * (1.25 if kind == "target" else 1.0), zorder=6 if kind == "target" else 4,
                        **STYLE[kind]))


def frame(ax, x0, x1, y):
    pad = 0.55
    ax.add_patch(FancyBboxPatch((x0 - pad, y - pad), (x1 - x0) + 2 * pad, 2 * pad,
                                boxstyle="round,pad=0,rounding_size=0.5", fc=FRAME_FILL, ec=FRAME, lw=0.8, zorder=2))


def title(ax, x, head, sub):
    ax.text(x, 3.0, head, fontsize=8, fontweight="bold", color=st.INK, va="center")
    ax.text(x, 1.95, sub, fontsize=7, color=st.MUTED, va="center")


def main() -> None:
    st.apply_style()
    fig = plt.figure(figsize=(st.TEXT_WIDTH_IN, 2.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")
    lab = dict(ha="right", va="center", fontsize=7.5, color=st.INK)

    # (a) window relocation: the window moves over a fixed protein
    title(ax, -7.0, "(a) Window relocation", "the window moves over a fixed protein; $T$ sits at relative position $r$")
    for i in range(N):
        bead(ax, i, 0, "target" if i == T else "res")
    ax.text(-1.0, 0, "full context", **lab)
    ax.text(T, 0.55, "$T$", ha="center", va="bottom", fontsize=7, color=st.INK)
    for r, y in zip((0.1, 0.5, 0.9), ROWS):
        s = T - round(r * (WB - 1))
        frame(ax, s, s + WB - 1, y)
        for i in range(N):
            bead(ax, i, y, "target" if i == T else ("res" if s <= i < s + WB else "ghost"))
        ax.text(-1.0, y, f"$r={r}$", **lab)

    # (b) sliding cassette: T moves with its native flank inside a fixed input
    title(ax, XB - 5.3, "(b) Sliding cassette", "$T$ moves with its native flank in a fixed input")
    c0 = XB + (WC - 1) / 2 - F
    for k in range(2 * F + 1):
        bead(ax, c0 + k, 0, "target" if k == F else "flank")
    ax.text(XB - 1.0, 0, "cassette", **lab)
    ax.text(c0 + F, 0.55, "$T$", ha="center", va="bottom", fontsize=7, color=st.INK)
    for c, y, tag in zip((F, (WC - 1) // 2, WC - 1 - F), ROWS, ("start", "center", "end")):
        frame(ax, XB, XB + WC - 1, y)
        for i in range(WC):
            d = abs(i - c)
            bead(ax, XB + i, y, "target" if d == 0 else ("flank" if d <= F else "carrier"))
        ax.text(XB - 1.0, y, tag, **lab)

    # legend and signal
    def mk(kind, label):
        s = STYLE[kind]
        return Line2D([], [], ls="", marker="o", markersize=5.8 if kind == "target" else 5.0,
                      markerfacecolor=s["fc"], markeredgecolor=s["ec"], markeredgewidth=s["lw"] + 0.2, label=label)
    handles = [mk("target", "target $T$"), mk("res", "native residue"), mk("ghost", "cut by the window"),
               mk("flank", r"native flank ($\pm f$)"), mk("carrier", "shuffled carrier"),
               Patch(fc=FRAME_FILL, ec=FRAME, lw=0.8, label="model input")]
    xmid = (-7.4 + XB + WC + 0.2) / 2
    rounded = HandlerPatch(patch_func=lambda legend, orig_handle, xdescent, ydescent, width, height, fontsize:
                           FancyBboxPatch((-xdescent + 0.5, -ydescent + 0.5), width - 1, height - 1,
                                          boxstyle="round,pad=0,rounding_size=2.2"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(xmid, ROWS[-1] - 1.15),
              bbox_transform=ax.transData, ncol=6, frameon=False, fontsize=7, handlelength=1.3,
              handletextpad=0.45, columnspacing=1.1, borderaxespad=0, handler_map={Patch: rounded})
    ax.text(xmid, ROWS[-1] - 3.05, r"signal: embedding fidelity, $\cos(h_{\mathrm{window}}(T),\,h_{\mathrm{full}}(T))$,"
            r" or masked recovery of $T$", ha="center", va="center", fontsize=7, color=st.MUTED)

    ax.set_xlim(-7.4, XB + WC + 0.2)
    ax.set_ylim(ROWS[-1] - 3.5, 3.4)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"figure -> {FIG}")


if __name__ == "__main__":
    main()
