"""Shared matplotlib style for the camera-ready composite figures (Okabe-Ito palette, print sizing)."""
from __future__ import annotations

import matplotlib as mpl

TEXT_WIDTH_IN = 6.5          # TMLR text width; figures are drawn at 1:1 print scale

# Entities keep one color across panels (validated with the dataviz palette checker).
TRAINED = "#0072B2"          # trained ESM-2
RANDOM_INIT = "#D55E00"      # randomly-initialized ESM-2 (same architecture)
FULL_CTX = "#009E73"         # full-context reference
CENTERED = "#0072B2"         # centered window (deployment figure)
PARTNER_EDGE = "#56B4E9"     # partner flush at the window edge
DOMAIN_START = "#E69F00"     # residue near the start of its domain
DOMAIN_MID = "#009E73"       # residue at the domain midpoint
DOMAIN_END = "#CC79A7"       # residue near the end of its domain
INK = "#222222"
MUTED = "#6b6b6b"
GRAY_RAMP = ("#c4c4c4", "#8f8f8f", "#5c5c5c", "#1f1f1f")   # ordinal (light -> dark)


def apply_style() -> None:
    mpl.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def panel(ax, letter: str, title: str) -> None:
    """Bold panel letter plus a short panel title, left-aligned above the axes."""
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold", pad=4)


def ygrid(ax) -> None:
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.5)
    ax.set_axisbelow(True)
