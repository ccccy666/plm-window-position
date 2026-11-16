"""Camera-ready Figure 2: window-position behavior across eight pLMs (no GPU; stored artifacts only).

(a) Embedding fidelity versus the target's relative position for every backbone.
(b) Center-edge gap (95% paired-bootstrap CI) for the seven bidirectional encoders, colored by position-encoding
    class: the classes overlap (the two learned-absolute encoders are the least and the most affected).
Values are pinned by tests/test_figdata.py::test_taxonomy_rows_match_table2.
Usage: python scripts/phase3/fig_taxonomy.py   (calibrate env)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from calibrate import figdata as fd  # noqa: E402
from calibrate import figstyle as st  # noqa: E402

TAXO = ROOT / "experiments/phase1_diagnosis/t1a_fidelity"
B1 = ROOT / "experiments/phase3/b1_pe_axis"
OUT = ROOT / "results/figures/fig2_taxonomy.png"
REL = list(fd.TAXO_REL)
CLASS_COLOR = {"rotary": "#0072B2", "T5 relative": "#E69F00", "learned absolute": "#CC79A7", "causal": "#D55E00"}
MODELS = [  # (key, label, class, marker), grouped by class
    ("esm2", "ESM-2", "rotary", "o"), ("esmc", "ESM C", "rotary", "s"), ("saprot", "SaProt", "rotary", "^"),
    ("prott5", "ProtT5", "T5 relative", "D"), ("ankh", "Ankh", "T5 relative", "P"),
    ("protbert", "ProtBERT", "learned absolute", "v"), ("esm1b", "ESM-1b", "learned absolute", "X"),
    ("progen2", "ProGen2", "causal", "*"),
]


def main() -> None:
    st.apply_style()
    t = fd.taxonomy_rows(TAXO, B1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(st.TEXT_WIDTH_IN, 2.55), gridspec_kw=dict(width_ratios=[1.3, 1]))

    for key, label, cls, marker in MODELS:
        a.plot(REL, t.loc[key, "curve"], color=CLASS_COLOR[cls], marker=marker, markersize=4.5, lw=1.3)
    a.set(xlabel="relative position of the residue in the window", ylabel="embedding fidelity",
          xticks=REL, ylim=(0.76, 1.0))
    st.ygrid(a)
    st.panel(a, "a", "Fidelity across window placements")

    bidir = sorted([m for m in MODELS if m[2] != "causal"], key=lambda m: t.loc[m[0], "gap_pp"])
    for y, (key, label, cls, marker) in enumerate(bidir):
        g, lo, hi = t.loc[key, ["gap_pp", "ci_lo", "ci_hi"]]
        b.plot([lo, hi], [y, y], color=CLASS_COLOR[cls], lw=1.6, solid_capstyle="round")
        b.plot([g], [y], color=CLASS_COLOR[cls], marker=marker, markersize=5.5, linestyle="none")
        b.text(hi + 0.25, y, f"{g:+.2f}", va="center", fontsize=6.3, color=st.INK)
    b.set(yticks=range(len(bidir)), yticklabels=[m[1] for m in bidir], xlim=(0, 11),
          xlabel="center $-$ edge fidelity gap (pp)")
    b.grid(axis="x", color="#e6e6e6", lw=0.5)
    b.set_axisbelow(True)
    st.panel(b, "b", "Gap by encoder (95% CI)")

    blank = Line2D([], [], linestyle="none", marker="none", label=" ")
    columns = []                                   # one legend column per encoding class (column-major fill)
    for cls in CLASS_COLOR:
        col = [Line2D([], [], color=CLASS_COLOR[cls], marker=marker, markersize=4.5, lw=1.3,
                      label=f"{label} ({'causal, rotary' if cls == 'causal' else cls})")
               for key, label, c, marker in MODELS if c == cls]
        columns.append(col + [blank] * (3 - len(col)))
    fig.legend(handles=[h for col in columns for h in col], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.19), columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(w_pad=1.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"figure -> {OUT}")


if __name__ == "__main__":
    main()
