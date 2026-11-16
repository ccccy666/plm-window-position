"""Figure 5 (camera-ready): when windowing cuts the context a task needs, from stored artifacts (no GPU).

(a) partner-out contact probe at W=256: full context / centered / partner at edge / partner out
(b) the same probe across window sizes W = 128 / 256 / 384
(c) share of confident long-range contacts split across tiles vs tile overlap and W
(d) recovered fraction of the edge loss by within-domain offset: domain-centered window vs aligned tiles

Also writes two appendix figures: per-pair partner-out scatter, and split rate vs contact separation.
Values are pinned by tests/test_figdata.py.
Usage: python scripts/phase3/fig_deployment_composite.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from calibrate import figdata as fd  # noqa: E402
from calibrate import figstyle as st  # noqa: E402

PAIRS = ROOT / "experiments/phase3/b1_contacts_partner_out/pairs.parquet"
MULTIW = ROOT / "rebuttal/r1_partner_out_multiW.json"
TILING = ROOT / "rebuttal/r1_tiling.json"
DOMAIN = ROOT / "experiments/phase3/c1_domain_offsets/residues.parquet"
FIGDIR = ROOT / "results/figures"
COND = (("full", "full context", st.FULL_CTX), ("center", "centered (both in)", st.CENTERED),
        ("partner_edge", "partner at edge", st.PARTNER_EDGE))


def bar_labels(ax, xs, ys, fmt, dy, fontsize=6.3):
    for x, y in zip(xs, ys):
        ax.text(x, y + dy, fmt.format(y), ha="center", va="bottom", fontsize=fontsize)


def main_figure() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(st.TEXT_WIDTH_IN, 4.6))
    (a, b), (c, d) = axes

    # (a) partner-out at W=256
    m = fd.partner_out_means(PAIRS)
    names = ["full", "center", "partner_edge", "partner_out"]
    ys = [m[k] for k in names]
    cols = [st.FULL_CTX, st.CENTERED, st.PARTNER_EDGE, st.MUTED]
    a.bar(range(4), ys, 0.62, color=cols)
    bar_labels(a, range(4), ys, "{:.2f}", 0.015)
    a.annotate("", xy=(1.66, m["partner_edge"]), xytext=(1.34, m["center"]),
               arrowprops=dict(arrowstyle="->", color=st.INK, lw=0.7))
    a.text(1.72, 0.70, f"$-${(m['center'] - m['partner_edge']) * 100:.1f} pp", ha="left", fontsize=6.5)
    a.set(xticks=range(4), xticklabels=["full\ncontext", "centered\n(both in)", "partner\nat edge",
                                        "partner\nout"], ylabel="predicted contact probability", ylim=(0, 1.05))
    st.ygrid(a)
    st.panel(a, "a", "Cutting a partner's context ($W{=}256$)")

    # (b) across window sizes
    mw = fd.partner_multiW(MULTIW)
    ws = sorted(mw)
    xs = np.arange(len(ws))
    wbar = 0.26
    for i, (key, label, color) in enumerate(COND):
        b.bar(xs + (i - 1) * wbar, [mw[w][key] for w in ws], wbar, color=color, label=label)
    for x, w in zip(xs, ws):
        b.text(x + wbar, mw[w]["partner_edge"] + 0.02, f"$-${mw[w]['gap_pp']:.0f}", ha="center",
               va="bottom", fontsize=6.3)
    b.set(xticks=xs, xticklabels=[f"$W{{=}}{w}$" for w in ws], ylabel="predicted contact probability",
          ylim=(0, 1.25), yticks=np.arange(0, 1.01, 0.2))
    b.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.8)
    st.ygrid(b)
    st.panel(b, "b", "Smaller windows cut more")

    # (c) tiling overlap x W
    t = fd.tiling_split_rates(TILING)
    ov = ["no overlap", "50% overlap", "75% overlap"]
    xs = np.arange(3)
    for i, (w, color) in enumerate(zip(sorted(t), st.GRAY_RAMP[1:])):
        vals = list(t[w])
        c.bar(xs + (i - 1) * wbar, vals, wbar, color=color, label=f"$W{{=}}{w}$")
        bar_labels(c, xs + (i - 1) * wbar, vals, "{:.1f}", 0.6, fontsize=5.8)
    c.set(xticks=xs, xticklabels=ov, ylabel="contacts split across tiles (%)", ylim=(0, 46))
    c.legend(loc="upper right")
    st.ygrid(c)
    st.panel(c, "c", "Overlap prevents splits")

    # (d) domain-aware windows
    rf = fd.domain_recovered_fraction(DOMAIN)
    us = [0.1, 0.25, 0.5, 0.75, 0.9]
    d.axhline(100, color=st.MUTED, lw=0.6, ls="--")
    d.axhline(0, color=st.MUTED, lw=0.6, ls=":")
    d.text(0.92, 102, "= centered on the residue", ha="right", va="bottom", fontsize=6.3, color=st.MUTED)
    d.text(0.92, -3, "= residue at the window edge", ha="right", va="top", fontsize=6.3, color=st.MUTED)
    for pol, label, color, marker in (("dom_center", "window centered on the domain", st.DOMAIN_MID, "s"),
                                      ("dom_start", "tile aligned to domain start", st.DOMAIN_START, "o"),
                                      ("dom_end", "tile aligned to domain end", st.DOMAIN_END, "^")):
        d.plot(us, [rf[pol][u] * 100 for u in us], color=color, marker=marker, label=label)
    d.set(xticks=us, xticklabels=["0.1\n(start)", "0.25", "0.5\n(mid)", "0.75", "0.9\n(end)"],
          xlabel="position of the residue within its domain", ylabel="edge loss recovered (%)",
          ylim=(-14, 118))
    d.legend(loc="lower center", bbox_to_anchor=(0.5, 0.12))
    st.ygrid(d)
    st.panel(d, "d", "Domain-aware windows")

    fig.tight_layout(h_pad=1.5, w_pad=1.6)
    fig.savefig(FIGDIR / "fig5_deployment.png")
    plt.close(fig)


def appendix_figures() -> None:
    df = pd.read_parquet(PAIRS)
    fig, ax = plt.subplots(figsize=(3.1, 3.0))
    ax.scatter(df.center, df.partner_edge, s=4, color=st.CENTERED, alpha=0.35, lw=0)
    ax.plot([0, 1], [0, 1], color=st.MUTED, lw=0.7, ls="--")
    ax.set(xlabel="centered window (both endpoints in)", ylabel="partner at the window edge",
           xlim=(0, 1.01), ylim=(0, 1.01))
    fig.savefig(FIGDIR / "figA_partner_scatter.png")
    plt.close(fig)

    sep = fd.split_by_separation(TILING)
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.plot(range(len(sep)), [p for _, _, p in sep], color=st.INK, marker="o")
    for i, (_, n, p) in enumerate(sep):
        ax.text(i + 0.1, p - 2, f"n={n}", ha="left", va="top", fontsize=6)
    ax.set(xticks=range(len(sep)), xticklabels=[s for s, _, _ in sep], ylim=(0, 62),
           xlabel="contact separation $|i-j|$ (residues)", ylabel="split by non-overlapping tiling (%)")
    st.ygrid(ax)
    fig.savefig(FIGDIR / "figA_split_separation.png")
    plt.close(fig)


if __name__ == "__main__":
    st.apply_style()
    main_figure()
    appendix_figures()
    print(f"-> {FIGDIR}/fig5_deployment.png, figA_partner_scatter.png, figA_split_separation.png")
