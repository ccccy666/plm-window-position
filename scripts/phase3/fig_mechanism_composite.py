"""Figure 3 (camera-ready): testing the candidate mechanisms, from stored artifacts (no GPU).

(a) information-preserved cassette sweep (n=200): recovery relative to the centered placement
(b) cassette flank sweep: recovery vs position, one line per flank f
(c) controlled truncation-cliff probe: recovery vs absolute position
(d) model x sequence 2x2: center - edge fidelity gap
(e) attention mass on window-dropped residues, centered vs edge window
(f) fidelity vs window position for residues near the start, middle, and end of their domain

Values are pinned by tests/test_figdata.py. Output: results/figures/fig3_mechanism.png
Usage: python scripts/phase3/fig_mechanism_composite.py
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

PILOT = ROOT / "experiments/phase0_pilot/p1b/position_sweep.parquet"
FLANK = ROOT / "experiments/phase1_diagnosis/t1b_mechanism/mechanism.parquet"
CLIFF = ROOT / "experiments/phase3/b2_cliff/cliff.parquet"
RANDINIT = ROOT / "rebuttal/reviewer2/r2_randinit_randseq.json"
ATTN = ROOT / "experiments/phase3/a_mechgen/esm2/attention.parquet"
DOMAIN = ROOT / "experiments/phase3/c1_domain_offsets/residues.parquet"
OUT = ROOT / "results/figures/fig3_mechanism.png"
REL = [0.1, 0.3, 0.5, 0.7, 0.9]


def cliff_relative(n_boot: int = 2000, seed: int = 42):
    """Per-cassette recovery minus that cassette's in-crop mean (pp), with a paired bootstrap CI."""
    df = pd.read_parquet(CLIFF)
    piv = df.pivot_table(index="cass", columns="pos", values="recovery")
    in_cols = [p for p in piv.columns if p <= 1022]
    rel = piv.sub(piv[in_cols].mean(axis=1), axis=0).to_numpy() * 100
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rel), size=(n_boot, len(rel)))
    boot = rel[idx].mean(axis=1)
    return (piv.columns.to_numpy(), rel.mean(axis=0),
            np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0))


def main() -> None:
    st.apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(st.TEXT_WIDTH_IN, 4.35))
    (a, b, c), (d, e, f) = axes

    # (a) pilot cassette sweep
    prof = fd.pilot_cassette_profile(PILOT)
    x = prof.index.to_numpy()
    a.axhline(0, color=st.MUTED, lw=0.6, ls=":")
    a.fill_between(x, prof.ci_lo, prof.ci_hi, color="#d0d0d0", lw=0)
    a.plot(x, prof.mean_pp, color=st.INK, marker="o")
    a.set(xlabel="relative position of the cassette", ylabel="recovery vs. centered (pp)",
          xticks=[0.05, 0.35, 0.65, 0.95], ylim=(-1.5, 4.0))
    st.panel(a, "a", "Context fixed")

    # (b) flank sweep
    curves, _ = fd.flank_sweep(FLANK)
    for color, (flank, row) in zip(st.GRAY_RAMP, curves.iterrows()):
        b.plot(REL, row[REL].to_numpy(), color=color, marker="o")
        b.text(0.93, row[0.9], f"f = {flank}", va="center", fontsize=6.5, color=st.INK)
    b.set(xlabel="relative position of the cassette", ylabel="masked recovery",
          xticks=REL, xlim=(0.05, 1.08), ylim=(0, 0.7))
    st.panel(b, "b", "Context fixed: flank sweep")

    # (c) controlled cliff, relative to each cassette's in-crop mean
    pos, mean_rel, lo, hi = cliff_relative()
    c.axhline(0, color=st.MUTED, lw=0.6, ls=":")
    c.fill_between(pos, lo, hi, color="#d0d0d0", lw=0)
    c.plot(pos, mean_rel, color=st.INK, marker="o")
    c.axvline(1022, color=st.RANDOM_INIT, lw=0.9, ls="--")
    c.text(1180, 4.3, "training crop\n(1022)", fontsize=6.5, color=st.RANDOM_INIT, va="top")
    c.set(xlabel="absolute position of the target", ylabel="recovery vs. in-crop mean (pp)",
          ylim=(-5, 4.5), xticks=[300, 1100, 1900])
    st.panel(c, "c", "Crossing the training crop")

    # (d) learned, not architectural
    g = fd.randinit_gaps(RANDINIT)
    xs = np.arange(2)
    w = 0.36
    for off, model, color, label in ((-w / 2, "trained", st.TRAINED, "trained ESM-2"),
                                     (w / 2, "random_init", st.RANDOM_INIT, "random-init ESM-2")):
        vals = [g[f"{model}/{s}"] for s in ("natural", "random_seq")]
        ys = np.array([v["gap_pp"] for v in vals])
        err = np.array([[v["gap_pp"] - v["ci_lo"] for v in vals], [v["ci_hi"] - v["gap_pp"] for v in vals]])
        d.bar(xs + off, ys, w, color=color, label=label, yerr=err, ecolor=st.INK,
              error_kw=dict(lw=0.6, capsize=1.5))
        for xi, yi, top in zip(xs + off, ys, ys + err[1]):
            d.text(xi, top + 0.08, f"{yi:+.2f}", ha="center", va="bottom", fontsize=6.3)
    d.axhline(3.0, color=st.MUTED, lw=0.6, ls="--")
    d.text(1.45, 2.92, "benchmark tier (3 pp)", ha="right", va="top", fontsize=6.3, color=st.MUTED)
    d.set(xticks=xs, xticklabels=["natural\nsequences", "random\nsequences"],
          ylabel="center $-$ edge fidelity gap (pp)", ylim=(-0.2, 4.6))
    d.legend(loc="upper right")
    st.ygrid(d)
    st.panel(d, "d", "Training $\\times$ sequence")

    # (e) attention locality (slope chart)
    att = fd.attention_masses(ATTN)
    for model, color, label in (("trained", st.TRAINED, "trained"), ("random_init", st.RANDOM_INIT, "random-init")):
        ctr, edg = att[model]
        e.plot([0, 1], [ctr, edg], color=color, marker="o", markersize=5, label=label)
        e.text(-0.08, ctr, f"{ctr:.3f}", ha="right", va="center", fontsize=6.3)
        e.text(1.08, edg, f"{edg:.3f}", ha="left", va="center", fontsize=6.3)
    e.set(xticks=[0, 1], xticklabels=["centered\nwindow", "edge\nwindow"], xlim=(-0.45, 1.45),
          ylabel="attention mass on dropped residues", ylim=(0.1, 0.7))
    e.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
    st.panel(e, "e", "Attention locality")

    # (f) own domain
    prof_u = fd.domain_side_profiles(DOMAIN)
    for u, color, marker, label in ((0.1, st.DOMAIN_START, "o", "near domain start"),
                                    (0.5, st.DOMAIN_MID, "s", "domain midpoint"),
                                    (0.9, st.DOMAIN_END, "^", "near domain end")):
        f.plot(REL, prof_u.loc[u, REL].to_numpy(), color=color, marker=marker, label=label)
    f.set(xlabel="relative position in the window", ylabel="embedding fidelity",
          xticks=REL, ylim=(0.86, 0.985))
    f.legend(loc="lower center")
    st.panel(f, "f", "Own-domain context")

    for ax in (a, b, c, f):
        st.ygrid(ax)
    fig.tight_layout(h_pad=1.4, w_pad=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
