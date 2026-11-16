"""P2 analysis: truncation cliff (H0b).

Primary test = BRCA1 (the only v0.1 long assay that straddles the 1022 crop):
ESM-2 zero-shot Spearman rho for in-crop vs out-of-crop mutations, within one
assay (controls for assay difficulty). Secondary = pooled cross-assay (confounded).
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SCORES = (_ROOT / "experiments/phase0_pilot/p2/scores.parquet")
FIG = (_ROOT / "results/figures/p2_truncation_cliff.png")
TAB = (_ROOT / "results/tables/p2_assay_rho.csv")
CROP = 1022


def rho(df):
    if len(df) < 10:
        return np.nan
    return spearmanr(df["esm_score"], df["dms_score"]).statistic


def boot_rho_diff(d_in, d_out, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    base = rho(d_in) - rho(d_out)
    diffs = []
    ai, ao = d_in.reset_index(drop=True), d_out.reset_index(drop=True)
    for _ in range(n):
        bi = ai.iloc[rng.integers(0, len(ai), len(ai))]
        bo = ao.iloc[rng.integers(0, len(ao), len(ao))]
        diffs.append(rho(bi) - rho(bo))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return base, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), min(1.0, p)


def main() -> None:
    df = pd.read_parquet(SCORES)

    # --- per-assay overall rho (sanity: is the scoring even working?) ---
    rows = []
    for a, g in df.groupby("DMS_id"):
        rows.append(dict(DMS_id=a, seq_len=g.seq_len.iloc[0], n=len(g),
                         n_out=(~g.in_crop).sum(), rho=rho(g),
                         rho_in=rho(g[g.in_crop]), rho_out=rho(g[~g.in_crop])))
    rt = pd.DataFrame(rows).sort_values("seq_len")
    TAB.parent.mkdir(parents=True, exist_ok=True)
    rt.to_csv(TAB, index=False)
    print("=== Per-assay Spearman rho (ESM-2 wt-marginals vs DMS) ===")
    print(rt.round(3).to_string(index=False))

    # --- primary: BRCA1 within-assay cliff ---
    brca = df[df.DMS_id.str.startswith("BRCA1")]
    d_in, d_out = brca[brca.in_crop], brca[~brca.in_crop]
    print(f"\n=== BRCA1 within-assay (n_in={len(d_in)}, n_out={len(d_out)}) ===")
    base, lo, hi, p = boot_rho_diff(d_in, d_out)
    print(f"  rho_in={rho(d_in):.3f}  rho_out={rho(d_out):.3f}")
    print(f"  delta_rho(in-out)={base:.3f}  CI=[{lo:.3f},{hi:.3f}]  p={p:.3f}  "
          f"(gate: |delta|>=0.10 & p<0.01)")

    # --- pooled cross-assay (confounded) ---
    pin, pout = df[df.in_crop], df[~df.in_crop]
    print(f"\n=== Pooled cross-assay (CONFOUNDED by protein/assay) ===")
    print(f"  rho_in(all in-crop, n={len(pin)})={rho(pin):.3f}  "
          f"rho_out(all out-crop, n={len(pout)})={rho(pout):.3f}")

    # --- figure: BRCA1 rho vs position bin ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    brca_s = brca.copy()
    brca_s["bin"] = (brca_s["pos"] // 200) * 200
    binned = brca_s.groupby("bin").apply(
        lambda g: pd.Series({"rho": rho(g), "n": len(g)}), include_groups=False
    ).reset_index()
    binned = binned[binned["n"] >= 20]
    axes[0].plot(binned["bin"], binned["rho"], marker="o", color="#0072B2")
    axes[0].axvline(CROP, ls="--", c="#d62728", label="train crop (1022 aa)")
    axes[0].set(title="BRCA1: Spearman ρ vs mutation position", xlabel="residue position",
                ylabel="Spearman ρ (200-aa bins)")
    axes[0].legend(); axes[0].grid(True, alpha=0.25, lw=0.5)
    # per-assay rho vs protein length; labels offset off the markers to avoid overlap (A10)
    axes[1].scatter(rt["seq_len"], rt["rho"], c=(rt["n_out"] > 0), cmap="coolwarm",
                    s=45, edgecolor="k", lw=0.4, zorder=3)
    rt2 = rt.reset_index(drop=True)
    # disambiguate duplicate gene names (e.g. two SPIKE / two POLG) by the first differing token
    from collections import defaultdict
    groups = defaultdict(list)
    for d in rt2["DMS_id"]:
        groups[d.split("_")[0]].append(d)
    label = {}
    for gene, ids in groups.items():
        if len(ids) == 1:
            label[ids[0]] = gene
            continue
        toks = [d.split("_") for d in ids]
        idx = next((j for j in range(1, min(len(t) for t in toks))
                    if len(set(t[j] for t in toks)) == len(toks)), 1)
        for d, t in zip(ids, toks):
            label[d] = f"{gene} ({t[idx]})"
    # crowded SPIKE pair (near-coincident in x and y): place labels to the RIGHT, in the
    # open space, off the markers and staggered up/down so neither touches a point (A10).
    OFF = {
        "SPIKE (expr)": (11, 5, "left", "bottom"),   # upper point (rho>0) -> label up-right
        "SPIKE (bind)": (11, -5, "left", "top"),     # lower point (rho<0) -> label down-right
    }
    for i, r in rt2.iterrows():
        lab = label[r["DMS_id"]]
        if lab in OFF:
            dx, dy, ha, va = OFF[lab]
        else:
            # default: alternate label offset up/down so near-coincident points don't collide
            dy = 9 if (i % 2 == 0) else -12
            dx, ha, va = 0, "center", ("bottom" if dy > 0 else "top")
        axes[1].annotate(lab, (r["seq_len"], r["rho"]), xytext=(dx, dy),
                         textcoords="offset points", fontsize=7, ha=ha, va=va)
    axes[1].axvline(CROP, ls="--", c="#d62728")
    axes[1].margins(x=0.12, y=0.18)
    axes[1].set(title="Per-assay ρ vs protein length", xlabel="protein length (aa)",
                ylabel="overall Spearman ρ")
    axes[1].grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    print(f"\nfigure -> {FIG}")


if __name__ == "__main__":
    main()
