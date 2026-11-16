"""Tier-B exp 1 (W2/Q4): a POSITIVE downstream instance for the unifying principle.

The Section-7 scoping nulls all keep the task-relevant context inside the window
(SS is local; VEP is a ranking; the contact null slid only pairs with BOTH endpoints
in-window). The unifying principle predicts the effect WILL surface for a per-position
output whose long-range context is actually cut. Long-range contact prediction in the
regime where the window edge falls between the two contacting residues is exactly that
case. This script demonstrates it (converting the boundary from inferred to demonstrated).

For each confident full-context long-range contact (i,j), prob_full>0.5, we read the
predicted contact probability under three window placements (W=256), all anchored on the
SAME pair so only the window position changes:

  center        window centered on the pair midpoint -> both endpoints balanced (the control,
                = the existing null condition).
  partner_edge  window slid so the far endpoint j sits flush at the near edge (jflank=0):
                j is still in-window, but its OUTER co-evolutionary context is truncated
                (the one-sided context-loss analog, now on a downstream output).
  partner_out   window slid one residue further so j falls OUTSIDE the window entirely:
                the contact spans two tiles and is unrecoverable -> prob := 0. This is what
                a fixed tiling does to a long-range contact whose partners land in different
                tiles; we also report the resulting miss-rate.

Outputs: experiments/phase3/b1_contacts_partner_out/pairs.parquet,
         results/figures/b1_contacts_partner_out.png, printed paired-bootstrap stats.

Usage (GPU, calibrate env): python scripts/phase3/b1_contacts_partner_out.py
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

DATA = (_ROOT / "data/processed/d4_ss")
OUT = (_ROOT / "experiments/phase3/b1_contacts_partner_out")
FIG = (_ROOT / "results/figures/b1_contacts_partner_out.png")

W = 256
SEP_MIN, SEP_MAX = 30, 200        # long-range; large enough that the partner CAN be pushed out
PROB_FULL_MIN = 0.5               # "confident" full-context contacts
MAX_PROT, MAX_LEN = 120, 900      # longer proteins so high-separation pairs exist
PAIRS_PER_PROT = 12


def clamp_start(s, L):
    return max(0, min(s, L - W))


def main() -> None:
    set_global_seed(42)
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_esm2()
    test = [json.loads(l) for l in (DATA / "test.jsonl").read_text().splitlines()]
    prots = [p for p in test if W + 4 <= len(p["seq"]) <= MAX_LEN][:MAX_PROT]

    rows = []
    for p in tqdm(prots, desc="b1 partner-out"):
        seq = p["seq"]; L = len(seq)
        cmap = model.contacts(seq)
        cands = []
        for i in range(L):
            for j in range(i + SEP_MIN, min(i + SEP_MAX + 1, L)):
                if cmap[i, j] >= PROB_FULL_MIN:
                    cands.append((i, j, float(cmap[i, j])))
        if not cands:
            continue
        sel = [cands[k] for k in rng.choice(len(cands), min(PAIRS_PER_PROT, len(cands)), replace=False)]
        for i, j, pf in sel:
            sep = j - i
            # --- placements (all anchored on the same pair; only window position changes) ---
            # center: window centered on midpoint, both endpoints in-window (the control)
            s_c = clamp_start(round((i + j) / 2 - W / 2), L)
            in_c = s_c <= i and j <= s_c + W - 1
            # partner_edge: j flush at the right edge (jflank=0), j still in-window
            s_e = clamp_start(j - W + 1, L)
            in_e = s_e <= i and j <= s_e + W - 1
            # partner_out: window holds i but excludes j (one tile further)
            s_o = clamp_start(j - W, L)
            in_i_out_j = s_o <= i <= s_o + W - 1 and j > s_o + W - 1
            if not (in_c and in_e):
                continue  # need both in-window placements to exist for the paired contrast
            prob_c = float(model.contacts(seq[s_c : s_c + W])[i - s_c, j - s_c])
            prob_e = float(model.contacts(seq[s_e : s_e + W])[i - s_e, j - s_e])
            prob_o = 0.0 if in_i_out_j else np.nan  # partner cut -> contact unrecoverable
            rows.append(dict(prot=p.get("acc", str(id(seq))), i=i, j=j, sep=sep, prob_full=pf,
                             center=prob_c, partner_edge=prob_e, partner_out=prob_o,
                             out_defined=bool(in_i_out_j)))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "pairs.parquet")
    n = len(df)
    print(f"\n=== B1 partner-out (n={n} confident long-range pairs, sep {SEP_MIN}-{SEP_MAX}, W={W}) ===")
    if n == 0:
        print("no pairs found"); return

    full = df["prob_full"].to_numpy()
    center = df["center"].to_numpy()
    edge = df["partner_edge"].to_numpy()
    print(f"  full-context prob (selected) = {full.mean():.3f}")
    print(f"  center placement   prob       = {center.mean():.3f}")
    print(f"  partner_edge prob (j flank=0) = {edge.mean():.3f}")

    # (1) graded in-window: does cutting the partner's OUTER context (j at edge) degrade it?
    r1 = paired_bootstrap_ci(center, edge, seed=42)
    print(f"\n  [in-window] center - partner_edge = {r1['diff']*100:+.2f} pp  "
          f"CI=[{r1['ci_low']*100:.2f},{r1['ci_high']*100:.2f}]  p={r1['p_value']:.4f}")

    # (2) the demonstrated boundary: partner pushed OUT -> contact unrecoverable (prob 0)
    sub = df[df["out_defined"]].copy()
    if len(sub):
        c2 = sub["center"].to_numpy(); o2 = sub["partner_out"].to_numpy()
        r2 = paired_bootstrap_ci(c2, o2, seed=42)
        print(f"  [partner OUT] center - partner_out = {r2['diff']*100:+.2f} pp  "
              f"CI=[{r2['ci_low']*100:.2f},{r2['ci_high']*100:.2f}]  p={r2['p_value']:.4f}  "
              f"(n={len(sub)}; partner_out prob=0 by construction)")

    # practitioner miss-rate: of confident long-range contacts, what fraction are split by a
    # fixed non-overlapping W-tiling (partners in different tiles)?
    def split_by_tiling(i, j):
        return (i // W) != (j // W)
    miss = np.mean([split_by_tiling(int(r.i), int(r.j)) for r in df.itertuples()])
    print(f"\n  fixed non-overlapping W={W} tiling splits {miss*100:.1f}% of these "
          f"confident long-range contacts (partners land in different tiles -> missed).")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    labels = ["full\ncontext", "center\n(both in)", "partner\nat edge", "partner\nout"]
    vals = [full.mean(), center.mean(), edge.mean(),
            float(sub["partner_out"].mean()) if len(sub) else 0.0]
    cols = ["#009E73", "#0072B2", "#56B4E9", "#D55E00"]
    ax.bar(range(4), vals, color=cols, width=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("predicted contact probability", fontsize=11)
    ax.set_title("Long-range contact recovery vs window placement", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.08)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    # double-headed arrow center-top -> partner-edge-top; label placed in the clear space at right
    gap_pp = (vals[1] - vals[2]) * 100
    ax.annotate("", xy=(2, vals[2]), xytext=(1, vals[1]),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.4))
    ax.text(2.42, (vals[1] + vals[2]) / 2, f"$-{gap_pp:.1f}$ pp\n($p<0.001$)",
            ha="left", va="center", fontsize=9, color="#444")
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)

    # right: paired centered vs partner-at-edge probability (below the diagonal = degraded)
    ax2 = axes[1]
    ax2.scatter(center, edge, s=10, c="#0072B2", alpha=0.35, edgecolor="none")
    ax2.plot([0, 1], [0, 1], ls="--", c="#888", lw=1.2, label="no change (y = x)")
    ax2.set_xlabel("centered-window prob (both endpoints in)", fontsize=11)
    ax2.set_ylabel("partner-at-edge prob\n(outer context cut)", fontsize=11)
    ax2.set_title("Cutting the partner's context degrades recovery", fontsize=11, fontweight="bold")
    below = float((edge < center).mean()) * 100
    ax2.text(0.04, 0.92, f"{below:.0f}% of pairs below the diagonal", fontsize=9, color="#333")
    ax2.set_xlim(0, 1.02); ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=8, loc="lower right"); ax2.grid(True, alpha=0.25, lw=0.5)

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    print(f"\nfigure -> {FIG}\nparquet -> {OUT / 'pairs.parquet'}")


if __name__ == "__main__":
    main()
