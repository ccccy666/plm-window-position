"""Reviewer-1 RC1 (GPU): is the long-range-contact partner-out degradation W-specific?

Re-runs the partner-out contact probe (cf. scripts/phase3/b1_contacts_partner_out.py) at
W in {128, 384} and combines with the stored W=256 result. For each confident full-context
long-range contact (prob>0.5, sep 30-200) it reads the predicted contact probability under three
placements anchored on the same pair: pair centered (both in-window, ~ the null), partner flush at
the edge (its outer context cut, still in-window), and partner pushed outside (unrecoverable, 0).

Shows the in-window degradation (center - partner_edge) and the partner-out collapse hold across
window sizes -> the failure is a general windowing property, not a W=256 artifact.

Usage (GPU, calibrate env): python rebuttal/r1_partner_out_multiW.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[1]

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = _ROOT
sys.path.insert(0, str(ROOT / "src"))
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

DATA = ROOT / "data/processed/d4_ss/test.jsonl"
STORED256 = ROOT / "experiments/phase3/b1_contacts_partner_out/pairs.parquet"
OUTJSON = ROOT / "rebuttal/r1_partner_out_multiW.json"
OUTFIG = ROOT / "rebuttal/r1_partner_out_multiW.png"

SEP_MIN, SEP_MAX = 30, 200
PROB_FULL_MIN = 0.5
MAX_PROT, MAX_LEN = 120, 900
PAIRS_PER_PROT = 12
NEW_WS = [128, 384]


def clamp_start(s, L, W):
    return max(0, min(s, L - W))


def run_W(model, prots, cand_by_prot, W):
    rows = []
    for p in prots:
        seq = p["seq"]; L = len(seq)
        for (i, j, pf) in cand_by_prot[p["acc"]]:
            s_c = clamp_start(round((i + j) / 2 - W / 2), L, W)
            in_c = s_c <= i and j <= s_c + W - 1
            s_e = clamp_start(j - W + 1, L, W)
            in_e = s_e <= i and j <= s_e + W - 1
            s_o = clamp_start(j - W, L, W)
            in_i_out_j = s_o <= i <= s_o + W - 1 and j > s_o + W - 1
            if not (in_c and in_e):
                continue
            prob_c = float(model.contacts(seq[s_c:s_c + W])[i - s_c, j - s_c])
            prob_e = float(model.contacts(seq[s_e:s_e + W])[i - s_e, j - s_e])
            rows.append(dict(i=i, j=j, sep=j - i, prob_full=pf, center=prob_c,
                             partner_edge=prob_e, out_defined=bool(in_i_out_j)))
    return pd.DataFrame(rows)


def summarize(df, W):
    c = df["center"].to_numpy(); e = df["partner_edge"].to_numpy()
    r = paired_bootstrap_ci(c, e, seed=42)
    sub = df[df["out_defined"]]
    return dict(W=W, n=int(len(df)), full=float(df["prob_full"].mean()),
                center=float(c.mean()), partner_edge=float(e.mean()),
                gap_pp=r["diff"] * 100, ci_lo=r["ci_low"] * 100, ci_hi=r["ci_high"] * 100,
                p=r["p_value"], n_out_defined=int(len(sub)),
                pct_below_diag=float((e < c).mean()) * 100)


def main() -> None:
    set_global_seed(42)
    rng = np.random.default_rng(42)
    model = load_esm2()
    test = [json.loads(l) for l in DATA.read_text().splitlines()]
    prots = [p for p in test if 384 + 4 <= len(p["seq"]) <= MAX_LEN][:MAX_PROT]

    # select confident long-range candidate pairs ONCE per protein (W-independent)
    cand_by_prot = {}
    for p in tqdm(prots, desc="select candidates"):
        seq = p["seq"]; L = len(seq)
        cmap = model.contacts(seq)
        cands = [(i, j, float(cmap[i, j]))
                 for i in range(L) for j in range(i + SEP_MIN, min(i + SEP_MAX + 1, L))
                 if cmap[i, j] >= PROB_FULL_MIN]
        if cands:
            sel = [cands[k] for k in rng.choice(len(cands), min(PAIRS_PER_PROT, len(cands)),
                                                replace=False)]
            cand_by_prot[p["acc"]] = sel
        else:
            cand_by_prot[p["acc"]] = []

    summaries = []
    for W in NEW_WS:
        df = run_W(model, prots, cand_by_prot, W)
        s = summarize(df, W)
        summaries.append(s)
        print(f"\n=== W={W}: n={s['n']}  full={s['full']:.3f} center={s['center']:.3f} "
              f"edge={s['partner_edge']:.3f}  center-edge={s['gap_pp']:+.2f}pp "
              f"CI[{s['ci_lo']:.2f},{s['ci_hi']:.2f}] p={s['p']:.4f}  "
              f"({s['pct_below_diag']:.0f}% below diag)")

    # stored W=256 (same method), from the committed partner-out parquet
    d256 = pd.read_parquet(STORED256)
    s256 = summarize(d256.rename(columns={}), 256)
    summaries.append(s256)
    summaries = sorted(summaries, key=lambda d: d["W"])
    print(f"\n=== W=256 (stored): n={s256['n']} center-edge={s256['gap_pp']:+.2f}pp "
          f"CI[{s256['ci_lo']:.2f},{s256['ci_hi']:.2f}] p={s256['p']:.4f}")

    OUTJSON.write_text(json.dumps(summaries, indent=2, default=float))

    # ---- figure: grouped bars full/center/edge across W, with the gap annotated ----
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    Ws = [s["W"] for s in summaries]
    x = np.arange(len(Ws)); w = 0.26
    ax.bar(x - w, [s["full"] for s in summaries], w, color="#009E73", label="full context")
    ax.bar(x, [s["center"] for s in summaries], w, color="#0072B2", label="centered (both in)")
    ax.bar(x + w, [s["partner_edge"] for s in summaries], w, color="#56B4E9",
           label="partner at edge")
    for k, s in enumerate(summaries):
        ax.text(k + w, s["partner_edge"] + 0.01,
                f"$-${s['gap_pp']:.0f}pp", ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_xticks(x); ax.set_xticklabels([f"W={s['W']}\n(n={s['n']})" for s in summaries], fontsize=9)
    ax.set_ylabel("predicted contact probability", fontsize=11)
    ax.set_title("Partner-out degradation holds across window sizes", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0); ax.legend(fontsize=8, loc="upper right"); ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(OUTFIG, dpi=200)
    print(f"\njson -> {OUTJSON}\nfigure -> {OUTFIG}")


if __name__ == "__main__":
    main()
