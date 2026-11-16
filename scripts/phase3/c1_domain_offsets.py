"""Camera-ready experiment C1 (GPU): does a domain-containing window protect residues AWAY from the
domain midpoint, and is the residue's own domain the unit of informative context?

Follow-up to the Reviewer-3 domain-aware ablation (rebuttal/reviewer3/r3_domain_windowing.py), whose
targets were domain MIDPOINTS -- so its domain-aware window (centered on the domain) coincided with the
centered window (fidelity 0.9773 vs 0.9770) and could not separate "retain the domain" from "center
the residue".

PRE-REGISTERED (fixed before the run, 2026-11-16; do not change after seeing results)
-------------------------------------------------------------------------------------
Data    : UniProt curated `ft_domain` boundaries for the benchmark pool (cached in
          rebuttal/reviewer3/domains_cache.json; no network). Domains D=[d0,d1] (0-based, inclusive)
          with 50 <= |D| <= W-20. Residues at within-domain offsets u in {0.1, 0.25, 0.5, 0.75, 0.9}:
          t = d0 + round(u * (|D|-1)). A residue is kept only if every window below fits inside the
          protein: MARGIN <= t <= L-MARGIN (MARGIN = round(0.9 W)), d0 + W <= L, d1 - W + 1 >= 0, and
          the domain-centered window exists.
Model   : ESM-2 650M; embedding fidelity = cos(h_window(t), h_full(t)); W = 256 (as in the taxonomy).
Windows : position-only rel in {0.1, 0.3, 0.5, 0.7, 0.9}   (start = t - round(rel * W));
          dom_center = Reviewer-3 rule (window centered on D, clipped to contain D);
          dom_start  = window starting at d0   (tile boundary aligned to the domain start);
          dom_end    = window ending at d1     (tile boundary aligned to the domain end).
Unit    : protein (bootstrap resamples proteins; 10,000 resamples, seed 42).

Q1 (practical): for off-midpoint residues (u != 0.5), the recovered fraction of the edge->center gap,
    RF = (mean P - mean edge) / (mean center - mean edge), edge = mean(f(rel 0.1), f(rel 0.9)).
    Decision: a policy is recommended as a GENERAL policy iff RF >= 0.68 (the low end of the
    centrality-weighted multi-crop) AND its 95% bootstrap CI lower bound >= 0.50.
    Primary policy: dom_center (what the paper describes). Secondary: dom_start, dom_end.
    Sanity check: at u = 0.5, dom_center should reproduce the Reviewer-3 result (RF ~ 1).
Q2 (mechanism): domain-side asymmetry A(t) = f(rel 0.1) - f(rel 0.9). At u = 0.1 the domain lies to the
    right of t (kept by the rel-0.1 window, mostly cut by the rel-0.9 window); at u = 0.9 it lies to the
    left. DiD = A(u=0.1) - A(u=0.9), per domain with both residues, averaged per protein.
    Prediction if the residue's own domain is the informative unit: DiD > 0. Decision: p < 0.01.

Outputs: experiments/phase3/c1_domain_offsets/{residues.parquet, summary.json}
Usage  : python scripts/phase3/c1_domain_offsets.py [--dry]    (calibrate env; --dry = counts only)
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = _ROOT
sys.path.insert(0, str(ROOT / "src"))
from calibrate.seed import set_global_seed  # noqa: E402

from calibrate.domains import (  # noqa: E402  (design + analysis live in src; see tests/test_domains.py)
    REL, SEED, W, build_residues, summarize_domain,
)

OUT = ROOT / "experiments/phase3/c1_domain_offsets"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    set_global_seed(SEED)

    res = build_residues()
    print(f"eligible residues: {len(res)}  domains: {res.groupby(['acc', 'd0']).ngroups}  "
          f"proteins: {res['acc'].nunique()}")
    print(res.groupby("u").size().rename("n_residues").to_string())
    both = res[res.u.isin([0.1, 0.9])].groupby(["acc", "d0"])["u"].nunique()
    print(f"domains with both u=0.1 and u=0.9 (for Q2): {(both == 2).sum()}")
    if args.dry:
        return

    from tqdm import tqdm
    from calibrate.models import load_esm2

    model = load_esm2()
    full_cache, win_cache = {}, {}

    def full_emb(acc, seq):
        if acc not in full_cache:
            full_cache[acc] = model.embed([seq])[0]
        return full_cache[acc]

    def win_emb(acc, seq, s):
        key = (acc, s)
        if key not in win_cache:
            win_cache[key] = model.embed([seq[s:s + W]])[0]
        return win_cache[key]

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    out = []
    for r in tqdm(res.itertuples(index=False), total=len(res), desc="residues"):
        h_full = full_emb(r.acc, r.seq)[r.t]
        rec = dict(acc=r.acc, d0=r.d0, d1=r.d1, size=r.size, u=r.u, t=r.t, L=r.L)
        for rel in REL:
            s = r.t - round(rel * W)
            rec[f"f_{rel}"] = cos(win_emb(r.acc, r.seq, s)[r.t - s], h_full)
        for pol in ("dom_center", "dom_start", "dom_end"):
            s = getattr(r, f"s_{pol}")
            rec[f"f_{pol}"] = cos(win_emb(r.acc, r.seq, s)[r.t - s], h_full)
            rec[f"rel_{pol}"] = (r.t - s) / W
        out.append(rec)
    df = pd.DataFrame(out)
    df["edge"] = (df["f_0.1"] + df["f_0.9"]) / 2
    df["A"] = df["f_0.1"] - df["f_0.9"]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "residues.parquet")

    summary = summarize_domain(df)
    print("\n=== center - edge by within-domain offset u (per-protein means) ===")
    for u, r in summary["center_minus_edge"].items():
        print(f"  u={u:<4} n_prot={r['n_prot']:3d}  center-edge {r['gap_pp']:+.2f}pp "
              f"CI[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] p={r['p']:.4f}")
    print("\n=== Q1: recovered fraction of the edge->center gap (protein bootstrap) ===")
    for pol, q in summary["Q1"].items():
        print(f"  {pol:10s} off-midpoint RF={q['rf_off_midpoint']*100:5.1f}% "
              f"CI[{q['ci'][0]*100:5.1f},{q['ci'][1]*100:5.1f}] -> {q['verdict']}")
        print("             per u: " + "  ".join(
            f"u={u}: RF={v['rf']*100:5.1f}% (rel {v['mean_rel']:.2f})" for u, v in q["per_u"].items()))
    q2 = summary["Q2"]
    print("\n=== Q2: domain-side asymmetry A = f(rel .1) - f(rel .9) ===")
    for u, v in q2["mean_A_by_u"].items():
        print(f"  u={u:<4} mean A = {v:+.2f}pp")
    print(f"  DiD = {q2['did_pp']:+.2f}pp CI[{q2['ci'][0]:+.2f},{q2['ci'][1]:+.2f}] p={q2['p']:.4f} "
          f"(n_dom={q2['n_domains']}, n_prot={q2['n_prot']}) -> {q2['verdict']}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n-> {OUT}/residues.parquet, summary.json")


if __name__ == "__main__":
    main()
