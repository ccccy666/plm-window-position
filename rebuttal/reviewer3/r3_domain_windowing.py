"""Reviewer-3 RC, step 2 (GPU): domain-aware windowing ablation (embedding fidelity, ESM-2).

Reads rebuttal/reviewer3/targets.json (built by r3_fetch_domains.py).

PRIMARY (requested ablation) on `containable` targets (residue = midpoint of a curated domain D,
|D| <= W): embedding fidelity (cosine of windowed vs full-context embedding) under
  edge          : position-only, target at rel 0.1 / 0.9 (avg)   -- the effect's bad case
  center        : position-only, target at rel 0.5               -- best position-only case
  domain_aware  : window positioned to fully contain D           -- retain the residue's domain
Reports the recovered fraction of the edge->center gap by domain-aware windowing.

SECONDARY (rebut concern 1) on `large_domain` targets (|D| >= 2W, so EVERY placement stays inside
the same domain -- no boundary crossed): the center-edge gap. If it persists, the effect is
intra-domain and NOT an artifact of ignoring domain boundaries.

Usage (GPU, calibrate env): python rebuttal/reviewer3/r3_domain_windowing.py
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
from tqdm import tqdm

ROOT = _ROOT
sys.path.insert(0, str(ROOT / "src"))
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

TARGETS = ROOT / "rebuttal/reviewer3/targets.json"
OUTJSON = ROOT / "rebuttal/reviewer3/r3_domain_windowing.json"
OUTFIG = ROOT / "rebuttal/reviewer3/r3_domain_windowing.png"


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def domain_aware_start(d0, d1, L, W):
    lo, hi = max(0, d1 - W + 1), min(d0, L - W)
    if lo > hi:
        return None
    return int(np.clip(round((d0 + d1) / 2 - W / 2), lo, hi))


def main() -> None:
    set_global_seed(42)
    cfg = json.loads(TARGETS.read_text())
    W = cfg["W"]
    model = load_esm2()

    def fid_at(seq, t, s):
        full = model.embed([seq])[0][t]
        return cos(model.embed([seq[s:s + W]])[0][t - s], full)

    # ---- PRIMARY: containable domains ----
    prim = {"edge": [], "center": [], "domain_aware": []}
    for c in tqdm(cfg["containable"], desc="containable"):
        seq, t, d0, d1 = c["seq"], c["t"], c["d0"], c["d1"]; L = len(seq)
        e = np.mean([fid_at(seq, t, t - round(r * W)) for r in (0.1, 0.9)])
        ctr = fid_at(seq, t, t - round(0.5 * W))
        w0 = domain_aware_start(d0, d1, L, W)
        if w0 is None:
            continue
        da = fid_at(seq, t, w0)
        prim["edge"].append(e); prim["center"].append(ctr); prim["domain_aware"].append(da)
    edge = np.array(prim["edge"]); center = np.array(prim["center"]); da = np.array(prim["domain_aware"])
    r_ec = paired_bootstrap_ci(center, edge, seed=42)
    r_dae = paired_bootstrap_ci(da, edge, seed=42)
    r_dac = paired_bootstrap_ci(da, center, seed=42)
    recov = (da.mean() - edge.mean()) / (center.mean() - edge.mean() + 1e-9) * 100
    print(f"\n=== PRIMARY containable (n={len(edge)}) ===")
    print(f"  edge={edge.mean():.4f}  center={center.mean():.4f}  domain_aware={da.mean():.4f}")
    print(f"  center-edge         = {r_ec['diff']*100:+.2f}pp CI[{r_ec['ci_low']*100:.2f},{r_ec['ci_high']*100:.2f}] p={r_ec['p_value']:.4f}")
    print(f"  domain_aware-edge   = {r_dae['diff']*100:+.2f}pp CI[{r_dae['ci_low']*100:.2f},{r_dae['ci_high']*100:.2f}] p={r_dae['p_value']:.4f}")
    print(f"  domain_aware-center = {r_dac['diff']*100:+.2f}pp CI[{r_dac['ci_low']*100:.2f},{r_dac['ci_high']*100:.2f}] p={r_dac['p_value']:.4f}")
    print(f"  domain-aware recovers {recov:.0f}% of the edge->center gap")

    # ---- SECONDARY: large domains (all placements intra-domain) ----
    le, lc = [], []
    for c in tqdm(cfg["large_domain"], desc="large_domain"):
        seq, t = c["seq"], c["t"]
        le.append(np.mean([fid_at(seq, t, t - round(r * W)) for r in (0.1, 0.9)]))
        lc.append(fid_at(seq, t, t - round(0.5 * W)))
    sec = None
    if len(le) >= 10:
        le = np.array(le); lc = np.array(lc)
        r_l = paired_bootstrap_ci(lc, le, seed=42)
        sec = dict(n=len(le), edge=float(le.mean()), center=float(lc.mean()),
                   gap_pp=r_l["diff"] * 100, ci_lo=r_l["ci_low"] * 100, ci_hi=r_l["ci_high"] * 100,
                   p=r_l["p_value"])
        print(f"\n=== SECONDARY large_domain, all windows intra-domain (n={sec['n']}) ===")
        print(f"  edge={sec['edge']:.4f} center={sec['center']:.4f}  center-edge={sec['gap_pp']:+.2f}pp "
              f"CI[{sec['ci_lo']:.2f},{sec['ci_hi']:.2f}] p={sec['p']:.4f}  (effect persists intra-domain)")

    out = dict(
        primary=dict(n=len(edge), edge=float(edge.mean()), center=float(center.mean()),
                     domain_aware=float(da.mean()),
                     center_minus_edge_pp=r_ec["diff"] * 100, center_edge_p=r_ec["p_value"],
                     domain_aware_minus_edge_pp=r_dae["diff"] * 100, dae_p=r_dae["p_value"],
                     domain_aware_minus_center_pp=r_dac["diff"] * 100, dac_p=r_dac["p_value"],
                     recovered_fraction_pct=recov),
        secondary=sec)
    OUTJSON.write_text(json.dumps(out, indent=2, default=float))

    # ---- figure ----
    ncol = 2 if sec else 1
    fig, axes = plt.subplots(1, ncol, figsize=(9.5 if sec else 5.2, 4.3))
    ax = axes[0] if sec else axes
    labels = ["edge\n(position-only)", "domain-\naware", "center\n(position-only)"]
    vals = [edge.mean(), da.mean(), center.mean()]
    cols = ["#D55E00", "#009E73", "#0072B2"]
    ax.bar(range(3), vals, color=cols, width=0.6)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("embedding fidelity (cosine to full context)", fontsize=10)
    ax.set_title(f"Domain-aware windowing recovers {recov:.0f}% of the edge loss\n(containable domains, n={len(edge)})",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(min(vals) - 0.01, 1.0); ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    if sec:
        ax2 = axes[1]
        ax2.bar([0, 1], [sec["edge"], sec["center"]], color=["#D55E00", "#0072B2"], width=0.5)
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["edge", "center"], fontsize=9)
        ax2.set_ylabel("embedding fidelity", fontsize=10)
        ax2.set_title(f"Effect persists with NO boundary crossed\n(large domains, all windows intra-domain, "
                      f"n={sec['n']}: {sec['gap_pp']:+.1f}pp, p={sec['p']:.3f})", fontsize=10, fontweight="bold")
        ax2.set_ylim(min(sec["edge"], sec["center"]) - 0.01, 1.0); ax2.grid(True, axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(OUTFIG, dpi=200)
    print(f"\njson -> {OUTJSON}\nfigure -> {OUTFIG}")


if __name__ == "__main__":
    main()
