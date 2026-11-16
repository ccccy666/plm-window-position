"""T2.A — characterize the training-free multi-crop remedy at the recovery level (ESM-2).

Extends the pilot D5 with a centrality-weighted ensemble and a clean recovered-fraction
analysis. On the controlled window-relocation probe, compare per-residue masked recovery:
  - single_edge            : target at window edge (rel 0.1/0.9) — the loss a residue hits.
  - single_center          : target centered (rel 0.5) — the upper bound (needs re-tiling).
  - multicrop_uniform      : equal-weight ensemble over placements {0.1..0.9} (D5).
  - multicrop_weighted     : centrality-weighted ensemble (w ∝ min(r,1-r); trust balanced
                             -context placements) — the new, still training-free strategy.

Reports, per window size, the recovered fraction (0 at edge, 1 at center) of each strategy
with a paired bootstrap CI vs the edge baseline. Writes a per-backbone JSON so other
backbones (ESM C in the esmc env) can be combined later (`t2a_combine.py`).

Usage: python scripts/phase2/t2a_remedy.py
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
from calibrate.remedy import aggregate_distributions, centrality_weights, recovered_fraction
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
GFP = (_ROOT / "data/processed/p1_0/gfp.fasta")
OUT = (_ROOT / "experiments/phase2_remedy")
FIG = (_ROOT / "results/figures/t2a_remedy.png")

WINDOWS = [256, 384]
PLACEMENTS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_TARGETS = 120
REC_MIN = 0.30
AA = set("ACDEFGHIKLMNPQRSTVWY")
STRATS = ["single_edge", "single_center", "multicrop_uniform", "multicrop_weighted"]


def parse_fasta(text):
    seqs, cur = [], []
    for ln in text.splitlines():
        if ln.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur = []
        elif ln.strip():
            cur.append(ln.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def collect(model) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    pool = [s for s in parse_fasta(SP.read_text()) + parse_fasta(GFP.read_text()) if set(s) <= AA]
    rng.shuffle(pool)
    margin = round(0.9 * max(WINDOWS))
    w_weighted = centrality_weights(PLACEMENTS)

    rows, n = [], 0
    for prot in tqdm(pool, desc="T2.A ESM-2"):
        if n >= N_TARGETS:
            break
        if len(prot) < 2 * margin + 1:
            continue
        t = int(rng.integers(margin, len(prot) - margin))
        c0 = prot[t - max(WINDOWS) // 2 : t - max(WINDOWS) // 2 + max(WINDOWS)]
        if float(model.recovery_prob(c0, [max(WINDOWS) // 2])[0]) < REC_MIN:
            continue
        ti = model.aa_token_index(prot[t])
        for W in WINDOWS:
            dists = {}
            for r in PLACEMENTS:
                s = t - round(r * W)
                dists[r] = model.masked_distribution(prot[s : s + W], [t - s])[0]
            stacked = np.array([dists[r] for r in PLACEMENTS])
            rows.append(dict(
                target=n, W=W,
                single_edge=float(np.mean([dists[0.1][ti], dists[0.9][ti]])),
                single_center=float(dists[0.5][ti]),
                multicrop_uniform=float(aggregate_distributions(stacked)[ti]),
                multicrop_weighted=float(aggregate_distributions(stacked, w_weighted)[ti]),
            ))
        n += 1
    print(f"used {n} targets")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, backbone: str) -> dict:
    out = {"backbone": backbone, "n_targets": int(df["target"].nunique()),
           "placements": PLACEMENTS, "by_W": {}}
    print(f"\n=== T2.A remedy ({backbone}); paired vs single_edge ===")
    for W in WINDOWS:
        g = df[df.W == W]
        means = {s: float(g[s].mean()) for s in STRATS}
        rec = {}
        for s in ["single_center", "multicrop_uniform", "multicrop_weighted"]:
            bc = paired_bootstrap_ci(g[s].to_numpy(), g["single_edge"].to_numpy(), seed=42)
            rec[s] = dict(
                mean=means[s], gain_pp=bc["diff"] * 100,
                ci_pp=[bc["ci_low"] * 100, bc["ci_high"] * 100], p=bc["p_value"],
                recovered_fraction=recovered_fraction(means[s], means["single_edge"],
                                                       means["single_center"]),
            )
        out["by_W"][W] = dict(edge=means["single_edge"], center=means["single_center"],
                              strategies=rec)
        print(f"  W={W}: edge={means['single_edge']:.3f} center={means['single_center']:.3f}")
        for s in ["multicrop_uniform", "multicrop_weighted"]:
            r = rec[s]
            print(f"    {s:18s} {r['mean']:.3f}  +{r['gain_pp']:.2f}pp "
                  f"(p={r['p']:.3f})  recovered={r['recovered_fraction']*100:.0f}%")
    return out


def figure(df: pd.DataFrame, summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # left: mean recovery per strategy, grouped by W
    ax = axes[0]
    width = 0.2
    for k, W in enumerate(WINDOWS):
        g = df[df.W == W]
        ax.bar(np.arange(len(STRATS)) + k * width, [g[s].mean() for s in STRATS],
               width, label=f"W={W}")
    ax.set_xticks(np.arange(len(STRATS)) + width / 2)
    ax.set_xticklabels([s.replace("_", "\n") for s in STRATS], fontsize=8)
    ax.set(title="Mean masked recovery by strategy (ESM-2)", ylabel="mean recovery")
    ax.legend(fontsize=8)
    # right: recovered fraction (uniform vs weighted; center=100% reference)
    ax = axes[1]
    xs = np.arange(len(WINDOWS))
    for j, s in enumerate(["multicrop_uniform", "multicrop_weighted"]):
        fr = [summary["by_W"][W]["strategies"][s]["recovered_fraction"] * 100 for W in WINDOWS]
        ax.bar(xs + j * 0.35, fr, 0.35, label=s.replace("multicrop_", ""))
    ax.axhline(100, ls="--", c="k", lw=1, label="centering (upper bound)")
    ax.set_xticks(xs + 0.175); ax.set_xticklabels([f"W={W}" for W in WINDOWS])
    ax.set(title="Recovered fraction of the edge loss (training-free)",
           ylabel="% of center−edge gap recovered")
    ax.legend(fontsize=8)
    fig.suptitle("T2.A — symmetric multi-crop recovers the window-edge loss without retraining; "
                 "centrality-weighting recovers more", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"figure -> {FIG}")


def main() -> None:
    set_global_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_esm2()
    df = collect(model)
    df.to_parquet(OUT / "esm2_remedy.parquet")
    summary = summarize(df, "esm2_650M")
    (OUT / "esm2_remedy.json").write_text(json.dumps(summary, indent=2, default=float))
    figure(df, summary)
    print(f"-> {OUT/'esm2_remedy.json'}")


if __name__ == "__main__":
    main()
