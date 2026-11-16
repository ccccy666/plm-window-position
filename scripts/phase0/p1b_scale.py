"""P1' scaling: does the window-edge inverted-U hold across ESM-2 sizes? (GPT scaling ablation)

Same non-terminal targets (selected once via 650M), window relocation, for ESM-2
{35M, 150M, 650M}. Reports center-edge recovery gap per size. Zero new deps (fair-esm).

Usage: python scripts/phase0/p1b_scale.py [--plot-only]   (--plot-only redraws the figure from scale.parquet)
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
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
GFP = (_ROOT / "data/processed/p1_0/gfp.fasta")
OUT = (_ROOT / "experiments/phase0_pilot/p1b")
FIG = (_ROOT / "results/figures/p1b_scale.png")

SIZES = ["esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"]
W = 256
REL_POS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_TARGETS = 60
REC_MIN = 0.30
AA = set("ACDEFGHIKLMNPQRSTVWY")


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


def crop_for(prot, t, r):
    s = t - round(r * W)
    return prot[s : s + W], t - s


def main() -> None:
    if "--plot-only" in sys.argv:        # redraw the figure from the stored run; no model needed
        plot(pd.read_parquet(OUT / "scale.parquet"))
        return
    set_global_seed(42)
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    pool = [s for s in parse_fasta(SP.read_text()) + parse_fasta(GFP.read_text()) if set(s) <= AA]
    rng.shuffle(pool)
    margin = round(0.9 * W)

    # select targets once with 650M (the reference)
    ref = load_esm2("esm2_t33_650M_UR50D")
    targets = []
    for prot in pool:
        if len(targets) >= N_TARGETS:
            break
        if len(prot) < 2 * margin + 1:
            continue
        t = int(rng.integers(margin, len(prot) - margin))
        crop, j = crop_for(prot, t, 0.5)
        if float(ref.recovery_prob(crop, [j])[0]) >= REC_MIN:
            targets.append((prot, t))
    print(f"selected {len(targets)} targets (650M screen)")
    del ref
    torch.cuda.empty_cache()

    rows = []
    for size in SIZES:
        try:
            model = load_esm2(size)
        except Exception as e:
            print(f"  {size}: load failed ({e}); skip")
            continue
        for (prot, t) in tqdm(targets, desc=size):
            for r in REL_POS:
                crop, j = crop_for(prot, t, r)
                rows.append(dict(size=size, target=id(prot), rel_pos=r,
                                 recovery=float(model.recovery_prob(crop, [j])[0])))
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "scale.parquet")
    plot(df)


def plot(df: pd.DataFrame) -> None:
    print("\n=== center-edge gap by ESM-2 size ===")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    _mk = {"esm2_t12_35M_UR50D": "o", "esm2_t30_150M_UR50D": "s", "esm2_t33_650M_UR50D": "^"}
    _cl = {"esm2_t12_35M_UR50D": "#56B4E9", "esm2_t30_150M_UR50D": "#0072B2",
           "esm2_t33_650M_UR50D": "#D55E00"}
    for size in SIZES:
        g = df[df["size"] == size]
        if g.empty:
            continue
        mk, cl = _mk.get(size, "o"), _cl.get(size, None)
        piv = g.pivot_table(index="target", columns="rel_pos", values="recovery")
        gap = paired_bootstrap_ci(piv[0.5].values, piv[[0.1, 0.9]].mean(axis=1).values, seed=42)
        tag = "ESM-2 " + size.split("_")[2]
        print(f"  {tag}: center={piv[0.5].mean():.3f} edge={piv[[0.1,0.9]].mean(axis=1).mean():.3f} "
              f"gap={gap['diff']*100:+.2f}pp (p={gap['p_value']:.3f})")
        ax.plot(piv.mean(axis=0).index, piv.mean(axis=0).values, marker=mk, color=cl, ms=6,
                lw=1.8, label=tag)
    ax.set(xlabel="target relative position in window", ylabel="masked recovery")
    ax.legend(framealpha=0.9); ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    print(f"\nfigure -> {FIG}")


if __name__ == "__main__":
    main()
