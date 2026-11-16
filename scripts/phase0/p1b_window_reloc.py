"""P1'.2a window relocation — in-distribution confirmation of crop-position brittleness.

Slide the crop WINDOW over a real long protein so the same (non-terminal) target
residue lands at different relative positions in the window. Unlike the cassette
probe (constant flank), here a near-edge target loses its real context on that side
*within the window* — the practical windowing scenario. Targets are kept far from the
protein's real N/C termini, so any effect is model-side, not termini biology (D3).

Run + within-target paired analysis + figure. Usage: python scripts/phase0/p1b_window_reloc.py
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

GFP = (_ROOT / "data/processed/p1_0/gfp.fasta")
SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
OUT = (_ROOT / "experiments/phase0_pilot/p1b")
FIG = (_ROOT / "results/figures/p1b_window_reloc.png")

WINDOWS = [256, 384]
REL_POS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_TARGETS = 150
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


def crop_for(prot, t, r, W):
    s = t - round(r * W)
    return prot[s : s + W], t - s  # crop, target index within crop


def main() -> None:
    set_global_seed(42)
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_esm2()

    pool = [s for s in parse_fasta(SP.read_text()) + parse_fasta(GFP.read_text()) if set(s) <= AA]
    rng.shuffle(pool)
    Wmax = max(WINDOWS)
    margin = round(0.9 * Wmax)  # need this many residues both sides of target

    rows = []
    n = 0
    for prot in pool:
        if n >= N_TARGETS:
            break
        if len(prot) < 2 * margin + 1:
            continue
        t = int(rng.integers(margin, len(prot) - margin))  # interior, far from real termini
        crop, j = crop_for(prot, t, 0.5, Wmax)             # screen on centered window
        if float(model.recovery_prob(crop, [j])[0]) < REC_MIN:
            continue
        for W in WINDOWS:
            for r in REL_POS:
                c, j = crop_for(prot, t, r, W)
                rec = float(model.recovery_prob(c, [j])[0])
                rows.append(dict(target=n, W=W, rel_pos=r, recovery=rec,
                                 logprob=float(np.log(max(rec, 1e-9)))))
        n += 1
    print(f"used {n} non-terminal recoverable targets")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "window_reloc.parquet")

    # within-target paired: center (0.5) vs edges (0.1,0.9)
    print("\n=== window relocation: within-target center(0.5) - edge(0.1,0.9) ===")
    res_rows = []
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for W, g in df.groupby("W"):
        piv = g.pivot_table(index="target", columns="rel_pos", values="recovery")
        center = piv[0.5]
        edge = piv[[0.1, 0.9]].mean(axis=1)
        res = paired_bootstrap_ci(center.values, edge.values, seed=42)
        amp = (piv.mean(0).max() - piv.mean(0).min()) * 100
        res_rows.append(dict(W=W, n=piv.shape[0], center=center.mean(), edge=edge.mean(),
                             gap_center_minus_edge_pp=res["diff"] * 100,
                             ci_lo=res["ci_low"] * 100, ci_hi=res["ci_high"] * 100,
                             p=res["p_value"], amplitude_pp=amp))
        ax.plot(piv.mean(0).index, piv.mean(0).values, marker="o", label=f"W={W}")
    rdf = pd.DataFrame(res_rows)
    print(rdf.round(3).to_string(index=False))
    ax.set(title="P1'.2a window relocation (in-distribution)", xlabel="target rel_pos in window",
           ylabel="masked recovery")
    ax.legend()
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"\nfigure -> {FIG}")
    print(f"max |center-edge gap| = {rdf['gap_center_minus_edge_pp'].abs().max():.2f} pp; "
          f"min p = {rdf['p'].min():.4f}; max amplitude = {rdf['amplitude_pp'].max():.2f} pp")


if __name__ == "__main__":
    main()
