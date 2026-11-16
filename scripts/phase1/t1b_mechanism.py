"""T1.B mechanism: is the window-edge effect one-sided context-loss or a terminal/sink bias?

A recoverable native site is wrapped in a CONSTANT, symmetric +/-f native flank (the cassette)
and slid across positions in a shuffled carrier (W=256). The center residue ALWAYS keeps its
+/-f context; only its position (distance to BOS/EOS, and how much shuffled carrier is on each
side) changes. We sweep f.

Prediction (context-loss hypothesis): the center-edge gap SHRINKS toward 0 as f grows (more of the
residue's own context travels with it) -> the window-edge brittleness is one-sided context-loss, NOT
an intrinsic terminal-token/position bias. If a gap persists at large f -> a real terminal effect.

Usage: python scripts/phase1/t1b_mechanism.py
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.models import load_esm2
from calibrate.probes.protniah import insert_needle, shuffle_sequence
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

GFP = (_ROOT / "data/processed/p1_0/gfp.fasta")
SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
OUT = (_ROOT / "experiments/phase1_diagnosis/t1b_mechanism")
FIG = (_ROOT / "results/figures/t1b_mechanism.png")

FLANKS = [10, 20, 40, 80]
MAX_F = max(FLANKS)
CARRIER_L = 256
REL_POS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_SITES = 50
NATIVE_REC_MIN = 0.30
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


def main() -> None:
    set_global_seed(42)
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_esm2()

    pool = [s for s in parse_fasta(SP.read_text()) + parse_fasta(GFP.read_text()) if set(s) <= AA]
    carrier_src = [s for s in pool if len(s) >= CARRIER_L]
    rng.shuffle(pool)

    # recoverable sites with >= MAX_F native flank on both sides
    sites = []
    for prot in pool:
        if len(sites) >= N_SITES:
            break
        if len(prot) < 2 * MAX_F + 1:
            continue
        for t in range(MAX_F, len(prot) - MAX_F):
            if float(model.recovery_prob(prot, [t])[0]) >= NATIVE_REC_MIN:
                sites.append((prot, t)); break
    print(f"selected {len(sites)} sites with >= {MAX_F} native flank")

    rows = []
    for f in FLANKS:
        for (prot, t) in tqdm(sites, desc=f"flank={f}"):
            cassette = prot[t - f : t + f + 1]
            src = carrier_src[int(rng.integers(0, len(carrier_src)))]
            start = int(rng.integers(0, len(src) - CARRIER_L + 1))
            carrier = shuffle_sequence(src[start : start + CARRIER_L], rng)
            for rp in REL_POS:
                seq, idx = insert_needle(carrier, cassette, rp)
                center = idx[f]
                rows.append(dict(flank=f, site=id(prot), rel_pos=rp,
                                 rec=float(model.recovery_prob(seq, [center])[0])))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "mechanism.parquet")

    print("\n=== center(0.5) - edge(0.1,0.9) gap vs flank (context held constant) ===")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    gaps = []
    for f in FLANKS:
        g = df[df.flank == f].pivot_table(index="site", columns="rel_pos", values="rec")
        res = paired_bootstrap_ci(g[0.5].values, g[[0.1, 0.9]].mean(axis=1).values, seed=42)
        gaps.append(res["diff"] * 100)
        print(f"  f={f:>3}: center={g[0.5].mean():.3f} edge={g[[0.1,0.9]].mean(axis=1).mean():.3f} "
              f"gap=+{res['diff']*100:.2f}pp (p={res['p_value']:.3f})")
        ax[0].plot(REL_POS, g.mean(axis=0)[REL_POS].values, marker="o", label=f"f={f}")
    ax[0].set(title="recovery vs. position, by flank $f$", xlabel="relative position",
              ylabel="center recovery")
    ax[0].legend(); ax[0].grid(True, alpha=0.25, lw=0.5)
    ax[1].plot(FLANKS, gaps, marker="s", color="#D55E00")
    ax[1].axhline(0, ls=":", c="gray")
    ax[1].set(title="center$-$edge gap stays null across flank $f$",
              xlabel="cassette flank $f$ (each side)", ylabel="center$-$edge gap (pp)")
    ax[1].grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=200)
    print(f"\nfigure -> {FIG}")
    print(f"gap trend over f={FLANKS}: {[round(x,2) for x in gaps]} pp")


if __name__ == "__main__":
    main()
