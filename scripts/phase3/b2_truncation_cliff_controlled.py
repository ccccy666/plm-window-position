"""Tier-B exp 2 (B1/Q2): a CONTROLLED truncation-cliff test, replacing the n=1 BRCA1 evidence.

Across the whole ProteinGym substitutions benchmark, BRCA1 is the ONLY assay whose single
mutants straddle the 1022 training crop (verified: 9 proteins >1022aa, only BRCA1 has >=10
singles on each side), so the cliff test cannot be broadened with a second DMS assay. We
instead run the clean controlled test the pilot flagged but never executed: hold a residue's
local context CONSTANT and slide it across ABSOLUTE positions straddling 1022, asking whether
masked recovery drops once the residue sits beyond the training crop.

Design (content held constant; only absolute position varies):
  - Cassette = a recoverable site (high native-context recovery) + a constant +/- f native flank.
  - A single fixed shuffled carrier of length L_TOTAL (constant amino-acid composition, no
    structure) is the background; for each absolute position p we overwrite carrier[p-f : p+f]
    with the cassette and mask the center residue at absolute index p.
  - Total length is FIXED at L_TOTAL and both sides keep >= MIN_SIDE residues of carrier, so the
    only thing that changes across placements is the residue's ABSOLUTE index -> this isolates a
    training-crop / absolute-position effect from one-sided context-loss and from sequence length.

If rotary encodings extrapolate (no cliff), recovery is flat across 1022; a cliff would show a
drop for p>1022. n ~ 60 cassettes, paired within-cassette in-crop vs out-of-crop.

Outputs: experiments/phase3/b2_cliff/cliff.parquet,
         results/figures/b2_truncation_cliff_controlled.png, printed paired-bootstrap stats.

Usage (GPU, calibrate env): python scripts/phase3/b2_truncation_cliff_controlled.py
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
OUT = (_ROOT / "experiments/phase3/b2_cliff")
FIG = (_ROOT / "results/figures/b2_truncation_cliff_controlled.png")

CROP = 1022
L_TOTAL = 2200
F = 40                                   # cassette half-width (constant native flank)
MIN_SIDE = 300                           # keep >= this many carrier residues on each side
POSITIONS = [300, 600, 900, 1100, 1400, 1700, 1900]   # absolute index of the target; straddles 1022
N_CASS = 60
REC_MIN_NATIVE = 0.5                     # select recoverable sites (dynamic range)
AA = list("ACDEFGHIKLMNPQRSTVWY")


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

    pool = [s for s in parse_fasta(SP.read_text()) + parse_fasta(GFP.read_text())
            if set(s) <= set(AA) and len(s) >= 2 * F + 1]
    rng.shuffle(pool)

    # fixed shuffled carrier (constant composition, no structure); reused for every cassette/position
    bg_src = "".join(pool)
    carrier = list(bg_src)
    rng.shuffle(carrier)
    carrier = carrier[:L_TOTAL]
    if len(carrier) < L_TOTAL:
        carrier += list(rng.choice(AA, L_TOTAL - len(carrier)))

    # select recoverable cassettes: high native full-context recovery at the center residue
    cassettes = []
    for prot in pool:
        if len(cassettes) >= N_CASS:
            break
        t = int(rng.integers(F, len(prot) - F))
        if float(model.recovery_prob(prot, [t])[0]) >= REC_MIN_NATIVE:
            cassettes.append(prot[t - F : t + F + 1])   # width 2F+1, center index F
    print(f"selected {len(cassettes)} recoverable cassettes (native recovery >= {REC_MIN_NATIVE})")

    rows = []
    for ci, cass in enumerate(tqdm(cassettes, desc="b2 cliff")):
        for p in POSITIONS:
            if not (MIN_SIDE <= p <= L_TOTAL - MIN_SIDE):
                continue
            seq = carrier.copy()
            seq[p - F : p + F + 1] = list(cass)              # implant cassette; target center at index p
            rec = float(model.recovery_prob("".join(seq), [p])[0])
            rows.append(dict(cass=ci, pos=p, in_crop=p <= CROP, recovery=rec))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "cliff.parquet")

    piv = df.pivot_table(index="cass", columns="pos", values="recovery")
    by_pos = {p: float(df[df.pos == p].recovery.mean()) for p in POSITIONS}
    print(f"\n=== B2 controlled truncation cliff (n={piv.shape[0]} cassettes, L={L_TOTAL}, crop={CROP}) ===")
    print("  recovery by absolute position:", {p: round(v, 3) for p, v in by_pos.items()})

    in_pos = [p for p in POSITIONS if p <= CROP]
    out_pos = [p for p in POSITIONS if p > CROP]
    in_rec = piv[in_pos].mean(axis=1).to_numpy()
    out_rec = piv[out_pos].mean(axis=1).to_numpy()
    r = paired_bootstrap_ci(in_rec, out_rec, seed=42)
    print(f"  in-crop (p<=1022) mean = {in_rec.mean():.3f}  out-crop (p>1022) mean = {out_rec.mean():.3f}")
    print(f"  Delta(in-out) = {r['diff']*100:+.2f} pp  CI=[{r['ci_low']*100:.2f},{r['ci_high']*100:.2f}]  "
          f"p={r['p_value']:.4f}  (cliff gate: drop >= +10pp & p<0.01)")

    # adjacent-step test for a discontinuity right at the boundary (900 -> 1100, the step over 1022)
    if 900 in piv.columns and 1100 in piv.columns:
        rb = paired_bootstrap_ci(piv[900].to_numpy(), piv[1100].to_numpy(), seed=42)
        print(f"  boundary step 900->1100: {rb['diff']*100:+.2f} pp  "
              f"CI=[{rb['ci_low']*100:.2f},{rb['ci_high']*100:.2f}]  p={rb['p_value']:.4f}")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    means = [by_pos[p] for p in POSITIONS]
    sems = [df[df.pos == p].recovery.std() / np.sqrt(piv.shape[0]) for p in POSITIONS]
    ax.errorbar(POSITIONS, means, yerr=sems, marker="o", color="#0072B2", lw=1.8, ms=7,
                capsize=4, label="masked recovery (constant local context)")
    ax.axvline(CROP, ls="--", c="#D55E00", lw=1.4, label=f"training crop ({CROP} aa)")
    ax.set_xlabel("target absolute position in the sequence (aa)", fontsize=11)
    ax.set_ylabel("masked recovery", fontsize=11)
    ax.set_title("No truncation cliff: recovery is flat across the training crop",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.25 + 0.02)
    ax.legend(fontsize=9, loc="lower left"); ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    print(f"\nfigure -> {FIG}\nparquet -> {OUT / 'cliff.parquet'}")


if __name__ == "__main__":
    main()
