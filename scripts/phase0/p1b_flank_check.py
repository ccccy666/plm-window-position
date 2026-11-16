"""P1' (sliding-cassette) feasibility micro-check: the flank size f*.

P1.0 showed a zero-flank inserted motif is recovered at chance. The sliding-cassette
redesign transplants a *recoverable native site WITH +/-f native flank* to varying
positions. This script asks: how much native flank f must travel with a high-recovery
site for its masked recovery to survive transplantation into a shuffled carrier?

If recovery rises toward its native value as f grows -> the cassette probe is viable
(use cassette ~f*). If it stays at chance even for large f -> fall back to per-position
pseudo-perplexity (insertion-free).

Usage: python scripts/phase0/p1b_flank_check.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.models import load_esm2
from calibrate.probes.protniah import insert_needle, shuffle_sequence
from calibrate.seed import set_global_seed

GFP_FASTA = (_ROOT / "data/processed/p1_0/gfp.fasta")
SP_FASTA = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
FIG = (_ROOT / "results/figures/p1b_flank_feasibility.png")

F_VALUES = [0, 3, 6, 10, 15, 25]
MAX_F = max(F_VALUES)
CARRIER_L = 256
N_SITES = 50
MIN_NATIVE_REC = 0.25
REL_POS = 0.5


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
    model = load_esm2()

    pool = parse_fasta(SP_FASTA.read_text()) + parse_fasta(GFP_FASTA.read_text())
    pool = [s for s in pool if 120 <= len(s) <= 300 and set(s) <= set("ACDEFGHIKLMNPQRSTVWY")]
    proteins = [pool[i] for i in rng.choice(len(pool), min(12, len(pool)), replace=False)]
    carrier_src = [s for s in pool if len(s) >= CARRIER_L]

    # 1) find high-recovery native sites (single-mask per position)
    sites = []  # (protein, pos, native_rec)
    for prot in proteins:
        positions = list(range(MAX_F, len(prot) - MAX_F))
        for p in positions:
            rec = float(model.recovery_prob(prot, [p])[0])
            if rec >= MIN_NATIVE_REC:
                sites.append((prot, p, rec))
    sites.sort(key=lambda t: -t[2])
    sites = sites[:N_SITES]
    print(f"found {len(sites)} native sites with recovery >= {MIN_NATIVE_REC}")
    if not sites:
        print("No recoverable native sites found -> recovery-based cassette likely unviable.")
        return

    # 2) sweep flank size f: transplant cassette into a shuffled carrier at p=0.5
    rows = {f: [] for f in F_VALUES}
    native_recs = []
    for prot, pos, native in sites:
        native_recs.append(native)
        for f in F_VALUES:
            cassette = prot[pos - f : pos + f + 1]      # center is the original site
            src = carrier_src[int(rng.integers(0, len(carrier_src)))]
            start = int(rng.integers(0, len(src) - CARRIER_L + 1))
            carrier = shuffle_sequence(src[start : start + CARRIER_L], rng)
            seq, idx = insert_needle(carrier, cassette, REL_POS)
            center = idx[f]                              # f-th cassette residue = original site
            rows[f].append(float(model.recovery_prob(seq, [center])[0]))

    native_mean = float(np.mean(native_recs))
    means = {f: float(np.mean(rows[f])) for f in F_VALUES}
    fracs = {f: means[f] / native_mean for f in F_VALUES}
    print(f"\nnative recovery (in full context) mean = {native_mean:.3f}")
    print(f"{'f':>4}{'transplant_rec':>16}{'frac_of_native':>16}")
    for f in F_VALUES:
        print(f"{f:>4}{means[f]:>16.3f}{fracs[f]:>16.2f}")

    f_star = next((f for f in F_VALUES if fracs[f] >= 0.8), None)
    print(f"\nf* (first f reaching >=80% of native) = {f_star}")
    print("chance ~ 0.05")

    # figure
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(F_VALUES, [means[f] for f in F_VALUES], marker="o", label="transplanted (shuffled carrier)")
    ax.axhline(native_mean, ls="--", c="green", label=f"native context ({native_mean:.2f})")
    ax.axhline(0.05, ls=":", c="gray", label="chance ~0.05")
    ax.set(title="P1' feasibility: does recovery survive transplantation?",
           xlabel="native flank size f (each side)", ylabel="mean center-residue recovery")
    ax.legend()
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"figure -> {FIG}")


if __name__ == "__main__":
    main()
