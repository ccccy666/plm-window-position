"""Reviewer-2 core RC (R2-A + R2-B): is the window-edge inverted-U an ARCHITECTURAL property of
Transformers or a LEARNED property of protein LMs, and does it need natural sequences?

2x2 design on the SAME n=120 seed-42 target positions and window protocol as the taxonomy:
  model in {trained ESM-2 650M, random-init ESM-2 650M (same architecture, no pretrained weights)}
  x sequence in {natural proteins, random-amino-acid sequences (matched lengths)}
For each cell we measure the embedding-fidelity center-edge gap (cosine of the target residue's
windowed vs full-context embedding), exactly as in scripts/phase1/t1a_fidelity.py.

Interpretation:
  - random-init also shows the inverted-U  -> the SHAPE is largely architectural/mechanical
    (attention + asymmetric context truncation); the novel/pLM-specific parts are the taxonomy,
    the PE gradient, the downstream scoping, and the remedy.
  - random-init is much weaker/absent      -> the effect needs learned representations (learned
    attention locality; see the r2_attention_locality.py theory) -> a learned pLM property.
  - natural vs random-seq gap              -> whether natural-sequence statistics are required.

Usage (GPU, calibrate env): python rebuttal/reviewer2/r2_randinit_randseq.py
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
import torch
from tqdm import tqdm

ROOT = _ROOT
sys.path.insert(0, str(ROOT / "src"))
from calibrate.models import ESM2Wrapper, load_esm2
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

SP = ROOT / "data/processed/p1_0/swissprot_sample.fasta"
OUTJSON = ROOT / "rebuttal/reviewer2/r2_randinit_randseq.json"
OUTFIG = ROOT / "rebuttal/reviewer2/r2_randinit_randseq.png"

W = 256
REL = [0.1, 0.3, 0.5, 0.7, 0.9]
MARGIN = round(0.9 * W)
N = 120
AA = list("ACDEFGHIKLMNPQRSTVWY")


def parse_fasta(text):
    s, c = [], []
    for l in text.splitlines():
        if l.startswith(">"):
            if c:
                s.append("".join(c)); c = []
        elif l.strip():
            c.append(l.strip())
    if c:
        s.append("".join(c))
    return s


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def build_random_esm2():
    """Same architecture as esm2_t33_650M_UR50D, random weights (no pretrained state_dict)."""
    import esm
    model = esm.model.esm2.ESM2(num_layers=33, embed_dim=1280, attention_heads=20, alphabet="ESM-1b")
    model = model.eval().half().cuda()
    return ESM2Wrapper(model, model.alphabet, "cuda")


def fidelity_gap(wrapper, targets, seqs):
    """center-edge embedding-fidelity gap over targets; seqs[i] is the carrier for targets[i]."""
    rows = []
    for (t, _), seq in zip(targets, seqs):
        full = wrapper.embed([seq])[0][t]
        fid = {}
        for r in REL:
            s = t - round(r * W)
            win = seq[s:s + W]
            fid[r] = cos(wrapper.embed([win])[0][t - s], full)
        rows.append(fid)
    center = np.array([r[0.5] for r in rows])
    edge = np.array([(r[0.1] + r[0.9]) / 2 for r in rows])
    res = paired_bootstrap_ci(center, edge, seed=42)
    return dict(n=len(rows), center=float(center.mean()), edge=float(edge.mean()),
                gap_pp=res["diff"] * 100, ci_lo=res["ci_low"] * 100, ci_hi=res["ci_high"] * 100,
                p=res["p_value"], curve={str(r): float(np.mean([x[r] for x in rows])) for r in REL})


def main() -> None:
    set_global_seed(42)
    rng = np.random.default_rng(42)
    OUTJSON.parent.mkdir(parents=True, exist_ok=True)

    pool = [s for s in parse_fasta(SP.read_text()) if set(s) <= set(AA)]
    rng.shuffle(pool)
    targets = []
    for prot in pool:
        if len(targets) >= N:
            break
        if len(prot) >= 2 * MARGIN + 1:
            targets.append((int(rng.integers(MARGIN, len(prot) - MARGIN)), prot))
    natural = [prot for (_, prot) in targets]
    tgt = [(t, prot) for (t, prot) in targets]
    # random-AA carriers of matched lengths (fixed seed for reproducibility)
    rs = np.random.default_rng(43)
    randseq = ["".join(rs.choice(AA, len(prot))) for prot in natural]
    print(f"{len(tgt)} targets; W={W}")

    results = {}
    for mtag, builder in [("trained", load_esm2), ("random_init", build_random_esm2)]:
        wrapper = builder()
        for stag, seqs in [("natural", natural), ("random_seq", randseq)]:
            r = fidelity_gap(wrapper, tgt, seqs)
            results[f"{mtag}/{stag}"] = r
            print(f"  {mtag:11s} {stag:10s}: center={r['center']:.4f} edge={r['edge']:.4f} "
                  f"gap={r['gap_pp']:+.2f}pp CI[{r['ci_lo']:.2f},{r['ci_hi']:.2f}] p={r['p']:.4f}")
        del wrapper
        torch.cuda.empty_cache()

    OUTJSON.write_text(json.dumps(results, indent=2, default=float))

    # ---- figure: 2x2 grouped bars of the center-edge gap ----
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    seqtypes = ["natural", "random_seq"]
    x = np.arange(len(seqtypes)); wdt = 0.36
    trained = [results[f"trained/{s}"]["gap_pp"] for s in seqtypes]
    randm = [results[f"random_init/{s}"]["gap_pp"] for s in seqtypes]
    b1 = ax.bar(x - wdt / 2, trained, wdt, color="#0072B2", label="trained ESM-2")
    b2 = ax.bar(x + wdt / 2, randm, wdt, color="#D55E00", label="random-init ESM-2")
    for bars, vals in [(b1, trained), (b2, randm)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (0.05 if v >= 0 else -0.15),
                    f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, c="k", lw=0.8)
    ax.axhline(3.0, ls="--", c="#888", lw=1.0, label="benchmark tier (3 pp)")
    ax.set_xticks(x); ax.set_xticklabels(["natural\nsequences", "random\nsequences"], fontsize=10)
    ax.set_ylabel("center − edge fidelity gap (pp)", fontsize=11)
    ax.set_title("Is the inverted-U architectural or learned?", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(OUTFIG, dpi=200)
    print(f"\njson -> {OUTJSON}\nfigure -> {OUTFIG}")


if __name__ == "__main__":
    main()
