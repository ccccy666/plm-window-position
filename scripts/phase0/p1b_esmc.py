"""P1' window relocation on ESM C 300M — a DIFFERENT bidirectional family (gate: >=2).

Runs in the `esmc` env (EvolutionaryScale esm SDK; no fair-esm). Masked recovery via
raw forward: mask token id 32 (<mask>), read softmax(sequence_logits) of the true id.
Same protocol as ESM-2 window relocation (non-terminal recoverable targets).

Usage: conda run -n esmc python scripts/phase0/p1b_esmc.py
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
from calibrate.seed import set_global_seed       # safe: no `esm` import
from calibrate.stats import paired_bootstrap_ci   # safe: numpy/scipy only

from esm.models.esmc import ESMC

SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
GFP = (_ROOT / "data/processed/p1_0/gfp.fasta")
OUT = (_ROOT / "experiments/phase0_pilot/p1b")
FIG = (_ROOT / "results/figures/p1b_esmc_window_reloc.png")

WINDOWS = [256, 384]
REL_POS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_TARGETS = 50
REC_MIN = 0.30
MASK_ID = 32
AA = set("ACDEFGHIKLMNPQRSTVWY")

_model = None
_tok = None


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


@torch.no_grad()
def recovery(seq: str, k: int) -> float:
    ids = _tok.encode(seq)           # [<cls> ... <eos>]
    true_id = ids[k + 1]
    ids[k + 1] = MASK_ID
    lg = _model(torch.tensor([ids]).cuda()).sequence_logits[0]
    p = torch.softmax(lg[k + 1].float(), -1)
    return float(p[true_id].item())


def crop_for(prot, t, r, W):
    s = t - round(r * W)
    return prot[s : s + W], t - s


def main() -> None:
    global _model, _tok
    set_global_seed(42)
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    _model = ESMC.from_pretrained("esmc_300m").to("cuda").eval()
    _tok = _model.tokenizer

    pool = [s for s in parse_fasta(SP.read_text()) + parse_fasta(GFP.read_text()) if set(s) <= AA]
    rng.shuffle(pool)
    margin = round(0.9 * max(WINDOWS))

    rows, n = [], 0
    for prot in tqdm(pool, desc="ESM C window-reloc"):
        if n >= N_TARGETS:
            break
        if len(prot) < 2 * margin + 1:
            continue
        t = int(rng.integers(margin, len(prot) - margin))
        crop, j = crop_for(prot, t, 0.5, max(WINDOWS))
        if recovery(crop, j) < REC_MIN:
            continue
        for W in WINDOWS:
            for r in REL_POS:
                c, j = crop_for(prot, t, r, W)
                rows.append(dict(target=n, W=W, rel_pos=r, recovery=recovery(c, j)))
        n += 1
    print(f"used {n} non-terminal recoverable targets")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "esmc_window_reloc.parquet")

    print("\n=== ESM C window relocation: center(0.5) - edge(0.1,0.9) ===")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for W, g in df.groupby("W"):
        piv = g.pivot_table(index="target", columns="rel_pos", values="recovery")
        res = paired_bootstrap_ci(piv[0.5].values, piv[[0.1, 0.9]].mean(axis=1).values, seed=42)
        amp = (piv.mean(axis=0).max() - piv.mean(axis=0).min()) * 100
        print(f"  W={W}: n={piv.shape[0]} center={piv[0.5].mean():.3f} edge={piv[[0.1,0.9]].mean(axis=1).mean():.3f} "
              f"gap={res['diff']*100:+.2f}pp CI=[{res['ci_low']*100:.2f},{res['ci_high']*100:.2f}] p={res['p_value']:.3f} amp={amp:.2f}pp")
        ax.plot(piv.mean(axis=0).index, piv.mean(axis=0).values, marker="o", label=f"W={W}")
    ax.set(title="ESM C 300M window relocation (different bidirectional family)",
           xlabel="target rel_pos in window", ylabel="masked recovery")
    ax.legend()
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"\nfigure -> {FIG}")


if __name__ == "__main__":
    main()
