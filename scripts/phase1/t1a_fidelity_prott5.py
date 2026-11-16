"""T1.A / A2 — embedding-fidelity for ProtT5 (relative position encoding), the missing
position-encoding axis in the taxonomy. Same targets/protocol as t1a_fidelity.py (seed 42,
N=120). Encoder-only half-precision model; per-residue embeddings = encoder hidden states.

ProtT5 uses a T5 relative-position bias (NOT rotary, NOT learned-absolute) and was trained with
a 512-token crop — the cleanest "RoPE vs relative vs learned-absolute" contrast.

Run (calibrate env; offline once cached):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/phase1/t1a_fidelity_prott5.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

from transformers import T5EncoderModel, T5Tokenizer

SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
OUT = (_ROOT / "experiments/phase1_diagnosis/t1a_fidelity")
MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"
W, REL, MARGIN, N = 256, [0.1, 0.3, 0.5, 0.7, 0.9], round(0.9 * 256), 120
AA = set("ACDEFGHIKLMNPQRSTVWY")


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


def main():
    set_global_seed(42); rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    tok = T5Tokenizer.from_pretrained(MODEL, do_lower_case=False)
    m = T5EncoderModel.from_pretrained(MODEL).to("cuda").eval()

    @torch.no_grad()
    def embed(seq):
        # ProtT5 expects space-separated residues; rare AAs (U,Z,O,B) -> X; appends </s>
        s = " ".join(re.sub(r"[UZOB]", "X", seq))
        enc = tok(s, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"].cuda(); mask = enc["attention_mask"].cuda()
        h = m(input_ids=ids, attention_mask=mask).last_hidden_state[0]  # (len(seq)+1, H)
        return h[: len(seq)].float().cpu().numpy()                      # drop trailing </s>

    pool = [s for s in parse_fasta(SP.read_text()) if set(s) <= AA]
    rng.shuffle(pool)
    targets = []
    for prot in pool:
        if len(targets) >= N:
            break
        if len(prot) >= 2 * MARGIN + 1:
            targets.append((prot, int(rng.integers(MARGIN, len(prot) - MARGIN))))

    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    rows = []
    for prot, t in tqdm(targets, desc="prott5"):
        full = embed(prot)[t]
        for r in REL:
            s = t - round(r * W)
            rows.append(dict(backbone="prott5", target=id(prot), rel_pos=r,
                             fidelity=cos(embed(prot[s : s + W])[t - s], full)))
    df = pd.DataFrame(rows); df.to_parquet(OUT / "fidelity_prott5.parquet")
    g = df.pivot_table(index="target", columns="rel_pos", values="fidelity")
    res = paired_bootstrap_ci(g[0.5].values, g[[0.1, 0.9]].mean(axis=1).values, seed=42)
    from scipy.stats import spearmanr
    sp = np.mean([spearmanr(REL, [g[r].iloc[i] for r in REL]).statistic for i in range(len(g))])
    print(f"prott5: center={g[0.5].mean():.4f} edge={g[[0.1,0.9]].mean(axis=1).mean():.4f} "
          f"gap=+{res['diff']*100:.2f}pp p={res['p_value']:.3f} Spearman(rel,fid)={sp:+.2f}")
    print("curve:", {r: round(float(g[r].mean()), 4) for r in REL})


if __name__ == "__main__":
    main()
