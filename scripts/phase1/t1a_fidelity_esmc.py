"""T1.A embedding-fidelity for ESM C 300M (esmc env). Same targets/protocol as t1a_fidelity.py.
Usage: conda run -n esmc python scripts/phase1/t1a_fidelity_esmc.py"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

from esm.models.esmc import ESMC

SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
OUT = (_ROOT / "experiments/phase1_diagnosis/t1a_fidelity")
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
    m = ESMC.from_pretrained("esmc_300m").to("cuda").eval()
    tok = m.tokenizer

    @torch.no_grad()
    def embed(seq):
        ids = torch.tensor([tok.encode(seq)]).cuda()
        h = m(ids).embeddings[0]  # (L+2, H)
        return h[1 : 1 + len(seq)].float().cpu().numpy()

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
    for prot, t in tqdm(targets, desc="esmc"):
        full = embed(prot)[t]
        for r in REL:
            s = t - round(r * W)
            rows.append(dict(backbone="esmc", target=id(prot), rel_pos=r,
                             fidelity=cos(embed(prot[s : s + W])[t - s], full)))
    df = pd.DataFrame(rows); df.to_parquet(OUT / "fidelity_esmc.parquet")
    g = df.pivot_table(index="target", columns="rel_pos", values="fidelity")
    res = paired_bootstrap_ci(g[0.5].values, g[[0.1, 0.9]].mean(axis=1).values, seed=42)
    print(f"esmc: center={g[0.5].mean():.4f} edge={g[[0.1,0.9]].mean(axis=1).mean():.4f} "
          f"gap=+{res['diff']*100:.2f}pp p={res['p_value']:.3f}")
    print("curve:", {r: round(float(g[r].mean()), 4) for r in REL})


if __name__ == "__main__":
    main()
