"""T1.A embedding-fidelity for ProGen2-764M (progen env). Causal -> expect MONOTONIC (more left
context = more faithful), contrasting bidirectional's inverted-U. Same targets/protocol.
Usage: conda run -n progen python scripts/phase1/t1a_fidelity_progen.py"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

os.environ.setdefault("HF_HOME", str(_ROOT / "data/cache/models/hf"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

from transformers import AutoModelForCausalLM, AutoTokenizer

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
    tok = AutoTokenizer.from_pretrained("hugohrban/progen2-base", trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-base",
                                             trust_remote_code=True).eval().half().cuda()

    @torch.no_grad()
    def embed(seq):  # ProGen2: 1 token/AA, no special tokens added
        ids = tok(seq, return_tensors="pt").input_ids.cuda()
        h = m(ids, output_hidden_states=True).hidden_states[-1][0]  # (L, H)
        return h[: len(seq)].float().cpu().numpy()

    pool = [s for s in parse_fasta(SP.read_text()) if set(s) <= AA]
    rng.shuffle(pool)
    targets = []  # same selection as other backbones, then drop >2000aa (ProGen2 max-ctx 2048)
    for prot in pool:
        if len(targets) >= N:
            break
        if len(prot) >= 2 * MARGIN + 1:
            targets.append((prot, int(rng.integers(MARGIN, len(prot) - MARGIN))))
    targets = [(p, t) for (p, t) in targets if len(p) <= 2000]
    print(f"progen2: {len(targets)} targets (<=2000aa; ProGen2 context limit)")

    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    rows = []
    for prot, t in tqdm(targets, desc="progen"):
        full = embed(prot)[t]
        for r in REL:
            s = t - round(r * W)
            rows.append(dict(backbone="progen2", target=id(prot), rel_pos=r,
                             fidelity=cos(embed(prot[s : s + W])[t - s], full)))
    df = pd.DataFrame(rows); df.to_parquet(OUT / "fidelity_progen.parquet")
    g = df.pivot_table(index="target", columns="rel_pos", values="fidelity")
    res = paired_bootstrap_ci(g[0.5].values, g[[0.1, 0.9]].mean(axis=1).values, seed=42)
    print(f"progen2: center={g[0.5].mean():.4f} edge={g[[0.1,0.9]].mean(axis=1).mean():.4f} "
          f"gap=+{res['diff']*100:.2f}pp p={res['p_value']:.3f}")
    print("curve:", {r: round(float(g[r].mean()), 4) for r in REL})


if __name__ == "__main__":
    main()
