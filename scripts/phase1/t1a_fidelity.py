"""T1.A — uniform embedding-fidelity position probe (cross-architecture taxonomy).

Masked recovery is not usable across all backbones (SaProt seq-only recovery is floored;
ProtT5/ESM C need bespoke heads). Embedding-fidelity is a uniform metric: cosine between a
target residue's WINDOWED embedding and its FULL-CONTEXT embedding, vs window position. It
measures representation-level position instability and works for any model that returns hidden
states. Bidirectional encoders -> inverted-U (center most faithful).

Backbones here (calibrate env): esm2 (fair-esm), saprot (transformers, sequence-only "X#").
ESM C / ProGen2 / ProtT5 run in their own envs with the same metric (separate runners).

Usage: python scripts/phase1/t1a_fidelity.py [esm2|saprot|both]
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
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
OUT = (_ROOT / "experiments/phase1_diagnosis/t1a_fidelity")
FIG = (_ROOT / "results/figures/t1a_fidelity.png")
W = 256
REL = [0.1, 0.3, 0.5, 0.7, 0.9]
MARGIN = round(0.9 * W)
N = 120
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


def get_embedder(backbone):
    if backbone == "esm2":
        from calibrate.models import load_esm2
        m = load_esm2()
        return lambda seq: m.embed([seq])[0]
    if backbone == "saprot":
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
        m = AutoModelForMaskedLM.from_pretrained("westlake-repl/SaProt_650M_AF2").eval().half().cuda()

        @torch.no_grad()
        def emb(seq):
            ids = tok("".join(c + "#" for c in seq), return_tensors="pt")["input_ids"].cuda()
            h = m(ids, output_hidden_states=True).hidden_states[-1][0]
            return h[1 : 1 + len(seq)].float().cpu().numpy()
        return emb
    raise ValueError(backbone)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_backbone(backbone, targets, embed):
    rows = []
    for prot, t in tqdm(targets, desc=backbone):
        full = embed(prot)[t]
        for r in REL:
            s = t - round(r * W)
            rows.append(dict(backbone=backbone, target=id(prot), rel_pos=r,
                             fidelity=cos(embed(prot[s : s + W])[t - s], full)))
    return rows


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    backbones = ["esm2", "saprot"] if which == "both" else [which]
    set_global_seed(42)
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)

    pool = [s for s in parse_fasta(SP.read_text()) if set(s) <= AA]
    rng.shuffle(pool)
    targets = []
    for prot in pool:
        if len(targets) >= N:
            break
        if len(prot) >= 2 * MARGIN + 1:
            targets.append((prot, int(rng.integers(MARGIN, len(prot) - MARGIN))))

    all_rows = []
    for b in backbones:
        all_rows += run_backbone(b, targets, get_embedder(b))
    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT / f"fidelity_{which}.parquet")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    print("\n=== embedding-fidelity (cosine to full context) center vs edge ===")
    for b in backbones:
        g = df[df.backbone == b].pivot_table(index="target", columns="rel_pos", values="fidelity")
        res = paired_bootstrap_ci(g[0.5].values, g[[0.1, 0.9]].mean(axis=1).values, seed=42)
        print(f"  {b}: center={g[0.5].mean():.4f} edge={g[[0.1,0.9]].mean(axis=1).mean():.4f} "
              f"gap=+{res['diff']*100:.2f}pp p={res['p_value']:.3f}")
        ax.plot(REL, g.mean(axis=0)[REL].values, marker="o", label=b)
    ax.set(title="T1.A embedding-fidelity vs window position (uniform taxonomy probe)",
           xlabel="target rel_pos in window", ylabel="cosine to full-context embedding")
    ax.legend()
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=130)
    print(f"figure -> {FIG}")


if __name__ == "__main__":
    main()
