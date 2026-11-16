"""T2.A — multi-crop remedy at the recovery level on ESM C 300M (a 2nd bidirectional family).

Runs in the `esmc` env (EvolutionaryScale esm SDK; no fair-esm). Masked recovery via raw
forward: mask id 32, softmax(sequence_logits). Same strategies as the ESM-2 runner
(`t2a_remedy.py`); writes a per-backbone JSON for `t2a_combine.py`.

Usage: conda run -n esmc python scripts/phase2/t2a_remedy_esmc.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.remedy import aggregate_distributions, centrality_weights, recovered_fraction
from calibrate.seed import set_global_seed         # safe: no `esm` import
from calibrate.stats import paired_bootstrap_ci     # safe: numpy/scipy only

from esm.models.esmc import ESMC

SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
GFP = (_ROOT / "data/processed/p1_0/gfp.fasta")
OUT = (_ROOT / "experiments/phase2_remedy")

WINDOWS = [256, 384]
PLACEMENTS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_TARGETS = 120
REC_MIN = 0.30
MASK_ID = 32
AA = set("ACDEFGHIKLMNPQRSTVWY")
STRATS = ["single_edge", "single_center", "multicrop_uniform", "multicrop_weighted"]

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
def masked_dist(seq: str, k: int):
    """Return (distribution over vocab, true_id) for masking position k."""
    ids = _tok.encode(seq)
    true_id = ids[k + 1]
    ids[k + 1] = MASK_ID
    lg = _model(torch.tensor([ids]).cuda()).sequence_logits[0]
    p = torch.softmax(lg[k + 1].float(), -1).cpu().numpy()
    return p, true_id


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
    w_weighted = centrality_weights(PLACEMENTS)

    rows, n = [], 0
    for prot in tqdm(pool, desc="T2.A ESM C"):
        if n >= N_TARGETS:
            break
        if len(prot) < 2 * margin + 1:
            continue
        t = int(rng.integers(margin, len(prot) - margin))
        Wm = max(WINDOWS)
        c0 = prot[t - Wm // 2 : t - Wm // 2 + Wm]
        p0, tid0 = masked_dist(c0, Wm // 2)
        if float(p0[tid0]) < REC_MIN:
            continue
        for W in WINDOWS:
            dists, ti = {}, None
            for r in PLACEMENTS:
                s = t - round(r * W)
                p, tid = masked_dist(prot[s : s + W], t - s)
                dists[r] = p; ti = tid
            stacked = np.array([dists[r] for r in PLACEMENTS])
            rows.append(dict(
                target=n, W=W,
                single_edge=float(np.mean([dists[0.1][ti], dists[0.9][ti]])),
                single_center=float(dists[0.5][ti]),
                multicrop_uniform=float(aggregate_distributions(stacked)[ti]),
                multicrop_weighted=float(aggregate_distributions(stacked, w_weighted)[ti]),
            ))
        n += 1
    print(f"used {n} targets")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "esmc_remedy.parquet")

    summary = {"backbone": "esmc_300m", "n_targets": int(df["target"].nunique()),
               "placements": PLACEMENTS, "by_W": {}}
    print("\n=== T2.A remedy (esmc_300m); paired vs single_edge ===")
    for W in WINDOWS:
        g = df[df.W == W]
        means = {s: float(g[s].mean()) for s in STRATS}
        rec = {}
        for s in ["single_center", "multicrop_uniform", "multicrop_weighted"]:
            bc = paired_bootstrap_ci(g[s].to_numpy(), g["single_edge"].to_numpy(), seed=42)
            rec[s] = dict(mean=means[s], gain_pp=bc["diff"] * 100,
                          ci_pp=[bc["ci_low"] * 100, bc["ci_high"] * 100], p=bc["p_value"],
                          recovered_fraction=recovered_fraction(means[s], means["single_edge"],
                                                                means["single_center"]))
        summary["by_W"][W] = dict(edge=means["single_edge"], center=means["single_center"],
                                  strategies=rec)
        print(f"  W={W}: edge={means['single_edge']:.3f} center={means['single_center']:.3f}")
        for s in ["multicrop_uniform", "multicrop_weighted"]:
            r = rec[s]
            print(f"    {s:18s} {r['mean']:.3f}  +{r['gain_pp']:.2f}pp "
                  f"(p={r['p']:.3f})  recovered={r['recovered_fraction']*100:.0f}%")
    (OUT / "esmc_remedy.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"-> {OUT/'esmc_remedy.json'}")


if __name__ == "__main__":
    main()
