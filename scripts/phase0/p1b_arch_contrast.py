"""P4 / D2 architecture contrast: bidirectional (ESM-2) vs causal (ProGen2).

Same non-terminal targets, same window-relocation protocol. Prediction:
  - ESM-2 (bidirectional MLM): symmetric **inverted-U** (center best; uses both sides).
  - ProGen2 (causal RoPE): **monotonic increase** with rel_pos — only left context is
    visible, and a larger rel_pos = more real left context. Right context is irrelevant.
A clean monotonic-vs-U contrast is the core D2 evidence under the small-effect reframe.

Runs in the `progen` env (has both fair-esm ESM-2 and transformers ProGen2).
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

os.environ.setdefault("HF_HOME", str(_ROOT / "data/cache/models/hf"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
GFP = (_ROOT / "data/processed/p1_0/gfp.fasta")
OUT = (_ROOT / "experiments/phase0_pilot/p1b")
FIG = (_ROOT / "results/figures/p1b_arch_contrast.png")

WINDOWS = [256, 384]
REL_POS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_TARGETS = 80
REC_MIN = 0.30
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

    esm = load_esm2()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("hugohrban/progen2-base", trust_remote_code=True)
    pm = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-base",
                                              trust_remote_code=True).eval().half().cuda()

    @torch.no_grad()
    def progen_logprob(crop: str, j: int) -> float:
        ids = tok(crop, return_tensors="pt").input_ids.cuda()
        lp = torch.log_softmax(pm(input_ids=ids).logits[0].float(), dim=-1)
        return float(lp[j - 1, ids[0, j]].item())  # token j predicted by logits[j-1]

    pool = [s for s in parse_fasta(SP.read_text()) + parse_fasta(GFP.read_text()) if set(s) <= AA]
    rng.shuffle(pool)
    margin = round(0.9 * max(WINDOWS))

    rows, n = [], 0
    for prot in tqdm(pool, desc="arch contrast"):
        if n >= N_TARGETS:
            break
        if len(prot) < 2 * margin + 1:
            continue
        t = int(rng.integers(margin, len(prot) - margin))
        c0 = prot[t - max(WINDOWS) // 2 : t - max(WINDOWS) // 2 + max(WINDOWS)]
        if float(esm.recovery_prob(c0, [max(WINDOWS) // 2])[0]) < REC_MIN:
            continue
        for W in WINDOWS:
            for r in REL_POS:
                s = t - round(r * W)
                crop = prot[s : s + W]
                j = t - s
                rows.append(dict(target=n, W=W, rel_pos=r,
                                 esm_rec=float(esm.recovery_prob(crop, [j])[0]),
                                 progen_lp=progen_logprob(crop, j)))
        n += 1
    print(f"used {n} targets")

    df = pd.DataFrame(rows)
    df["progen_prob"] = np.exp(df["progen_lp"])
    df.to_parquet(OUT / "arch_contrast.parquet")

    # shape tests: ESM-2 inverted-U (center-edge>0); ProGen2 monotonic (Spearman rel_pos vs lp >0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    print("\n=== architecture contrast (within-target) ===")
    for W, g in df.groupby("W"):
        e = g.pivot_table(index="target", columns="rel_pos", values="esm_rec")
        ecg = paired_bootstrap_ci(e[0.5].values, e[[0.1, 0.9]].mean(1).values, seed=42)
        # ProGen2 per-target monotonic trend
        rhos = [spearmanr(REL_POS, g[g.target == ti].sort_values("rel_pos")["progen_lp"]).statistic
                for ti in g.target.unique()]
        rho_mean = float(np.nanmean(rhos))
        frac_pos = float(np.mean(np.array(rhos) > 0))
        print(f"  W={W}: ESM-2 center-edge = {ecg['diff']*100:+.2f}pp (p={ecg['p_value']:.3f}) | "
              f"ProGen2 monotonic Spearman(rel_pos,lp) mean={rho_mean:+.2f}, frac>0={frac_pos:.2f}")
        axes[0].plot(e.mean().index, e.mean().values, marker="o", label=f"W={W}")
        pg = g.groupby("rel_pos")["progen_prob"].mean()
        axes[1].plot(pg.index, pg.values, marker="s", label=f"W={W}")
    axes[0].set(title="ESM-2 (bidirectional): masked recovery", xlabel="target rel_pos", ylabel="recovery")
    axes[1].set(title="ProGen2 (causal): P(residue | left ctx)", xlabel="target rel_pos", ylabel="prob")
    axes[0].legend(); axes[1].legend()
    fig.suptitle("D2 architecture contrast: bidirectional inverted-U vs causal monotonic")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"\nfigure -> {FIG}")


if __name__ == "__main__":
    main()
