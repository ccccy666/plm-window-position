"""Reviewer-2 RC, mechanistic prong (R2-C empirical): WHY the inverted-U, and why it is learned.

Theory: a residue's windowed embedding drops the out-of-window tokens; the perturbation to its
representation scales with the ATTENTION MASS the residue places on those dropped tokens. A centered
window drops the FARTHEST tokens on both sides; an edge window drops NEAR-side tokens. Under attention
LOCALITY (near tokens weighted more) the edge placement removes more attention mass -> larger
perturbation -> lower cosine -> center > edge (inverted-U). Under UNIFORM attention, center and edge
remove equal total mass -> no gap. So the inverted-U requires attention locality.

This script measures, for query = the target residue (full-context forward, attention averaged over
layers+heads), the attention mass on the window-dropped residues at center vs edge placements, for
BOTH the trained and the random-init ESM-2. Prediction consistent with r2_randinit_randseq.py:
  - trained: edge drops MORE mass than center (learned locality) -> asymmetry -> the gap;
  - random-init: attention is diffuse -> center ~ edge dropped mass -> little asymmetry -> ~no gap.
We also correlate the per-target dropped-mass asymmetry with the per-target fidelity gap (trained).

Memory: need_head_weights materializes (layers,heads,T,T); we cap to targets with full length <= 700
and process one sequence at a time.

Usage (GPU, calibrate env): python rebuttal/reviewer2/r2_attention_locality.py
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
OUTJSON = ROOT / "rebuttal/reviewer2/r2_attention_locality.json"
OUTFIG = ROOT / "rebuttal/reviewer2/r2_attention_locality.png"

W = 256
MARGIN = round(0.9 * W)
N = 120
MAXLEN_ATTN = 700          # cap for the L x L attention materialization
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
    import esm
    m = esm.model.esm2.ESM2(num_layers=33, embed_dim=1280, attention_heads=20, alphabet="ESM-1b")
    return ESM2Wrapper(m.eval().half().cuda(), m.alphabet, "cuda")


@torch.no_grad()
def query_attention_over_residues(wrapper, seq, t):
    """Attention FROM query=residue t to every residue key, averaged over layers and heads.
    Returns a length-L vector summing to ~1 over residue keys (BOS/EOS excluded)."""
    toks = wrapper._tokens([seq])
    out = wrapper.model(toks, need_head_weights=True, repr_layers=[])
    att = out["attentions"][0]                 # (layers, heads, T, T) with BOS/EOS
    row = att[:, :, t + 1, :].float().mean(dim=(0, 1))   # (T,) query = t (+1 BOS offset)
    row = row[1:1 + len(seq)]                   # keep residue keys only
    del out, att
    return (row / (row.sum() + 1e-9)).cpu().numpy()


@torch.no_grad()
def dropped_mass_and_fidelity(wrapper, targets):
    """For each target: attention mass on window-dropped residues + fidelity, at center vs edge."""
    rows = []
    for (t, seq) in tqdm(targets, desc="attn"):
        a = query_attention_over_residues(wrapper, seq, t)     # (L,)
        full = wrapper.embed([seq])[0][t]
        rec = {}
        for tag, rels in [("center", [0.5]), ("edge", [0.1, 0.9])]:
            dm, fd = [], []
            for r in rels:
                s = t - round(r * W)
                keep = np.zeros(len(seq), dtype=bool); keep[s:s + W] = True
                dm.append(float(a[~keep].sum()))               # mass on dropped residues
                fd.append(cos(wrapper.embed([seq[s:s + W]])[0][t - s], full))
            rec[f"drop_{tag}"] = float(np.mean(dm))
            rec[f"fid_{tag}"] = float(np.mean(fd))
        rows.append(rec)
    return rows


def summarize(rows):
    dc = np.array([r["drop_center"] for r in rows]); de = np.array([r["drop_edge"] for r in rows])
    fc = np.array([r["fid_center"] for r in rows]); fe = np.array([r["fid_edge"] for r in rows])
    md = paired_bootstrap_ci(de, dc, seed=42)                  # edge - center dropped mass
    mf = paired_bootstrap_ci(fc, fe, seed=42)                  # center - edge fidelity
    # correlation: does more dropped-mass asymmetry -> bigger fidelity gap? (per target)
    asym = de - dc; gap = fc - fe
    corr = float(np.corrcoef(asym, gap)[0, 1]) if len(rows) > 2 else float("nan")
    return dict(n=len(rows), drop_center=float(dc.mean()), drop_edge=float(de.mean()),
                drop_edge_minus_center_pp=md["diff"] * 100, drop_p=md["p_value"],
                fidelity_gap_pp=mf["diff"] * 100, fidelity_p=mf["p_value"],
                corr_asym_vs_gap=corr)


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
    attn_targets = [(t, p) for (t, p) in targets if len(p) <= MAXLEN_ATTN]
    print(f"attention subset: {len(attn_targets)} / {len(targets)} targets (len<= {MAXLEN_ATTN})")

    results = {}
    for mtag, builder in [("trained", load_esm2), ("random_init", build_random_esm2)]:
        wrapper = builder()
        rows = dropped_mass_and_fidelity(wrapper, attn_targets)
        s = summarize(rows)
        results[mtag] = s
        print(f"  {mtag:11s}: dropped mass center={s['drop_center']:.3f} edge={s['drop_edge']:.3f} "
              f"(edge-center={s['drop_edge_minus_center_pp']:+.2f}pp p={s['drop_p']:.4f}); "
              f"fidelity gap={s['fidelity_gap_pp']:+.2f}pp p={s['fidelity_p']:.4f}; "
              f"corr(asym,gap)={s['corr_asym_vs_gap']:+.2f}")
        del wrapper
        torch.cuda.empty_cache()

    OUTJSON.write_text(json.dumps(results, indent=2, default=float))

    # ---- figure: dropped attention mass, center vs edge, trained vs random-init ----
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    models = ["trained", "random_init"]
    x = np.arange(len(models)); wdt = 0.36
    ctr = [results[m]["drop_center"] for m in models]
    edg = [results[m]["drop_edge"] for m in models]
    ax.bar(x - wdt / 2, ctr, wdt, color="#0072B2", label="centered window")
    ax.bar(x + wdt / 2, edg, wdt, color="#56B4E9", label="edge window")
    for k, m in enumerate(models):
        ax.text(k, max(ctr[k], edg[k]) + 0.01,
                f"$\\Delta$={results[m]['drop_edge_minus_center_pp']:+.1f}pp", ha="center",
                va="bottom", fontsize=9, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(["trained ESM-2", "random-init ESM-2"], fontsize=10)
    ax.set_ylabel("target's attention mass on\nwindow-dropped residues", fontsize=10)
    ax.set_title("Learned attention locality drives the inverted-U", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(OUTFIG, dpi=200)
    print(f"\njson -> {OUTJSON}\nfigure -> {OUTFIG}")


if __name__ == "__main__":
    main()
