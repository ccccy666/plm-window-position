"""Camera-ready experiment C (GPU): does each encoder's attention asymmetry predict the size of its window effect?

Motivation (exploratory, noticed after group A): over the five group-A encoders, the center-edge fidelity gap and
the attention asymmetry (attention mass an edge window drops minus what a centered window drops) have the same
rank order (Spearman 1.00, exact p = 0.017, n = 5). ESM-1b and Ankh have B1 gaps, but their attention has not been
measured, so they are an out-of-sample test.

PRE-REGISTERED (fixed 2026-11-16 before measuring ESM-1b or Ankh attention; do not change after seeing results)
---------------------------------------------------------------------------------------------------------------
Asymmetry: as A2: the target's attention row from the full-context pass, averaged over layers and heads and
           renormalized over residue keys; dropped mass of the centered window (rel 0.5) and of the edge windows
           (mean of rel 0.1 and 0.9); asymmetry = edge - center, in pp. Same targets as B1 (ESM-1b: the 98 in
           proteins <= 1022 aa; Ankh: 120).
Gaps     : group-A trained/natural gaps (ESM-2, ESM C, SaProt, ProtT5, ProtBERT) and the B1 gaps (ESM-1b, Ankh).
P1       : ESM-1b's asymmetry is the smallest of the seven (below SaProt's 8.05 pp).
P2       : Ankh's asymmetry lies between ProtT5's (10.22 pp) and ProtBERT's (16.00 pp).
P3       : decision. Spearman rho(gap, asymmetry) over the seven encoders has exact two-sided permutation p < 0.05
           (rho >= 0.79). "The window effect's size tracks the learned attention asymmetry across encoders" is
           supported iff P3 holds.
Sensitivity: the same correlation with every encoder restricted to the 98 targets common to all seven.

Outputs  : experiments/phase3/c_attention_vs_gap/{<model>/attention.parquet, summary.json}
Usage    : python scripts/phase3/c_attention_vs_gap.py esm1b|ankh [--smoke]   (esm1b: calibrate env; ankh: progen)
           python scripts/phase3/c_attention_vs_gap.py analyze                (any env, no GPU)
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = _ROOT
sys.path.insert(0, str(ROOT / "src"))
from calibrate import mechgen as mg  # noqa: E402
from calibrate.seed import set_global_seed  # noqa: E402

MODELS = ROOT / "data/cache/models"
A = ROOT / "experiments/phase3/a_mechgen"
B1 = ROOT / "experiments/phase3/b1_pe_axis"
OUT = ROOT / "experiments/phase3/c_attention_vs_gap"
A_MODELS = ["esm2", "esmc", "saprot", "prott5", "protbert"]
NEW = ["esm1b", "ankh"]
MAXLEN = {"esm1b": 1022, "ankh": None}


def rows_mean(rows, offset, length):
    import torch
    return mg.residue_row(torch.stack(rows).mean(0).cpu().numpy(), offset, length)


def load_esm1b():
    import torch
    from transformers import AutoTokenizer, EsmModel
    path = MODELS / "esm1b_t33_650M_UR50S"
    tok = AutoTokenizer.from_pretrained(path)
    m = EsmModel.from_pretrained(path, add_pooling_layer=False).eval().half().cuda()
    default_impl = m.config._attn_implementation

    def ids_of(seq):
        return tok(seq, return_tensors="pt")["input_ids"].cuda()

    @torch.no_grad()
    def attn_row(seq, t, full_reference=False):
        q, rows = t + 1, []
        m.set_attn_implementation("eager")
        try:
            if full_reference:
                att = torch.stack(m(ids_of(seq), output_attentions=True).attentions)[:, 0]
                return mg.residue_row(att[:, :, q, :].float().mean(dim=(0, 1)).cpu().numpy(), 1, len(seq))

            def hook(mod, inp, out):
                rows.append(out[1][0, :, q, :].float().mean(0))   # fp16 -> fp32 copy; weights then discarded
            hs = [layer.attention.self.register_forward_hook(hook) for layer in m.encoder.layer]
            try:
                m(ids_of(seq))
            finally:
                for h in hs:
                    h.remove()
        finally:
            m.set_attn_implementation(default_impl)
        return rows_mean(rows, 1, len(seq))
    return SimpleNamespace(attn_row=attn_row)


def load_ankh():
    import torch
    from transformers import AutoTokenizer, T5EncoderModel
    path = MODELS / "ankh-base"
    tok = AutoTokenizer.from_pretrained(path)
    m = T5EncoderModel.from_pretrained(path).eval().cuda()

    def enc(seq):
        e = tok([list(seq)], is_split_into_words=True, add_special_tokens=True, return_tensors="pt")
        return dict(input_ids=e["input_ids"].cuda(), attention_mask=e["attention_mask"].cuda())

    @torch.no_grad()
    def attn_row(seq, t, full_reference=False):
        q, rows = t, []
        if full_reference:
            att = torch.stack(m(**enc(seq), output_attentions=True).attentions)[:, 0]
            return mg.residue_row(att[:, :, q, :].float().mean(dim=(0, 1)).cpu().numpy(), 0, len(seq))

        def hook(mod, inp, out):          # (attn_output, present_kv, position_bias, attn_weights)
            a = out[3]
            rows.append(a[0, :, q, :].float().mean(0))
            return out[:3] + (a[:, :, q:q + 1, :].clone(),) + tuple(out[4:])
        hs = [blk.layer[0].SelfAttention.register_forward_hook(hook) for blk in m.encoder.block]
        try:
            m(**enc(seq), output_attentions=True)
        finally:
            for h in hs:
                h.remove()
        return rows_mean(rows, 0, len(seq))
    return SimpleNamespace(attn_row=attn_row)


def measure(model, smoke):
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    set_global_seed(42)
    targets = mg.benchmark_targets()
    keep = [k for k, (p, _) in enumerate(targets) if MAXLEN[model] is None or len(p) <= MAXLEN[model]]
    bb = load_esm1b() if model == "esm1b" else load_ankh()
    if smoke:
        short = min(keep, key=lambda k: len(targets[k][0]))
        p, t = targets[short]
        a, ref = bb.attn_row(p, t), bb.attn_row(p, t, full_reference=True)
        print(f"[{model}] hook-vs-full max|diff| = {np.abs(a - ref).max():.2e}; row sum {a.sum():.4f}")
        keep = keep[:3]
    rows = []
    for k in keep:
        p, t = targets[k]
        row = bb.attn_row(p, t)
        rows.append(dict(target=k, L=len(p), row_sum=float(row.sum()),
                         drop_center=mg.dropped_mass(row, mg.window_start(t, 0.5)),
                         drop_edge=float(np.mean([mg.dropped_mass(row, mg.window_start(t, r)) for r in (0.1, 0.9)]))))
    df = pd.DataFrame(rows)
    print(f"[{model}] {len(df)} targets in {time.time() - t0:.0f}s; non-finite {int((~np.isfinite(df.drop_edge)).sum())}; "
          f"asymmetry {100 * (df.drop_edge - df.drop_center).mean():+.2f} pp", flush=True)
    if not smoke:
        (OUT / model).mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT / model / "attention.parquet")


def spearman_exact(x, y):
    rk = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    rx, ry = rk(np.asarray(x)), rk(np.asarray(y))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    perm = np.array([np.corrcoef(rx, np.array(pm))[0, 1] for pm in itertools.permutations(range(len(x)))])
    return rho, float(np.mean(np.abs(perm) >= abs(rho) - 1e-12))


def per_target(model):
    """Per-target gap and asymmetry (pp), indexed by benchmark target."""
    if model in A_MODELS:
        f = pd.read_parquet(A / model / "fidelity.parquet")
        f = f[(f.model == "trained") & (f.seqtype == "natural")]
        a = pd.read_parquet(A / model / "attention.parquet")
        a = a[a.model == "trained"].set_index("target")
    else:
        f = pd.read_parquet(B1 / model / "fidelity.parquet")
        a = pd.read_parquet(OUT / model / "attention.parquet").set_index("target")
    g = f.pivot_table(index="target", columns="rel", values="fidelity")
    gap = (g[0.5] - g[[0.1, 0.9]].mean(axis=1)) * 100
    return pd.DataFrame(dict(gap=gap, asym=(a.drop_edge - a.drop_center) * 100)).dropna()


def analyze():
    data = {m: per_target(m) for m in A_MODELS + NEW}
    tab = pd.DataFrame({m: dict(n=len(d), gap=d.gap.mean(), asym=d.asym.mean()) for m, d in data.items()}).T
    rho, p = spearman_exact(tab.gap, tab.asym)
    r = float(np.corrcoef(tab.gap, tab.asym)[0, 1])
    common = sorted(set.intersection(*[set(d.index) for d in data.values()]))
    sub = pd.DataFrame({m: dict(gap=d.loc[common].gap.mean(), asym=d.loc[common].asym.mean())
                        for m, d in data.items()}).T
    rho_c, p_c = spearman_exact(sub.gap, sub.asym)
    others = tab.drop(index=["esm1b"])
    p1 = bool(tab.loc["esm1b", "asym"] < others.asym.min())
    p2 = bool(tab.loc["prott5", "asym"] < tab.loc["ankh", "asym"] < tab.loc["protbert", "asym"])
    within = {}
    for m in NEW:
        d = data[m]
        within[m] = dict(r=float(np.corrcoef(d.gap, d.asym)[0, 1]), n=len(d))
    summary = dict(table=tab.to_dict(orient="index"), spearman=rho, spearman_p=p, pearson=r,
                   P1_esm1b_smallest=p1, P2_ankh_between=p2, P3_supported=bool(p < 0.05),
                   common_targets=dict(n=len(common), spearman=rho_c, spearman_p=p_c,
                                       table=sub.to_dict(orient="index")),
                   within_model_corr=within)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(tab.sort_values("gap").round(2).to_string())
    print(f"\nSpearman rho = {rho:.2f} (exact two-sided p = {p:.4f}), Pearson r = {r:.2f}, n = {len(tab)}")
    print(f"P1 ESM-1b smallest asymmetry: {p1}  (ESM-1b {tab.loc['esm1b', 'asym']:.2f}, next {others.asym.min():.2f})")
    print(f"P2 Ankh between ProtT5 and ProtBERT: {p2}  (Ankh {tab.loc['ankh', 'asym']:.2f})")
    print(f"P3 (decision): {'SUPPORTED' if p < 0.05 else 'not supported'}")
    print(f"common {len(common)} targets: Spearman {rho_c:.2f} (p = {p_c:.4f})")
    print("within-model per-target r: " + ", ".join(f"{m} {v['r']:+.2f} (n={v['n']})" for m, v in within.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["esm1b", "ankh", "analyze"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.step == "analyze":
        analyze()
    else:
        measure(args.step, args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
