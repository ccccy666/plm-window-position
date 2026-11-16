"""Camera-ready group B1 (GPU): a second model in each non-rotary position-encoding class.

PRE-REGISTERED (fixed 2026-11-16 before either model is run; do not change after seeing results)
------------------------------------------------------------------------------------------------
Models   : ESM-1b 650M (learned absolute positions; ESM-2's predecessor with the same size and UniRef50 training
           family, i.e. the closest available contrast to rotary ESM-2), calibrate env, fp16 as ESM-2;
           Ankh-base (T5 relative position bias; a second relative-bias encoder next to ProtT5), progen env, fp32
           encoder as ProtT5.
Targets  : the 120 taxonomy targets (calibrate.mechgen); ESM-1b only the 98 in proteins of <= 1022 residues
           (its learned position table ends at 1026).
Probe    : last-layer embedding fidelity at relative positions {0.1, 0.3, 0.5, 0.7, 0.9}, W = 256 (as Table 2):
           Position Gap, Shift Sensitivity, Robust Accuracy (calibrate.metrics).
Decision : "absolute > rotary" replicates iff ESM-1b's per-target gap exceeds ESM-2's on the same 98 targets
           (paired bootstrap, p < 0.01). "relative > rotary" replicates iff Ankh's gap exceeds ESM-2's on the 120
           targets (p < 0.01). Comparisons with ProtT5 / ProtBERT are descriptive.
Index    : ESM-1b only, windows fed with each token's full-context position id (as A3 for ProtBERT); descriptive.

Outputs  : experiments/phase3/b1_pe_axis/<model>/{fidelity.parquet, summary.json}
Usage    : python scripts/phase3/b1_pe_axis.py esm1b|ankh [--smoke]   (esm1b: calibrate env; ankh: progen env)
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import argparse
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
from calibrate import metrics  # noqa: E402
from calibrate.seed import set_global_seed  # noqa: E402

MODELS = ROOT / "data/cache/models"
OUTROOT = ROOT / "experiments/phase3/b1_pe_axis"
TAXO = ROOT / "experiments/phase1_diagnosis/t1a_fidelity"
MAXLEN = {"esm1b": 1022, "ankh": None}
P_BAR = 0.01


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_esm1b():
    import torch
    from transformers import AutoTokenizer, EsmModel
    path = MODELS / "esm1b_t33_650M_UR50S"
    tok = AutoTokenizer.from_pretrained(path)
    m = EsmModel.from_pretrained(path, add_pooling_layer=False).eval().half().cuda()
    off = m.config.pad_token_id + 1                     # HF ESM: <cls> sits at padding_idx + 1

    def ids_of(seq):
        return tok(seq, return_tensors="pt")["input_ids"].cuda()

    @torch.no_grad()
    def embed(seq, position_ids=None):
        pos = None if position_ids is None else torch.tensor(np.asarray([position_ids])).cuda()
        return m(ids_of(seq), position_ids=pos).last_hidden_state[0][1:1 + len(seq)].float().cpu().numpy()

    def embed_preserved(prot, s):
        return embed(prot[s:s + mg.W], mg.preserved_position_ids(len(prot), s, offset=off))
    return SimpleNamespace(embed=embed, embed_preserved=embed_preserved, offset=off, tok=tok)


def load_ankh():
    import torch
    from transformers import AutoTokenizer, T5EncoderModel
    path = MODELS / "ankh-base"
    tok = AutoTokenizer.from_pretrained(path)
    m = T5EncoderModel.from_pretrained(path).eval().cuda()

    @torch.no_grad()
    def embed(seq):
        e = tok([list(seq)], is_split_into_words=True, add_special_tokens=True, return_tensors="pt")
        assert e["input_ids"].shape[1] == len(seq) + 1, "Ankh: expected one token per residue + </s>"
        h = m(input_ids=e["input_ids"].cuda(), attention_mask=e["attention_mask"].cuda()).last_hidden_state[0]
        return h[:len(seq)].float().cpu().numpy()
    return SimpleNamespace(embed=embed, embed_preserved=None)


def esm2_matrix():
    """Stored ESM-2 taxonomy fidelities, (120, 5) in target order (row order of the stored run)."""
    d = pd.read_parquet(TAXO / "fidelity_both.parquet")
    return d[d.backbone == "esm2"]["fidelity"].to_numpy().reshape(-1, len(mg.REL))


def summarize(M):
    return dict(n=M.shape[0], position_gap_pp=metrics.position_gap(M, list(mg.REL)) * 100,
                shift_sensitivity=metrics.shift_sensitivity(M), robust_accuracy=metrics.robust_accuracy(M),
                curve=[float(x) for x in M.mean(0)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["esm1b", "ankh"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)                            # GPU job; leave the CPU to the ProtBERT run
    t0 = time.time()
    set_global_seed(42)
    targets = mg.benchmark_targets()
    keep = [k for k, (p, _) in enumerate(targets) if MAXLEN[args.model] is None or len(p) <= MAXLEN[args.model]]
    if args.smoke:
        keep = keep[:3]
    bb = load_esm1b() if args.model == "esm1b" else load_ankh()
    if args.smoke and args.model == "esm1b":           # explicit default ids must equal the implicit ones
        p, t = targets[keep[0]]
        s = mg.window_start(t, 0.5)
        win = p[s:s + mg.W]
        default = np.arange(len(win) + 2) + bb.offset
        print(f"explicit-vs-implicit ids max|diff| = {np.abs(bb.embed(win, default) - bb.embed(win)).max():.2e}")
    rows = []
    for k in keep:
        p, t = targets[k]
        full = bb.embed(p)[t]
        for rel in mg.REL:
            s = mg.window_start(t, rel)
            rec = dict(target=k, L=len(p), rel=rel, fidelity=cos(bb.embed(p[s:s + mg.W])[t - s], full))
            if bb.embed_preserved is not None:
                rec["fidelity_preserved"] = cos(bb.embed_preserved(p, s)[t - s], full)
            rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"[{args.model}] {len(keep)} targets, non-finite {int((~np.isfinite(df.fidelity)).sum())}, "
          f"{time.time() - t0:.0f}s", flush=True)
    M = df.pivot_table(index="target", columns="rel", values="fidelity")[list(mg.REL)].to_numpy()
    if args.smoke:
        print(df.groupby("rel")[[c for c in df.columns if c.startswith("fidelity")]].mean().round(4).to_string())
        return 0

    E = esm2_matrix()[keep]
    g_new, g_e2 = M[:, 2] - M[:, [0, 4]].mean(1), E[:, 2] - E[:, [0, 4]].mean(1)
    vs = mg.gap_summary(g_new, g_e2)                    # paired per-target gap difference (pp)
    summary = dict(model=args.model, n_targets=len(keep), metrics=summarize(M), esm2_same_targets=summarize(E),
                   gap_minus_esm2=vs, verdict=("replicates" if vs["gap_pp"] > 0 and vs["p"] < P_BAR
                                               else "does not replicate"), runtime_s=time.time() - t0)
    if "fidelity_preserved" in df:
        P = df.pivot_table(index="target", columns="rel", values="fidelity_preserved")[list(mg.REL)].to_numpy()
        g_pres = P[:, 2] - P[:, [0, 4]].mean(1)
        d = mg.gap_summary(g_new, g_pres)
        summary["index_preserved"] = dict(metrics=summarize(P), standard_minus_preserved=d,
                                          index_share=d["gap_pp"] / (g_new.mean() * 100))
    out = OUTROOT / args.model
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "fidelity.parquet")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    m, e = summary["metrics"], summary["esm2_same_targets"]
    print(f"[{args.model}] gap {m['position_gap_pp']:+.2f}pp (ESM-2 same targets {e['position_gap_pp']:+.2f}) "
          f"diff {vs['gap_pp']:+.2f} CI[{vs['ci_lo']:+.2f},{vs['ci_hi']:+.2f}] p={vs['p']:.4f} -> "
          f"{summary['verdict']}; shift {m['shift_sensitivity']:.3f} robust {m['robust_accuracy']:.3f}", flush=True)
    if "index_preserved" in summary:
        ip = summary["index_preserved"]
        print(f"[{args.model}] index-preserved gap {ip['metrics']['position_gap_pp']:+.2f}pp; index share "
              f"{ip['index_share'] * 100:.0f}%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
