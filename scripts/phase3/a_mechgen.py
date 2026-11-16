"""Camera-ready group A (GPU): do the ESM-2 mechanism results hold on the other bidirectional encoders?

PRE-REGISTERED (fixed 2026-11-16 before any non-ESM-2 run; do not change after seeing results)
-----------------------------------------------------------------------------------------------
Backbones : ESM-2 650M and SaProt 650M sequence-only (calibrate env), ESM C 300M (esmc env), ProtT5-XL
            encoder and ProtBERT (progen env). Same 120 taxonomy targets (calibrate.mechgen), W = 256,
            relative positions {0.1, 0.3, 0.5, 0.7, 0.9}, last-layer embedding fidelity, each backbone's
            taxonomy precision (ESM-2/SaProt/ProtBERT fp16, ESM C bf16, ProtT5 fp32).
Gate      : the trained/natural center-edge gap must reproduce the stored taxonomy run within 0.05 pp
            (Table 2: ESM-2 3.29, ESM C 3.04, SaProt 2.82, ProtT5 5.49, ProtBERT 8.68); otherwise stop.
Random-init: same architecture and config, library-default initialization, constructed after
            set_global_seed(42) and after the trained model (the procedure of the ESM-2 runs).
A1 (2x2)  : center-edge gap for {trained, random-init} x {natural, uniform random sequences (seed 43)}.
            "Learned, not architectural" replicates iff BOTH control cells (random-init/natural and
            trained/random-seq) are below one third of the trained/natural gap.
A2 (attn) : target's attention row from the full-context pass, averaged over layers and heads and
            renormalized over residue keys, on ALL 120 targets (no length cap; rows are read per layer by
            hooks). Dropped mass at the centered vs the edge windows. Replicates iff the trained model's
            edge-center difference is > 0 at p < 0.01, the random-init model's |edge-center| is below one
            third of the trained difference, and corr(per-target asymmetry, fidelity gap) > 0 at
            permutation p < 0.01 for the trained model.
A3 (ProtBERT): windows fed with each token's full-context position id (index-preserved). Reports the gap
            with and without index preservation; share of the gap due to the absolute-index change
            = (standard - preserved) / standard. Descriptive, no decision bar.
A4 (domain): the C1 design and decision rules, unchanged (calibrate.domains): dom_center recommended as a
            general policy iff RF >= 0.68 with CI-low >= 0.50; own-domain asymmetry iff DiD > 0 at p < 0.01.

Outputs   : experiments/phase3/a_mechgen/<backbone>/{fidelity,attention,domain_residues}.parquet,
            summary.json
Usage     : python scripts/phase3/a_mechgen.py BACKBONE [--smoke] [--skip-domain]
            (esm2|saprot: calibrate env; esmc: esmc env; prott5|protbert: progen env;
             run offline: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = _ROOT
sys.path.insert(0, str(ROOT / "src"))
from calibrate import domains as dm  # noqa: E402
from calibrate import mechgen as mg  # noqa: E402
from calibrate.seed import set_global_seed  # noqa: E402

OUTROOT = ROOT / "experiments/phase3/a_mechgen"
TAXO = ROOT / "experiments/phase1_diagnosis/t1a_fidelity"
TAXO_FILE = {"esm2": "fidelity_both.parquet", "saprot": "fidelity_both.parquet",
             "esmc": "fidelity_esmc.parquet", "prott5": "fidelity_prott5.parquet",
             "protbert": "fidelity_protbert.parquet"}
GATE_PP = 0.05
DEVICE = "cuda"   # --cpu switches ProtBERT to CPU fp32 (two GPU resets on 2026-11-16 were both during ProtBERT)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def layer_mean_row(rows, offset, length):
    import torch
    return mg.residue_row(torch.stack(rows).mean(0).cpu().numpy(), offset, length)


# ----------------------------------------------------------------------------------------- backbones
def load_esm2(random_init):
    import torch
    from calibrate.models import ESM2Wrapper, load_esm2 as _load
    if random_init:
        import esm
        m = esm.model.esm2.ESM2(num_layers=33, embed_dim=1280, attention_heads=20, alphabet="ESM-1b")
        w = ESM2Wrapper(m.eval().half().cuda(), m.alphabet, "cuda")
    else:
        w = _load()

    def embed(seq):
        return w.embed([seq])[0]

    @torch.no_grad()
    def attn_row(seq, t, full_reference=False):
        q, rows = t + 1, []
        if full_reference:  # no hooks: the r2_attention_locality.py path (short proteins only)
            att = w.model(w._tokens([seq]), need_head_weights=True, repr_layers=[])["attentions"][0]
            return mg.residue_row(att[:, :, q, :].float().mean(dim=(0, 1)).cpu().numpy(), 1, len(seq))

        def hook(mod, inp, out):
            x, a = out                                  # a: (heads, B, T, T)
            rows.append(a[:, 0, q, :].float().mean(0))
            return x, a[:, :, q:q + 1, :].clone()      # clone: a view would keep the full T x T alive
        hs = [layer.register_forward_hook(hook) for layer in w.model.layers]
        try:
            w.model(w._tokens([seq]), need_head_weights=True, repr_layers=[])
        finally:
            for h in hs:
                h.remove()
        return layer_mean_row(rows, 1, len(seq))
    return SimpleNamespace(embed=embed, attn_row=attn_row, model=w.model)


def load_saprot(random_init):
    import torch
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
    name = "westlake-repl/SaProt_650M_AF2"
    tok = AutoTokenizer.from_pretrained(name)
    m = (AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(name)) if random_init
         else AutoModelForMaskedLM.from_pretrained(name)).eval().half().cuda()
    default_impl = m.config._attn_implementation

    def ids_of(seq):
        return tok("".join(c + "#" for c in seq), return_tensors="pt")["input_ids"].cuda()

    @torch.no_grad()
    def embed(seq):
        h = m(ids_of(seq), output_hidden_states=True).hidden_states[-1][0]
        return h[1:1 + len(seq)].float().cpu().numpy()

    @torch.no_grad()
    def attn_row(seq, t, full_reference=False):
        q, rows = t + 1, []
        m.set_attn_implementation("eager")
        try:
            if full_reference:
                att = torch.stack(m(ids_of(seq), output_attentions=True).attentions)[:, 0]
                return mg.residue_row(att[:, :, q, :].float().mean(dim=(0, 1)).cpu().numpy(), 1, len(seq))

            def hook(mod, inp, out):
                rows.append(out[1][0, :, q, :].float().mean(0))
            hs = [layer.attention.self.register_forward_hook(hook) for layer in m.esm.encoder.layer]
            try:
                m(ids_of(seq))
            finally:
                for h in hs:
                    h.remove()
        finally:
            m.set_attn_implementation(default_impl)
        return layer_mean_row(rows, 1, len(seq))
    return SimpleNamespace(embed=embed, attn_row=attn_row, model=m)


def load_esmc(random_init):
    import torch
    import torch.nn.functional as F
    from esm.models.esmc import ESMC
    if random_init:
        from esm.tokenization import get_esmc_model_tokenizers
        m = ESMC(d_model=960, n_heads=15, n_layers=30, tokenizer=get_esmc_model_tokenizers(),
                 use_flash_attn=True).eval().to("cuda").to(torch.bfloat16)
    else:
        m = ESMC.from_pretrained("esmc_300m").to("cuda").eval()
    tok = m.tokenizer
    check_errs = []

    def ids_of(seq):
        return torch.tensor([tok.encode(seq)]).cuda()

    @torch.no_grad()
    def embed(seq):
        return m(ids_of(seq)).embeddings[0][1:1 + len(seq)].float().cpu().numpy()

    @torch.no_grad()
    def attn_row(seq, t, full_reference=False):
        """sdpa returns no weights: recompute the query row from q, k (+ mask) and check P @ V."""
        q, rows = t + 1, []
        orig = F.scaled_dot_product_attention

        def wrapped(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
            out = orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                       scale=scale, **kw)
            sc = scale if scale is not None else query.shape[-1] ** -0.5
            s = (query[:, :, q:q + 1, :].float() @ key.float().transpose(-1, -2)) * sc
            if attn_mask is not None:
                mk = attn_mask[..., q:q + 1, :] if attn_mask.shape[-2] > 1 else attn_mask
                s = s.masked_fill(~mk, float("-inf")) if mk.dtype == torch.bool else s + mk.float()
            p = s.softmax(-1)
            rows.append(p[0, :, 0, :].mean(0))
            ref = out[:, :, q:q + 1, :].float()
            check_errs.append(float((p @ value.float() - ref).abs().max() / (ref.abs().max() + 1e-6)))
            return out
        torch.nn.functional.scaled_dot_product_attention = wrapped
        try:
            m(ids_of(seq))
        finally:
            torch.nn.functional.scaled_dot_product_attention = orig
        return layer_mean_row(rows, 1, len(seq))
    return SimpleNamespace(embed=embed, attn_row=attn_row, model=m, check_errs=check_errs)


def load_protbert(random_init):
    os.environ.setdefault("HF_HOME", str(ROOT / "data/cache/models/hf"))
    import torch
    from transformers import AutoConfig, AutoModel, BertTokenizer
    name = "Rostlab/prot_bert"
    tok = BertTokenizer.from_pretrained(name, do_lower_case=False)
    m = (AutoModel.from_config(AutoConfig.from_pretrained(name)) if random_init
         else AutoModel.from_pretrained(name)).eval()
    m = m.half().cuda() if DEVICE == "cuda" else m.float()

    def ids_of(seq):
        return tok(" ".join(seq), return_tensors="pt").input_ids.to(DEVICE)

    @torch.no_grad()
    def embed(seq):
        return m(ids_of(seq)).last_hidden_state[0][1:1 + len(seq)].float().cpu().numpy()

    @torch.no_grad()
    def embed_preserved(prot, s):
        win = prot[s:s + mg.W]
        pos = torch.tensor(np.asarray([mg.preserved_position_ids(len(prot), s)])).to(DEVICE)
        return m(ids_of(win), position_ids=pos).last_hidden_state[0][1:1 + len(win)].float().cpu().numpy()

    @torch.no_grad()
    def attn_row(seq, t, full_reference=False):
        q, rows = t + 1, []
        if full_reference:
            att = torch.stack(m(ids_of(seq), output_attentions=True).attentions)[:, 0]
            return mg.residue_row(att[:, :, q, :].float().mean(dim=(0, 1)).cpu().numpy(), 1, len(seq))

        def hook(mod, inp, out):
            a = out[1]
            rows.append(a[0, :, q, :].float().mean(0))
            return (out[0], a[:, :, q:q + 1, :].clone()) + tuple(out[2:])
        hs = [layer.attention.self.register_forward_hook(hook) for layer in m.encoder.layer]
        try:
            m(ids_of(seq), output_attentions=True)
        finally:
            for h in hs:
                h.remove()
        return layer_mean_row(rows, 1, len(seq))
    return SimpleNamespace(embed=embed, attn_row=attn_row, model=m, embed_preserved=embed_preserved)


def load_prott5(random_init):
    import torch
    from transformers import T5Config, T5EncoderModel, T5Tokenizer
    name = "Rostlab/prot_t5_xl_half_uniref50-enc"
    tok = T5Tokenizer.from_pretrained(name, do_lower_case=False)
    m = (T5EncoderModel(T5Config.from_pretrained(name)) if random_init
         else T5EncoderModel.from_pretrained(name)).to("cuda").eval()

    def enc(seq):
        e = tok(" ".join(re.sub(r"[UZOB]", "X", seq)), return_tensors="pt", add_special_tokens=True)
        return dict(input_ids=e["input_ids"].cuda(), attention_mask=e["attention_mask"].cuda())

    @torch.no_grad()
    def embed(seq):
        return m(**enc(seq)).last_hidden_state[0][:len(seq)].float().cpu().numpy()

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
        return layer_mean_row(rows, 0, len(seq))
    return SimpleNamespace(embed=embed, attn_row=attn_row, model=m)


LOADERS = dict(esm2=load_esm2, saprot=load_saprot, esmc=load_esmc, protbert=load_protbert,
               prott5=load_prott5)


# ----------------------------------------------------------------------------------------- steps
def fidelity_rows(bb, tag, seqtype, targets, seqs, preserved=False):
    rows = []
    for i, ((_, t), seq) in enumerate(zip(targets, seqs)):
        full = bb.embed(seq)[t]
        for rel in mg.REL:
            s = mg.window_start(t, rel)
            rec = dict(model=tag, seqtype=seqtype, target=i, L=len(seq), t=t, rel=rel,
                       fidelity=cos(bb.embed(seq[s:s + mg.W])[t - s], full))
            if preserved:
                rec["fidelity_preserved"] = cos(bb.embed_preserved(seq, s)[t - s], full)
            rows.append(rec)
    return rows


def center_edge(df, col="fidelity"):
    g = df.pivot_table(index="target", columns="rel", values=col)
    return g[0.5].values, g[[0.1, 0.9]].mean(axis=1).values, {str(r): float(g[r].mean()) for r in mg.REL}


def gate(name, df):
    stored = pd.read_parquet(TAXO / TAXO_FILE[name])
    if "backbone" in stored:
        stored = stored[stored.backbone == name]
    old = stored["fidelity"].to_numpy().reshape(-1, len(mg.REL))     # stored row order = target order
    new = df.sort_values(["target", "rel"])["fidelity"].to_numpy().reshape(-1, len(mg.REL))
    c_old, e_old = old[:, 2], old[:, [0, 4]].mean(1)
    c_new, e_new = new[:, 2], new[:, [0, 4]].mean(1)
    old_pp, new_pp = (c_old - e_old).mean() * 100, (c_new - e_new).mean() * 100
    return dict(stored_gap_pp=float(old_pp), new_gap_pp=float(new_pp), diff_pp=float(new_pp - old_pp),
                max_abs_fidelity_diff=float(np.abs(old - new).max()),
                passed=bool(abs(new_pp - old_pp) <= GATE_PP))


def attention_rows(bb, tag, targets, fid):
    """fid: per-target (center, edge) fidelity from the A1 natural run of the same model."""
    rows = []
    for i, (prot, t) in enumerate(targets):
        row = bb.attn_row(prot, t)
        de = np.mean([mg.dropped_mass(row, mg.window_start(t, r)) for r in (0.1, 0.9)])
        rows.append(dict(model=tag, target=i, L=len(prot), row_sum=float(row.sum()),
                         drop_center=mg.dropped_mass(row, mg.window_start(t, 0.5)), drop_edge=float(de),
                         fid_center=fid[i][0], fid_edge=fid[i][1]))
    return rows


def domain_rows(bb, res):
    full, win = {}, {}
    out = []
    for r in res.itertuples(index=False):
        if r.acc not in full:
            full[r.acc] = bb.embed(r.seq)
        hf = full[r.acc][r.t]
        rec = dict(acc=r.acc, d0=r.d0, d1=r.d1, size=r.size, u=r.u, t=r.t, L=r.L)
        for rel in dm.REL:
            s = r.t - round(rel * dm.W)
            rec[f"f_{rel}"] = cos(bb.embed(r.seq[s:s + dm.W])[r.t - s], hf)
        for pol in dm.POLICIES:
            s = getattr(r, f"s_{pol}")
            if (r.acc, s) not in win:
                win[(r.acc, s)] = bb.embed(r.seq[s:s + dm.W])
            rec[f"f_{pol}"] = cos(win[(r.acc, s)][r.t - s], hf)
            rec[f"rel_{pol}"] = (r.t - s) / dm.W
        out.append(rec)
    return pd.DataFrame(out)


def release():
    """Free GPU memory once the caller has dropped its last reference to a backbone (closures included)."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def checkpoint(out, fid_rows, attn_rows):
    """Persist partial results after each step, so a crash does not lose the finished steps."""
    out.mkdir(parents=True, exist_ok=True)
    if fid_rows:
        pd.DataFrame(fid_rows).to_parquet(out / "fidelity.partial.parquet")
    if attn_rows:
        pd.DataFrame(attn_rows).to_parquet(out / "attention.partial.parquet")


# ----------------------------------------------------------------------------------------- verdicts
def a1_verdict(cells):
    tn = cells["trained/natural"]["gap_pp"]
    ctrl = {k: cells[k]["gap_pp"] for k in ("random_init/natural", "trained/random_seq")}
    ok = all(v < tn / 3 for v in ctrl.values())
    return dict(trained_natural_pp=tn, controls_pp=ctrl, bar_pp=tn / 3,
                verdict="replicates" if ok else "does not replicate")


def a2_verdict(tr, ri):
    ok = (tr["drop_edge_minus_center_pp"] > 0 and tr["drop_p"] < 0.01
          and abs(ri["drop_edge_minus_center_pp"]) < tr["drop_edge_minus_center_pp"] / 3
          and tr["corr_asym_vs_gap"] > 0 and tr["corr_p"] < 0.01)
    return "replicates" if ok else "does not replicate"


def summarize_attention(df):
    return mg.attention_summary(df.drop_center, df.drop_edge, df.fid_center, df.fid_edge)


# ----------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone", choices=sorted(LOADERS))
    ap.add_argument("--smoke", action="store_true", help="3 targets + hook-vs-full checks, no outputs")
    ap.add_argument("--skip-domain", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="ProtBERT only: CPU fp32, 6 threads")
    args = ap.parse_args()
    global DEVICE
    if args.cpu:
        if args.backbone != "protbert":
            ap.error("--cpu is implemented for protbert only")
        import torch
        DEVICE = "cpu"
        torch.set_num_threads(6)
    name, t0 = args.backbone, time.time()
    out = OUTROOT / name
    set_global_seed(42)
    targets = mg.benchmark_targets()
    natural = [p for p, _ in targets]
    randseq = mg.random_carriers(natural)
    if args.smoke:
        keep = sorted(range(len(targets)), key=lambda i: len(targets[i][0]))[:2] + [
            max(range(len(targets)), key=lambda i: len(targets[i][0]))]
        targets = [targets[i] for i in keep]
        natural, randseq = [natural[i] for i in keep], [randseq[i] for i in keep]
    print(f"[{name}] {len(targets)} targets, lengths {min(map(len, natural))}-{max(map(len, natural))}",
          flush=True)
    summary = dict(backbone=name, n_targets=len(targets))
    fid_rows, attn_rows = [], []

    # ---- trained model ----
    bb = LOADERS[name](random_init=False)
    nat = fidelity_rows(bb, "trained", "natural", targets, natural, preserved=(name == "protbert"))
    fid_rows += nat
    dnat = pd.DataFrame(nat)
    if not args.smoke:
        summary["gate"] = gate(name, dnat)
        print(f"[{name}] gate: {summary['gate']}", flush=True)
        if not summary["gate"]["passed"]:
            out.mkdir(parents=True, exist_ok=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
            print(f"[{name}] GATE FAILED -- stopping", flush=True)
            return 1
    fid_rows += fidelity_rows(bb, "trained", "random_seq", targets, randseq)
    if not args.smoke:
        checkpoint(out, fid_rows, attn_rows)
    c, e, _ = center_edge(dnat)
    fid_tr = list(zip(c, e))
    if args.smoke:
        short = min(range(len(targets)), key=lambda i: len(targets[i][0]))
        p, t = targets[short]
        a_hook, a_ref = bb.attn_row(p, t), bb.attn_row(p, t, full_reference=True)
        print(f"[{name}] smoke: hook-vs-full max|diff| = {np.abs(a_hook - a_ref).max():.2e} "
              f"(row sums {a_hook.sum():.4f}/{a_ref.sum():.4f})", flush=True)
    attn_rows += attention_rows(bb, "trained", targets, fid_tr)
    if not args.smoke:
        checkpoint(out, fid_rows, attn_rows)
    if getattr(bb, "check_errs", None):
        summary["esmc_sdpa_check_max_rel_err"] = max(bb.check_errs)
        print(f"[{name}] sdpa P@V check: max rel err {max(bb.check_errs):.2e}", flush=True)
    if not args.skip_domain:
        res = dm.build_residues()
        if args.smoke:
            res = res.head(5)
        ddf = domain_rows(bb, res)
        if not args.smoke:
            out.mkdir(parents=True, exist_ok=True)
            ddf.to_parquet(out / "domain_residues.parquet")
            summary["A4"] = dm.summarize_domain(ddf)
    bb = None
    release()
    print(f"[{name}] trained done ({time.time() - t0:.0f}s)", flush=True)

    # ---- random-init model (constructed after the trained one, as in the ESM-2 runs) ----
    bb = LOADERS[name](random_init=True)
    rnat = fidelity_rows(bb, "random_init", "natural", targets, natural)
    fid_rows += rnat + fidelity_rows(bb, "random_init", "random_seq", targets, randseq)
    c, e, _ = center_edge(pd.DataFrame(rnat))
    attn_rows += attention_rows(bb, "random_init", targets, list(zip(c, e)))
    if not args.smoke:
        checkpoint(out, fid_rows, attn_rows)
    bb = None
    release()
    print(f"[{name}] random-init done ({time.time() - t0:.0f}s)", flush=True)

    fdf, adf = pd.DataFrame(fid_rows), pd.DataFrame(attn_rows)
    bad = int((~np.isfinite(fdf.fidelity)).sum()) + int((~np.isfinite(adf.drop_edge)).sum())
    print(f"[{name}] non-finite values: {bad}; attention row sums in "
          f"[{adf.row_sum.min():.4f}, {adf.row_sum.max():.4f}]", flush=True)
    if args.smoke:
        print(fdf.groupby(["model", "seqtype", "rel"]).fidelity.mean().unstack().round(4).to_string())
        print(adf[["model", "L", "drop_center", "drop_edge"]].round(3).to_string())
        return 0

    cells = {}
    for (mt, st), g in fdf.groupby(["model", "seqtype"]):
        ce, ed, curve = center_edge(g)
        cells[f"{mt}/{st}"] = dict(mg.gap_summary(ce, ed), curve=curve)
    summary["A1"] = dict(cells=cells, **a1_verdict(cells))
    att = {}
    for mt, g in adf.groupby("model"):
        att[mt] = dict(all=summarize_attention(g), le700=summarize_attention(g[g.L <= 700]))
    summary["A2"] = dict(att, verdict=a2_verdict(att["trained"]["all"], att["random_init"]["all"]))
    if name == "protbert":
        g = fdf[(fdf.model == "trained") & (fdf.seqtype == "natural")]
        cs, es, curve_s = center_edge(g)
        cp, ep, curve_p = center_edge(g, "fidelity_preserved")
        std, pres = mg.gap_summary(cs, es), mg.gap_summary(cp, ep)
        diff = mg.gap_summary(cs - es, cp - ep)
        summary["A3"] = dict(standard=dict(std, curve=curve_s), preserved=dict(pres, curve=curve_p),
                             standard_minus_preserved=diff,
                             index_share=diff["gap_pp"] / std["gap_pp"])
    if "A4" in summary:
        q1, q2 = summary["A4"]["Q1"]["dom_center"], summary["A4"]["Q2"]
        summary["A4_verdict"] = dict(dom_center=q1["verdict"], own_domain=q2["verdict"])
    summary["runtime_s"] = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    fdf.to_parquet(out / "fidelity.parquet")
    adf.to_parquet(out / "attention.parquet")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[{name}] A1 {summary['A1']['verdict']}: " + ", ".join(
        f"{k} {v['gap_pp']:+.2f}pp (p={v['p']:.3f})" for k, v in cells.items()), flush=True)
    tr = att["trained"]["all"]
    print(f"[{name}] A2 {summary['A2']['verdict']}: trained drop center {tr['drop_center']:.3f} edge "
          f"{tr['drop_edge']:.3f} ({tr['drop_edge_minus_center_pp']:+.2f}pp p={tr['drop_p']:.4f}), "
          f"r={tr['corr_asym_vs_gap']:+.2f} (p={tr['corr_p']:.4f}); random-init "
          f"{att['random_init']['all']['drop_edge_minus_center_pp']:+.2f}pp", flush=True)
    if "A3" in summary:
        a3 = summary["A3"]
        print(f"[{name}] A3 standard {a3['standard']['gap_pp']:+.2f}pp, preserved "
              f"{a3['preserved']['gap_pp']:+.2f}pp, index share {a3['index_share']*100:.0f}%", flush=True)
    if "A4" in summary:
        q1, q2 = summary["A4"]["Q1"], summary["A4"]["Q2"]
        print(f"[{name}] A4 dom_center RF {q1['dom_center']['rf_off_midpoint']*100:.1f}% "
              f"CI[{q1['dom_center']['ci'][0]*100:.1f},{q1['dom_center']['ci'][1]*100:.1f}] "
              f"({q1['dom_center']['verdict']}); dom_start {q1['dom_start']['rf_off_midpoint']*100:.1f}%, "
              f"dom_end {q1['dom_end']['rf_off_midpoint']*100:.1f}%; DiD {q2['did_pp']:+.2f}pp "
              f"p={q2['p']:.4f} ({q2['verdict']})", flush=True)
    print(f"[{name}] done in {summary['runtime_s']:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
