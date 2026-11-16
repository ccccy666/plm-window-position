"""C1 domain-offset design and analysis (moved from scripts/phase3/c1_domain_offsets.py).

The pre-registered design and decision rules are documented in that script's docstring; this module
holds the residue set and the analysis so any backbone can be run through them unchanged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate.stats import paired_bootstrap_ci

ROOT = Path(__file__).resolve().parents[2]
SP = ROOT / "data/processed/p1_0/swissprot_sample.fasta"
CACHE = ROOT / "rebuttal/reviewer3/domains_cache.json"

W = 256
MARGIN = round(0.9 * W)
REL = [0.1, 0.3, 0.5, 0.7, 0.9]
OFFSETS = [0.1, 0.25, 0.5, 0.75, 0.9]
POLICIES = ("dom_center", "dom_start", "dom_end")
DMIN, DMAX = 50, W - 20
AA = set("ACDEFGHIKLMNPQRSTVWY")
N_BOOT, SEED = 10_000, 42
RF_BAR, RF_CI_BAR, P_BAR = 0.68, 0.50, 0.01


def parse_pool(path: Path = SP) -> list[tuple[str, str]]:
    seqs, acc, cur, a = [], [], [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur)); acc.append(a); cur = []
            m = re.match(r">\w+\|([^|]+)\|", line)
            a = m.group(1) if m else None
        elif line.strip():
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur)); acc.append(a)
    return [(x, s) for x, s in zip(acc, seqs) if x and set(s) <= AA and len(s) >= 461]


def domain_center_start(d0: int, d1: int, L: int) -> int | None:
    """Reviewer-3 rule: window centered on D, clipped so it contains D and stays inside the protein."""
    lo, hi = max(0, d1 - W + 1), min(d0, L - W)
    if lo > hi:
        return None
    return int(np.clip(round((d0 + d1) / 2 - W / 2), lo, hi))


def build_residues(pool_path: Path = SP, cache_path: Path = CACHE) -> pd.DataFrame:
    cache = json.loads(Path(cache_path).read_text())
    rows = []
    for acc, seq in parse_pool(pool_path):
        L = len(seq)
        for (d0, d1) in cache.get(acc) or []:
            size = d1 - d0 + 1
            if d0 < 0 or d1 >= L or not (DMIN <= size <= DMAX):
                continue
            s_c = domain_center_start(d0, d1, L)
            s_s, s_e = d0, d1 - W + 1
            if s_c is None or s_s + W > L or s_e < 0:
                continue
            for u in OFFSETS:
                t = d0 + round(u * (size - 1))
                if MARGIN <= t <= L - MARGIN:
                    rows.append(dict(acc=acc, seq=seq, L=L, d0=d0, d1=d1, size=size, u=u, t=t,
                                     s_dom_center=s_c, s_dom_start=s_s, s_dom_end=s_e))
    return pd.DataFrame(rows)


def protein_bootstrap_rf(df: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """Recovered fraction over pooled residues, with a protein-level bootstrap CI."""
    def rf(d):
        e, c, p = d["edge"].mean(), d["f_0.5"].mean(), d[col].mean()
        return (p - e) / (c - e)
    prots = df["acc"].unique()
    groups = {a: g for a, g in df.groupby("acc")}
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(prots, len(prots), replace=True)
        boot.append(rf(pd.concat([groups[a] for a in pick], ignore_index=True)))
    boot = np.array(boot)
    return float(rf(df)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize_domain(df: pd.DataFrame) -> dict:
    """Pre-registered C1 analysis: center-edge by offset, Q1 recovered fractions, Q2 domain-side DiD."""
    df = df.copy()
    df["edge"] = (df["f_0.1"] + df["f_0.9"]) / 2
    df["A"] = df["f_0.1"] - df["f_0.9"]
    summary = dict(n_residues=len(df), n_proteins=int(df.acc.nunique()),
                   n_domains=int(df.groupby(["acc", "d0"]).ngroups),
                   prereg=dict(rf_bar=RF_BAR, rf_ci_bar=RF_CI_BAR, p_bar=P_BAR, unit="protein"))

    summary["center_minus_edge"] = {}
    for u, g in df.groupby("u"):
        pm = g.groupby("acc")[["f_0.5", "edge"]].mean()
        r = paired_bootstrap_ci(pm["f_0.5"].values, pm["edge"].values, n_boot=N_BOOT, seed=SEED)
        summary["center_minus_edge"][str(u)] = dict(n_prot=len(pm), gap_pp=r["diff"] * 100,
                                                    ci=[r["ci_low"] * 100, r["ci_high"] * 100],
                                                    p=r["p_value"])

    summary["Q1"] = {}
    off = df[df.u != 0.5]
    for pol in POLICIES:
        col = f"f_{pol}"
        rf_off, lo, hi = protein_bootstrap_rf(off, col)
        verdict = "RECOMMEND (general)" if (rf_off >= RF_BAR and lo >= RF_CI_BAR) else "not general"
        per_u = {str(u): dict(rf=float((g[col].mean() - g.edge.mean()) /
                                       (g["f_0.5"].mean() - g.edge.mean())),
                              mean_rel=float(g[f"rel_{pol}"].mean()))
                 for u, g in df.groupby("u")}
        summary["Q1"][pol] = dict(rf_off_midpoint=rf_off, ci=[lo, hi], verdict=verdict, per_u=per_u)

    a = df[df.u.isin([0.1, 0.9])].pivot_table(index=["acc", "d0"], columns="u", values="A").dropna()
    a["did"] = a[0.1] - a[0.9]
    did_prot = a.groupby(level="acc")["did"].mean()
    r = paired_bootstrap_ci(did_prot.values, np.zeros(len(did_prot)), n_boot=N_BOOT, seed=SEED)
    q2_verdict = ("domain IS an informative unit" if (r["diff"] > 0 and r["p_value"] < P_BAR)
                  else "no domain-side asymmetry at the pre-registered bar")
    summary["Q2"] = dict(n_domains=len(a), n_prot=len(did_prot), did_pp=r["diff"] * 100,
                         ci=[r["ci_low"] * 100, r["ci_high"] * 100], p=r["p_value"],
                         verdict=q2_verdict,
                         mean_A_by_u={str(u): float(g.A.mean() * 100) for u, g in df.groupby("u")})
    return summary
