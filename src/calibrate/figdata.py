"""Summary values behind the camera-ready composite figures (no model, no GPU).

Each function reads one stored artifact and returns exactly the numbers a figure panel plots and the paper
quotes, so that `tests/test_figdata.py` can pin them to the text.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PILOT_EDGES = (0.05, 0.95)
FLANK_EDGES = (0.1, 0.9)
CENTER = 0.5


def _edge_gap_pp(piv: pd.DataFrame, edges: tuple[float, float]) -> float:
    return float(((piv[CENTER] - piv[list(edges)].mean(axis=1)) * 100).mean())


def pilot_cassette_edge_gaps(path: Path) -> dict[tuple[str, int], float]:
    """Center - edge(0.05, 0.95) recovery gap (pp) per (carrier type, carrier length) stratum."""
    df = pd.read_parquet(path)
    out = {}
    for (carrier, length), g in df.groupby(["carrier_type", "L"]):
        piv = g.pivot_table(index="cassette", columns="rel_pos", values="center_rec")
        out[(carrier, int(length))] = _edge_gap_pp(piv, PILOT_EDGES)
    return out


def pilot_cassette_profile(path: Path, n_boot: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Recovery relative to the centered placement (pp), averaged over the four carrier strata per
    cassette, with a cassette-level bootstrap 95% CI. Index: rel_pos; columns: mean_pp, ci_lo, ci_hi."""
    df = pd.read_parquet(path)
    piv = df.pivot_table(index=["cassette", "carrier_type", "L"], columns="rel_pos", values="center_rec")
    delta = piv.sub(piv[CENTER], axis=0) * 100
    per_cassette = delta.groupby(level="cassette").mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_cassette), size=(n_boot, len(per_cassette)))
    rows = {}
    for rel in per_cassette.columns:
        x = per_cassette[rel].to_numpy()
        boot = x[idx].mean(axis=1)
        rows[float(rel)] = dict(mean_pp=float(x.mean()), ci_lo=float(np.percentile(boot, 2.5)),
                                ci_hi=float(np.percentile(boot, 97.5)))
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def flank_sweep(path: Path) -> tuple[pd.DataFrame, dict[int, float]]:
    """Mean recovery by (flank f, rel_pos) and the center - edge(0.1, 0.9) gap (pp) per flank."""
    df = pd.read_parquet(path)
    curves = df.groupby(["flank", "rel_pos"])["rec"].mean().unstack("rel_pos")
    gaps = {}
    for f, g in df.groupby("flank"):
        piv = g.pivot_table(index="site", columns="rel_pos", values="rec")
        gaps[int(f)] = _edge_gap_pp(piv, FLANK_EDGES)
    return curves, gaps


def cliff_profile(path: Path) -> tuple[pd.Series, float, float]:
    """Mean recovery by absolute position, and the in-crop / out-of-crop means."""
    df = pd.read_parquet(path)
    by_pos = df.groupby("pos")["recovery"].mean()
    return by_pos, float(df[df.in_crop].recovery.mean()), float(df[~df.in_crop].recovery.mean())


def randinit_gaps(path: Path) -> dict[str, dict]:
    """Center - edge fidelity gap (pp) with CI and p for the model x sequence 2x2."""
    d = json.loads(Path(path).read_text())
    return {k: dict(gap_pp=v["gap_pp"], ci_lo=v["ci_lo"], ci_hi=v["ci_hi"], p=v["p"]) for k, v in d.items()}


def attention_masses(path: Path) -> dict[str, tuple[float, float]]:
    """Target's attention mass on window-dropped residues: (centered window, edge window) per model,
    averaged over the per-target rows of a group-A attention parquet."""
    d = pd.read_parquet(path).groupby("model")[["drop_center", "drop_edge"]].mean()
    return {m: (float(d.loc[m, "drop_center"]), float(d.loc[m, "drop_edge"])) for m in ("trained", "random_init")}


def _domain_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["edge"] = (df["f_0.1"] + df["f_0.9"]) / 2
    return df


def domain_side_profiles(path: Path) -> pd.DataFrame:
    """Mean fidelity by within-domain offset u (rows) and position-only rel placement (columns)."""
    df = _domain_frame(path)
    cols = {f"f_{r}": r for r in (0.1, 0.3, 0.5, 0.7, 0.9)}
    return df.groupby("u")[list(cols)].mean().rename(columns=cols)


def domain_asymmetry_pp(path: Path) -> dict[float, float]:
    """Domain-side asymmetry A = f(rel 0.1) - f(rel 0.9), mean pp per within-domain offset u."""
    df = _domain_frame(path)
    a = (df["f_0.1"] - df["f_0.9"]) * 100
    return {float(u): float(v) for u, v in a.groupby(df["u"]).mean().items()}


def domain_recovered_fraction(path: Path) -> dict[str, dict]:
    """Recovered fraction of the edge->center gap per domain-window policy, per u and pooled off-midpoint."""
    df = _domain_frame(path)

    def rf(d: pd.DataFrame, col: str) -> float:
        e, c = d["edge"].mean(), d["f_0.5"].mean()
        return float((d[col].mean() - e) / (c - e))

    out = {}
    for pol in ("dom_center", "dom_start", "dom_end"):
        col = f"f_{pol}"
        res = {float(u): rf(g, col) for u, g in df.groupby("u")}
        res["off_midpoint"] = rf(df[df.u != CENTER], col)
        out[pol] = res
    return out


def partner_out_means(path: Path) -> dict[str, float]:
    """Mean predicted contact probability under the four W=256 placements of the partner-out probe."""
    df = pd.read_parquet(path)
    return dict(full=float(df.prob_full.mean()), center=float(df.center.mean()),
                partner_edge=float(df.partner_edge.mean()),
                partner_out=float(df.loc[df.out_defined, "partner_out"].mean()))


def partner_multiW(path: Path) -> dict[int, dict]:
    """Per window size: full / centered / partner-at-edge means and the center - edge gap (pp, CI)."""
    rows = json.loads(Path(path).read_text())
    keys = ("n", "full", "center", "partner_edge", "gap_pp", "ci_lo", "ci_hi", "p")
    return {int(r["W"]): {k: r[k] for k in keys} for r in rows}


def tiling_split_rates(path: Path) -> dict[int, tuple[float, float, float]]:
    """Percent of confident long-range contacts split by a tiling: (non-overlap, 50%, 75% overlap) per W."""
    grid = json.loads(Path(path).read_text())["overlap_grid"]
    order = ("non-overlap (s=W)", "50% overlap (s=W/2)", "75% overlap (s=W/4)")
    return {int(w): tuple(g[k]["miss_all"] * 100 for k in order) for w, g in grid.items()}


def split_by_separation(path: Path) -> list[tuple[str, int, float]]:
    """(separation bin, n contacts, percent split) for the non-overlapping W=256 tiling."""
    rows = json.loads(Path(path).read_text())["miss_by_separation_W256_nonoverlap"]
    return [(r["sep_bin"], int(r["n"]), float(r["miss_pct"])) for r in rows]


TAXO_FILES = ("fidelity_both.parquet", "fidelity_esmc.parquet", "fidelity_prott5.parquet",
              "fidelity_protbert.parquet", "fidelity_progen.parquet")
TAXO_REL = (0.1, 0.3, 0.5, 0.7, 0.9)


def _taxonomy_row(model: str, M: pd.DataFrame) -> dict:
    from calibrate.stats import paired_bootstrap_ci
    r = paired_bootstrap_ci(M[0.5].values, M[[0.1, 0.9]].mean(axis=1).values, seed=42)
    return dict(model=model, n=len(M), curve=[float(M[x].mean()) for x in TAXO_REL],
                gap_pp=r["diff"] * 100, ci_lo=r["ci_low"] * 100, ci_hi=r["ci_high"] * 100)


def taxonomy_rows(taxo_dir: Path, b1_dir: Path) -> pd.DataFrame:
    """Per-backbone fidelity curve and center-edge gap (95% CI): the taxonomy runs plus ESM-1b and Ankh (B1)."""
    rows = []
    stored = pd.concat([pd.read_parquet(Path(taxo_dir) / f) for f in TAXO_FILES], ignore_index=True)
    for b, g in stored.groupby("backbone"):
        rows.append(_taxonomy_row(b, g.pivot_table(index="target", columns="rel_pos", values="fidelity")))
    for b in ("esm1b", "ankh"):
        g = pd.read_parquet(Path(b1_dir) / b / "fidelity.parquet")
        rows.append(_taxonomy_row(b, g.pivot_table(index="target", columns="rel", values="fidelity")))
    return pd.DataFrame(rows).set_index("model")
