"""ProteinGym crop-shift (VEP) analysis: Spearman rho of model scores vs DMS per placement, per mutant.

Rows of the stored run (experiments/phase0_pilot/p2b/crop_shift.parquet) were written mutant by mutant, one row
per placement, so a mutant id can be reconstructed from row order when the file lacks one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from calibrate.stats import paired_bootstrap_ci

REL = (0.1, 0.5, 0.9)


def crop_shift_assays(d: pd.DataFrame, rel: tuple[float, ...] = REL) -> pd.DataFrame:
    """Per assay: rho at the center and (mean of) the edge placements, their difference, and the mean
    per-mutant std of the score across placements (Shift Sensitivity, in nats)."""
    d = d.reset_index(drop=True).copy()
    if "mut" not in d:
        d["mut"] = np.arange(len(d)) // len(rel)
    per = d.groupby("mut")
    assert (per.rel_pos.apply(tuple) == tuple(rel)).all(), "each mutant must have one row per placement"
    assert (per[["DMS_id", "pos", "dms"]].nunique() == 1).all().all(), "rows of one mutant must agree"
    out = []
    for a, g in d.groupby("DMS_id"):
        piv = g.pivot_table(index="mut", columns="rel_pos", values="score")
        dms = g.groupby("mut")["dms"].first().loc[piv.index]
        rho = {r: spearmanr(piv[r], dms).statistic for r in rel}
        out.append(dict(DMS_id=a, n=len(piv), rho_center=rho[0.5], rho_edge=(rho[rel[0]] + rho[rel[-1]]) / 2,
                        shift_sensitivity=float(piv.std(axis=1).mean())))
    t = pd.DataFrame(out).set_index("DMS_id")
    t["delta_rho"] = t.rho_center - t.rho_edge
    return t


def delta_rho_summary(t: pd.DataFrame) -> dict:
    """Center - edge rho over assays (paired bootstrap) and the mean Shift Sensitivity."""
    b = paired_bootstrap_ci(t.rho_center.values, t.rho_edge.values, seed=42)
    return dict(n=len(t), delta_rho=b["diff"], ci=[b["ci_low"], b["ci_high"]], p=b["p_value"],
                shift=float(t.shift_sensitivity.mean()))
