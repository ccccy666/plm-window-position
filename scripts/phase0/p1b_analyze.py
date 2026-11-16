"""P1' position-sweep analysis (D1: crop-position brittleness).

Key test is WITHIN-CASSETTE (paired): each cassette is measured at every position,
so per-cassette (mid - edge) controls for the large cassette-to-cassette variance.
Reports Position Gap (pp), log-prob gap (nats), amplitude, paired bootstrap CI/p,
vs the pre-registered dual threshold (3pp benchmark / 10pp method; 0.3 nats).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.stats import paired_bootstrap_ci

RAW = (_ROOT / "experiments/phase0_pilot/p1b/position_sweep.parquet")
FIG = (_ROOT / "results/figures/p1b_position_sweep.png")
TAB = (_ROOT / "results/tables/p1b_position_gap.csv")

EDGE = [0.05, 0.95]
MID = [0.35, 0.5, 0.65]


def main() -> None:
    df = pd.read_parquet(RAW)
    out = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for (ct, L), g in df.groupby(["carrier_type", "L"]):
        piv = g.pivot_table(index="cassette", columns="rel_pos", values="center_rec")
        edge = piv[EDGE].mean(axis=1)
        mid = piv[MID].mean(axis=1)
        res = paired_bootstrap_ci(mid.values, edge.values, seed=42)  # mid - edge
        amp_pp = (piv.mean(axis=0).max() - piv.mean(axis=0).min()) * 100
        # logprob gap (nats), paired
        pivl = g.pivot_table(index="cassette", columns="rel_pos", values="center_logprob")
        nats = float((pivl[MID].mean(axis=1) - pivl[EDGE].mean(axis=1)).mean())
        out.append(dict(carrier_type=ct, L=L, n=piv.shape[0],
                        mid_mean=mid.mean(), edge_mean=edge.mean(),
                        gap_mid_minus_edge_pp=res["diff"] * 100,
                        ci_lo_pp=res["ci_low"] * 100, ci_hi_pp=res["ci_high"] * 100,
                        p=res["p_value"], amplitude_pp=amp_pp, gap_nats=nats))
        ax = axes[0] if ct == "real" else axes[1]
        m = g.groupby("rel_pos")["center_rec"].mean()
        ax.plot(m.index, m.values, marker="o", label=f"L={L}")
        ax.set(title=f"{ct} carrier", xlabel="rel_pos in window", ylabel="center-residue recovery")
        ax.legend()

    res_df = pd.DataFrame(out)
    TAB.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(TAB, index=False)
    print("=== P1' within-cassette position gap (mid - edge; +=center better) ===")
    print(res_df.round(3).to_string(index=False))

    fig.suptitle("P1' position sweep (D1): fixed cassette, constant flank, slid across window")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"\nfigure -> {FIG}")

    # verdict vs pre-registered dual threshold
    max_abs_gap = res_df["gap_mid_minus_edge_pp"].abs().max()
    max_amp = res_df["amplitude_pp"].max()
    sig = res_df[(res_df["p"] < 0.01)]
    print("\n=== D1 read (pre-registered: benchmark >=3pp/0.3nats, method >=10pp, p<0.01) ===")
    print(f"  max |mid-edge gap| = {max_abs_gap:.2f} pp ; max amplitude across bins = {max_amp:.2f} pp")
    print(f"  strata with p<0.01: {len(sig)}/{len(res_df)}")
    print(f"  max |gap_nats| = {res_df['gap_nats'].abs().max():.3f} nats")


if __name__ == "__main__":
    main()
