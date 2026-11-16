"""P1.0 analysis (Risk B instrument check).

Questions:
  Q1: does a position effect appear in MLM recovery ALONE (not just attention)?
  Q2: does it survive in an in-distribution carrier (real / gfp), not just shuffled?
Outputs an amplitude table, a paired real-vs-shuffled bootstrap, and a figure.
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.stats import amplitude_pp, paired_bootstrap_ci

RAW = (_ROOT / "experiments/phase0_pilot/p1_0/raw.parquet")
FIG = (_ROOT / "results/figures/p1_0_micropilot.png")
TAB = (_ROOT / "results/tables/p1_0_amplitude.csv")


def main() -> None:
    df = pd.read_parquet(RAW)

    # --- amplitude across rel_pos, per (carrier_type, L), for both metrics ---
    rows = []
    for (ct, L), g in df.groupby(["carrier_type", "L"]):
        rec_by_pos = g.groupby("rel_pos")["recovery"].mean()
        att_by_pos = g.groupby("rel_pos")["attn_mass"].mean()
        rows.append(
            dict(
                carrier_type=ct,
                L=L,
                recovery_mean=g["recovery"].mean(),
                recovery_amp_pp=amplitude_pp(rec_by_pos.values),
                attn_amp_pp=amplitude_pp(att_by_pos.values),
                uniform_attn_baseline=1.0 / (L + 2),
            )
        )
    amp = pd.DataFrame(rows).sort_values(["carrier_type", "L"]).reset_index(drop=True)
    TAB.parent.mkdir(parents=True, exist_ok=True)
    amp.to_csv(TAB, index=False)
    print("=== Amplitude across rel_pos (pp) and mean levels ===")
    print(amp.round(4).to_string(index=False))

    # --- paired real vs shuffled (same carrier window) ---
    print("\n=== Paired real - shuffled recovery (bootstrap) ===")
    piv = df[df.paired_id.notna()].pivot_table(
        index="paired_id", columns="carrier_type", values="recovery"
    )
    piv = piv.dropna()
    res = paired_bootstrap_ci(piv["real"].values, piv["shuffled_real"].values, seed=42)
    print(f"  n_pairs={len(piv)}  diff(pp)={res['diff']*100:.2f}  "
          f"CI=[{res['ci_low']*100:.2f},{res['ci_high']*100:.2f}]pp  p={res['p_value']:.3f}")

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (ct, L), g in df.groupby(["carrier_type", "L"]):
        lab = f"{ct} L={L}"
        m = g.groupby("rel_pos")[["recovery", "attn_mass"]].mean()
        axes[0].plot(m.index, m["recovery"], marker="o", label=lab)
        axes[1].plot(m.index, m["attn_mass"], marker="o", label=lab)
    axes[0].set(title="MLM recovery of needle vs position", xlabel="rel_pos p/L",
                ylabel="recovery prob")
    axes[0].axhline(0.05, ls="--", c="gray", lw=1, label="1/20 chance")
    axes[1].set(title="Norm. attention into needle vs position", xlabel="rel_pos p/L",
                ylabel="mean attn weight")
    axes[0].legend(fontsize=7)
    fig.suptitle("P1.0 micro-pilot (Risk B): is the instrument trustworthy?")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"\nfigure -> {FIG}")

    # --- verdict heuristic ---
    indist = amp[amp.carrier_type.isin(["real", "gfp"])]
    max_rec_amp = indist["recovery_amp_pp"].max()
    print("\n=== Risk-B read ===")
    print(f"  max in-distribution recovery amplitude = {max_rec_amp:.2f} pp (gate wants >=10pp)")
    print(f"  mean recovery level = {amp['recovery_mean'].mean():.3f} (vs 1/20=0.05 chance)")


if __name__ == "__main__":
    main()
