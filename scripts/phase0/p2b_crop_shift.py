"""D4 — crop-shift sensitivity on ProteinGym (does window position hurt VEP?).

For each interior single mutation, score it (ESM-2 wt-marginals) with the mutation
placed at rel-pos {0.1,0.5,0.9} of a W-window. Tests whether centering the mutation
in the inference window gives better DMS correlation than edge placement (the real-
task consequence of the P1' inverted-U), and the per-mutation Shift Sensitivity.

Usage: python scripts/phase0/p2b_crop_shift.py
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
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.data.proteingym import parse_mutant
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed
from calibrate.scoping import crop_shift_assays, delta_rho_summary  # per-mutant analysis (tests/test_scoping.py)

META = (_ROOT / "data/processed/proteingym_long_meta.parquet")
OUT = (_ROOT / "experiments/phase0_pilot/p2b")
FIG = (_ROOT / "results/figures/p2b_crop_shift.png")

W = 256
REL_POS = [0.1, 0.5, 0.9]
MARGIN = round(0.9 * W)
N_PER_ASSAY = 120
MIN_FOR_RHO = 40


def main() -> None:
    """Score (GPU) and analyze; `--analyze-only` re-analyzes the stored crop_shift.parquet without a model."""
    if "--analyze-only" in sys.argv:
        analyze(pd.read_parquet(OUT / "crop_shift.parquet"))
        return
    set_global_seed(42)
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(META)
    model = load_esm2()

    rows, mut_id = [], 0
    for _, row in meta.iterrows():
        seq = row["target_seq"]
        df = pd.read_csv(row["local_csv"])
        # interior single mutants (room to place at all rel_pos), WT-matched
        muts = []
        for _, r in df.iterrows():
            try:
                wt, pos, alt = parse_mutant(r["mutant"])
            except ValueError:
                continue
            if MARGIN < pos <= len(seq) - MARGIN and seq[pos - 1] == wt:
                muts.append((pos, wt, alt, float(r["DMS_score"])))
        if len(muts) < MIN_FOR_RHO:
            continue
        if len(muts) > N_PER_ASSAY:
            muts = [muts[i] for i in rng.choice(len(muts), N_PER_ASSAY, replace=False)]
        for pos, wt, alt, dms in muts:
            for rp in REL_POS:
                s = (pos - 1) - round(rp * W)
                crop = seq[s : s + W]
                j = (pos - 1) - s
                lp = model.position_logprobs(crop)
                score = float(lp[j, model.aa_token_index(alt)] - lp[j, model.aa_token_index(wt)])
                rows.append(dict(DMS_id=row["DMS_id"], mut=mut_id, pos=pos, dms=dms, rel_pos=rp, score=score))
            mut_id += 1
        print(f"  {row['DMS_id']}: {len(muts)} interior muts scored")

    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "crop_shift.parquet")
    analyze(d)


def analyze(d: pd.DataFrame) -> None:
    # per-assay Spearman per MUTANT at each placement + center-vs-edge + shift sensitivity
    rdf = crop_shift_assays(d, tuple(REL_POS)).reset_index()
    rdf.to_csv(str(_ROOT / "results/tables/p2b_crop_shift.csv"), index=False)
    print("\n=== per-assay (center vs edge placement) ===")
    print(rdf.round(3).to_string(index=False))
    s = delta_rho_summary(rdf.set_index("DMS_id"))
    print(f"\nΔρ(center-edge) over {s['n']} assays: mean={s['delta_rho']:.3f} "
          f"CI=[{s['ci'][0]:.3f},{s['ci'][1]:.3f}] p={s['p']:.3f} (gate Δρ>=0.03)")
    print(f"mean shift sensitivity (score std across placements) = {s['shift']:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, g in d.groupby("DMS_id"):
        m = g.groupby("rel_pos").apply(
            lambda x: spearmanr(x["score"], x["dms"]).statistic, include_groups=False)
        ax[0].plot(m.index, m.values, marker="o", alpha=0.7, label=a.split("_")[0])
    ax[0].set(title="Spearman ρ vs mutation placement in window", xlabel="mutation rel_pos",
              ylabel="ρ (ESM-2 vs DMS)")
    ax[0].legend(fontsize=6)
    ax[1].bar(range(len(rdf)), rdf["delta_rho"])
    ax[1].axhline(0.03, ls="--", c="red", label="gate Δρ=0.03")
    ax[1].set(title="Δρ (center - edge) per assay", xticks=range(len(rdf)),
              xlabel="assay", ylabel="Δρ")
    ax[1].set_xticklabels([a.split("_")[0] for a in rdf["DMS_id"]], rotation=60, fontsize=6)
    ax[1].legend()
    fig.suptitle("D4: crop-shift sensitivity on ProteinGym (ESM-2 wt-marginals)")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"\nfigure -> {FIG}")


if __name__ == "__main__":
    main()
