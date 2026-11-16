"""B-tier (B6) — report all three bias metrics (Position Gap / Shift Sensitivity / Robust
Accuracy) across all backbones, for the appendix (benchmark completeness). Pure analysis:
reads the persisted n=120 embedding-fidelity parquets; no model load.

Usage: python scripts/phase3/b6_bias_metrics.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate import metrics

EXP = (_ROOT / "experiments/phase1_diagnosis/t1a_fidelity")
TAB = (_ROOT / "results/tables/b6_bias_metrics.csv")
REL = [0.1, 0.3, 0.5, 0.7, 0.9]
FILES = ["fidelity_both.parquet", "fidelity_esmc.parquet", "fidelity_prott5.parquet",
         "fidelity_protbert.parquet", "fidelity_progen.parquet"]
ORDER = ["esm2", "esmc", "saprot", "prott5", "protbert", "progen2"]
LABEL = {"esm2": "ESM-2 650M", "esmc": "ESM C 300M", "saprot": "SaProt 650M",
         "prott5": "ProtT5", "protbert": "ProtBERT", "progen2": "ProGen2-764M"}


def main() -> None:
    df = pd.concat([pd.read_parquet(EXP / f) for f in FILES if (EXP / f).exists()],
                   ignore_index=True)
    rows = []
    for b in ORDER:
        sub = df[df.backbone == b]
        if sub.empty:
            continue
        M = sub.pivot_table(index="target", columns="rel_pos", values="fidelity")[REL].to_numpy()
        rows.append(dict(
            backbone=LABEL[b], n=M.shape[0],
            position_gap_pp=round(metrics.position_gap(M, REL) * 100, 2),
            shift_sensitivity=round(metrics.shift_sensitivity(M), 4),
            robust_accuracy=round(metrics.robust_accuracy(M), 4),
        ))
    tab = pd.DataFrame(rows)
    TAB.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(TAB, index=False)
    print(tab.to_string(index=False))
    print(f"\n-> {TAB}")


if __name__ == "__main__":
    main()
