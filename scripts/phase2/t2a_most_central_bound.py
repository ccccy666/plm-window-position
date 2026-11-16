"""Table 5, row "most central of the K windows": lower bound on its recovered fraction (no GPU).

The stored remedy runs keep, per target, the recovery at the edge, at the center, and of the uniform and
centrality-weighted mixtures over the five placements {0.1, ..., 0.9}. Masked recovery is linear in the
predictive distribution, so the mean recovery at rel 0.3 / 0.7 (0.2 W off center) follows exactly from them
(calibrate.remedy.off_center_recovery). The most central of K = 5 overlapping windows is at most 0.1 W off
center, so this is a lower bound on its recovery.
Usage: python scripts/phase2/t2a_most_central_bound.py   (calibrate env)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from calibrate.remedy import off_center_recovery, recovered_fraction  # noqa: E402
from calibrate.stats import paired_bootstrap_ci  # noqa: E402


def main() -> None:
    for b in ("esm2", "esmc"):
        d = pd.read_parquet(ROOT / f"experiments/phase2_remedy/{b}_remedy.parquet")
        for w, g in d.groupby("W"):
            e, c = g.single_edge.values, g.single_center.values
            m = off_center_recovery(g.multicrop_uniform.values, e, c)
            r = paired_bootstrap_ci(m, e, seed=42)
            rf = recovered_fraction(m.mean(), e.mean(), c.mean())
            print(f"{b} W={w}: most central of K=5 >= {rf * 100:.0f}% (+{r['diff'] * 100:.1f} pp over the edge, "
                  f"95% CI [{r['ci_low'] * 100:.1f}, {r['ci_high'] * 100:.1f}], p={r['p_value']:.4f})")


if __name__ == "__main__":
    main()
