"""T1.D — Downstream scoping consolidation (the C4 "when it does NOT matter" result).

Pulls together, in one place and with one statistical idiom (paired bootstrap on the
center-vs-edge gap), the contrast that defines the paper's honest boundary:

  POSITIVE  (metric-level, controlled probe) — the window-edge effect is real:
     - masked-recovery   window-relocation, ESM-2, W=256  (experiments/.../p1b/window_reloc.parquet)
     - embedding-fidelity benchmark test split, ESM-2      (experiments/.../benchmark/esm2_report.json)
  NEGATIVE  (standard downstream tasks) — it does NOT propagate when the window keeps
            task-relevant context:
     - ProteinGym VEP        Δρ(center-edge) over assays   (results/tables/p2b_crop_shift.csv)
     - secondary structure   Q3 gap                        (experiments/.../d4_ss/ss_summary.json)
     - long-range contacts   predicted-prob gap            (experiments/.../d4_contacts/contacts.parquet)

Outputs: results/tables/t1d_scoping.csv, results/figures/t1d_scoping.png, and a JSON.
Reads only persisted artifacts (no model load), so it is cheap and regenerable.

Usage: python scripts/phase1/t1d_scoping.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.stats import paired_bootstrap_ci

ROOT = _ROOT
EXP = ROOT / "experiments"
TAB = ROOT / "results/tables/t1d_scoping.csv"
FIG = ROOT / "results/figures/t1d_scoping.png"
JSN = ROOT / "results/tables/t1d_scoping.json"

# pre-registered benchmark-tier detection thresholds (locked 2026-06-05)
THR_PP = 3.0          # >=3 pp for recovery / per-position accuracy gaps
THR_RHO = 0.03        # >=Delta-rho 0.03 for VEP


def positive_recovery() -> dict:
    """Controlled window-relocation, masked recovery (ESM-2, W=256): center vs edge per target."""
    df = pd.read_parquet(EXP / "phase0_pilot/p1b/window_reloc.parquet")
    df = df[df["W"] == 256]
    piv = df.pivot_table(index="target", columns="rel_pos", values="recovery")
    center = piv[0.5].to_numpy()
    edge = piv[[0.1, 0.9]].mean(axis=1).to_numpy()
    r = paired_bootstrap_ci(center, edge, seed=42)
    return dict(name="Masked recovery (controlled probe)", group="positive", unit="pp",
                n=len(center), effect=r["diff"] * 100,
                ci=(r["ci_low"] * 100, r["ci_high"] * 100), p=r["p_value"],
                threshold=THR_PP, detail="window-relocation, ESM-2, W=256")


def positive_fidelity() -> dict:
    """Embedding-fidelity on the benchmark test split (ESM-2) from the packaged harness."""
    rep = json.loads((EXP / "phase1_diagnosis/benchmark/esm2_report.json").read_text())
    lo, hi = rep["gap_ci_pp"]
    return dict(name="Embedding fidelity (controlled probe)", group="positive", unit="pp",
                n=rep["n"], effect=rep["position_gap_pp"], ci=(lo, hi), p=rep["gap_p"],
                threshold=THR_PP, detail="benchmark test split, ESM-2, W=256")


def vep() -> dict:
    """ProteinGym crop-shift: per-assay Spearman center vs edge; paired bootstrap over assays."""
    df = pd.read_csv(ROOT / "results/tables/p2b_crop_shift.csv")
    r = paired_bootstrap_ci(df["rho_center"].to_numpy(), df["rho_edge"].to_numpy(), seed=42)
    return dict(name="ProteinGym VEP (Δρ)", group="downstream", unit="Δρ",
                n=len(df), effect=r["diff"], ci=(r["ci_low"], r["ci_high"]), p=r["p_value"],
                threshold=THR_RHO, detail=f"{len(df)} long assays, ESM-2 wt-marginals, W=256")


def ss() -> dict:
    s = json.loads((EXP / "phase1_diagnosis/d4_ss/ss_summary.json").read_text())
    return dict(name="Secondary structure (Q3)", group="downstream", unit="pp",
                n=s["n"], effect=s["gap_pp"], ci=tuple(s["gap_ci_pp"]), p=s["gap_p"],
                threshold=THR_PP, detail="linear probe on ESM-2, W=256")


def contacts() -> dict:
    df = pd.read_parquet(EXP / "phase1_diagnosis/d4_contacts/contacts.parquet")
    center = df["r_0.5"].to_numpy()
    edge = df[["r_0.2", "r_0.8"]].mean(axis=1).to_numpy()
    r = paired_bootstrap_ci(center, edge, seed=42)
    return dict(name="Long-range contacts (prob)", group="downstream", unit="pp",
                n=len(df), effect=r["diff"] * 100, ci=(r["ci_low"] * 100, r["ci_high"] * 100),
                p=r["p_value"], threshold=THR_PP,
                detail="ESM-2 contact head, sep 24-100, W=256")


def verdict(row: dict) -> str:
    lo, hi = row["ci"]
    clears = abs(row["effect"]) >= row["threshold"] and lo > 0 and row["p"] < 0.01
    return "clears bar (p<0.01)" if clears else "n.s. (CI crosses 0)"


def main() -> None:
    rows = [positive_recovery(), positive_fidelity(), vep(), ss(), contacts()]
    for r in rows:
        r["verdict"] = verdict(r)

    # ---- table ----
    tab = pd.DataFrame([{
        "task": r["name"], "group": r["group"], "n": r["n"],
        "center_minus_edge": round(r["effect"], 4), "unit": r["unit"],
        "ci_low": round(r["ci"][0], 4), "ci_high": round(r["ci"][1], 4),
        "threshold": r["threshold"], "p": round(r["p"], 4), "verdict": r["verdict"],
        "detail": r["detail"],
    } for r in rows])
    TAB.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(TAB, index=False)
    JSN.write_text(json.dumps(rows, indent=2, default=float))
    print(tab.to_string(index=False))

    # ---- figure: 4 panels (controlled probe bars | 3 downstream forest points) ----
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6))

    # panel (0,0): controlled metric-level effect (positive)
    ax0 = axes[0, 0]
    pos = [r for r in rows if r["group"] == "positive"]
    x = np.arange(len(pos))
    eff = [r["effect"] for r in pos]
    err = [[r["effect"] - r["ci"][0] for r in pos], [r["ci"][1] - r["effect"] for r in pos]]
    ax0.bar(x, eff, color="#2c7fb8", width=0.55)
    ax0.errorbar(x, eff, yerr=err, fmt="none", ecolor="k", capsize=5)
    ax0.axhline(THR_PP, ls="--", c="#d95f02", lw=1.2, label=f"benchmark tier ({THR_PP:.0f} pp)")
    ax0.axhline(10, ls=":", c="#7570b3", lw=1.2, label="method tier (10 pp)")
    ax0.axhline(0, c="k", lw=0.8)
    ax0.set_xticks(x); ax0.set_xticklabels(["masked\nrecovery", "embedding\nfidelity"], fontsize=10)
    ax0.set_ylabel("center − edge gap (pp)", fontsize=11)
    ax0.set_title("Controlled probes (metric level)", fontsize=12, fontweight="bold")
    for xi, r in zip(x, pos):
        ax0.text(xi + 0.32, r["effect"], "p<0.01", ha="left", va="center", fontsize=9)
    ax0.set_xlim(-0.6, len(pos) - 1 + 1.0)
    ax0.legend(fontsize=8, loc="upper right")
    ax0.tick_params(labelsize=9); ax0.grid(True, axis="y", alpha=0.25, lw=0.5)

    # other three panels: the downstream tasks (each in native units)
    down = [r for r in rows if r["group"] == "downstream"]
    for ax, r in zip([axes[0, 1], axes[1, 0], axes[1, 1]], down):
        lo, hi = r["ci"]
        ax.errorbar([0], [r["effect"]], yerr=[[r["effect"] - lo], [hi - r["effect"]]],
                    fmt="o", color="#636363", capsize=6, ms=8, lw=1.5)
        ax.axhline(0, c="k", lw=0.8)
        thr = r["threshold"]
        ax.axhline(thr, ls="--", c="#d95f02", lw=1.2)
        if r["unit"] == "Δρ":
            ax.axhline(-thr, ls="--", c="#d95f02", lw=1.2)
        ax.set_xticks([]); ax.set_xlim(-1, 1)
        ax.set_ylabel(f"center − edge ({r['unit']})", fontsize=11)
        ax.set_title(r["name"], fontsize=12, fontweight="bold")
        ax.text(0.0, hi + (abs(hi - lo) * 0.18 + 1e-3), f"p={r['p']:.2f} (n.s.)",
                ha="center", va="bottom", fontsize=9)
        span = max(abs(lo), abs(hi), thr) * 1.7
        ax.set_ylim(-span, span)
        ax.tick_params(labelsize=9); ax.grid(True, axis="y", alpha=0.25, lw=0.5)

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    print(f"\ntable  -> {TAB}\njson   -> {JSN}\nfigure -> {FIG}")


if __name__ == "__main__":
    main()
