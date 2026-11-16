"""Reviewer-1 RC2: computational cost / cost-benefit of the training-free multi-crop remedy.

Cost model for scoring every residue of a length-L protein with a width-W window:
  - single non-overlapping tiling (the baseline; each residue scored from one tile, wherever it
    lands):                      ceil(L/W) forward passes;
  - K-crop multi-crop (stride ~ W/K so each interior residue is covered by K windows):
                                 ~ K * ceil(L/W) forward passes;
  - per-residue centering (a distinct window centered on each residue, the 100% upper bound):
                                 ~ L forward passes  (== W x the single-tiling cost).
Recovered fractions of the center-edge gap are taken from the paper's Table 3 (ESM-2/ESM C):
uniform K-crop ~46-56%, centrality-weighted K-crop ~68-80% (same cost as uniform), centering 100%.

Also measures ESM-2 forward wall-clock at a few window sizes to turn pass-counts into seconds.

Usage (GPU, calibrate env): python rebuttal/r1_remedy_cost.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[1]

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = _ROOT
sys.path.insert(0, str(ROOT / "src"))
OUTJSON = ROOT / "rebuttal/r1_remedy_cost.json"
OUTTXT = ROOT / "rebuttal/r1_remedy_cost.txt"

AA = list("ACDEFGHIKLMNPQRSTVWY")


def passes(L, W, strategy, K=5):
    tiles = int(np.ceil(L / W))
    if strategy == "single_tiling":
        return tiles
    if strategy == "multicrop":
        return K * tiles
    if strategy == "centering":
        return L
    raise ValueError(strategy)


def main() -> None:
    import torch
    from calibrate.models import load_esm2
    from calibrate.seed import set_global_seed
    set_global_seed(42)
    rng = np.random.default_rng(42)
    model = load_esm2()

    # --- measure per-forward wall-clock at a few window sizes (masked recovery of 1 residue) ---
    timings = {}
    for W in [128, 256, 384]:
        seq = "".join(rng.choice(AA, W))
        for _ in range(2):  # warmup
            model.recovery_prob(seq, [W // 2])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        n = 10
        for _ in range(n):
            model.recovery_prob(seq, [W // 2])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        timings[W] = (time.perf_counter() - t0) / n
    print("per-forward wall-clock (s):", {k: round(v, 4) for k, v in timings.items()})

    # --- cost-benefit table for a representative protein ---
    recovered = {  # from paper Table 3 (ESM-2, W=256); weighted == uniform cost, higher recovery
        "single_tiling": 0.0, "uniform_multicrop": 0.53, "weighted_multicrop": 0.74,
        "centering": 1.0,
    }
    K = 5
    rows = []
    for L in [500, 1000, 2000]:
        for W in [256]:
            base = passes(L, W, "single_tiling")
            entries = {
                "single_tiling": passes(L, W, "single_tiling"),
                "uniform_multicrop": passes(L, W, "multicrop", K),
                "weighted_multicrop": passes(L, W, "multicrop", K),
                "centering": passes(L, W, "centering"),
            }
            for strat, nfwd in entries.items():
                rows.append(dict(L=L, W=W, strategy=strat, forward_passes=nfwd,
                                 x_over_single=round(nfwd / base, 1),
                                 recovered_frac=recovered.get(strat.replace("uniform_", "uniform_").replace("weighted_", "weighted_"), None),
                                 est_seconds=round(nfwd * timings[W], 2)))

    out = {"per_forward_seconds": timings, "K_crops": K,
           "recovered_fraction_note": "from paper Table 3 (ESM-2 W=256): uniform~0.46-0.56, weighted~0.68-0.80",
           "cost_table": rows,
           "decision_rule": [
               "weighted multi-crop and uniform multi-crop cost the SAME (K forward passes); "
               "weighted recovers more (68-80% vs 46-56%) -> always prefer weighted.",
               "K-crop multi-crop costs ~K x a single tiling and recovers most of the gap; "
               "per-residue centering reaches 100% but costs ~W x a single tiling "
               "(~W/K x more than K-crop) -> use centering only when the last 20-30% is worth "
               "an order-of-magnitude more compute.",
           ]}
    OUTJSON.write_text(json.dumps(out, indent=2, default=float))

    # readable summary
    lines = ["Remedy cost-benefit (ESM-2, K=5 crops)\n",
             f"per-forward wall-clock (s): " + ", ".join(f"W={k}:{v:.3f}" for k, v in timings.items()) + "\n"]
    lines.append(f"{'L':>5} {'strategy':>20} {'#fwd':>6} {'xSingle':>8} {'recov%':>7} {'~sec':>7}")
    for r in rows:
        rf = "" if r["recovered_frac"] is None else f"{r['recovered_frac']*100:.0f}"
        lines.append(f"{r['L']:>5} {r['strategy']:>20} {r['forward_passes']:>6} "
                     f"{r['x_over_single']:>7}x {rf:>7} {r['est_seconds']:>7}")
    txt = "\n".join(lines)
    OUTTXT.write_text(txt + "\n\nDecision rule:\n- " + "\n- ".join(out["decision_rule"]) + "\n")
    print(txt)
    print(f"\njson -> {OUTJSON}\ntxt  -> {OUTTXT}")


if __name__ == "__main__":
    main()
