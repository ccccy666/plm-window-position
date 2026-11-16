"""Apply the training-free multi-crop remedy to one residue (worked ESM-2 example).

When a residue can be scored from several overlapping windows (the natural situation
when tiling a long protein), aggregating those per-placement predictions recovers part
of the window-edge loss WITHOUT retraining. This demo scores one residue at five window
placements and compares edge / center / uniform multi-crop / centrality-weighted.

Run: conda run -n calibrate python release/examples/remedy_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from calibrate.models import load_esm2
from calibrate.remedy import aggregate_distributions, centrality_weights, recovered_fraction

PLACEMENTS = [0.1, 0.3, 0.5, 0.7, 0.9]
W = 256


def main() -> None:
    import json

    probes = [json.loads(l) for l in (ROOT / "release/data/probes.jsonl").read_text().splitlines()]
    p = next(x for x in probes if x["split"] == "test")     # any test protein
    seq, t = p["seq"], p["target"]
    model = load_esm2()
    true_idx = model.aa_token_index(seq[t])

    # score the SAME residue at each window placement
    dists = {}
    for r in PLACEMENTS:
        s = t - round(r * W)
        dists[r] = model.masked_distribution(seq[s : s + W], [t - s])[0]   # (vocab,)
    stacked = np.array([dists[r] for r in PLACEMENTS])

    edge = float(np.mean([dists[0.1][true_idx], dists[0.9][true_idx]]))
    center = float(dists[0.5][true_idx])
    uniform = float(aggregate_distributions(stacked)[true_idx])
    weighted = float(aggregate_distributions(stacked, centrality_weights(PLACEMENTS))[true_idx])

    print(f"protein {p['id']}  residue {t} ({seq[t]})  W={W}")
    print(f"  edge   (rel 0.1/0.9)      recovery = {edge:.3f}   (the bad case)")
    print(f"  center (rel 0.5)          recovery = {center:.3f}   (upper bound)")
    print(f"  uniform multi-crop        recovery = {uniform:.3f}   "
          f"(recovered {recovered_fraction(uniform, edge, center)*100:.0f}%)")
    print(f"  centrality-weighted       recovery = {weighted:.3f}   "
          f"(recovered {recovered_fraction(weighted, edge, center)*100:.0f}%)")


if __name__ == "__main__":
    main()
