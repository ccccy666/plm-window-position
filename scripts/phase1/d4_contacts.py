"""D4-contacts (option A): does crop position degrade LONG-RANGE contact prediction?

Long-range contacts inherently need both (sequence-distal) endpoints' context, so they
are the predicted venue where crop-position brittleness "matters". No ground-truth
structures needed: we take contacts ESM-2 is confident about in FULL context, then slide
the pair to different window positions and ask whether the predicted contact probability
degrades near the window edge (self-consistency under windowing).

Usage: python scripts/phase1/d4_contacts.py
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

DATA = (_ROOT / "data/processed/d4_ss")
OUT = (_ROOT / "experiments/phase1_diagnosis/d4_contacts")
FIG = (_ROOT / "results/figures/d4_contacts_position.png")

W = 256
REL_POS = [0.2, 0.35, 0.5, 0.65, 0.8]
SEP_MIN, SEP_MAX = 24, 100      # long-range, but placeable with both ends inside W
PROB_FULL_MIN = 0.5             # "confident" full-context contacts
MAX_PROT, MAX_LEN = 60, 700
PAIRS_PER_PROT = 8


def valid_s(c, rel, i, j, L):
    s = round(c - rel * W)
    if s < 0 or s + W > L or i < s or j >= s + W:
        return None
    return s


def main() -> None:
    set_global_seed(42)
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_esm2()
    test = [json.loads(l) for l in (DATA / "test.jsonl").read_text().splitlines()]
    prots = [p for p in test if len(p["seq"]) <= MAX_LEN][:MAX_PROT]

    rows = []
    for p in tqdm(prots, desc="d4 contacts"):
        seq = p["seq"]; L = len(seq)
        cmap = model.contacts(seq)
        # confident long-range pairs, placeable at every rel
        cands = []
        for i in range(L):
            for j in range(i + SEP_MIN, min(i + SEP_MAX + 1, L)):
                if cmap[i, j] < PROB_FULL_MIN:
                    continue
                c = (i + j) / 2
                if all(valid_s(c, r, i, j, L) is not None for r in REL_POS):
                    cands.append((i, j, float(cmap[i, j])))
        if not cands:
            continue
        sel = [cands[k] for k in rng.choice(len(cands), min(PAIRS_PER_PROT, len(cands)), replace=False)]
        for i, j, pf in sel:
            c = (i + j) / 2
            probw = {}
            for r in REL_POS:
                s = valid_s(c, r, i, j, L)
                wc = model.contacts(seq[s : s + W])
                probw[r] = float(wc[i - s, j - s])
            rows.append(dict(prob_full=pf, **{f"r_{r}": probw[r] for r in REL_POS}))

    df_rows = rows
    print(f"\n=== D4-contacts (n={len(df_rows)} confident long-range pairs, sep {SEP_MIN}-{SEP_MAX}, W={W}) ===")
    if not df_rows:
        print("no pairs found"); return
    by_rel = {r: float(np.mean([row[f"r_{r}"] for row in df_rows])) for r in REL_POS}
    full_mean = float(np.mean([row["prob_full"] for row in df_rows]))
    center = np.array([row["r_0.5"] for row in df_rows])
    edge = np.array([(row["r_0.2"] + row["r_0.8"]) / 2 for row in df_rows])
    res = paired_bootstrap_ci(center, edge, seed=42)
    print(f"  full-context prob (selected) = {full_mean:.3f}")
    print("  windowed contact prob by rel:", {r: round(v, 3) for r, v in by_rel.items()})
    print(f"  center(0.5)={center.mean():.3f}  edge(0.2,0.8)={edge.mean():.3f}  "
          f"gap=+{res['diff']*100:.2f}pp CI=[{res['ci_low']*100:.2f},{res['ci_high']*100:.2f}] p={res['p_value']:.3f}")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(REL_POS, [by_rel[r] for r in REL_POS], marker="o", label="windowed")
    ax.axhline(full_mean, ls="--", c="green", label=f"full context ({full_mean:.2f})")
    ax.set(title="D4 long-range contacts: predicted prob vs window position",
           xlabel="contact-pair center rel_pos in window", ylabel="predicted contact prob")
    ax.legend()
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=130)
    import pandas as pd
    pd.DataFrame(df_rows).to_parquet(OUT / "contacts.parquet")
    print(f"figure -> {FIG}")


if __name__ == "__main__":
    main()
