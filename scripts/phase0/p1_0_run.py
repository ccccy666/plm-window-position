"""P1.0 micro-pilot inference.

For each probe: (a) MLM recovery prob of the needle (mean over its residues),
(b) attention mass into the needle, normalized per query token and averaged over
the last-16 layers and all heads. Writes raw.parquet.

Usage: python scripts/phase0/p1_0_run.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed

PROBES = (_ROOT / "data/processed/p1_0/probes.jsonl")
OUT_DIR = (_ROOT / "experiments/phase0_pilot/p1_0")
LAST_K_LAYERS = 16


def main() -> None:
    set_global_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probes = [json.loads(line) for line in PROBES.read_text().splitlines()]
    model = load_esm2()
    layers = list(range(model.num_layers - LAST_K_LAYERS + 1, model.num_layers + 1))

    rows = []
    for p in tqdm(probes, desc="P1.0 inference"):
        seq, idx = p["sequence"], p["needle_indices"]
        rec = float(model.recovery_prob(seq, idx).mean())
        att = model.attention_into(seq, idx, layers=layers)  # (16, heads, n_targets)
        t = len(seq) + 2  # query tokens incl BOS/EOS
        att_norm = float((att / t).mean())  # mean attn weight a needle residue receives
        rows.append(
            dict(
                id=p["id"],
                carrier_type=p["carrier_type"],
                paired_id=p["paired_id"],
                L=p["L"],
                rel_pos=p["rel_pos"],
                recovery=rec,
                attn_mass=att_norm,
            )
        )

    df = pd.DataFrame(rows)
    out = OUT_DIR / "raw.parquet"
    df.to_parquet(out)
    (OUT_DIR / "config.yaml").write_text(
        "probe: P1.0\nmodel: esm2_t33_650M_UR50D\nseed: 42\n"
        f"last_k_layers: {LAST_K_LAYERS}\nn_probes: {len(df)}\n"
    )
    print(f"wrote {len(df)} rows -> {out}")
    print(df.groupby(["carrier_type", "L", "rel_pos"])[["recovery", "attn_mass"]].mean().round(4))


if __name__ == "__main__":
    main()
