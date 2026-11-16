"""P2 run: ESM-2 650M zero-shot wt-marginal scoring of long-protein DMS assays.

Full-sequence forward (RoPE extrapolation past the 1022 crop). Records per single
mutant: position, in/out-of-crop flag, DMS score, ESM-2 score. Writes scores.parquet.

Usage: python scripts/phase0/p2_run.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.data.proteingym import TRAIN_CROP, parse_mutant, score_full_sequence
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed

META = (_ROOT / "data/processed/proteingym_long_meta.parquet")
OUT_DIR = (_ROOT / "experiments/phase0_pilot/p2")


def safe_score(model, seq, mutants):
    try:
        return score_full_sequence(model, seq, mutants)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"    OOM at L={len(seq)} -> CPU fallback")
        model.model = model.model.float().cpu()
        model.device = "cpu"
        scores = score_full_sequence(model, seq, mutants)
        model.model = model.model.half().cuda()
        model.device = "cuda"
        torch.cuda.empty_cache()
        return scores


def main() -> None:
    set_global_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(META).sort_values("seq_len").reset_index(drop=True)
    model = load_esm2()

    all_rows = []
    for _, row in meta.iterrows():
        seq = row["target_seq"]
        df = pd.read_csv(row["local_csv"])
        # keep single mutants whose WT matches target_seq at the stated position
        recs = []
        for _, r in df.iterrows():
            try:
                wt, pos, alt = parse_mutant(r["mutant"])
            except ValueError:
                continue
            if 1 <= pos <= len(seq) and seq[pos - 1] == wt:
                recs.append((r["mutant"], pos, float(r["DMS_score"])))
        if not recs:
            print(f"  {row['DMS_id']}: 0 usable mutants, skip")
            continue
        muts = [m for m, _, _ in recs]
        scores = safe_score(model, seq, muts)
        for (m, pos, dms), sc in zip(recs, scores):
            all_rows.append(
                dict(DMS_id=row["DMS_id"], seq_len=row["seq_len"], pos=pos,
                     in_crop=pos <= TRAIN_CROP, dms_score=dms, esm_score=sc)
            )
        n_out = sum(1 for _, p, _ in recs if p > TRAIN_CROP)
        print(f"  {row['DMS_id']} (L={row['seq_len']}): {len(recs)} mutants, {n_out} out-of-crop")

    out = pd.DataFrame(all_rows)
    out.to_parquet(OUT_DIR / "scores.parquet")
    (OUT_DIR / "config.yaml").write_text(
        "probe: P2\nmodel: esm2_t33_650M_UR50D\nscoring: wt_marginals_full_seq\n"
        f"train_crop: {TRAIN_CROP}\nseed: 42\nn_rows: {len(out)}\n"
    )
    print(f"\nwrote {len(out)} scored mutants -> {OUT_DIR/'scores.parquet'}")


if __name__ == "__main__":
    main()
