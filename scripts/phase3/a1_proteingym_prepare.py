"""A1 — expand the ProteinGym long-assay set for the VEP crop-shift (power up C4's weakest leg).

Lowers the length threshold from >1024aa to >460aa (the W=256 interior requirement, MARGIN=230,
so len>2*230) and downloads the per-assay DMS CSVs for those assays from the HF dataset
OATML-Markslab/ProteinGym (small files; per-file, robust; mirror-friendly). Rebuilds
proteingym_long_meta.parquet so scripts/phase0/p2b_crop_shift.py scores all of them.

Run (mirror recommended on flaky networks):
  HF_ENDPOINT=https://hf-mirror.com python scripts/phase3/a1_proteingym_prepare.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("HF_HOME", str(_ROOT / "data/cache/models/hf"))
from huggingface_hub import hf_hub_download

REPO = "OATML-Markslab/ProteinGym"
RAW = (_ROOT / "data/raw/proteingym")
SUBS = RAW / "substitutions"
PROCESSED = (_ROOT / "data/processed")
MIN_LEN = 460                 # W=256 interior requirement (len > 2*MARGIN, MARGIN=230)
MIN_SINGLES = 80              # need enough interior single mutants for a stable Spearman


def main() -> None:
    SUBS.mkdir(parents=True, exist_ok=True)
    # the HF repo's OWN reference (filenames match its CSVs + carries target_seq)
    ref_path = hf_hub_download(REPO, "ProteinGym_reference_file_substitutions.csv", repo_type="dataset")
    ref = pd.read_csv(ref_path)
    seq_col = next(c for c in ("target_seq", "seq", "sequence") if c in ref.columns)
    ref["seq_len"] = ref[seq_col].str.len()
    fname_col = "DMS_filename" if "DMS_filename" in ref.columns else "DMS_id"
    singles_col = next((c for c in ("DMS_number_single_mutants", "DMS_total_number_mutants")
                        if c in ref.columns), None)

    longs = ref[ref["seq_len"] > MIN_LEN].copy()
    if singles_col:
        longs = longs[longs[singles_col] >= MIN_SINGLES]
    longs = longs.sort_values("seq_len", ascending=False)
    print(f"{len(longs)} candidate assays (len>{MIN_LEN}, singles>={MIN_SINGLES}) of {len(ref)} total")

    rows, ok, failed = [], 0, []
    for _, row in longs.iterrows():
        fname = str(row[fname_col])
        if not fname.endswith(".csv"):
            fname += ".csv"
        local = SUBS / fname
        if not local.exists():
            err = None
            for _ in range(3):                          # small CSVs over a flaky link -> retry
                try:
                    p = hf_hub_download(REPO, f"ProteinGym_substitutions/{fname}", repo_type="dataset")
                    local.write_bytes(Path(p).read_bytes()); err = None; break
                except Exception as e:
                    err = type(e).__name__
            if err:
                failed.append((fname, err)); continue
        r = row.to_dict()
        r["local_csv"] = str(local)
        r["target_seq"] = row[seq_col]
        rows.append(r); ok += 1

    meta = pd.DataFrame(rows)
    meta.to_parquet(PROCESSED / "proteingym_long_meta.parquet")
    print(f"\nmeta rebuilt: {ok} assays available (downloaded/cached); {len(failed)} failed")
    if failed:
        print("  failed:", failed[:10])
    print(meta[["DMS_id", "seq_len"]].sort_values("seq_len", ascending=False).head(40).to_string(index=False))


if __name__ == "__main__":
    main()
