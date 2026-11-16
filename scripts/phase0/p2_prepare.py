"""P2 data prep: download ProteinGym substitution DMS for proteins >1024aa.

Pulls only the long-protein assay CSVs (not the full zip) from the HF dataset
OATML-Markslab/ProteinGym, plus a metadata table carrying the WT target_seq.

Usage: python scripts/phase0/p2_prepare.py
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
MIN_LEN = 1024


def main() -> None:
    SUBS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Use the HF repo's OWN reference (v0.1) so filenames match its CSVs.
    ref_path = hf_hub_download(
        REPO, "ProteinGym_reference_file_substitutions.csv", repo_type="dataset"
    )
    ref = pd.read_csv(ref_path)
    seq_col = next(c for c in ("target_seq", "seq", "sequence") if c in ref.columns)
    ref["seq_len"] = ref[seq_col].str.len()
    fname_col = "DMS_filename" if "DMS_filename" in ref.columns else "DMS_id"
    longs = ref[ref["seq_len"] > MIN_LEN].copy()
    print(f"reference columns: {list(ref.columns)[:12]}")
    print(f"{len(longs)} assays with protein >{MIN_LEN}aa (of {len(ref)} total)")

    rows = []
    for _, row in longs.iterrows():
        fname = row[fname_col]
        if not str(fname).endswith(".csv"):
            fname = f"{fname}.csv"
        local = SUBS / fname
        if not local.exists():
            print(f"  downloading {fname}")
            p = hf_hub_download(
                REPO, f"ProteinGym_substitutions/{fname}", repo_type="dataset"
            )
            local.write_bytes(Path(p).read_bytes())
        r = row.to_dict()
        r["local_csv"] = str(local)
        r["target_seq"] = row[seq_col]
        rows.append(r)

    meta = pd.DataFrame(rows)
    meta.to_parquet(PROCESSED / "proteingym_long_meta.parquet")
    print(f"\nsaved metadata for {len(meta)} assays -> proteingym_long_meta.parquet")
    print(meta[["DMS_id", "seq_len"]].sort_values("seq_len", ascending=False).to_string(index=False))

    sample = pd.read_csv(rows[0]["local_csv"])
    print(f"\nsample {Path(rows[0]['local_csv']).name}: {len(sample)} rows; cols={list(sample.columns)}")
    print(sample.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
