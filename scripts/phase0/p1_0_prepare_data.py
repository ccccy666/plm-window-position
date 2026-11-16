"""P1.0 micro-pilot data prep (Risk B instrument check).

Fetches carriers (GFP P42212 + a reviewed Swiss-Prot sample, cached & idempotent)
and builds ~300 needle probes as JSONL. Carrier types:
  - real          : real Swiss-Prot window of length L, motif inserted (in-distribution)
  - shuffled_real : the SAME window, residues shuffled, then motif inserted (OOD control)
  - gfp           : native GFP (folded, in-distribution), motif inserted
`real` and `shuffled_real` are PAIRED (share paired_id) for a clean shuffled-vs-not test.

Usage: python scripts/phase0/p1_0_prepare_data.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import json
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.probes.protniah import insert_needle, make_nxst_motif, shuffle_sequence
from calibrate.seed import DEFAULT_SEED

OUT = (_ROOT / "data/processed/p1_0")
GFP_ACC = "P42212"
REL_POS = [0.1, 0.5, 0.9]
LENGTHS = [256, 1024]
N_CARRIERS_PER_L = 16
N_GFP_DRAWS = 32


def _fetch_text(url: str, dest: Path) -> str:
    if dest.exists():
        return dest.read_text()
    print(f"  fetching {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_text(r.text)
    return r.text


def _parse_fasta(text: str) -> list[tuple[str, str]]:
    out, acc, seq = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if acc is not None:
                out.append((acc, "".join(seq)))
            acc = line[1:].split()[0]
            seq = []
        elif line.strip():
            seq.append(line.strip())
    if acc is not None:
        out.append((acc, "".join(seq)))
    return out


def fetch_carriers() -> tuple[str, list[tuple[str, str]]]:
    OUT.mkdir(parents=True, exist_ok=True)
    gfp_fasta = _fetch_text(
        f"https://rest.uniprot.org/uniprotkb/{GFP_ACC}.fasta", OUT / "gfp.fasta"
    )
    gfp_seq = _parse_fasta(gfp_fasta)[0][1]

    sp_url = (
        "https://rest.uniprot.org/uniprotkb/search?"
        "query=(reviewed:true)+AND+(organism_id:9606)+AND+(length:[200 TO 2500])"
        "&format=fasta&size=500"
    )
    sp_fasta = _fetch_text(sp_url, OUT / "swissprot_sample.fasta")
    pool = [(a, s) for a, s in _parse_fasta(sp_fasta) if set(s) <= set("ACDEFGHIKLMNPQRSTVWY")]
    print(f"  GFP len={len(gfp_seq)}; Swiss-Prot pool={len(pool)} clean seqs")
    return gfp_seq, pool


def build_probes(gfp_seq: str, pool: list[tuple[str, str]]) -> list[dict]:
    rng = np.random.default_rng(DEFAULT_SEED)
    probes: list[dict] = []
    pid = 0

    for L in LENGTHS:
        candidates = [(a, s) for a, s in pool if len(s) >= L]
        if len(candidates) < N_CARRIERS_PER_L:
            print(f"  WARN: only {len(candidates)} seqs >= {L}aa")
        picks = [candidates[i] for i in rng.choice(len(candidates), N_CARRIERS_PER_L, replace=False)]
        for acc, seq in picks:
            start = int(rng.integers(0, len(seq) - L + 1))
            window = seq[start : start + L]
            shuffled = shuffle_sequence(window, rng)
            for rp in REL_POS:
                motif = make_nxst_motif(rng)
                paired = f"pair{pid}"
                pid += 1
                for ctype, carrier in (("real", window), ("shuffled_real", shuffled)):
                    s, idx = insert_needle(carrier, motif, rp)
                    probes.append(
                        dict(
                            id=f"{ctype}_{acc}_L{L}_p{rp}_{motif}",
                            carrier_type=ctype,
                            paired_id=paired,
                            source_acc=acc,
                            L=len(carrier),
                            rel_pos=rp,
                            motif=motif,
                            needle_indices=idx,
                            sequence=s,
                        )
                    )

    for _ in range(N_GFP_DRAWS):
        for rp in REL_POS:
            motif = make_nxst_motif(rng)
            s, idx = insert_needle(gfp_seq, motif, rp)
            probes.append(
                dict(
                    id=f"gfp_{GFP_ACC}_p{rp}_{motif}",
                    carrier_type="gfp",
                    paired_id=None,
                    source_acc=GFP_ACC,
                    L=len(gfp_seq),
                    rel_pos=rp,
                    motif=motif,
                    needle_indices=idx,
                    sequence=s,
                )
            )
    return probes


def main() -> None:
    gfp_seq, pool = fetch_carriers()
    probes = build_probes(gfp_seq, pool)
    out_path = OUT / "probes.jsonl"
    with out_path.open("w") as f:
        for p in probes:
            f.write(json.dumps(p) + "\n")
    from collections import Counter

    counts = Counter(p["carrier_type"] for p in probes)
    print(f"wrote {len(probes)} probes -> {out_path}")
    print(f"  by carrier_type: {dict(counts)}")
    print(f"  lengths: {sorted(set(p['L'] for p in probes))}")


if __name__ == "__main__":
    main()
