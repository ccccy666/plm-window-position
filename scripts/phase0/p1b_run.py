"""P1' position sweep — the clean D1 test (crop-position brittleness).

A fixed, recoverable cassette (a real high-recovery residue + its native +/-FLANK
flank) is slid to different relative positions in a length-L carrier. The cassette's
own local context is CONSTANT at every position; only the cassette's position in the
construct (and the carrier on each side) changes -> position is the only variable.
Carriers: shuffled (controlled/OOD) and real-window (in-distribution background).

Records center-residue masked recovery + true-residue log-prob (nats).
Usage: python scripts/phase0/p1b_run.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.models import load_esm2
from calibrate.probes.protniah import insert_needle, shuffle_sequence
from calibrate.seed import set_global_seed

GFP = (_ROOT / "data/processed/p1_0/gfp.fasta")
SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
OUT = (_ROOT / "experiments/phase0_pilot/p1b")

FLANK = 30
NATIVE_REC_MIN = 0.30
N_CASSETTES = 200
LENGTHS = [256, 512]
REL_POS = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
CARRIERS = ["shuffled", "real"]
AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_fasta(text):
    seqs, cur = [], []
    for ln in text.splitlines():
        if ln.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur = []
        elif ln.strip():
            cur.append(ln.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def main() -> None:
    set_global_seed(42)
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_esm2()

    pool = [s for s in parse_fasta(SP.read_text()) + parse_fasta(GFP.read_text())
            if set(s) <= AA]
    cas_pool = [s for s in pool if 120 <= len(s) <= 2500]
    carrier_pool = [s for s in pool if len(s) >= max(LENGTHS)]
    rng.shuffle(cas_pool)

    # 1) collect recoverable cassettes (center residue native recovery >= threshold)
    cassettes = []  # (cassette_str, center_aa, native_rec, acc_idx)
    for pi, prot in enumerate(cas_pool):
        if len(cassettes) >= N_CASSETTES:
            break
        for t in range(FLANK, len(prot) - FLANK):
            rec = float(model.recovery_prob(prot, [t])[0])
            if rec >= NATIVE_REC_MIN:
                cassettes.append((prot[t - FLANK : t + FLANK + 1], prot[t], rec, pi))
                break  # one cassette per protein to diversify
    print(f"selected {len(cassettes)} recoverable cassettes (native rec >= {NATIVE_REC_MIN})")

    # 2) slide each cassette across positions x carrier x L
    rows = []
    for ci, (cas, center_aa, native, _) in enumerate(tqdm(cassettes, desc="P1' sweep")):
        for L in LENGTHS:
            src = carrier_pool[int(rng.integers(0, len(carrier_pool)))]
            start = int(rng.integers(0, len(src) - L + 1))
            window = src[start : start + L]
            carriers = {"real": window, "shuffled": shuffle_sequence(window, rng)}
            for ctype in CARRIERS:
                for rp in REL_POS:
                    seq, idx = insert_needle(carriers[ctype], cas, rp)
                    center = idx[FLANK]
                    rec = float(model.recovery_prob(seq, [center])[0])
                    rows.append(dict(cassette=ci, native_rec=native, carrier_type=ctype,
                                     L=L, rel_pos=rp, center_rec=rec,
                                     center_logprob=float(np.log(max(rec, 1e-9)))))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "position_sweep.parquet")
    (OUT / "config.yaml").write_text(
        f"probe: P1prime_position_sweep\nmodel: esm2_t33_650M_UR50D\nseed: 42\n"
        f"flank: {FLANK}\nn_cassettes: {len(cassettes)}\nlengths: {LENGTHS}\n"
        f"rel_pos: {REL_POS}\nn_rows: {len(df)}\n"
    )
    print(f"\nwrote {len(df)} rows -> {OUT/'position_sweep.parquet'}")
    print(df.groupby(["carrier_type", "L", "rel_pos"])["center_rec"].mean().round(3))


if __name__ == "__main__":
    main()
