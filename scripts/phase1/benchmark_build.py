"""T1.C — build the versioned crop-position benchmark snapshot + family-aware split.

Fixes a reproducible probe set: real proteins + one non-terminal target residue each,
clustered by MMseqs2 (30% id / 80% cov) into families and split train/test by family
(no family crosses the split). The eval harness runs any backbone's embedder over this
fixed set and reports the bias metrics. Snapshot = data/processed/benchmark/.

Usage: python scripts/phase1/benchmark_build.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.seed import set_global_seed

SP = (_ROOT / "data/processed/p1_0/swissprot_sample.fasta")
OUT = (_ROOT / "data/processed/benchmark")
W, REL = 256, [0.1, 0.3, 0.5, 0.7, 0.9]
MARGIN = round(0.9 * W)
MIN_LEN, MAX_LEN, N_PROT = 480, 1500, 200
AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_fasta(text):
    s, c, name = [], [], None
    for l in text.splitlines():
        if l.startswith(">"):
            if name is not None:
                s.append((name, "".join(c)))
            name = l[1:].split()[0]; c = []
        elif l.strip():
            c.append(l.strip())
    if name is not None:
        s.append((name, "".join(c)))
    return s


def mmseqs_clusters(fasta: Path, workdir: Path) -> dict[str, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    pref = workdir / "clust"
    subprocess.run(
        ["mmseqs", "easy-cluster", str(fasta), str(pref), str(workdir / "tmp"),
         "--min-seq-id", "0.3", "-c", "0.8", "-v", "1"],
        check=True,
    )
    member2rep = {}
    for line in (Path(str(pref) + "_cluster.tsv")).read_text().splitlines():
        rep, mem = line.split("\t")
        member2rep[mem] = rep
    return member2rep


def main() -> None:
    set_global_seed(42)
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    pool = [(n, s) for n, s in parse_fasta(SP.read_text())
            if set(s) <= AA and MIN_LEN <= len(s) <= MAX_LEN]
    rng.shuffle(pool)
    pool = pool[:N_PROT]
    print(f"{len(pool)} benchmark proteins (len {MIN_LEN}-{MAX_LEN})")

    # one non-terminal target per protein
    probes = []
    for acc, seq in pool:
        t = int(rng.integers(MARGIN, len(seq) - MARGIN))
        probes.append(dict(id=acc, seq=seq, target=t))

    # family clustering + split
    fa = OUT / "proteins.fasta"
    fa.write_text("".join(f">{p['id']}\n{p['seq']}\n" for p in probes))
    m2r = mmseqs_clusters(fa, OUT / "mmseqs")
    reps = sorted({m2r.get(p["id"], p["id"]) for p in probes})
    rng.shuffle(reps)
    test_reps = set(reps[: max(1, int(0.3 * len(reps)))])
    for p in probes:
        rep = m2r.get(p["id"], p["id"])
        p["cluster"] = rep
        p["split"] = "test" if rep in test_reps else "train"

    with (OUT / "probes.jsonl").open("w") as f:
        for p in probes:
            f.write(json.dumps(p) + "\n")
    manifest = dict(
        name="crop-position-benchmark", version="0.1", date="2026-06-08",
        n_probes=len(probes), n_families=len(reps),
        n_train=sum(p["split"] == "train" for p in probes),
        n_test=sum(p["split"] == "test" for p in probes),
        window_W=W, rel_pos_grid=REL, seed=42,
        mmseqs="easy-cluster --min-seq-id 0.3 -c 0.8",
        probe="window-relocation: place target at rel_pos in W-window; metric vs placement",
        bias_metrics=["position_gap", "shift_sensitivity", "robust_accuracy"],
        source="UniProt reviewed human (pre-2024 snapshot)",
        git_sha=subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
    )
    (OUT / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"wrote {len(probes)} probes -> {OUT/'probes.jsonl'}")
    print(f"families={len(reps)}  train={manifest['n_train']}  test={manifest['n_test']}")


if __name__ == "__main__":
    main()
