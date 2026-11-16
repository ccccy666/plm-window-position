# Provenance

## Snapshot
- **Built by:** `scripts/phase1/benchmark_build.py` (regenerable).
- **Seed:** 42 (global; see `src/calibrate/seed.py`).
- **Snapshot git SHA (at build):** `92ee3d9` (recorded in `data/manifest.yaml`).
- **Probe set:** 200 proteins, 1 non-terminal target each; 200 singleton MMseqs2 families
  (`easy-cluster --min-seq-id 0.3 -c 0.8`) split 140 train / 60 test, family-disjoint.
- **Source:** UniProt reviewed human, pre-2024 snapshot (pre ESM-2 cutoff).

## Environments (3 conda envs; fair-esm and the `esm` SDK cannot coexist)
| env | purpose | key packages |
|---|---|---|
| `calibrate` | ESM-2, SaProt, ESM-1b; harness/metrics/remedy | torch 2.11.0+cu128, transformers 5.10.1, fair-esm 2.0.0, numpy 2.4.6, scikit-learn 1.9.0 |
| `esmc` | ESM C 300M | EvolutionaryScale `esm` SDK 3.2.1 (no fair-esm) |
| `progen` | ProGen2-764M, ProtBERT, ProtT5, Ankh | transformers 4.44.2 |

- **Hardware (pilot):** single RTX 5080 16 GB (Blackwell sm_120, WSL2); CUDA cu128 build.
- **Stats:** paired bootstrap (10k resamples, seed 42); pre-registered thresholds (benchmark tier
  ≥3 pp or Δρ≥0.03; method tier ≥10 pp; p<0.01) with Holm–Bonferroni across placement bins.

## Reproduction entrypoint
`conda run -n calibrate python release/reproduce.py` — headline ESM-2 benchmark report from the
shipped snapshot + the full multi-env pipeline manifest.

> The `git_sha` in `data/manifest.yaml` (`92ee3d9`) is the snapshot-build revision; later analysis
> commits do not change the snapshot. Re-running `benchmark_build.py` at seed 42 regenerates an
> identical probe set.
