"""Evaluate ANY protein LM on the crop-position benchmark.

The only thing a backbone must provide is an embedding callback:

    embed_fn(seq: str) -> np.ndarray of shape (len(seq), hidden)

i.e. per-residue embeddings for a single sequence. Everything else — the fixed probe
set, the window-relocation protocol, and the bias metrics (Position Gap / Shift
Sensitivity / Robust Accuracy) — is shared by the harness in `calibrate.benchmark`.

This file shows the contract with a worked ESM-2 example (run in the `calibrate` env).
For other backbones, write the analogous `embed_fn` in that backbone's environment:
  - ESM C (esmc env):   model(t).embeddings  -> (L, H)   [see scripts/phase1/t1a_fidelity_esmc.py]
  - ProGen2 (progen):   hidden_states[-1]     -> (L, H)   [see scripts/phase1/t1a_fidelity_progen.py]

Run: conda run -n calibrate python release/examples/eval_any_backbone.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo root
sys.path.insert(0, str(ROOT / "src"))

from calibrate.benchmark import fidelity_matrix, load_probes, report

PROBES = Path(__file__).resolve().parents[1] / "data" / "probes.jsonl"  # shipped snapshot


def esm2_embed_fn():
    """A concrete embed_fn for ESM-2 650M (calibrate env)."""
    from calibrate.models import load_esm2

    model = load_esm2()
    return lambda seq: model.embed([seq])[0]        # (L, hidden)


def main() -> None:
    probes = load_probes("test", path=PROBES)        # 60 family-disjoint test probes
    print(f"loaded {len(probes)} test probes from {PROBES}")

    embed_fn = esm2_embed_fn()                        # <-- swap in your backbone here
    M = fidelity_matrix(embed_fn, probes)             # (n_probes, n_placements) cosine matrix
    rep = report(M)

    print("\n=== crop-position benchmark report ===")
    print(f"  Position Gap     : {rep['position_gap_pp']:+.2f} pp "
          f"(95% CI [{rep['gap_ci_pp'][0]:.2f}, {rep['gap_ci_pp'][1]:.2f}], p={rep['gap_p']:.4f})")
    print(f"  Shift Sensitivity: {rep['shift_sensitivity']:.4f}")
    print(f"  Robust Accuracy  : {rep['robust_accuracy']:.4f}")
    print(f"  fidelity curve   : { {k: round(v, 4) for k, v in rep['curve'].items()} }")


if __name__ == "__main__":
    main()
