"""T1.C — evaluate a backbone on the crop-position benchmark snapshot (test split).

ESM-2 here (calibrate env). Other backbones: same harness with their embed_fn in their env
(esmc/progen). Reports the bias metrics from the shared module on the fixed probe set.

Usage: python scripts/phase1/benchmark_eval.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.benchmark import fidelity_matrix, load_probes, report
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed

OUT = (_ROOT / "experiments/phase1_diagnosis/benchmark")


def main() -> None:
    set_global_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    probes = load_probes("test")
    print(f"benchmark test split: {len(probes)} probes")
    m = load_esm2()
    M = fidelity_matrix(lambda s: m.embed([s])[0], probes)
    rep = report(M)
    rep["backbone"] = "esm2_650M"
    print(json.dumps({k: v for k, v in rep.items() if k != "curve"}, indent=2, default=float))
    print("curve:", {k: round(v, 4) for k, v in rep["curve"].items()})
    (OUT / "esm2_report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(f"-> {OUT/'esm2_report.json'}")


if __name__ == "__main__":
    main()
