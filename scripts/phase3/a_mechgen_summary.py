"""Cross-backbone summary of camera-ready group A (reads experiments/phase3/a_mechgen/*/summary.json).

Prints the A1-A4 tables next to the ESM-2 numbers already in the paper and writes
experiments/phase3/a_mechgen/summary_all.json. No GPU.
Usage: python scripts/phase3/a_mechgen_summary.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import json
from pathlib import Path

ROOT = _ROOT
OUT = ROOT / "experiments/phase3/a_mechgen"
ORDER = ["esm2", "esmc", "saprot", "prott5", "protbert"]
PAPER_2X2 = ROOT / "rebuttal/reviewer2/r2_randinit_randseq.json"
PAPER_ATTN = ROOT / "rebuttal/reviewer2/r2_attention_locality.json"
PAPER_C1 = ROOT / "experiments/phase3/c1_domain_offsets/summary.json"


def pp(x, p=None):
    return f"{x:+.2f}" + (f" (p={p:.3f})" if p is not None else "")


def main() -> None:
    runs = {b: json.loads((OUT / b / "summary.json").read_text()) for b in ORDER
            if (OUT / b / "summary.json").exists()}
    if "esm2" in runs:  # domain analysis for ESM-2 is the stored C1 run
        runs["esm2"]["A4"] = json.loads(PAPER_C1.read_text())
    paper2, papera = json.loads(PAPER_2X2.read_text()), json.loads(PAPER_ATTN.read_text())

    print("=== gate: trained/natural gap vs stored taxonomy (pp) ===")
    for b, s in runs.items():
        g = s.get("gate", {})
        print(f"  {b:9s} stored {g.get('stored_gap_pp', float('nan')):.3f}  new {g.get('new_gap_pp', float('nan')):.3f}"
              f"  max|dfid| {g.get('max_abs_fidelity_diff', float('nan')):.1e}  passed={g.get('passed')}")

    print("\n=== A1: center-edge fidelity gap (pp), trained/random-init x natural/random ===")
    cols = ["trained/natural", "random_init/natural", "trained/random_seq", "random_init/random_seq"]
    print(f"  {'backbone':9s} " + " | ".join(f"{c:>22s}" for c in cols) + " | verdict")
    for b, s in runs.items():
        if "A1" not in s:
            continue
        c = s["A1"]["cells"]
        print(f"  {b:9s} " + " | ".join(f"{pp(c[k]['gap_pp'], c[k]['p']):>22s}" for k in cols)
              + f" | {s['A1']['verdict']} (bar {s['A1']['bar_pp']:.2f})")
    print("  paper ESM-2: " + ", ".join(f"{k} {v['gap_pp']:+.2f}" for k, v in paper2.items()))

    print("\n=== A2: attention mass on window-dropped residues (all targets) ===")
    for b, s in runs.items():
        if "A2" not in s:
            continue
        tr, ri = s["A2"]["trained"]["all"], s["A2"]["random_init"]["all"]
        print(f"  {b:9s} n={tr['n']:3d} trained center {tr['drop_center']:.3f} edge {tr['drop_edge']:.3f} "
              f"edge-center {pp(tr['drop_edge_minus_center_pp'], tr['drop_p'])}  r={tr['corr_asym_vs_gap']:+.2f} "
              f"(p={tr['corr_p']:.4f}) | random-init {pp(ri['drop_edge_minus_center_pp'])} | {s['A2']['verdict']}")
    if "esm2" in runs and "A2" in runs["esm2"]:
        sub = runs["esm2"]["A2"]["trained"]["le700"]
        print(f"  ESM-2 <=700 re-run: n={sub['n']} center {sub['drop_center']:.3f} edge {sub['drop_edge']:.3f} "
              f"r={sub['corr_asym_vs_gap']:+.2f} | paper: n={papera['trained']['n']} center "
              f"{papera['trained']['drop_center']:.3f} edge {papera['trained']['drop_edge']:.3f} "
              f"r={papera['trained']['corr_asym_vs_gap']:+.2f}")

    if "protbert" in runs and "A3" in runs["protbert"]:
        a3 = runs["protbert"]["A3"]
        print("\n=== A3: ProtBERT, standard vs index-preserved windows ===")
        for k in ("standard", "preserved"):
            print(f"  {k:9s} gap {pp(a3[k]['gap_pp'], a3[k]['p'])}  curve "
                  + " ".join(f"{float(v):.3f}" for v in a3[k]["curve"].values()))
        d = a3["standard_minus_preserved"]
        print(f"  standard - preserved {d['gap_pp']:+.2f}pp CI[{d['ci_lo']:+.2f},{d['ci_hi']:+.2f}] "
              f"p={d['p']:.3f}; index share {a3['index_share'] * 100:.0f}%")

    print("\n=== A4: domain offsets (C1 design) ===")
    for b, s in runs.items():
        if "A4" not in s:
            continue
        q1, q2 = s["A4"]["Q1"], s["A4"]["Q2"]
        cme = " ".join(f"{float(v['gap_pp']):+.1f}" for v in s["A4"]["center_minus_edge"].values())
        print(f"  {b:9s} center-edge by u [{cme}] | dom_center {q1['dom_center']['rf_off_midpoint'] * 100:5.1f}% "
              f"CI[{q1['dom_center']['ci'][0] * 100:5.1f},{q1['dom_center']['ci'][1] * 100:5.1f}] "
              f"start {q1['dom_start']['rf_off_midpoint'] * 100:5.1f}% end {q1['dom_end']['rf_off_midpoint'] * 100:5.1f}% "
              f"| DiD {q2['did_pp']:+.1f} CI[{q2['ci'][0]:+.1f},{q2['ci'][1]:+.1f}] p={q2['p']:.4f} "
              f"| A by u " + " ".join(f"{float(v):+.1f}" for v in q2["mean_A_by_u"].values()))

    (OUT / "summary_all.json").write_text(json.dumps(runs, indent=2, default=float))
    print(f"\n-> {OUT / 'summary_all.json'}")


if __name__ == "__main__":
    main()
