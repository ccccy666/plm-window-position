"""Reproduce entrypoint for the window-position benchmark release.

What runs here (`calibrate` env, from the shipped snapshot):
  * the headline benchmark evaluation - ESM-2 on the 60-probe family-disjoint test
    split - reproducing the Position Gap / Shift Sensitivity / Robust Accuracy;
  * then it prints the full pipeline: which script produces which table or figure of the paper.

The model runs span three conda environments (a single env cannot import fair-esm and the
EvolutionaryScale `esm` SDK together, and ProtBERT / ProtT5 / Ankh / ProGen2 need transformers 4.44).
Scripts marked "stored" read stored outputs only and need no model or GPU.

Run: conda run -n calibrate python release/reproduce.py          (--list: print the pipeline only)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "examples"))

# (env, script, what it produces) in paper order; "stored" = reads stored outputs, no model
PIPELINE = [
    # Section 4 - what changes
    ("calibrate", "scripts/phase1/t1a_fidelity.py",                   "Table 2: ESM-2 and SaProt fidelity"),
    ("esmc",      "scripts/phase1/t1a_fidelity_esmc.py",              "Table 2: ESM C fidelity"),
    ("progen",    "scripts/phase1/t1a_fidelity_protbert.py",          "Table 2: ProtBERT fidelity"),
    ("progen",    "scripts/phase1/t1a_fidelity_prott5.py",            "Table 2: ProtT5 fidelity"),
    ("progen",    "scripts/phase1/t1a_fidelity_progen.py",            "Table 2: ProGen2 fidelity"),
    ("calibrate", "scripts/phase3/b1_pe_axis.py esm1b",               "Table 2: ESM-1b fidelity"),
    ("progen",    "scripts/phase3/b1_pe_axis.py ankh",                "Table 2: Ankh-base fidelity"),
    ("calibrate", "scripts/phase3/b6_bias_metrics.py",                "Table 2: three bias metrics (stored)"),
    ("calibrate", "scripts/phase3/fig_taxonomy.py",                   "Fig. 2 (stored)"),
    ("calibrate", "scripts/phase0/p1b_window_reloc.py",               "Sec. 4: masked recovery (ESM-2)"),
    ("calibrate", "scripts/phase0/p1b_scale.py [--plot-only]",        "Fig. 5, App. A.1: ESM-2 scales"),
    ("calibrate", "scripts/phase3/b4_multiseed.py",                   "App. A.2: seed stability"),
    # Section 5 - why
    ("calibrate", "scripts/phase0/p1b_run.py",                        "Sec. 5.1: cassette sweep, n = 200"),
    ("calibrate", "scripts/phase1/t1b_mechanism.py",                  "Sec. 5.1: cassette flank sweep"),
    ("calibrate", "scripts/phase3/b2_truncation_cliff_controlled.py", "Sec. 5.2: controlled crop cliff"),
    ("calibrate", "scripts/phase0/p2_run.py, p2_analyze.py",          "Sec. 5.2: BRCA1 across the crop"),
    ("calibrate", "rebuttal/reviewer2/r2_randinit_randseq.py",        "Sec. 5.3, Fig. 3d: 2x2 (ESM-2)"),
    ("per model", "scripts/phase3/a_mechgen.py BACKBONE",             "Table 3: mechanism tests, 5 encoders"),
    ("calibrate", "scripts/phase3/a_mechgen_summary.py",              "Table 3 (stored)"),
    ("per model", "scripts/phase3/c_attention_vs_gap.py esm1b|ankh",  "Sec. 5.4: attention vs effect size"),
    ("calibrate", "scripts/phase3/c1_domain_offsets.py",              "Sec. 5.5, 7.1: domain offsets"),
    ("calibrate", "scripts/phase3/fig_mechanism_composite.py",        "Fig. 3 (stored)"),
    # Section 6 - when it matters
    ("calibrate", "scripts/phase0/p2b_crop_shift.py [--analyze-only]", "Table 4: ProteinGym VEP"),
    ("calibrate", "scripts/phase1/d4_ss_run.py",                      "Table 4: secondary structure"),
    ("calibrate", "scripts/phase1/d4_contacts.py",                    "Table 4: long-range contacts"),
    ("calibrate", "scripts/phase1/t1d_scoping.py",                    "Table 4, Fig. 6 (stored)"),
    ("calibrate", "scripts/phase3/b1_contacts_partner_out.py",        "Sec. 6.2: partner-out contacts"),
    ("calibrate", "rebuttal/r1_partner_out_multiW.py",                "Sec. 6.2: window size"),
    ("calibrate", "scripts/phase3/b2_contacts_afdb.py",               "Sec. 6.2, App. A.5: AlphaFold check"),
    ("calibrate", "rebuttal/r1_tiling_analysis.py",                   "Sec. 6.3: tiling split rates (stored)"),
    ("calibrate", "scripts/phase3/fig_deployment_composite.py",       "Figs. 4, 7, 8 (stored)"),
    # Section 7 - what to do
    ("calibrate", "rebuttal/reviewer3/r3_domain_windowing.py",        "Sec. 7.1: domain windows, midpoints"),
    ("calibrate", "scripts/phase2/t2a_remedy.py",                     "Table 5: ESM-2 window choices"),
    ("esmc",      "scripts/phase2/t2a_remedy_esmc.py",                "Table 5: ESM C window choices"),
    ("calibrate", "scripts/phase2/t2a_combine.py",                    "Table 5 (stored)"),
    ("calibrate", "scripts/phase2/t2a_most_central_bound.py",         "Table 5: most central window (stored)"),
    ("calibrate", "rebuttal/r1_remedy_cost.py",                       "Table 5 cost, Table 6"),
]


def print_pipeline() -> None:
    print("== full pipeline (run each in its conda env; 'per model' = see the script's Usage) ==")
    for env, script, artifact in PIPELINE:
        print(f"  [{env:9s}] {script:52s} -> {artifact}")
    print("\nSee README.md for environment setup and provenance.")


def main() -> None:
    if "--list" not in sys.argv:
        print("== headline reproduction: ESM-2 on the shipped benchmark test split ==\n")
        import eval_any_backbone
        eval_any_backbone.main()
        print()
    print_pipeline()


if __name__ == "__main__":
    main()
