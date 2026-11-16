# Analyses from the review period

These scripts were run during the TMLR review (June 2026). The camera-ready paper reports their results; the
file names follow the review threads (`r1` = reviewer 1, and so on). The stored outputs (`*.json`) sit next
to the scripts, and the camera-ready figure scripts read them.

| Script | Output | In the paper |
|---|---|---|
| `r1_tiling_analysis.py` (no GPU) | `r1_tiling.json` | Sec. 6.3, Fig. 4, Fig. 8: how often a tiling splits a confident long-range contact |
| `r1_partner_out_multiW.py` | `r1_partner_out_multiW.json` | Sec. 6.2, Fig. 4: partner-out contact loss at W = 128, 256, 384 |
| `r1_remedy_cost.py` | `r1_remedy_cost.json` | Table 5 (cost column), Table 6 |
| `reviewer2/r2_randinit_randseq.py` | `r2_randinit_randseq.json` | Sec. 5.3, Fig. 3d: trained vs random-init × natural vs random sequences (ESM-2) |
| `reviewer2/r2_attention_locality.py` | `r2_attention_locality.json` | Attention locality for the reviewed version (proteins up to 700 residues); the camera-ready Sec. 5.4 uses all 120 targets (`scripts/phase3/a_mechgen.py`) |
| `reviewer3/r3_fetch_domains.py` (network) | `domains_cache.json`, `targets.json` | UniProt-curated domain boundaries used by all domain analyses |
| `reviewer3/r3_domain_windowing.py` | `r3_domain_windowing.json` | Sec. 7.1: windows that contain the domain, for residues at domain midpoints |

The camera-ready experiments extend these: `scripts/phase3/a_mechgen.py` repeats the 2×2, the attention
analysis, and the domain analysis on five encoders, and `scripts/phase3/c1_domain_offsets.py` moves the
target residue away from the domain midpoint.
