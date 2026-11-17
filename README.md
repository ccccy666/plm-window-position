# When does window position matter in protein language models?

Code, benchmark, and stored results for

> Yifan Chen. **When Does Window Position Matter in Protein Language Models? A Controlled Benchmark and
> Cross-Architecture Taxonomy.** *Transactions on Machine Learning Research*, 2026.
> [OpenReview](https://openreview.net/forum?id=kEpikXuZfK)

Protein language models (pLMs) read long proteins through fixed-length windows, so the same residue can sit at
the edge or at the center of its window. The paper measures what that changes, why, and when it matters:

- **What changes.** All seven bidirectional encoders tested (ESM-2, ESM-1b, ESM C, SaProt, ProtT5, Ankh,
  ProtBERT) represent a residue closest to its full-context representation when it sits near the center of the
  window: a center-favored inverted-U whose size varies five-fold across encoders without following the type of
  position encoding. The causal ProGen2 depends monotonically on left context.
- **Why.** The effect is one-sided loss of informative context, not an intrinsic position bias or a
  training-crop cliff. It is learned: in each of five encoders tested, a randomly-initialized network of the
  same architecture shows essentially none. Trained encoders attend locally, so an edge window removes more of
  the context a residue attends to, and the cost is largest when the cut removes the residue's own domain.
- **When it matters.** Window position matters where the window cuts context that an output depends on.
  Variant-effect rankings show no significant change, because the edge shifts an assay's scores together, and
  neither do secondary structure and long-range contacts while their context stays in the window. Once the
  context beyond a contact partner is cut, the model's confident contact predictions lose 20 points of
  probability, and they vanish once the partner leaves the window: a non-overlapping tiling of 256-residue
  windows separates the partners of 19% of these contacts, a 50%-overlapping tiling 0.8%.
- **What to do.** For per-residue representations, a window centered on the residue's domain recovers 98% of
  the edge loss; for masked-recovery scoring, taking each residue from its most central overlapping window
  recovers most of it at the cost of averaging over those windows.

## Contents

| Path | What it holds |
|---|---|
| `release/data/` | The benchmark snapshot: 200 human Swiss-Prot proteins with one non-terminal target residue each, a family-disjoint MMseqs2 split (140 train / 60 test), its manifest, and the secondary-structure labels. |
| `src/calibrate/` | The package: evaluation harness (`benchmark.py`), bias metrics (`metrics.py`), window choices (`remedy.py`), model wrappers (`models.py`), statistics (`stats.py`), and the analysis helpers used by the scripts and tests. |
| `scripts/phase0`–`phase3/` | Experiment scripts, in the order they were run: pilot, taxonomy and scoping, window choices, camera-ready experiments and figures. |
| `rebuttal/` | Analyses run during the review; the paper reports them (see `rebuttal/README.md`). |
| `experiments/` | Stored outputs (parquet, json) of every run reported in the paper. |
| `results/` | The paper's figures and the tables written by the scripts. |
| `tests/` | The test suite. Besides unit tests, it pins the paper's numbers to the stored outputs. |
| `data/processed/` | Inputs built by the scripts (benchmark build, secondary-structure set, pilot carriers). |

## Quick start: no GPU needed

The figures and the numbers in the tables are recomputed from the stored outputs on a CPU (the scripts marked
"stored" in `python release/reproduce.py --list`).

```bash
conda create -n calibrate python=3.11 -y && conda activate calibrate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; enough for these scripts
pip install -r requirements-pilot.txt && pip install -e .

python -m pytest                        # tests that need model weights skip when the weights are absent
python release/reproduce.py --list      # which script produces which table or figure
python scripts/phase3/fig_taxonomy.py   # e.g. redraw Fig. 2 from experiments/
```

## Evaluate another model on the benchmark

A model only has to provide per-residue embeddings for one sequence:

```python
from calibrate.benchmark import fidelity_matrix, load_probes, report

probes = load_probes("test", path="release/data/probes.jsonl")   # 60 family-disjoint test probes
M = fidelity_matrix(embed_fn, probes)   # embed_fn(seq: str) -> array of shape (len(seq), hidden)
print(report(M))                        # Position Gap (with bootstrap CI), Shift Sensitivity, Robust Accuracy
```

`release/examples/eval_any_backbone.py` is a worked example with ESM-2 650M, and `python release/reproduce.py`
runs it (Position Gap +3.58 pp, 95% CI [2.77, 4.47], on the test split).

## Reproducing the paper

`python release/reproduce.py --list` prints the full list. The main entries:

| Paper | Script | Env |
|---|---|---|
| Fig. 1 (schematic of the two probes) | `scripts/phase3/fig1_schematic.py` | calibrate |
| Table 2, Fig. 2 (taxonomy) | `scripts/phase1/t1a_fidelity*.py`, `scripts/phase3/b1_pe_axis.py esm1b\|ankh`; redraw: `scripts/phase3/fig_taxonomy.py`, `scripts/phase3/b6_bias_metrics.py` | per model |
| Sec. 5.1 (fixed-context cassette) | `scripts/phase0/p1b_run.py`, `scripts/phase1/t1b_mechanism.py` | calibrate |
| Sec. 5.2 (training-crop cliff) | `scripts/phase3/b2_truncation_cliff_controlled.py`, `scripts/phase0/p2_run.py` | calibrate |
| Sec. 5.3–5.5, Table 3 (learned; attention locality; own domain) | `scripts/phase3/a_mechgen.py BACKBONE`, `scripts/phase3/c_attention_vs_gap.py`, `scripts/phase3/c1_domain_offsets.py`; summary: `scripts/phase3/a_mechgen_summary.py` | per model |
| Fig. 3 (mechanisms) | `scripts/phase3/fig_mechanism_composite.py` | calibrate |
| Table 4 (standard tasks), Fig. 6 | `scripts/phase0/p2b_crop_shift.py`, `scripts/phase1/d4_ss_run.py`, `scripts/phase1/d4_contacts.py`; consolidated: `scripts/phase1/t1d_scoping.py` | calibrate |
| Sec. 6.2–6.3 (cut context, tiling) | `scripts/phase3/b1_contacts_partner_out.py`, `rebuttal/r1_partner_out_multiW.py`, `rebuttal/r1_tiling_analysis.py`, `scripts/phase3/b2_contacts_afdb.py` | calibrate |
| Fig. 4, Figs. 7–8 (deployment) | `scripts/phase3/fig_deployment_composite.py` (writes `fig5_deployment.png` = Fig. 4) | calibrate |
| Table 5 (window choices), Table 6 | `scripts/phase2/t2a_remedy.py`, `scripts/phase2/t2a_remedy_esmc.py`, `scripts/phase2/t2a_combine.py`, `scripts/phase2/t2a_most_central_bound.py`, `rebuttal/r1_remedy_cost.py` | calibrate, esmc |
| Fig. 5 (ESM-2 scales) | `scripts/phase0/p1b_scale.py` (`--plot-only` redraws from the stored run) | calibrate |

The effect-size thresholds were fixed before the main probes were run, and each camera-ready
experiment (`scripts/phase3/a_mechgen.py`, `b1_pe_axis.py`, `b2_contacts_afdb.py`, `c_attention_vs_gap.py`,
`c1_domain_offsets.py`) states its decision rules in its docstring, written before the run. Target sampling
and the bootstrap use seed 42.

## Environments and models

Three conda environments: fair-esm and the EvolutionaryScale `esm` SDK both import as `esm`, and ProtBERT,
ProtT5, Ankh, and ProGen2 run under transformers 4.44:

| Env | Setup | Models |
|---|---|---|
| `calibrate` | `scripts/setup_pilot_env.sh` (torch 2.11 cu128, transformers 5.10, fair-esm 2.0) | ESM-2, ESM-1b, SaProt; all stored-output scripts |
| `esmc` | `scripts/setup_esmc_env.sh` (clone of `calibrate` with the `esm` SDK 3.2.1 instead of fair-esm) | ESM C 300M |
| `progen` | `scripts/setup_progen_env.sh` (clone of `calibrate` with transformers 4.44.2) | ProtBERT, ProtT5, Ankh, ProGen2 |

| Model | Checkpoint | How the scripts load it |
|---|---|---|
| ESM-2 650M (35M, 150M for Fig. 5) | fair-esm `esm2_t33_650M_UR50D` | torch hub, cached under `TORCH_HOME` (default `data/cache/models/torch`) |
| ESM-1b 650M | `facebook/esm1b_t33_650M_UR50S` | local copy in `data/cache/models/esm1b_t33_650M_UR50S` |
| SaProt 650M (sequence-only input) | `westlake-repl/SaProt_650M_AF2` | Hugging Face Hub |
| ESM C 300M | `esmc_300m` | `esm` SDK |
| ProtBERT | `Rostlab/prot_bert` | Hugging Face Hub |
| ProtT5-XL encoder | `Rostlab/prot_t5_xl_half_uniref50-enc` | Hugging Face Hub |
| Ankh-base | `ElnaggarLab/ankh-base` | local copy in `data/cache/models/ankh-base` |
| ProGen2-base | `hugohrban/progen2-base` | Hugging Face Hub (`trust_remote_code`) |

The model runs used a single 16 GB GPU; each script sets its own precision. `release/PROVENANCE.md` records
the package versions.

## Data and licenses

- Code: MIT (`LICENSE`).
- Benchmark snapshot, stored outputs, and figures: CC BY 4.0 (`LICENSE-DATA`).
- Third-party sources, which keep their own terms: protein sequences and domain annotations from UniProtKB
  (CC BY 4.0); deep mutational scanning assays from ProteinGym and the original assay publications; the
  contact labels in `experiments/phase3/b2_contacts_afdb/` are derived from AlphaFold DB models (CC BY 4.0).

## Citation

```bibtex
@article{chen2026window,
  title   = {When Does Window Position Matter in Protein Language Models? A Controlled Benchmark and Cross-Architecture Taxonomy},
  author  = {Chen, Yifan},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=kEpikXuZfK}
}
```
