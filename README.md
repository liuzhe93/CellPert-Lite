# CellPert-Lite v1.1 Research Release

CellPert-Lite predicts **perturbation-level, population-average gene-expression response vectors** from precomputed gene embeddings. The research workflow uses **GenePT (3,072 features)** and **UCE-derived input features (5,120 features)** with a gated direct-prediction architecture. The baseline `cellpert` CLI remains available for Ridge/MLP benchmarking and a synthetic smoke test.

> **Scope:** This release packages the original research scripts and a stable dispatcher; it is not a fully self-contained data release. K562, RPE1, HepG2 and Jurkat processed matrices, pretrained embeddings and model checkpoints are **not included**. External data access, licenses, preprocessing provenance and compute requirements must be resolved before full reproduction. We do not claim state-of-the-art performance or independent clinical validation.

## Install

```bash
git clone https://github.com/liuzhe93/CellPert-Lite.git
cd CellPert-Lite
python -m pip install -e '.[research,dev]'
python -m pytest -q
cellpert demo --outdir results/demo --seed 42
```

The public GitHub `main` branch may still contain v1.0 until this release is reviewed and merged. The commands below describe **this v1.1 release package**.

## Research input contract

Place the following preprocessed files relative to the repository root:

| File | Required arrays | Verified study dimensions |
|---|---|---|
| `data/processed/benchmark_genept_uce_common/response_matrix_common.npz` | `perturbations`, `genes`, `delta` | 1,950 targets × 8,563 response genes |
| `results/step03_genept_uce_common/aligned_features/GenePT_3large.npz` | `genes`, `embeddings` | 1,950 × 3,072 |
| `results/step03_genept_uce_common/aligned_features/UCE_input_tokens.npz` | `genes`, `embeddings` | 1,950 × 5,120 |

The `genes` field of both embedding files refers to **perturbation target IDs**, not response-output genes. Target order must match `response_matrix_common.npz['perturbations']` exactly. The 1,950 rows represent perturbation targets, **not 1,950 single cells**. UCE features are precomputed; this package does not establish an end-to-end raw UCE extraction recipe.

```bash
python scripts/preflight.py --root .
```

## Reproduce experiments

`research.py` dispatches to the **original analysis scripts** without modifying model fitting, splitting, metric calculations or checkpoint semantics. Use `--dry-run` to inspect a command and `--script-help` after the selected command for the underlying script's options.

```bash
python scripts/research.py k562-unseen --dry-run
python scripts/research.py k562-unseen --root . --seeds 42 43 44 45 46
python scripts/research.py external-transfer --root .
python scripts/research.py external-analysis --root .
python scripts/research.py response-diagnostics --root .
```

**Order matters:** external transfer requires K562 checkpoints from Step10A and preprocessed external response matrices from Step11B-2/2A. `external-prepare` alone may not reproduce the exact expanded threshold dataset used for the reported analysis; consult `docs/REPRODUCIBILITY.md` and the original script configuration. These commands are not a guarantee of successful execution without the required datasets.

### Figure generation

```bash
python scripts/research.py figure2 --script-help
python scripts/research.py figure5 --script-help
python scripts/research.py figure6 --script-help
python scripts/research.py figure7 --script-help
python scripts/research.py figure8 --terms results/step13M_figure8_publication/Figure8_plotted_terms.csv --audit results/step13M_figure8_publication/Figure8_enrichment_audit.csv --outdir results/step13M_figure8_publication_v3
```

Figure 1 is a schematic, not a model-derived plot. Figure 8 v3 renders **already selected** enriched terms; it does not rerun enrichment or independently select terms. See `docs/REPRODUCIBILITY.md` for all figure inputs and interpretation limits.

## Architecture and evaluation

Four variants: `gated_direct`, `no_gate_direct`, `genept_only_direct`, `uce_only_direct`. The gated model uses two linear projections with LayerNorm and GELU, feature-wise sigmoid gating, and a direct 8,563-gene response head. The gated model is called `gated_no_residual` in the original Step09B architecture code. **It is not a Ridge-residual CCRT model.** See `scripts/step09B_ccrt_formal_ablation.py` and `scripts/step09C_direct_ablation_transfer.py` for executable definitions.

The Step10A evaluation splits by perturbation **target**, with inner K562 model selection; external transfer reloads frozen K562 weights and performs prediction replay checks. The RPE1 analysis is exploratory because prior RPE1 results informed research design. Seeds are repeated computational splits, not independent biological replicates.

## Package vs research scripts

- `cellpert/`: original v1.0 baseline CLI and reusable baseline functions.
- `scripts/research.py`: stable public entry point to the research workflow.
- `scripts/step*.py`: original research implementations, preserved to avoid silently changing published computations.
- `scripts/preflight.py`: K562 input validation.
- `tests/`: baseline and release-interface smoke tests.

See `docs/REPRODUCIBILITY.md` and `docs/RELEASE_NOTES_v1.1.md`. Do not commit raw `.h5ad`, large `.npz`, model checkpoints, tokens, or individual-level data unless their redistribution is explicitly permitted.
