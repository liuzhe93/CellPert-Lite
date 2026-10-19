# CellPert-Lite v1.0

A runnable starter framework for predicting **population-average gene-expression changes** after a perturbation. Includes synthetic smoke test, AnnData preprocessing, held-out-perturbation split, No-change / Train-mean / Ridge / MLP baselines, metrics, plots, CLI and tests.

> **Research status:** this is an engineering baseline, not a validated state-of-the-art virtual-cell method. Synthetic demo results are not biological evidence. v1.0 does not yet implement UCE extraction, exact GenePert replication, cross-context validation, or calibrated uncertainty.

## Install

Python 3.10/3.11 recommended.

```bash
git clone https://github.com/<your-account>/CellPert-Lite.git
cd CellPert-Lite
conda env create -f environment.yml
conda activate cellpert-lite
pytest -q
cellpert demo --outdir results/demo --seed 42
```

Alternatively: `pip install -e ".[singlecell,dev]"`.

## Demo outputs

`metrics.csv`, `predictions.npz`, `benchmark.png`, `model_bundle.joblib`, `gene_embeddings_used.npz`, and `run_summary.json`.

## Real `.h5ad` workflow

Inspect labels:
```bash
cellpert inspect-data --input data/k562.h5ad --perturbation-key gene_target
```

Prepare perturbation mean-delta matrix:
```bash
cellpert prepare-h5ad --input data/k562.h5ad --outdir results/prepared \
  --perturbation-key gene_target --control-label control --min-cells 20
```

The input requires `adata.obs[gene_target]` with perturbation labels and a consistent control label. It uses `adata.X` unless `--layer` is specified. Matrix values should be documented (counts or normalized expression). This starter calculates group means; it does not correct batch/guide effects or perform replicate-aware differential expression. For very large `.h5ad` files, the starter currently densifies the expression matrix and may require substantial RAM.

## Precomputed gene embeddings

Use an `.npz` with:
- `genes`: gene symbols/IDs, shape `(n_genes,)`
- `embeddings`: float matrix, shape `(n_genes, embedding_dim)`

GenePT resources: https://github.com/yiqunchen/GenePT

Train/evaluate:
```bash
cellpert train --response results/prepared/response_matrix.npz \
  --gene-embeddings data/gene_embeddings.npz --outdir results/benchmark --seed 42
```

Predict a perturbation with the trained Ridge model:
```bash
cellpert predict --model results/benchmark/model_bundle.joblib \
  --gene-embeddings data/gene_embeddings.npz --perturbation STAT1 --out results/STAT1.csv
```

The target perturbation must have a gene embedding of the same dimension. Predictions are average response changes across the genes retained in training.

## Important scientific caveats

- If no real embeddings are supplied to `demo`, it uses deterministic random vectors strictly for software smoke tests. Do not interpret these as biological embeddings.
- Train/test split is by perturbation label, not by individual cell. For publication, use the official split and preprocessing of the chosen benchmark, compare GenePert and appropriate competing methods, repeat across seeds, and report uncertainty.
- `PCC-delta` in this starter is a flattened correlation between observed and predicted response matrices. Verify and implement the exact benchmark definition before comparing published numbers.
- DEG recovery here ranks absolute group-mean effects; it is not a significance-tested DEG list.
- The MLP is a gene-embedding-only baseline, not the proposed CCRT method. v1.0 does not yet include UCE features, context-conditioned training, GRN/pathway constraints, cross-context evaluation or calibrated uncertainty.
- Do not claim state-of-the-art performance based on the synthetic demo.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Please include OS, Python version, exact command and traceback in bug reports.
