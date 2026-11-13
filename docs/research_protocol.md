# Research protocol and known gaps

## v1.0 target
Establish a reproducible baseline for held-out perturbation response prediction.

## Required before manuscript claims
1. Use official benchmark preprocessing and splits where available.
2. Reproduce GenePert using its official implementation and documented settings.
3. Add control-only UCE context embeddings without test-perturbation leakage.
4. Evaluate held-out perturbations and held-out cellular contexts separately.
5. Add validated uncertainty calibration on a separate calibration split.
6. Repeat with fixed seeds and report mean, dispersion and confidence intervals.
7. Audit gene identifiers, control labels, guide/replicate metadata and normalization.

## Current limitations
The v1.0 `train` command uses a random perturbation-level holdout, not an official benchmark split. `prepare-h5ad` computes group mean deltas and does not handle replicate-level DE or batch correction. The prediction model only uses gene embeddings. The current software is a starting point, not yet the proposed CCRT method.
