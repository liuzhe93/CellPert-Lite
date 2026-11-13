#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Filter K562 response targets to those shared by both GenePT embeddings."""

import json
from pathlib import Path

import numpy as np


RESPONSE_PATH = Path("data/processed/k562/response_matrix.npz")

LARGE_PATH = Path(
    "data/processed/embeddings/genept_protein_3large.npz"
)

ADA_PATH = Path(
    "data/processed/embeddings/genept_ada.npz"
)

OUTPUT_PATH = Path(
    "data/processed/k562/response_matrix_genept_common.npz"
)

AUDIT_PATH = Path(
    "data/processed/k562/genept_common_target_audit.json"
)


def normalize_ids(values):
    """Normalize identifiers for matching only."""
    return np.asarray(
        [str(x).strip().upper() for x in values],
        dtype=str,
    )


def load_embedding_genes(path):
    """Load and validate gene IDs from a converted embedding NPZ."""
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        if not {"genes", "embeddings"}.issubset(data.files):
            raise ValueError(f"Invalid embedding NPZ: {path}")

        genes = data["genes"].astype(str)
        embeddings = data["embeddings"]

        if embeddings.ndim != 2 or embeddings.shape[0] != len(genes):
            raise ValueError(f"Inconsistent embedding shape: {path}")

        if not np.isfinite(embeddings).all():
            raise ValueError(f"NaN/Inf in embeddings: {path}")

    normalized = normalize_ids(genes)

    # Do not silently collapse collisions at this stage.
    if len(set(normalized.tolist())) != len(normalized):
        raise ValueError(
            f"Duplicate normalized IDs in {path}; "
            "resolve them in the conversion step first."
        )

    return set(normalized.tolist()), list(embeddings.shape)


def main():
    for path in [RESPONSE_PATH, LARGE_PATH, ADA_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    with np.load(RESPONSE_PATH, allow_pickle=False) as data:
        required = {"perturbations", "genes", "delta"}
        if not required.issubset(data.files):
            raise ValueError(
                f"Response NPZ must contain {sorted(required)}"
            )

        original_targets = data["perturbations"].astype(str)
        expression_genes = data["genes"].astype(str)
        delta = np.asarray(data["delta"], dtype=np.float32)

    if delta.shape != (len(original_targets), len(expression_genes)):
        raise ValueError(
            f"Response shape {delta.shape} does not match "
            f"{len(original_targets)} perturbations x "
            f"{len(expression_genes)} genes"
        )

    if not np.isfinite(delta).all():
        raise ValueError("Response matrix contains NaN/Inf")

    normalized_targets = normalize_ids(original_targets)

    if any(not x for x in normalized_targets):
        raise ValueError("Empty perturbation identifier detected")

    if len(set(normalized_targets.tolist())) != len(normalized_targets):
        raise ValueError(
            "Duplicate perturbation IDs after normalization; "
            "inspect the original response matrix before filtering."
        )

    large_genes, large_shape = load_embedding_genes(LARGE_PATH)
    ada_genes, ada_shape = load_embedding_genes(ADA_PATH)

    in_large = np.asarray(
        [gene in large_genes for gene in normalized_targets],
        dtype=bool,
    )

    in_ada = np.asarray(
        [gene in ada_genes for gene in normalized_targets],
        dtype=bool,
    )

    common = in_large & in_ada

    if not common.any():
        raise ValueError("No common perturbation targets found")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Keep response labels normalized consistently with embedding IDs.
    np.savez_compressed(
        OUTPUT_PATH,
        perturbations=normalized_targets[common],
        genes=expression_genes,
        delta=delta[common],
    )

    audit = {
        "response_input": str(RESPONSE_PATH),
        "large_embedding_input": str(LARGE_PATH),
        "ada_embedding_input": str(ADA_PATH),
        "original_perturbation_count": int(len(normalized_targets)),
        "large_embedding_shape": large_shape,
        "ada_embedding_shape": ada_shape,
        "large_matched": int(in_large.sum()),
        "ada_matched": int(in_ada.sum()),
        "common_matched": int(common.sum()),
        "large_coverage": float(in_large.mean()),
        "ada_coverage": float(in_ada.mean()),
        "common_coverage": float(common.mean()),
        "missing_from_large": normalized_targets[~in_large].tolist(),
        "missing_from_ada": normalized_targets[~in_ada].tolist(),
        "missing_from_both": normalized_targets[
            (~in_large) & (~in_ada)
        ].tolist(),
        "output_response_shape": list(delta[common].shape),
        "output_file": str(OUTPUT_PATH),
        "note": (
            "Matching uses uppercase-normalized identifiers. "
            "This does not independently verify biological aliases."
        ),
    }

    AUDIT_PATH.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== Shared GenePT target filtering =====")
    print(f"Original targets: {len(normalized_targets)}")
    print(f"3-large matched: {int(in_large.sum())}")
    print(f"ada-002 matched: {int(in_ada.sum())}")
    print(f"Shared targets: {int(common.sum())}")
    print(f"Shared coverage: {common.mean():.2%}")
    print(f"Filtered response shape: {delta[common].shape}")
    print(f"Saved response: {OUTPUT_PATH}")
    print(f"Saved audit: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
