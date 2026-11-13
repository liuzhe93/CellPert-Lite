
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert the official GenePT ada-002 gene embedding pickle
to the NPZ format required by CellPert-Lite.

Features
--------
1. Read a trusted GenePT pickle dictionary.
2. Normalize identifiers using strip().upper().
3. Prefer an exact uppercase raw key when a collision exists.
4. Retain a unique raw key if it has no surrounding whitespace.
5. Exclude unresolved ambiguous identifiers instead of
   silently overwriting or averaging their embeddings.
6. Save an ID audit JSON file.
7. Calculate perturbation-target coverage against a response NPZ.

Important
---------
Uppercase normalization is a string-matching rule, not proof
that two biological identifiers represent the same entity.
Excluded collisions must be reviewed before scientific use.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert GenePT ada-002 embeddings to NPZ."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original GenePT pickle file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output NPZ path.",
    )

    parser.add_argument(
        "--response",
        default=None,
        help="Optional CellPert-Lite response_matrix.npz.",
    )

    return parser.parse_args()


def load_pickle(path):
    """Load a trusted official GenePT pickle file."""
    with open(path, "rb") as handle:
        obj = pickle.load(handle)

    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected a dictionary, received {type(obj)}"
        )

    print(f"Original dictionary entries: {len(obj)}")

    return obj


def collect_groups(obj):
    """Group valid vectors by normalized identifier."""
    groups = {}
    skipped = 0

    for key, value in obj.items():

        # Do not convert non-string keys into gene symbols.
        if not isinstance(key, str):
            skipped += 1
            continue

        gene = key.strip().upper()

        if not gene:
            skipped += 1
            continue

        try:
            vector = np.asarray(
                value, dtype=np.float32
            ).squeeze()

        except (TypeError, ValueError):
            skipped += 1
            continue

        if (
            vector.ndim != 1
            or vector.size == 0
            or not np.isfinite(vector).all()
        ):
            skipped += 1
            continue

        groups.setdefault(gene, []).append(
            (key, vector)
        )

    return groups, skipped


def resolve_groups(groups):
    """
    Resolve normalized IDs deterministically.

    Rules:
    - One exact uppercase raw key: select it.
    - One unique raw key without surrounding whitespace:
      retain it, even if its original case differs.
    - Multiple conflicting raw keys without a unique exact
      uppercase key: exclude the normalized identifier.
    """
    vectors = {}
    audit_rows = []

    duplicate_groups = 0
    unresolved = 0
    dimensions = set()

    for gene, entries in groups.items():

        if len(entries) > 1:
            duplicate_groups += 1

        exact = [
            (key, vector)
            for key, vector in entries
            if key == gene
        ]

        selected_key = ""
        selected_vector = None

        if len(exact) == 1:

            selected_key, selected_vector = exact[0]

            status = "selected_exact_raw_key"

        elif len(entries) == 1:

            raw_key, raw_vector = entries[0]

            if raw_key == raw_key.strip():

                selected_key = raw_key
                selected_vector = raw_vector

                status = (
                    "selected_unique_normalized_key"
                )

            else:
                status = "excluded_key_has_whitespace"

        else:

            # Do not arbitrarily choose between conflicting IDs.
            status = "excluded_ambiguous_normalized_id"

        if selected_vector is not None:

            vectors[gene] = selected_vector

            dimensions.add(
                selected_vector.shape[0]
            )

        else:
            unresolved += 1

        # Audit all duplicate groups and excluded identifiers.
        if len(entries) > 1 or selected_vector is None:

            audit_rows.append({
                "normalized_id": gene,
                "original_keys": [
                    key for key, _ in entries
                ],
                "selected_key": selected_key,
                "status": status,
            })

    if not vectors:
        raise ValueError(
            "No valid gene embeddings remain."
        )

    if len(dimensions) != 1:
        raise ValueError(
            "Inconsistent embedding dimensions among "
            f"selected entries: {sorted(dimensions)}"
        )

    genes = np.asarray(
        list(vectors.keys()), dtype=str
    )

    embeddings = np.stack(
        [vectors[gene] for gene in genes]
    ).astype(np.float32)

    summary = {
        "policy": "prefer_unique_exact_uppercase_key",
        "duplicate_groups": duplicate_groups,
        "excluded_normalized_ids": unresolved,
        "retained_gene_count": len(genes),
        "embedding_shape": list(embeddings.shape),
        "records": audit_rows,
    }

    return genes, embeddings, summary


def save_outputs(
    genes,
    embeddings,
    summary,
    output_path,
    skipped,
):
    """Save embeddings and their audit report."""
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    np.savez_compressed(
        output_path,
        genes=genes,
        embeddings=embeddings,
    )

    summary["skipped_invalid_entries"] = skipped

    audit_path = Path(
        str(output_path) + ".id_audit.json"
    )

    audit_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n===== Conversion summary =====")
    print(f"Output: {output_path}")
    print(f"Genes: {len(genes)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Duplicate groups: {summary['duplicate_groups']}")
    print(
        "Excluded ambiguous IDs: "
        f"{summary['excluded_normalized_ids']}"
    )
    print(f"Skipped invalid entries: {skipped}")
    print(f"ID audit: {audit_path}")

    return output_path


def evaluate_coverage(genes, response_path):
    """Report matching coverage for perturbation targets."""
    response_path = Path(response_path)

    if not response_path.exists():
        raise FileNotFoundError(
            f"Response file not found: {response_path}"
        )

    with np.load(
        response_path, allow_pickle=False
    ) as response:

        required = {"perturbations", "genes", "delta"}

        if not required.issubset(response.files):
            raise ValueError(
                f"Response file must contain {required}"
            )

        targets = np.char.upper(
            np.char.strip(
                response["perturbations"].astype(str)
            )
        )

        delta_shape = response["delta"].shape

        if delta_shape != (
            len(targets), len(response["genes"])
        ):
            raise ValueError(
                "Response matrix dimensions are inconsistent."
            )

    available = set(genes.tolist())

    matched = np.asarray(
        [target in available for target in targets],
        dtype=bool,
    )

    n_matched = int(matched.sum())
    n_missing = int((~matched).sum())

    print("\n===== Perturbation coverage =====")
    print(f"Response file: {response_path}")
    print(f"Perturbation targets: {len(targets)}")
    print(f"Matched targets: {n_matched}")
    print(f"Missing targets: {n_missing}")
    print(
        f"Coverage: {n_matched / len(targets):.2%}"
        if len(targets)
        else "Coverage: undefined (no targets)"
    )

    if n_missing:
        print(
            "Missing examples:",
            targets[~matched][:30].tolist(),
        )


def main():
    args = parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input pickle not found: {input_path}"
        )

    obj = load_pickle(input_path)

    groups, skipped = collect_groups(obj)

    genes, embeddings, summary = resolve_groups(groups)

    save_outputs(
        genes=genes,
        embeddings=embeddings,
        summary=summary,
        output_path=args.output,
        skipped=skipped,
    )

    if args.response:
        evaluate_coverage(
            genes=genes,
            response_path=args.response,
        )


if __name__ == "__main__":
    main()

