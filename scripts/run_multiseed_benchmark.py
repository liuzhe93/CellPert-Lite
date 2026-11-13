#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run paired multi-seed CellPert-Lite K562 benchmarks."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEEDS = [42, 123, 2025, 3407, 2026]

DEFAULT_RESPONSE = (
    "data/processed/k562/response_matrix_genept_common.npz"
)

DEFAULT_EMBEDDINGS = {
    "genept_3large":
        "data/processed/embeddings/genept_protein_3large.npz",
    "genept_ada":
        "data/processed/embeddings/genept_ada.npz",
}


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_one(response, embedding, outdir, seed, test_size):
    """Run one benchmark through the installed CellPert-Lite CLI."""
    metrics_path = outdir / "metrics.csv"

    if metrics_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {outdir}\n"
            "Choose a new --outdir or remove the old experiment "
            "directory after backing it up."
        )

    outdir.mkdir(parents=True, exist_ok=True)

    command = [
        "cellpert", "train",
        "--response", str(response),
        "--gene-embeddings", str(embedding),
        "--outdir", str(outdir),
        "--seed", str(seed),
        "--test-size", str(test_size),
    ]

    print("\n" + "=" * 78)
    print("Running:", " ".join(command))
    print("=" * 78)

    subprocess.run(command, check=True)

    required_files = [
        outdir / "metrics.csv",
        outdir / "split_manifest.json",
        outdir / "run_summary.json",
        outdir / "predictions.npz",
    ]

    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise RuntimeError(
            "Benchmark finished without expected files: "
            + ", ".join(missing)
        )

    metrics = pd.read_csv(metrics_path)

    if "model" not in metrics.columns:
        raise ValueError(f"'model' column missing in {metrics_path}")

    metrics["seed"] = seed
    metrics["embedding"] = embedding.name
    metrics["run_dir"] = str(outdir)

    return metrics


def check_paired_split(path_a, path_b, seed):
    """Require identical perturbation-level splits within each seed."""
    a = read_json(path_a)
    b = read_json(path_b)

    same_train = (
        a["train_perturbations"] == b["train_perturbations"]
    )
    same_test = (
        a["test_perturbations"] == b["test_perturbations"]
    )

    if not same_train or not same_test:
        raise RuntimeError(
            f"Split mismatch for seed={seed}: "
            f"same_train={same_train}, same_test={same_test}"
        )

    print(
        f"[PASS] seed={seed}: paired train/test perturbations identical "
        f"(train={a['n_train']}, test={a['n_test']})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run paired five-seed K562 GenePT benchmarks."
    )

    parser.add_argument(
        "--response",
        default=DEFAULT_RESPONSE,
    )
    parser.add_argument(
        "--outdir",
        default="results/k562_multiseed",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--large-embedding",
        default=DEFAULT_EMBEDDINGS["genept_3large"],
    )
    parser.add_argument(
        "--ada-embedding",
        default=DEFAULT_EMBEDDINGS["genept_ada"],
    )

    args = parser.parse_args()

    response = Path(args.response)
    large_embedding = Path(args.large_embedding)
    ada_embedding = Path(args.ada_embedding)
    root = Path(args.outdir)

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Random seeds must be unique.")

    for path in [response, large_embedding, ada_embedding]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {root}\n"
            "Use a new --outdir to preserve existing results."
        )

    root.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in args.seeds:
        seed_dir = root / f"seed_{seed}"

        large_dir = seed_dir / "genept_3large"
        ada_dir = seed_dir / "genept_ada"

        # Same response, seed, and test size; only the embedding differs.
        large_metrics = run_one(
            response,
            large_embedding,
            large_dir,
            seed,
            args.test_size,
        )

        ada_metrics = run_one(
            response,
            ada_embedding,
            ada_dir,
            seed,
            args.test_size,
        )

        check_paired_split(
            large_dir / "split_manifest.json",
            ada_dir / "split_manifest.json",
            seed,
        )

        all_results.extend(
            [large_metrics, ada_metrics]
        )

    raw = pd.concat(all_results, ignore_index=True)

    raw_path = root / "metrics_by_seed.csv"
    raw.to_csv(raw_path, index=False)

    metric_columns = [
        col for col in raw.columns
        if col not in {"model", "seed", "embedding", "run_dir"}
        and pd.api.types.is_numeric_dtype(raw[col])
    ]

    summary_rows = []

    for (embedding, model), group in raw.groupby(
        ["embedding", "model"], sort=True
    ):
        row = {
            "embedding": embedding,
            "model": model,
            "n_runs": int(group["seed"].nunique()),
        }

        for metric in metric_columns:
            values = pd.to_numeric(
                group[metric], errors="coerce"
            ).to_numpy(dtype=float)

            finite = values[np.isfinite(values)]

            row[f"{metric}_mean"] = (
                float(np.mean(finite))
                if len(finite)
                else float("nan")
            )

            row[f"{metric}_sd"] = (
                float(np.std(finite, ddof=1))
                if len(finite) > 1
                else float("nan")
            )

            row[f"{metric}_valid_n"] = int(len(finite))

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = root / "metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Paired differences: 3-large minus ada, on the same seed.
    paired = raw.pivot_table(
        index=["seed", "model"],
        columns="embedding",
        values=metric_columns,
        aggfunc="first",
    )

    paired_rows = []

    for (seed, model), row in paired.iterrows():
        result = {
            "seed": int(seed),
            "model": model,
        }

        for metric in metric_columns:
            try:
                large_value = row[(metric, "genept_protein_3large.npz")]
                ada_value = row[(metric, "genept_ada.npz")]
            except KeyError:
                continue

            result[f"{metric}_large_minus_ada"] = (
                float(large_value - ada_value)
                if pd.notna(large_value) and pd.notna(ada_value)
                else float("nan")
            )

        paired_rows.append(result)

    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(
            root / "paired_differences_by_seed.csv",
            index=False,
        )

    run_info = {
        "response": str(response),
        "large_embedding": str(large_embedding),
        "ada_embedding": str(ada_embedding),
        "seeds": args.seeds,
        "test_size": args.test_size,
        "n_seeds": len(args.seeds),
        "paired_split_check": "passed for every seed",
        "summary_note": (
            "SD is the sample standard deviation across random splits. "
            "These runs are repeated computational splits, not "
            "independent biological replicates."
        ),
    }

    (root / "multiseed_summary.json").write_text(
        json.dumps(run_info, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("MULTI-SEED BENCHMARK COMPLETE")
    print("=" * 78)
    print(f"Raw metrics: {raw_path}")
    print(f"Mean/SD summary: {summary_path}")
    print(f"Run configuration: {root / 'multiseed_summary.json'}")

    paired_path = root / "paired_differences_by_seed.csv"
    if paired_path.exists():
        print(f"Paired differences: {paired_path}")

    print("\nSummary:")
    show_columns = [
        c for c in [
            "embedding",
            "model",
            "n_runs",
            "mse_mean",
            "mse_sd",
            "mae_mean",
            "mae_sd",
            "pcc_delta_mean",
            "pcc_delta_sd",
        ]
        if c in summary.columns
    ]
    print(summary[show_columns].to_string(index=False))


if __name__ == "__main__":
    main()
