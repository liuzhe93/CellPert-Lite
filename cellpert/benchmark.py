
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reproducible benchmarks for perturbation-response prediction."""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from .data import (
    make_synthetic_dataset,
    load_response_npz,
    load_gene_embeddings,
)
from .models import ResponsePredictor
from .metrics import response_metrics


def align_embeddings(target_genes, embedding_data):
    """Align embedding rows to perturbation IDs in response data."""
    target_genes = np.asarray(target_genes).astype(str)
    embedding_genes = np.asarray(
        embedding_data["genes"]
    ).astype(str)
    embeddings = np.asarray(
        embedding_data["embeddings"],
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional.")

    if len(embedding_genes) != embeddings.shape[0]:
        raise ValueError(
            "Embedding gene count does not match embedding rows."
        )

    if len(set(embedding_genes.tolist())) != len(embedding_genes):
        raise ValueError("Duplicate embedding identifiers detected.")

    lookup = {
        gene: i for i, gene in enumerate(embedding_genes)
    }

    missing = [
        gene for gene in target_genes if gene not in lookup
    ]

    if missing:
        raise ValueError(
            f"Missing embeddings for {len(missing)} IDs; "
            f"examples: {missing[:10]}"
        )

    aligned = embeddings[
        [lookup[gene] for gene in target_genes]
    ]

    if not np.isfinite(aligned).all():
        raise ValueError("Aligned embeddings contain NaN or Inf.")

    return aligned


def run_benchmark(
    response,
    embedding_data,
    outdir,
    seed=42,
    test_size=0.2,
):
    """Run baseline models with a reproducible perturbation-level split."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    perts = np.asarray(response["perturbations"]).astype(str)
    genes = np.asarray(response["genes"]).astype(str)
    delta = np.asarray(response["delta"], dtype=np.float32)

    if delta.ndim != 2:
        raise ValueError("Response delta must be two-dimensional.")

    if delta.shape != (len(perts), len(genes)):
        raise ValueError(
            "Response matrix dimensions do not match perturbations/genes."
        )

    if not np.isfinite(delta).all():
        raise ValueError("Response matrix contains NaN or Inf.")

    if len(set(perts.tolist())) != len(perts):
        raise ValueError("Duplicate perturbation IDs detected.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    # Align each perturbation to exactly one embedding row.
    emb = align_embeddings(perts, embedding_data)

    # Split perturbation IDs, not individual expression measurements.
    idx = np.arange(len(perts))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    train_perts = perts[train_idx]
    test_perts = perts[test_idx]

    if set(train_perts) & set(test_perts):
        raise RuntimeError("Train/test perturbation leakage detected.")

    # Save the exact split so different embedding experiments can
    # be verified to use the same perturbations.
    split_manifest = {
        "seed": int(seed),
        "test_size": float(test_size),
        "n_perturbations": int(len(perts)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_perturbations": train_perts.tolist(),
        "test_perturbations": test_perts.tolist(),
    }

    (outdir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2),
        encoding="utf-8",
    )

    rows = []
    predictions = {}
    models = {}

    model_names = [
        "no_change",
        "train_mean",
        "ridge_gene",
        "mlp_gene",
    ]

    for name in model_names:
        model = ResponsePredictor(
            name,
            random_state=seed,
        )

        model.fit(
            emb[train_idx],
            delta[train_idx],
        )

        pred = model.predict(emb[test_idx])

        if pred.shape != delta[test_idx].shape:
            raise ValueError(
                f"{name}: prediction shape {pred.shape} does not "
                f"match test response shape {delta[test_idx].shape}."
            )

        if not np.isfinite(pred).all():
            raise ValueError(
                f"{name}: predictions contain NaN or Inf."
            )

        metrics = response_metrics(
            delta[test_idx],
            pred,
        )

        rows.append({
            "model": name,
            **metrics,
        })

        predictions[name] = pred
        models[name] = model

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(
        outdir / "metrics.csv",
        index=False,
    )

    # Preserve the original prediction keys for compatibility.
    np.savez_compressed(
        outdir / "predictions.npz",
        test_perturbations=test_perts,
        genes=genes,
        y_true=delta[test_idx],
        **{
            f"pred_{name}": pred
            for name, pred in predictions.items()
        },
    )

    # This bundle intentionally remains the Ridge baseline.
    # It is NOT necessarily the best model and is not selected
    # using the held-out test results.
    bundle = {
        "model": models["ridge_gene"],
        "model_name": "ridge_gene",
        "genes": genes,
        "gene_embedding_dim": int(emb.shape[1]),
        "feature_mode": "gene_embedding_only",
        "seed": int(seed),
        "training_perturbations": train_perts,
        "note": (
            "Ridge baseline fitted on the training split only. "
            "Not a final model selected through validation."
        ),
    }

    joblib.dump(
        bundle,
        outdir / "model_bundle.joblib",
    )

    np.savez_compressed(
        outdir / "gene_embeddings_used.npz",
        genes=perts,
        embeddings=emb,
    )

    # Plot the existing core metrics. NaN values, such as the
    # no-change PCC, remain undefined and are not replaced with zero.
    plot_metrics = [
        "mse",
        "mae",
        "pcc_delta",
    ]

    fig, axes = plt.subplots(
        1, len(plot_metrics), figsize=(13, 4)
    )

    for ax, metric in zip(axes, plot_metrics):
        values = metrics_df[metric].to_numpy(dtype=float)
        x = np.arange(len(metrics_df))
        valid = np.isfinite(values)

        ax.bar(x[valid], values[valid])
        ax.set_xticks(x)
        ax.set_xticklabels(
            metrics_df["model"],
            rotation=35,
            ha="right",
        )
        ax.set_title(metric)

        if (~valid).any():
            ax.text(
                0.02,
                0.95,
                "NaN = undefined",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(
        outdir / "benchmark.png",
        dpi=180,
    )
    plt.close(fig)

    meta = {
        "seed": int(seed),
        "test_size": float(test_size),
        "n_perturbations": int(len(perts)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_genes": int(len(genes)),
        "embedding_dim": int(emb.shape[1]),
        "split_file": "split_manifest.json",
        "metrics_file": "metrics.csv",
        "prediction_file": "predictions.npz",
        "model_bundle": "model_bundle.joblib",
        "model_bundle_name": "ridge_gene",
        "model_selection": (
            "No validation-based model selection performed."
        ),
        "global_pcc_definition": (
            "Pearson correlation after flattening all test "
            "perturbations and measured genes."
        ),
        "perturbation_pcc_definition": (
            "Pearson correlation calculated separately per "
            "perturbation; undefined rows excluded from mean/median."
        ),
        "topk_definition": (
            "Overlap of top-K genes ranked by absolute response. "
            "Rows with constant absolute predictions are excluded; "
            "valid fractions are reported."
        ),
        "warning": (
            "A synthetic input is a software smoke test only. "
            "Held-out perturbations from the same cell context do "
            "not establish cross-cell-context generalization."
        ),
    }

    (outdir / "run_summary.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    return metrics_df


def run_demo(
    outdir,
    seed=42,
    n_perturbations=120,
    n_genes=250,
    embedding_dim=48,
    gene_embedding_path=None,
):
    """Run the synthetic smoke test."""
    toy = make_synthetic_dataset(
        seed,
        n_perturbations,
        n_genes,
        embedding_dim,
    )

    if gene_embedding_path:
        external = load_gene_embeddings(
            gene_embedding_path
        )

        emb = align_embeddings(
            toy["perturbations"],
            external,
        )

        embedding_data = {
            "genes": toy["perturbations"],
            "embeddings": emb,
        }
    else:
        embedding_data = {
            "genes": toy["genes"],
            "embeddings": toy["embeddings"],
        }

    response = {
        "perturbations": toy["perturbations"],
        "genes": toy["genes"],
        "delta": toy["delta"],
    }

    return run_benchmark(
        response,
        embedding_data,
        outdir,
        seed=seed,
    )


def run_response_benchmark(
    response_path,
    embedding_path,
    outdir,
    seed=42,
    test_size=0.2,
):
    """Load real response and embedding NPZ files and run benchmarks."""
    return run_benchmark(
        load_response_npz(response_path),
        load_gene_embeddings(embedding_path),
        outdir,
        seed=seed,
        test_size=test_size,
    )

