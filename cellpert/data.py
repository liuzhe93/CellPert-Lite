from pathlib import Path
import json
import numpy as np
from scipy import sparse

def make_synthetic_dataset(seed=42, n_perturbations=120, n_genes=250, embedding_dim=48):
    """Synthetic smoke-test data only; random embeddings are not biological embeddings."""
    rng = np.random.default_rng(seed)
    genes = np.array([f"GENE{i:04d}" for i in range(n_genes)])
    perturbations = genes[:n_perturbations].copy()
    embeddings = rng.normal(size=(n_genes, embedding_dim)).astype(np.float32)
    latent = rng.normal(size=(embedding_dim, 32))
    z = embeddings[:n_perturbations] @ latent
    basis = rng.normal(size=(32, n_genes))
    delta = (z @ basis / np.sqrt(32)).astype(np.float32)
    delta += rng.normal(scale=0.15, size=delta.shape).astype(np.float32)
    delta[rng.random(delta.shape) < 0.65] *= 0.15
    return {"genes": genes, "perturbations": perturbations,
            "embeddings": embeddings, "delta": delta}

def load_response_npz(path):
    d = np.load(path, allow_pickle=True)
    required = {"perturbations", "genes", "delta"}
    if not required.issubset(d.files):
        raise ValueError(f"Response NPZ requires keys {sorted(required)}")
    out = {"perturbations": d["perturbations"].astype(str),
           "genes": d["genes"].astype(str), "delta": np.asarray(d["delta"], np.float32)}
    if out["delta"].shape != (len(out["perturbations"]), len(out["genes"])):
        raise ValueError("delta shape does not match perturbations x genes")
    if not np.isfinite(out["delta"]).all():
        raise ValueError("delta contains NaN/Inf")
    return out

def load_gene_embeddings(path):
    d = np.load(path, allow_pickle=True)
    required = {"genes", "embeddings"}
    if not required.issubset(d.files):
        raise ValueError(f"Embedding NPZ requires keys {sorted(required)}")
    genes = d["genes"].astype(str)
    emb = np.asarray(d["embeddings"], np.float32)
    if emb.ndim != 2 or emb.shape[0] != len(genes):
        raise ValueError("embeddings must have shape (len(genes), dimension)")
    if len(set(genes)) != len(genes):
        raise ValueError("Duplicate gene IDs in embedding file")
    if not np.isfinite(emb).all():
        raise ValueError("Embedding contains NaN/Inf")
    return {"genes": genes, "embeddings": emb}

def save_response_npz(path, perturbations, genes, delta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, perturbations=np.asarray(perturbations, str),
                        genes=np.asarray(genes, str), delta=np.asarray(delta, np.float32))

def inspect_h5ad(input_path, perturbation_key=None):
    try:
        import anndata as ad
    except ImportError as e:
        raise RuntimeError("Install optional dependencies: pip install -e '.[singlecell]'") from e
    a = ad.read_h5ad(input_path, backed="r")
    out = {"input": str(input_path), "n_cells": int(a.n_obs), "n_genes": int(a.n_vars),
           "obs_columns": list(a.obs.columns), "var_names_preview": a.var_names[:10].astype(str).tolist()}
    if perturbation_key:
        if perturbation_key not in a.obs:
            out["label_error"] = f"{perturbation_key!r} not found in adata.obs"
        else:
            out["label_counts"] = a.obs[perturbation_key].astype(str).value_counts().to_dict()
    return out

def prepare_h5ad(
    input_path,
    outdir,
    perturbation_key="perturbation",
    control_label="control",
    min_cells=20,
    layer=None,
):
    """Sparse-safe export of perturbation mean-minus-control mean responses."""
    from pathlib import Path
    import json
    import numpy as np
    import anndata as ad
    from scipy import sparse

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_path)

    if perturbation_key not in adata.obs.columns:
        raise KeyError(
            f"{perturbation_key!r} not found in adata.obs"
        )

    labels = adata.obs[perturbation_key].astype(str).to_numpy()

    if control_label not in set(labels):
        raise ValueError(
            f"Control label {control_label!r} not found"
        )

    control_mask = labels == control_label
    n_control = int(control_mask.sum())

    if n_control < min_cells:
        raise ValueError(
            f"Only {n_control} control cells; "
            f"min_cells={min_cells}"
        )

    matrix = adata.layers[layer] if layer else adata.X

    # Crucial: preserve sparse format; never densify the full matrix.
    if sparse.issparse(matrix):
        matrix = matrix.tocsr().astype(np.float32)

        def group_mean(mask):
            return np.asarray(
                matrix[mask].mean(axis=0)
            ).ravel().astype(np.float32)

        matrix_mode = "sparse"

    else:
        matrix = np.asarray(matrix, dtype=np.float32)

        if matrix.ndim != 2 or matrix.shape[0] != adata.n_obs:
            raise ValueError("Unexpected expression matrix shape")

        if not np.isfinite(matrix).all():
            raise ValueError("Expression matrix contains NaN/Inf")

        def group_mean(mask):
            return matrix[mask].mean(axis=0).astype(np.float32)

        matrix_mode = "dense"

    control_mean = group_mean(control_mask)

    perturbations = []
    deltas = []
    cell_counts = {}

    unique_labels = sorted(set(labels) - {control_label})

    for i, label in enumerate(unique_labels, start=1):
        mask = labels == label
        n_cells = int(mask.sum())
        cell_counts[label] = n_cells

        if n_cells < min_cells:
            continue

        perturbed_mean = group_mean(mask)
        delta = perturbed_mean - control_mean

        perturbations.append(label)
        deltas.append(delta)

        if i % 200 == 0:
            print(
                f"Processed {i}/{len(unique_labels)} perturbations"
            )

    if not perturbations:
        raise ValueError(
            "No perturbations passed the minimum cell threshold"
        )

    genes = adata.var_names.astype(str).to_numpy()
    delta_matrix = np.vstack(deltas).astype(np.float32)

    # Save using the NPZ format expected by CellPert-Lite.
    np.savez_compressed(
        outdir / "response_matrix.npz",
        perturbations=np.asarray(perturbations, dtype=str),
        genes=np.asarray(genes, dtype=str),
        delta=delta_matrix,
    )

    (outdir / "gene_names.txt").write_text(
        "\n".join(genes) + "\n",
        encoding="utf-8",
    )

    summary = {
        "input": str(input_path),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "perturbation_key": perturbation_key,
        "control_label": control_label,
        "control_cells": n_control,
        "min_cells": int(min_cells),
        "n_perturbations_retained": len(perturbations),
        "cell_counts_by_label": cell_counts,
        "matrix_mode": matrix_mode,
        "response_definition": (
            "Mean expression in each perturbation group "
            "minus mean expression in control cells"
        ),
        "warning": (
            "Group-mean differences only; no replicate-aware "
            "differential expression or batch correction."
        ),
    }

    (outdir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Matrix mode: {matrix_mode}")
    print(f"Retained perturbations: {len(perturbations)}")
    print(f"Response matrix shape: {delta_matrix.shape}")
    print(f"Output: {outdir.resolve()}")

    return summary

