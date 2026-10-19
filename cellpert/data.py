from pathlib import Path
import json
import numpy as np

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

def prepare_h5ad(input_path, outdir, perturbation_key, control_label, min_cells=20, layer=None):
    try:
        import anndata as ad
    except ImportError as e:
        raise RuntimeError("Install optional dependencies: pip install -e '.[singlecell]'") from e
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(input_path)
    if perturbation_key not in a.obs:
        raise KeyError(f"{perturbation_key!r} not found in adata.obs")
    labels = a.obs[perturbation_key].astype(str).to_numpy()
    if control_label not in set(labels):
        raise ValueError(f"Control label {control_label!r} not found")
    x = a.layers[layer] if layer else a.X
    # Starter implementation densifies the matrix; use pseudobulk/chunking for very large files.
    if hasattr(x, "toarray"): x = x.toarray()
    x = np.asarray(x, np.float32)
    if not np.isfinite(x).all(): raise ValueError("Expression matrix contains NaN/Inf")
    ctrl = labels == control_label
    if ctrl.sum() < min_cells: raise ValueError(f"Only {ctrl.sum()} control cells (< min_cells)")
    control_mean = x[ctrl].mean(axis=0)
    perts, deltas, counts = [], [], {}
    for label in sorted(set(labels) - {control_label}):
        mask = labels == label; counts[label] = int(mask.sum())
        if mask.sum() < min_cells: continue
        perts.append(label); deltas.append(x[mask].mean(axis=0) - control_mean)
    if not perts: raise ValueError("No perturbations passed min_cells")
    genes = a.var_names.astype(str).to_numpy()
    save_response_npz(outdir / "response_matrix.npz", perts, genes, np.vstack(deltas))
    (outdir / "gene_names.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")
    summary = {"input": str(input_path), "n_cells": int(a.n_obs), "n_genes": int(a.n_vars),
               "perturbation_key": perturbation_key, "control_label": control_label,
               "control_cells": int(ctrl.sum()), "min_cells": int(min_cells),
               "n_perturbations_retained": len(perts), "cell_counts_by_label": counts,
               "layer": layer, "warning": "Group-mean delta only; no replicate-aware DE or batch correction."}
    (outdir / "preparation_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
