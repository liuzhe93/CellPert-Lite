from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from .data import make_synthetic_dataset, load_response_npz, load_gene_embeddings
from .models import ResponsePredictor
from .metrics import response_metrics

def align_embeddings(target_genes, embedding_data):
    lookup = {g:i for i,g in enumerate(embedding_data["genes"])}
    missing = [g for g in target_genes if g not in lookup]
    if missing: raise ValueError(f"Missing embeddings for {len(missing)} IDs; examples: {missing[:5]}")
    return embedding_data["embeddings"][[lookup[g] for g in target_genes]]

def run_benchmark(response, embedding_data, outdir, seed=42, test_size=0.2):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    perts = response["perturbations"].astype(str)
    genes = response["genes"].astype(str); delta = response["delta"]
    emb = align_embeddings(perts, embedding_data)
    idx = np.arange(len(perts))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    rows=[]; predictions={}; models={}
    for name in ["no_change", "train_mean", "ridge_gene", "mlp_gene"]:
        model = ResponsePredictor(name, random_state=seed).fit(emb[train_idx], delta[train_idx])
        pred = model.predict(emb[test_idx])
        rows.append({"model":name, **response_metrics(delta[test_idx], pred)})
        predictions[name] = pred; models[name] = model
    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir/"metrics.csv", index=False)
    np.savez_compressed(outdir/"predictions.npz",
        test_perturbations=perts[test_idx], genes=genes, y_true=delta[test_idx],
        **{f"pred_{k}":v for k,v in predictions.items()})
    # Fit final ridge on training split only, so the held-out test remains untouched.
    bundle={"model":models["ridge_gene"], "model_name":"ridge_gene",
            "genes":genes, "gene_embedding_dim":int(emb.shape[1]),
            "feature_mode":"gene_embedding_only", "seed":int(seed)}
    joblib.dump(bundle, outdir/"model_bundle.joblib")
    np.savez_compressed(outdir/"gene_embeddings_used.npz", genes=perts, embeddings=emb)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["mse","mae","pcc_delta"]):
        ax.bar(metrics["model"], metrics[metric])
        ax.set_title(metric); ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(outdir/"benchmark.png", dpi=180); plt.close(fig)
    meta={"seed":int(seed), "test_size":float(test_size),
          "n_perturbations":len(perts), "n_train":len(train_idx), "n_test":len(test_idx),
          "n_genes":len(genes), "embedding_dim":int(emb.shape[1]),
          "warning":"If input is synthetic, this is a software smoke test only."}
    (outdir/"run_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return metrics

def run_demo(outdir, seed=42, n_perturbations=120, n_genes=250, embedding_dim=48,
             gene_embedding_path=None):
    toy = make_synthetic_dataset(seed, n_perturbations, n_genes, embedding_dim)
    if gene_embedding_path:
        external=load_gene_embeddings(gene_embedding_path)
        emb=align_embeddings(toy["perturbations"], external)
        embedding_data={"genes":toy["perturbations"], "embeddings":emb}
    else:
        embedding_data={"genes":toy["genes"], "embeddings":toy["embeddings"]}
    response={"perturbations":toy["perturbations"], "genes":toy["genes"], "delta":toy["delta"]}
    return run_benchmark(response, embedding_data, outdir, seed=seed)

def run_response_benchmark(response_path, embedding_path, outdir, seed=42, test_size=0.2):
    return run_benchmark(load_response_npz(response_path), load_gene_embeddings(embedding_path),
                         outdir, seed=seed, test_size=test_size)
