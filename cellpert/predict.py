from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import load_gene_embeddings

def predict_one(model_path, embedding_path, perturbation, output_path):
    bundle=joblib.load(model_path); data=load_gene_embeddings(embedding_path)
    lookup={g:i for i,g in enumerate(data["genes"])}
    if perturbation not in lookup: raise ValueError(f"No embedding for perturbation {perturbation!r}")
    if bundle.get("feature_mode") != "gene_embedding_only":
        raise ValueError("This model bundle has an unsupported feature mode")
    x=data["embeddings"][lookup[perturbation]][None,:]
    if x.shape[1] != bundle["gene_embedding_dim"]: raise ValueError("Embedding dimension mismatch")
    pred=bundle["model"].predict(x)[0]
    out=pd.DataFrame({"gene":bundle["genes"], "predicted_delta":pred})
    out["abs_predicted_delta"]=np.abs(out["predicted_delta"])
    out=out.sort_values("abs_predicted_delta", ascending=False)
    path=Path(output_path); path.parent.mkdir(parents=True, exist_ok=True); out.to_csv(path,index=False)
    return out
