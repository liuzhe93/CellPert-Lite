import anndata as ad
import numpy as np
import pandas as pd

files = {
    "HepG2": "data/external_validation/HepG2/nadig_2024_hepg2.h5ad",
    "Jurkat": "data/external_validation/Jurkat/nadig_2024_jurkat.h5ad",
}

for name, path in files.items():
    print("\n" + "=" * 70)
    print("DATASET:", name)
    print("=" * 70)
    adata = ad.read_h5ad(path, backed="r")
    try:
        print("\nShape:", adata.shape)
        print("\n[1] Control distribution")
        print(adata.obs["is_control"].value_counts(dropna=False))
        print("\n[2] Perturbation types")
        print(adata.obs["perturbation_type"].value_counts(dropna=False).head(20))
        print("\n[3] Perturbation targets")
        target_counts = adata.obs["gene_name"].astype("string").value_counts()
        print("Unique targets:", len(target_counts))
        print("Targets >= 30 cells:", (target_counts >= 30).sum())
        print("Targets >= 50 cells:", (target_counts >= 50).sum())
        print("Targets >= 100 cells:", (target_counts >= 100).sum())
        print("\nTop 10 targets:")
        print(target_counts.head(10))
        print("\n[4] Batch distribution")
        print(adata.obs["batch_num"].value_counts(dropna=False).head(20))
        print("\n[5] Control by batch")
        print(pd.crosstab(adata.obs["batch_num"], adata.obs["is_control"]))
        print("\n[6] Expression matrix sample")
        n_sample = min(500, adata.n_obs)
        X = adata.X[:n_sample]
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X)
        print("Min:", np.min(X))
        print("Max:", np.max(X))
        print("Mean:", np.mean(X))
        print("Median:", np.median(X))
        print("Zero fraction:", np.mean(X == 0))
        print("Approximately integer:", np.mean(np.isclose(X, np.round(X), atol=1e-5)))
        print("\nGene examples:")
        print(adata.var_names[:10].tolist())
        print("\nGene ID columns:")
        print(adata.var[["ensembl_gene_id", "gene_name"]].head(10))
    finally:
        adata.file.close()

print("\nAUDIT COMPLETE")
