#!/usr/bin/env python3
"""Audit K562/RPE1 and construct batch-matched RPE1 perturbation responses.

IMPORTANT: RPE1 delta = weighted mean over eligible batches of
(mean log1p(CP10k) perturbed cells - mean log1p(CP10k) non-targeting cells).
This is NOT automatically equivalent to an existing K562 delta matrix.
"""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse


def strings(a):
    return np.asarray(a).astype(str)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_k562(path):
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        required = {"perturbations", "genes", "delta"}
        if not required.issubset(keys):
            raise ValueError(f"K562 NPZ missing {required - set(keys)}; available: {keys}")
        perts, genes = strings(z["perturbations"]), strings(z["genes"])
        y = z["delta"]
        if y.shape != (len(perts), len(genes)):
            raise ValueError(f"K562 delta shape {y.shape} incompatible with labels")
        return perts, genes, y, keys


def logcpm(X):
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()
    X = X.astype(np.float32)
    if X.nnz and (np.min(X.data) < 0 or not np.all(np.isfinite(X.data))):
        raise ValueError("Negative/nonfinite X entries: expected nonnegative counts")
    totals = np.asarray(X.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("Cells with zero library size: inspect filtering before build")
    X = X.multiply((10000.0 / totals)[:, None]).tocsr()
    X.data = np.log1p(X.data)
    return X


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rpe1", required=True)
    p.add_argument("--k562", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--mode", choices=["audit", "build"], default="audit")
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--min-cells-per-batch", type=int, default=5)
    p.add_argument("--min-total-cells", type=int, default=20)
    p.add_argument("--min-control-per-batch", type=int, default=30)
    args = p.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    kp, kg, ky, kkeys = load_k562(args.k562)
    a = ad.read_h5ad(args.rpe1, backed="r")
    try:
        for c in ("batch", "gene", "perturbation", "nperts"):
            if c not in a.obs:
                raise ValueError(f"RPE1 obs missing {c}")
        rg = strings(a.var_names)
        if len(set(rg)) != len(rg):
            raise ValueError("Duplicate RPE1 var_names; resolve before proceeding")
        if len(set(kg)) != len(kg):
            raise ValueError("Duplicate K562 gene names; resolve before proceeding")
        obs = a.obs
        labels = obs["gene"].astype(str)
        is_control = labels.eq("non-targeting") & obs["perturbation"].astype(str).eq("control") & obs["nperts"].eq(0)
        is_pert = obs["nperts"].eq(1) & ~labels.eq("non-targeting")
        if (is_control & is_pert).any():
            raise ValueError("Overlapping control and perturbation cells")
        rperts = sorted(set(labels[is_pert]))
        common_genes = sorted(set(kg) & set(rg))
        common_perts = sorted(set(kp) & set(rperts))
        batch_summary = pd.DataFrame({"batch": obs["batch"].astype(str), "control": is_control.astype(int), "perturbed": is_pert.astype(int)})
        batch_summary = batch_summary.groupby("batch", sort=True)[["control", "perturbed"]].sum().reset_index()
        batch_summary.to_csv(out / "rpe1_batch_counts.csv", index=False)
        counts = labels[is_pert].value_counts().rename_axis("perturbation").reset_index(name="n_cells")
        counts.to_csv(out / "rpe1_perturbation_counts.csv", index=False)
        kstat = {"shape": list(ky.shape), "dtype": str(ky.dtype), "min": float(np.nanmin(ky)), "max": float(np.nanmax(ky)), "mean": float(np.nanmean(ky)), "std": float(np.nanstd(ky)), "nan_count": int(np.isnan(ky).sum()), "npz_keys": kkeys}
        audit = {"k562": kstat, "rpe1_shape": list(a.shape), "rpe1_control_cells": int(is_control.sum()), "rpe1_perturbed_cells": int(is_pert.sum()), "rpe1_batches": int(obs["batch"].nunique()), "rpe1_perturbations": len(rperts), "k562_perturbations": len(kp), "common_perturbations": len(common_perts), "common_expression_genes": len(common_genes), "rpe1_x_encoding": "NOT VERIFIED; sample was count-like", "warning": "K562 delta normalization provenance must be independently verified before merging with RPE1."}
        write_json(out / "audit_summary.json", audit)
        pd.DataFrame({"perturbation": common_perts}).to_csv(out / "common_perturbations.csv", index=False)
        pd.DataFrame({"gene": common_genes}).to_csv(out / "common_expression_genes.csv", index=False)
        print(json.dumps(audit, indent=2))
        if args.mode == "audit":
            print("[DONE] Audit only. No expression matrix has been changed.")
            return
        if args.chunk_size < 1 or args.min_control_per_batch < 1 or args.min_cells_per_batch < 1:
            raise ValueError("Chunk size and minimum cell counts must be positive")
        # Store only common gene expression to keep memory bounded and outputs compatible.
        r_lookup = {g: i for i, g in enumerate(rg)}
        r_idx = np.asarray([r_lookup[g] for g in common_genes], dtype=int)
        batches = obs["batch"].astype(str).to_numpy()
        genes_cell = labels.to_numpy()
        control_arr = is_control.to_numpy()
        pert_arr = is_pert.to_numpy()
        target_index = {g: i for i, g in enumerate(common_perts)}
        weighted_sum = np.zeros((len(common_perts), len(common_genes)), dtype=np.float64)
        total_weight = np.zeros(len(common_perts), dtype=np.int64)
        contributing_batches = np.zeros(len(common_perts), dtype=np.int32)
        detail = []
        for b in sorted(set(batches)):
            ctrl_indices = np.flatnonzero((batches == b) & control_arr)
            if len(ctrl_indices) < args.min_control_per_batch:
                print(f"[SKIP] batch={b} controls={len(ctrl_indices)}")
                continue
            # control mean: log1p(CP10k) per cell, then average
            csum = np.zeros(len(common_genes), dtype=np.float64)
            for start in range(0, len(ctrl_indices), args.chunk_size):
                ii = ctrl_indices[start:start+args.chunk_size]
                X = logcpm(a.X[ii, :])[:, r_idx]
                csum += np.asarray(X.sum(axis=0)).ravel()
            cmean = csum / len(ctrl_indices)
            p_indices = np.flatnonzero((batches == b) & pert_arr & np.isin(genes_cell, common_perts))
            if len(p_indices) == 0:
                continue
            # Aggregate group sums in sparse form to avoid a dense cells x genes matrix.
            psum = {}
            pcount = Counter()
            for start in range(0, len(p_indices), args.chunk_size):
                ii = p_indices[start:start+args.chunk_size]
                X = logcpm(a.X[ii, :])[:, r_idx]
                local_names, inverse = np.unique(genes_cell[ii], return_inverse=True)
                selector = sparse.csr_matrix((np.ones(len(ii), dtype=np.float32), (inverse, np.arange(len(ii)))), shape=(len(local_names), len(ii)))
                means_sum = np.asarray((selector @ X).toarray(), dtype=np.float64)
                for j, name in enumerate(local_names):
                    if name not in psum:
                        psum[name] = means_sum[j]
                    else:
                        psum[name] += means_sum[j]
                    pcount[name] += int(np.sum(inverse == j))
            for name, n in pcount.items():
                if n < args.min_cells_per_batch:
                    continue
                j = target_index[name]
                diff = psum[name] / n - cmean
                weighted_sum[j] += diff * n
                total_weight[j] += n
                contributing_batches[j] += 1
                detail.append((b, name, n, len(ctrl_indices)))
            print(f"[BATCH] {b}: {len(ctrl_indices)} controls, {len(p_indices)} candidate perturbed cells", flush=True)
        valid = total_weight >= args.min_total_cells
        if not valid.any():
            raise RuntimeError("No perturbations passed minimum cells; adjust filters after reviewing audit")
        Y = (weighted_sum[valid] / total_weight[valid, None]).astype(np.float32)
        names = np.asarray(common_perts, dtype=str)[valid]
        np.savez_compressed(out / "rpe1_response_common_provisional.npz", perturbations=names, genes=np.asarray(common_genes, dtype=str), delta=Y)
        pd.DataFrame({"perturbation": common_perts, "eligible_cells": total_weight, "contributing_batches": contributing_batches, "retained": valid}).to_csv(out / "rpe1_response_qc.csv", index=False)
        pd.DataFrame(detail, columns=["batch", "perturbation", "n_perturbed_cells", "n_control_cells"]).to_csv(out / "rpe1_batch_perturbation_qc.csv", index=False)
        write_json(out / "rpe1_response_protocol.json", {"response": "per-batch mean(log1p(counts/sum(counts)*10000)) perturbation minus non-targeting control; batch differences weighted by perturbation cell counts", "source": args.rpe1, "output": "rpe1_response_common_provisional.npz", "shared_genes": len(common_genes), "retained_perturbations": int(valid.sum()), "min_control_per_batch": args.min_control_per_batch, "min_cells_per_batch": args.min_cells_per_batch, "min_total_cells": args.min_total_cells, "IMPORTANT": "PROVISIONAL; NOT proven same response scale as K562. No combined training until K562 provenance checked."})
        print(f"[DONE] Saved {Y.shape} provisional RPE1 response matrix")
    finally:
        a.file.close()

if __name__ == "__main__":
    main()
