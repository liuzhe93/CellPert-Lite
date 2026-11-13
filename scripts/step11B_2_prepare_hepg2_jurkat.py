#!/usr/bin/env python3
"""Build batch-matched log1p(CP10k) CRISPRi response matrices for HepG2/Jurkat.

Read-only input; backed, chunked; no training, no invented gene observations.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def norm_chunk(x, target_sum):
    if sparse.issparse(x):
        x = x.tocsr().astype(np.float64)
        x.sum_duplicates()
        if x.data.size and (np.min(x.data) < 0 or not np.all(np.isfinite(x.data))):
            raise ValueError('Invalid count matrix: negative or nonfinite entries')
        sums = np.asarray(x.sum(axis=1)).ravel()
        scales = np.divide(target_sum, sums, out=np.zeros_like(sums), where=sums > 0)
        x = x.multiply(scales[:, None]).tocsr()
        x.data = np.log1p(x.data)
        return x
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Invalid count matrix: negative or nonfinite entries')
    sums = x.sum(axis=1)
    scales = np.divide(target_sum, sums, out=np.zeros_like(sums), where=sums > 0)
    return np.log1p(x * scales[:, None])


def get_gene_names(a, name):
    if name == 'Jurkat':
        genes = a.var['gene_name'].astype(str).to_numpy()
    elif 'gene_name' in a.var:
        genes = a.var['gene_name'].astype(str).to_numpy()
    else:
        genes = a.var_names.astype(str).to_numpy()
    genes = np.array([g.strip() for g in genes], dtype=str)
    invalid = {'', 'nan', 'None', 'NA'}
    # Preserve only uniquely identifiable genes; ambiguous duplicated symbols are excluded.
    counts = pd.Series(genes).value_counts()
    keep = np.array([g not in invalid and counts[g] == 1 for g in genes])
    return genes, keep


def process(name, path, outroot, args, reference_genes):
    print(f'\n### {name}: {path}', flush=True)
    a = ad.read_h5ad(path, backed='r')
    try:
        for col in ['gene_name', 'is_control', 'batch_num']:
            if col not in a.obs.columns:
                raise KeyError(f'{name}: missing obs column {col}')
        raw_genes, keep = get_gene_names(a, name)
        genes = raw_genes[keep]
        if len(genes) == 0:
            raise ValueError('No valid unique gene symbols')
        col_idx = np.flatnonzero(keep)
        labels = a.obs['gene_name'].astype(str).str.strip().to_numpy()
        batches = a.obs['batch_num'].astype(str).to_numpy()
        ctrl_raw = a.obs['is_control']
        if pd.api.types.is_bool_dtype(ctrl_raw):
            controls = ctrl_raw.to_numpy(dtype=bool)
        else:
            values = ctrl_raw.astype(str).str.lower().str.strip()
            if not values.isin(['true', 'false', '1', '0']).all():
                raise ValueError('Unrecognized is_control values')
            controls = values.isin(['true', '1']).to_numpy()
        if not np.any(controls):
            raise ValueError('No control cells')
        if np.any(controls & (labels != 'non-targeting')):
            print('WARNING: some is_control=True labels differ from non-targeting', flush=True)
        # Batch-specific control eligibility fixed before inspecting expression.
        batch_control_counts = pd.Series(batches[controls]).value_counts().to_dict()
        eligible = {b for b, n in batch_control_counts.items() if n >= args.min_controls}
        print(f'Cells={a.n_obs:,}; genes={a.n_vars:,}; unique valid genes={len(genes):,}; '
              f'controls={controls.sum():,}; eligible batches={len(eligible)}/{len(set(batches))}', flush=True)
        if not eligible:
            raise ValueError('No batches pass minimum controls')

        # Sum vectors indexed by (batch, perturbation); one sparse matrix read per chunk.
        sums = {}
        counts = defaultdict(int)
        ctrl_sums = {}
        ctrl_counts = defaultdict(int)
        zero_library_cells = 0
        for start in range(0, a.n_obs, args.chunk_size):
            stop = min(start + args.chunk_size, a.n_obs)
            x = a.X[start:stop, col_idx]
            if sparse.issparse(x):
                raw_sum = np.asarray(x.sum(axis=1)).ravel()
            else:
                raw_sum = np.asarray(x).sum(axis=1)
            zero_library_cells += int(np.sum(raw_sum == 0))
            if not np.all(np.isfinite(raw_sum)):
                raise ValueError(f'Nonfinite library sizes at rows {start}:{stop}')
            x = norm_chunk(x, args.target_sum)
            bs = batches[start:stop]
            ls = labels[start:stop]
            cs = controls[start:stop]
            valid = np.isin(bs, list(eligible))
            # Restrict to eligible batches, then aggregate by group using sparse indicator.
            indices = np.flatnonzero(valid)
            if len(indices) == 0:
                continue
            keys = [(str(bs[i]), '__CONTROL__' if cs[i] else str(ls[i])) for i in indices]
            # exclude malformed non-control labels
            active = [j for j, key in enumerate(keys) if key[1] not in ('', 'nan', 'None', 'non-targeting') or cs[indices[j]]]
            if not active:
                continue
            indices = indices[active]
            keys = [keys[j] for j in active]
            unique_keys = list(dict.fromkeys(keys))
            mapping = {k: i for i, k in enumerate(unique_keys)}
            group_idx = np.fromiter((mapping[k] for k in keys), dtype=np.int32)
            indicator = sparse.csr_matrix((np.ones(len(indices), dtype=np.float64),
                                          (group_idx, np.arange(len(indices)))),
                                         shape=(len(unique_keys), len(indices)))
            sub = x[indices]
            grouped = indicator @ sub
            if sparse.issparse(grouped):
                grouped = grouped.toarray()
            grouped = np.asarray(grouped)
            local_n = np.bincount(group_idx, minlength=len(unique_keys))
            for j, key in enumerate(unique_keys):
                batch, label = key
                if label == '__CONTROL__':
                    if batch not in ctrl_sums:
                        ctrl_sums[batch] = np.zeros(len(genes), dtype=np.float64)
                    ctrl_sums[batch] += grouped[j]
                    ctrl_counts[batch] += int(local_n[j])
                else:
                    if key not in sums:
                        sums[key] = np.zeros(len(genes), dtype=np.float64)
                    sums[key] += grouped[j]
                    counts[key] += int(local_n[j])
            if (start // args.chunk_size + 1) % 25 == 0 or stop == a.n_obs:
                print(f'  processed {stop:,}/{a.n_obs:,} cells', flush=True)
        # Combine batch-matched effects weighted by perturbation-cell count.
        per_target = defaultdict(list)
        batch_rows = []
        for (batch, target), n in sorted(counts.items()):
            if n < args.min_batch_pert_cells:
                continue
            nc = ctrl_counts[batch]
            if nc < args.min_controls:
                continue
            effect = sums[(batch, target)] / n - ctrl_sums[batch] / nc
            per_target[target].append((n, effect))
            batch_rows.append({'perturbation': target, 'batch': batch,
                               'n_perturbed': n, 'n_controls': nc})
        targets, rows, target_qc = [], [], []
        for target in sorted(per_target):
            pieces = per_target[target]
            total = sum(n for n, _ in pieces)
            if total < args.min_cells:
                continue
            weighted = sum(n * effect for n, effect in pieces) / total
            targets.append(target)
            rows.append(weighted.astype(np.float32))
            target_qc.append({'perturbation': target, 'n_perturbed': total,
                              'n_batches': len(pieces)})
        if not targets:
            raise ValueError(f'{name}: no perturbations retained')
        delta = np.vstack(rows)
        if not np.isfinite(delta).all():
            raise ValueError('Nonfinite output delta')
        out = outroot / name
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'response_matrix_log1p_cp10k_batchmatched.npz',
                            perturbations=np.asarray(targets), genes=genes, delta=delta)
        pd.DataFrame(batch_rows).to_csv(out / 'batch_qc.csv', index=False)
        pd.DataFrame(target_qc).to_csv(out / 'perturbation_qc.csv', index=False)
        pd.DataFrame({'original_column': np.arange(a.n_vars), 'gene_symbol': raw_genes,
                      'retained_unique_symbol': keep}).to_csv(out / 'gene_qc.csv', index=False)
        pd.DataFrame([{'batch': b, 'n_controls': batch_control_counts.get(b, 0),
                       'eligible': b in eligible} for b in sorted(set(batches))]).to_csv(
                           out / 'control_batch_qc.csv', index=False)
        common = [g for g in reference_genes if g in set(genes)] if reference_genes is not None else []
        if reference_genes is not None:
            gene_to_col = {g: j for j, g in enumerate(genes)}
            cols = [gene_to_col[g] for g in common]
            np.savez_compressed(out / 'response_matrix_k562_gene_intersection.npz',
                                perturbations=np.asarray(targets), genes=np.asarray(common),
                                delta=delta[:, cols])
            pd.DataFrame({'k562_gene': reference_genes,
                          'observed_in_external': np.isin(reference_genes, genes)}).to_csv(
                              out / 'k562_gene_coverage.csv', index=False)
        summary = {'dataset': name, 'input': str(path), 'n_cells': int(a.n_obs),
                   'n_input_genes': int(a.n_vars), 'n_unique_symbol_genes': len(genes),
                   'n_controls': int(controls.sum()), 'n_batches': len(set(batches)),
                   'n_eligible_batches': len(eligible), 'min_controls': args.min_controls,
                   'min_batch_pert_cells': args.min_batch_pert_cells,
                   'min_total_pert_cells': args.min_cells,
                   'n_perturbations_retained': len(targets),
                   'n_k562_response_genes_observed': len(common) if reference_genes is not None else None,
                   'zero_library_cells': zero_library_cells,
                   'target_sum': args.target_sum,
                   'expression': 'log1p(raw_counts / per_cell_sum(raw_counts) * target_sum)',
                   'response': 'sum_batch(n_pert_batch * (mean_pert_batch - mean_ctrl_batch)) / sum_batch(n_pert_batch)',
                   'note': 'K562 used global controls; external datasets use batch-matched controls. No missing genes imputed.'}
        (out / 'preparation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'  SAVED: {len(targets)} targets x {len(genes)} observed genes; '
              f'{len(common)} K562 response genes observed', flush=True)
    finally:
        a.file.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--outdir', default='data/external_validation/step11B_2_responses')
    p.add_argument('--k562-response', default='data/processed/benchmark_genept_uce_common/response_matrix_common.npz')
    p.add_argument('--datasets', nargs='+', choices=['HepG2', 'Jurkat'], default=['HepG2', 'Jurkat'])
    p.add_argument('--chunk-size', type=int, default=512)
    p.add_argument('--target-sum', type=float, default=10000)
    p.add_argument('--min-controls', type=int, default=30)
    p.add_argument('--min-batch-pert-cells', type=int, default=5)
    p.add_argument('--min-cells', type=int, default=30)
    args = p.parse_args()
    if min(args.chunk_size, args.min_controls, args.min_batch_pert_cells, args.min_cells) < 1 or args.target_sum <= 0:
        p.error('Thresholds/chunk size/target sum must be positive')
    root = args.root.resolve()
    ref_path = root / args.k562_response
    if not ref_path.is_file():
        p.error(f'Missing K562 reference response: {ref_path}')
    with np.load(ref_path, allow_pickle=False) as z:
        reference_genes = z['genes'].astype(str)
    paths = {'HepG2': root / 'data/external_validation/HepG2/nadig_2024_hepg2.h5ad',
             'Jurkat': root / 'data/external_validation/Jurkat/nadig_2024_jurkat.h5ad'}
    for name in args.datasets:
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
        process(name, paths[name], root / args.outdir, args, reference_genes)
    print('\nDONE: external response matrices generated. Frozen models have NOT been run.', flush=True)


if __name__ == '__main__':
    main()
