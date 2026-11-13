#!/usr/bin/env python3
"""Build batch-matched RPE1 perturbation response matrices for K562 transfer.

No K562 data are changed. Only K562-overlapping perturbations are included.
RPE1 X is assumed to be raw UMI counts and is checked on sampled blocks.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rpe1', default='data/raw/replogle_rpe1/ReplogleWeissman2022_rpe1.h5ad')
    p.add_argument('--k562', default='data/processed/benchmark_genept_uce_common/response_matrix_common.npz')
    p.add_argument('--outdir', default='results/step08_figure3_rpe1_v2')
    p.add_argument('--perturbation-col', default='gene')
    p.add_argument('--batch-col', default='batch')
    p.add_argument('--control-label', default='non-targeting')
    p.add_argument('--min-controls', type=int, default=20)
    p.add_argument('--min-total-cells', type=int, default=20)
    p.add_argument('--min-batch-cells', type=int, nargs='+', default=[3,5,10])
    p.add_argument('--target-sum', type=float, default=10000.0)
    p.add_argument('--chunk-size', type=int, default=512)
    p.add_argument('--layer', default='X')
    p.add_argument('--skip-count-check', action='store_true', help='Use only if X/layer is independently verified as raw counts')
    return p.parse_args()


def normalize_counts(x, target_sum):
    if sparse.issparse(x):
        x = x.tocsr().astype(np.float64)
        if x.data.size and (not np.isfinite(x.data).all() or x.data.min() < 0):
            raise ValueError('Nonfinite or negative expression encountered')
        totals = np.asarray(x.sum(axis=1)).ravel()
        factors = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals>0)
        x = x.multiply(factors[:, None]).tocsr()
        x.data = np.log1p(x.data)
        return x
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Nonfinite or negative expression encountered')
    totals = x.sum(axis=1)
    factors = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals>0)
    return np.log1p(x * factors[:, None])


def get_chunk(a, layer, ids):
    # ids must be monotonically increasing for backed HDF5 fancy indexing.
    if layer == 'X':
        return a.X[ids, :]
    return a.layers[layer][ids, :]


def audit_count_like(a, layer, chunk_size):
    ids = np.arange(min(chunk_size, a.n_obs), dtype=np.int64)
    x = get_chunk(a, layer, ids)
    vals = x.data if sparse.issparse(x) else np.asarray(x).ravel()
    vals = np.asarray(vals)
    if vals.size == 0 or not np.isfinite(vals).all() or np.min(vals) < 0:
        raise ValueError('Expression is empty, nonfinite or negative')
    integer_fraction = float(np.mean(np.isclose(vals, np.rint(vals), atol=1e-5, rtol=0)))
    if integer_fraction < .999:
        raise ValueError(f'Expression is not count-like (integer fraction {integer_fraction:.6f}). Do not double-normalize.')
    return integer_fraction


def read_normalized(a, layer, ids, chunk_size, target_sum):
    for start in range(0, len(ids), chunk_size):
        chunk_ids = ids[start:start+chunk_size]
        yield chunk_ids, normalize_counts(get_chunk(a, layer, chunk_ids), target_sum)


def main():
    ar = args_parser()
    thresholds = sorted(set(ar.min_batch_cells))
    if not thresholds or min(thresholds) < 1 or ar.min_total_cells < 1 or ar.min_controls < 1 or ar.chunk_size < 1:
        raise ValueError('All cell and chunk thresholds must be positive')
    outdir = Path(ar.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with np.load(ar.k562, allow_pickle=False) as z:
        kperts = z['perturbations'].astype(str)
        kgenes = z['genes'].astype(str)
        if z['delta'].shape != (len(kperts), len(kgenes)):
            raise ValueError('K562 delta axes mismatch')
    if len(set(kperts)) != len(kperts) or len(set(kgenes)) != len(kgenes):
        raise ValueError('Duplicate K562 labels')
    a = ad.read_h5ad(ar.rpe1, backed='r')
    try:
        if ar.layer != 'X' and ar.layer not in a.layers:
            raise ValueError('Requested layer does not exist')
        if not ar.skip_count_check:
            frac = audit_count_like(a, ar.layer, ar.chunk_size)
            print(f'[COUNT AUDIT] integer-like fraction={frac:.6f}', flush=True)
        else:
            frac = None
        for col in (ar.perturbation_col, ar.batch_col):
            if col not in a.obs.columns:
                raise ValueError(f'Missing obs column: {col}')
        rgenes = a.var_names.astype(str).to_numpy()
        if len(set(rgenes)) != len(rgenes):
            raise ValueError('Duplicate RPE1 gene symbols; resolve before running')
        rg_index = {g:i for i,g in enumerate(rgenes)}
        common_genes = [g for g in kgenes if g in rg_index]
        common_idx = np.array([rg_index[g] for g in common_genes], dtype=np.int64)
        print(f'[GENES] K562={len(kgenes)} RPE1={len(rgenes)} common={len(common_genes)}', flush=True)
        labels = a.obs[ar.perturbation_col].astype(str).to_numpy()
        batches = a.obs[ar.batch_col].astype(str).to_numpy()
        kset = set(kperts)
        is_control = labels == ar.control_label
        if not np.any(is_control):
            raise ValueError('No control cells found')
        if 'nperts' in a.obs.columns:
            nperts = pd.to_numeric(a.obs['nperts'], errors='coerce').to_numpy()
            if not np.all(nperts[is_control] == 0):
                raise ValueError('Control label has cells with nperts != 0')
            allowed = np.isfinite(nperts) & (nperts == 1)
        else:
            allowed = np.ones(a.n_obs, dtype=bool)
        # Cross-check optional RPE1 perturbation metadata, if available.
        if 'perturbation' in a.obs.columns and ar.perturbation_col == 'gene':
            aux = a.obs['perturbation'].astype(str).to_numpy()
            if not np.all(aux[is_control] == 'control'):
                raise ValueError('Some non-targeting controls are not annotated perturbation=control')
        is_pert = ~is_control & allowed & np.isin(labels, kperts)
        total_overlap = int(np.unique(labels[is_pert]).size)
        print(f'[PERTURBATIONS] K562-overlapping RPE1 labels={total_overlap}', flush=True)
        # Global accumulators, one row per K562 perturbation; stored only for common response genes.
        npert, ng = len(kperts), len(common_genes)
        sums = {t:np.zeros((npert,ng), dtype=np.float64) for t in thresholds}
        counts = {t:np.zeros(npert, dtype=np.int64) for t in thresholds}
        batch_numbers = {t:np.zeros(npert, dtype=np.int32) for t in thresholds}
        pert_index = {p:i for i,p in enumerate(kperts)}
        qc_rows = []
        control_rows = []
        for bi, batch in enumerate(sorted(np.unique(batches))):
            ctrl_ids = np.flatnonzero((batches == batch) & is_control)
            if len(ctrl_ids) < ar.min_controls:
                control_rows.append({'batch':batch,'n_controls':len(ctrl_ids),'eligible':False})
                print(f'[BATCH {bi+1}] {batch}: controls={len(ctrl_ids)} SKIP', flush=True)
                continue
            control_rows.append({'batch':batch,'n_controls':len(ctrl_ids),'eligible':True})
            # Read each eligible batch's controls only once.
            ctrl_sum = np.zeros(ng, dtype=np.float64)
            for _, x in read_normalized(a, ar.layer, ctrl_ids, ar.chunk_size, ar.target_sum):
                if sparse.issparse(x):
                    ctrl_sum += np.asarray(x[:,common_idx].sum(axis=0)).ravel()
                else:
                    ctrl_sum += x[:,common_idx].sum(axis=0)
            ctrl_mean = ctrl_sum / len(ctrl_ids)
            p_ids = np.flatnonzero((batches == batch) & is_pert)
            if not len(p_ids):
                print(f'[BATCH {bi+1}] {batch}: no overlapping perturbed cells', flush=True)
                continue
            bc = pd.Series(labels[p_ids]).value_counts()
            eligible_names = bc.index[bc.to_numpy() >= thresholds[0]].to_numpy(dtype=str)
            if not len(eligible_names):
                print(f'[BATCH {bi+1}] {batch}: no groups with >= {thresholds[0]} cells', flush=True)
                continue
            mask = np.isin(labels[p_ids], eligible_names)
            p_ids = p_ids[mask]
            local_names = np.sort(eligible_names)
            local_index = {p:i for i,p in enumerate(local_names)}
            local_sums = np.zeros((len(local_names),ng), dtype=np.float64)
            local_counts = np.zeros(len(local_names), dtype=np.int64)
            # Stream eligible perturbed cells in sorted order; aggregate via sparse one-hot multiplication.
            for chunk_ids, x in read_normalized(a, ar.layer, p_ids, ar.chunk_size, ar.target_sum):
                codes = np.array([local_index[p] for p in labels[chunk_ids]], dtype=np.int32)
                m = sparse.csr_matrix((np.ones(len(codes)),(codes,np.arange(len(codes)))),shape=(len(local_names),len(codes)))
                xc = x[:,common_idx]
                block = m @ xc
                if sparse.issparse(block):
                    block = block.toarray()
                local_sums += np.asarray(block)
                local_counts += np.bincount(codes, minlength=len(local_names))
            for j, name in enumerate(local_names):
                n = int(local_counts[j])
                if n != int(bc[name]):
                    raise RuntimeError(f'Count mismatch for {name} batch {batch}')
                row_id = pert_index[name]
                effect_sum = local_sums[j] - n * ctrl_mean
                for t in thresholds:
                    if n >= t:
                        sums[t][row_id] += effect_sum
                        counts[t][row_id] += n
                        batch_numbers[t][row_id] += 1
                qc_rows.append({'perturbation':name,'batch':batch,'n_perturbed':n,'n_controls':len(ctrl_ids),**{f'eligible_batch{t}':bool(n>=t) for t in thresholds}})
            print(f'[BATCH {bi+1}/{len(np.unique(batches))}] {batch}: controls={len(ctrl_ids)}, groups>={thresholds[0]}={len(local_names)}', flush=True)
        pd.DataFrame(qc_rows).to_csv(outdir/'eligible_perturbation_batches.csv', index=False)
        pd.DataFrame(control_rows).to_csv(outdir/'control_batch_qc.csv', index=False)
        original_counts = pd.Series(labels[is_pert]).value_counts()
        summary = []
        for t in thresholds:
            valid = counts[t] >= ar.min_total_cells
            selected = np.flatnonzero(valid)
            d = sums[t][selected] / counts[t][selected,None]
            if not np.isfinite(d).all():
                raise ValueError(f'Nonfinite output for batch{t}')
            sub = outdir/f'batch{t}'
            sub.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sub/'rpe1_response_common_genes.npz',
                                perturbations=kperts[selected],
                                genes=np.asarray(common_genes,dtype=str),
                                delta=d.astype(np.float32))
            qc = pd.DataFrame({'perturbation':kperts,
                               'n_total_rpe1_cells':[int(original_counts.get(p,0)) for p in kperts],
                               'n_matched_cells':counts[t],
                               'n_eligible_batches':batch_numbers[t],
                               'retained':valid})
            qc.to_csv(sub/'perturbation_qc.csv',index=False)
            retained_qc = qc.loc[qc['retained']].copy()
            retained_qc.to_csv(sub/'retained_perturbations.csv',index=False)
            report = {'rpe1':ar.rpe1,'k562':ar.k562,'rpe1_cells':int(a.n_obs),
                      'rpe1_genes':int(a.n_vars),'k562_perturbations':int(len(kperts)),
                      'overlap_perturbations':total_overlap,'common_response_genes':ng,
                      'min_batch_cells':t,'min_total_cells':ar.min_total_cells,
                      'min_controls_per_batch':ar.min_controls,'retained_perturbations':int(valid.sum()),
                      'target_sum':ar.target_sum,'layer':ar.layer,'integer_like_fraction_sample':frac,
                      'response_definition':'weighted batch-matched mean(log1p(CP10k)) perturbed minus control; weights=n_perturbed',
                      'control_definition':f'{ar.perturbation_col} == {ar.control_label}',
                      'warning':'K562 uses global controls while RPE1 uses batch-matched controls; do not interpret as identical batch-adjustment.'}
            (sub/'audit_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
            summary.append({'threshold':t,'retained_perturbations':int(valid.sum()),
                            'total_matched_cells':int(counts[t][valid].sum()),
                            'mean_eligible_batches':float(batch_numbers[t][valid].mean()) if valid.any() else 0.0})
            print(f'[SAVE] batch{t}: {int(valid.sum())} perturbations x {ng} genes',flush=True)
        pd.DataFrame(summary).to_csv(outdir/'threshold_summary.csv',index=False)
        print('[DONE] '+str(outdir),flush=True)
    finally:
        a.file.close()


if __name__ == '__main__':
    main()
