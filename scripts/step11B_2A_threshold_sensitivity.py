#!/usr/bin/env python3
"""Step11B-2A: pre-prediction, metadata-only batch-threshold sensitivity.

Does not read expression matrices, train models, or modify Step11B-2 outputs.
Optionally rebuilds an expanded response set using the existing Step11B-2 script.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

DATA = {
    'HepG2': 'data/external_validation/HepG2/nadig_2024_hepg2.h5ad',
    'Jurkat': 'data/external_validation/Jurkat/nadig_2024_jurkat.h5ad',
}


def audit_dataset(name, path, out, batch_thresholds, min_controls, total_thresholds):
    print(f'\n[{name}] metadata: {path}', flush=True)
    a = ad.read_h5ad(path, backed='r')
    try:
        obs = a.obs[['gene_name', 'batch_num', 'is_control']].copy()
        n_cells = a.n_obs
    finally:
        a.file.close()
    labels = obs['gene_name'].astype('string').str.strip()
    batches = obs['batch_num'].astype('string').str.strip()
    raw_ctrl = obs['is_control']
    if pd.api.types.is_bool_dtype(raw_ctrl):
        ctrl = raw_ctrl.fillna(False).astype(bool)
    else:
        s = raw_ctrl.astype('string').str.lower().str.strip()
        if not s.isin(['true', 'false', '1', '0']).all():
            raise ValueError(f'{name}: invalid is_control values: {s.value_counts(dropna=False).to_dict()}')
        ctrl = s.isin(['true', '1'])
    if labels.isna().any() or batches.isna().any():
        raise ValueError(f'{name}: missing gene_name/batch_num')
    bad_labels = {'', 'nan', 'none', 'na', 'non-targeting'}
    malformed = (~ctrl) & labels.str.lower().isin(bad_labels)
    if malformed.any():
        print(f'  WARNING: {malformed.sum()} malformed noncontrol labels excluded')
    n_ctrl = (pd.DataFrame({'batch': batches[ctrl].to_numpy()})
              .groupby('batch', observed=True).size().rename('n_controls'))
    all_batches = pd.Index(sorted(batches.unique()), name='batch')
    ctrl_table = n_ctrl.reindex(all_batches, fill_value=0).reset_index()
    ctrl_table['eligible'] = ctrl_table['n_controls'] >= min_controls
    ctrl_table.to_csv(out / f'{name}_control_batch_qc.csv', index=False)
    eligible = set(ctrl_table.loc[ctrl_table.eligible, 'batch'])

    pert = pd.DataFrame({'target': labels[(~ctrl) & (~malformed)].to_numpy(),
                         'batch': batches[(~ctrl) & (~malformed)].to_numpy()})
    groups = pert.groupby(['target', 'batch'], observed=True).size().rename('n_cells').reset_index()
    groups['eligible_control'] = groups['batch'].isin(eligible)
    groups.to_csv(out / f'{name}_all_target_batch_counts.csv', index=False)
    raw_counts = groups.groupby('target', observed=True)['n_cells'].sum()
    rows, targets_rows = [], []
    for t in batch_thresholds:
        valid = groups.loc[groups['eligible_control'] & (groups['n_cells'] >= t)]
        sums = valid.groupby('target', observed=True)['n_cells'].sum()
        n_batches = valid.groupby('target', observed=True).size()
        merged = pd.DataFrame({'target': raw_counts.index, 'n_raw_cells': raw_counts.values})
        merged['n_effective_cells'] = merged['target'].map(sums).fillna(0).astype(int)
        merged['n_eligible_batches'] = merged['target'].map(n_batches).fillna(0).astype(int)
        merged['min_batch_pert_cells'] = t
        for total in total_thresholds:
            n_retained = int((merged.n_effective_cells >= total).sum())
            rows.append({'dataset': name, 'min_controls_per_batch': min_controls,
                         'min_batch_pert_cells': t, 'min_total_pert_cells': total,
                         'n_input_cells': n_cells, 'n_raw_targets': len(raw_counts),
                         'n_batches_total': len(all_batches), 'n_batches_eligible': len(eligible),
                         'n_target_batch_groups_raw': len(groups),
                         'n_target_batch_groups_eligible': len(valid),
                         'n_targets_with_any_eligible_batch': len(sums),
                         'n_targets_retained': n_retained,
                         'n_pert_cells_retained': int(merged.loc[merged.n_effective_cells >= total, 'n_effective_cells'].sum()),
                         'n_targets_raw_ge_total': int((raw_counts >= total).sum())})
        targets_rows.append(merged)
        print(f'  min_batch={t}: >=30 effective cells: {(merged.n_effective_cells >= 30).sum()} targets; '
              f'eligible target-batch groups={len(valid)}', flush=True)
    pd.concat(targets_rows, ignore_index=True).to_csv(out / f'{name}_target_retention_by_threshold.csv', index=False)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--datasets', nargs='+', choices=list(DATA), default=list(DATA))
    p.add_argument('--min-controls', type=int, default=30)
    p.add_argument('--batch-thresholds', nargs='+', type=int, default=[1, 3, 5])
    p.add_argument('--total-thresholds', nargs='+', type=int, default=[20, 30, 50, 100])
    p.add_argument('--outdir', default='results/step11B_2A_threshold_sensitivity')
    p.add_argument('--rebuild-expanded', action='store_true',
                   help='Also call Step11B-2 to generate separate expanded min_batch=1 matrices (no overwrite of main).')
    p.add_argument('--chunk-size', type=int, default=512)
    args = p.parse_args()
    if min([args.min_controls, args.chunk_size] + args.batch_thresholds + args.total_thresholds) < 1:
        p.error('Thresholds and chunk-size must be positive')
    root = args.root.resolve()
    out = root / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.datasets:
        path = root / DATA[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(audit_dataset(name, path, out, sorted(set(args.batch_thresholds)),
                                  args.min_controls, sorted(set(args.total_thresholds))))
    summary = pd.DataFrame(rows).sort_values(['dataset', 'min_batch_pert_cells', 'min_total_pert_cells'])
    summary.to_csv(out / 'threshold_summary.csv', index=False)
    print('\nThreshold summary (total>=30):\n' + summary.loc[summary.min_total_pert_cells == 30,
          ['dataset','min_batch_pert_cells','n_targets_retained','n_pert_cells_retained']].to_string(index=False), flush=True)
    (out / 'audit_settings.json').write_text(json.dumps(vars(args), indent=2, default=str), encoding='utf-8')
    if args.rebuild_expanded:
        script = root / 'scripts/step11B_2_prepare_hepg2_jurkat.py'
        if not script.is_file():
            raise FileNotFoundError(f'Cannot rebuild: {script}')
        if 1 not in args.batch_thresholds:
            raise ValueError('To rebuild expanded, include --batch-thresholds 1')
        # Separate destination protects existing min_batch=5 outputs.
        cmd = [sys.executable, '-u', str(script), '--root', str(root),
               '--outdir', 'data/external_validation/step11B_2A_expanded_minbatch1',
               '--datasets', *args.datasets, '--min-controls', str(args.min_controls),
               '--min-batch-pert-cells', '1', '--min-cells', '30',
               '--chunk-size', str(args.chunk_size)]
        print('\nRebuilding expanded response matrices in a NEW directory:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    print('\nDONE: Step11B-2A. No model predictions were performed.', flush=True)


if __name__ == '__main__':
    main()
