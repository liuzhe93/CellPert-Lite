#!/usr/bin/env python3
"""Step08D: paired, frozen-model K562 -> RPE1 transfer analysis.

Inputs: Step08C prediction NPZ files and original K562 response matrix.
No training or model selection. Perturbation-cluster bootstrap is descriptive,
not an independent biological-replicate significance test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODELS = ('ridge_fusion', 'mlp_fusion', 'ccrt')


def npz(path, required):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        missing = set(required) - set(z.files)
        if missing:
            raise ValueError(f'{path}: missing {sorted(missing)}; keys={z.files}')
        return {k: z[k] for k in required}


def unique(ids, label):
    ids = np.asarray(ids).astype(str)
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate {label} IDs')
    return ids


def row_pcc(y, pred):
    a = y.astype(np.float64) - y.mean(axis=1, keepdims=True, dtype=np.float64)
    b = pred.astype(np.float64) - pred.mean(axis=1, keepdims=True, dtype=np.float64)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(np.einsum('ij,ij->i', a, b), den,
                     out=np.full(len(y), np.nan), where=den > 1e-12)


def row_mse(y, pred):
    return np.mean((y.astype(np.float64) - pred.astype(np.float64)) ** 2, axis=1)


def safe_spearman(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 5 or np.std(a[good]) < 1e-12 or np.std(b[good]) < 1e-12:
        return np.nan, int(good.sum())
    return float(spearmanr(a[good], b[good]).statistic), int(good.sum())


def bootstrap_cluster_mean(d, n_boot, seed):
    """Resample perturbations (not seed-perturbation rows); retain all their seeds."""
    per_pert = d.groupby('perturbation', sort=True)['difference'].mean().to_numpy(dtype=float)
    per_pert = per_pert[np.isfinite(per_pert)]
    if not len(per_pert):
        return (np.nan,) * 3
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for j in range(n_boot):
        reps[j] = np.mean(per_pert[rng.integers(0, len(per_pert), len(per_pert))])
    return float(per_pert.mean()), float(np.quantile(reps, .025)), float(np.quantile(reps, .975))


def write_plot(df, out, fmt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    # Per-perturbation differences, not pseudo-replicate seed dots.
    specs = [
        ('mlp_fusion', 'ridge_fusion', 'mse', 'MLP - Ridge MSE'),
        ('ccrt', 'ridge_fusion', 'mse', 'CCRT - Ridge MSE'),
        ('mlp_fusion', 'ridge_fusion', 'pcc', 'MLP - Ridge PCC'),
        ('ccrt', 'ridge_fusion', 'pcc', 'CCRT - Ridge PCC'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (a, b, metric, title) in zip(axes.flat, specs):
        d = df[(df['threshold'] == 5) & df['model'].isin((a, b))]
        piv = d.pivot_table(index=['seed', 'perturbation'], columns='model', values=metric)
        diff = (piv[a] - piv[b]).groupby(level='perturbation').mean().dropna()
        ax.hist(diff.to_numpy(), bins=35, edgecolor='white', linewidth=.4)
        ax.axvline(0, color='black', lw=1, ls='--')
        ax.axvline(diff.mean(), color='tab:red', lw=1.5)
        ax.set_title(title)
        ax.set_xlabel('Mean paired difference per perturbation')
        ax.set_ylabel('Number of perturbations')
    fig.tight_layout()
    for extension in fmt:
        fig.savefig(out / f'Figure3D_paired_model_differences.{extension}', bbox_inches='tight')
    plt.close(fig)

    d = df[(df['threshold'] == 5) & df['model'].isin(MODELS)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    for ax, model in zip(axes, MODELS):
        t = d[d['model'] == model].groupby('perturbation', as_index=False).agg(
            transferability=('k562_rpe1_pcc', 'mean'), model_pcc=('pcc', 'mean'),
            mse_improvement=('mse_improvement_vs_zero', 'mean'))
        ax.scatter(t['transferability'], t['model_pcc'], s=8, alpha=.45, rasterized=True)
        rho, n = safe_spearman(t['transferability'], t['model_pcc'])
        ax.set_title(f'{model}\nSpearman rho={rho:.3f}, n={n}' if np.isfinite(rho)
                     else f'{model}\nSpearman unavailable')
        ax.set_xlabel('Observed K562-RPE1 response PCC')
        ax.axhline(0, lw=.7, color='grey', ls=':')
        ax.axvline(0, lw=.7, color='grey', ls=':')
    axes[0].set_ylabel('RPE1 prediction PCC')
    fig.tight_layout()
    for extension in fmt:
        fig.savefig(out / f'Figure3D_transferability_scatter.{extension}', bbox_inches='tight')
    plt.close(fig)

    # Matched-set sensitivity: same perturbations per seed across all thresholds.
    fig, ax = plt.subplots(figsize=(7, 4))
    t = pd.read_csv(out / 'matched_threshold_summary.csv')
    for model in MODELS:
        q = t[t['model'] == model].sort_values('threshold')
        ax.plot(q['threshold'], q['mean_pcc'], marker='o', label=model)
    ax.set_xticks([3, 5, 10], ['batch3', 'batch5', 'batch10'])
    ax.set_xlabel('Minimum perturbed cells per batch')
    ax.set_ylabel('Mean per-perturbation PCC')
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in fmt:
        fig.savefig(out / f'Figure3D_matched_threshold_sensitivity.{extension}', bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='.')
    ap.add_argument('--step08c', default='results/step08C_k562_to_rpe1_transfer')
    ap.add_argument('--outdir', default='results/step08D_transfer_analysis')
    ap.add_argument('--thresholds', nargs='+', type=int, default=[3, 5, 10])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()
    if args.bootstrap < 100 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Bootstrap >=100 and unique seeds required')
    root = Path(args.root).resolve()
    src = root / args.step08c
    out = root / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    kpath = root / 'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
    k = npz(kpath, ('perturbations', 'genes', 'delta'))
    kp = unique(k['perturbations'], 'K562 perturbation')
    kg = unique(k['genes'], 'K562 response gene')
    ky = k['delta'].astype(np.float32)
    if ky.shape != (len(kp), len(kg)) or not np.isfinite(ky).all():
        raise ValueError('Invalid K562 response')
    pidx = {p: i for i, p in enumerate(kp)}
    gidx = {g: i for i, g in enumerate(kg)}
    records = []
    sets = {}
    for threshold in args.thresholds:
        for seed in args.seeds:
            f = src / f'batch{threshold}' / f'seed_{seed}' / 'predictions.npz'
            req = ('perturbations', 'genes', 'y_true') + tuple('pred_' + m for m in MODELS)
            z = npz(f, req)
            perts = unique(z['perturbations'], f'{f} perturbation')
            genes = unique(z['genes'], f'{f} gene')
            if any(p not in pidx for p in perts) or any(g not in gidx for g in genes):
                raise ValueError(f'{f}: IDs not contained in K562 matrix')
            y = z['y_true'].astype(np.float32)
            if y.shape != (len(perts), len(genes)) or not np.isfinite(y).all():
                raise ValueError(f'{f}: invalid y_true')
            for m in MODELS:
                if z['pred_' + m].shape != y.shape or not np.isfinite(z['pred_' + m]).all():
                    raise ValueError(f'{f}: invalid {m} prediction')
            kk = ky[np.ix_([pidx[p] for p in perts], [gidx[g] for g in genes])]
            cross_pcc = row_pcc(kk, y)
            cross_mse = row_mse(kk, y)
            zero_mse = row_mse(y, np.zeros_like(y))
            # Check the per-seed perturbation subset matches the fixed Step05 outer train.
            split = root / 'results/step05_tuned_genept_uce/inner_splits' / f'seed_{seed}.json'
            m = json.loads(split.read_text())
            if not set(perts).issubset(set(m['outer_train'])):
                raise ValueError(f'{f}: includes K562 outer-test perturbations')
            sets[(threshold, seed)] = set(perts)
            for model in MODELS:
                pred = z['pred_' + model].astype(np.float32)
                pp = row_pcc(y, pred)
                mm = row_mse(y, pred)
                for i, perturbation in enumerate(perts):
                    records.append({'threshold': threshold, 'seed': seed, 'perturbation': perturbation,
                                    'model': model, 'pcc': pp[i], 'mse': mm[i],
                                    'zero_mse': zero_mse[i],
                                    'mse_improvement_vs_zero': zero_mse[i] - mm[i],
                                    'relative_mse_improvement': (zero_mse[i]-mm[i])/zero_mse[i]
                                        if zero_mse[i] > 1e-12 else np.nan,
                                    'k562_rpe1_pcc': cross_pcc[i],
                                    'k562_rpe1_mse': cross_mse[i],
                                    'k562_response_norm': float(np.sqrt(np.mean(kk[i].astype(float)**2))),
                                    'rpe1_response_norm': float(np.sqrt(zero_mse[i]))})
            print(f'[LOAD] batch{threshold} seed={seed} n={len(perts)}', flush=True)
    df = pd.DataFrame.from_records(records)
    df.to_csv(out / 'per_perturbation_transfer.csv', index=False)

    # Paired differences within identical seed and perturbation.
    comparisons = [('mlp_fusion', 'ridge_fusion'), ('ccrt', 'ridge_fusion'),
                   ('mlp_fusion', 'ccrt')]
    paired = []
    for th in args.thresholds:
        d = df[df.threshold == th]
        for metric in ('mse', 'pcc', 'mse_improvement_vs_zero'):
            piv = d.pivot(index=['seed', 'perturbation'], columns='model', values=metric)
            for a, b in comparisons:
                tmp = (piv[a] - piv[b]).dropna().rename('difference').reset_index()
                est, low, high = bootstrap_cluster_mean(tmp, args.bootstrap,
                                                        args.random_state + th)
                per_seed = tmp.groupby('seed')['difference'].mean()
                paired.append({'threshold': th, 'metric': metric, 'model_a': a,
                               'model_b': b, 'difference_definition': 'A minus B',
                               'n_seed_perturbation_pairs': len(tmp),
                               'n_unique_perturbations': tmp.perturbation.nunique(),
                               'mean_of_perturbation_means': est,
                               'bootstrap_ci95_low': low, 'bootstrap_ci95_high': high,
                               'mean_seed_difference': per_seed.mean(),
                               'sd_seed_difference': per_seed.std(ddof=1),
                               'seed_differences_json': json.dumps(per_seed.to_dict())})
    pd.DataFrame(paired).to_csv(out / 'paired_model_comparisons.csv', index=False)

    # Biological interpretation: observed response agreement vs model performance.
    association = []
    for (th, model), d in df.groupby(['threshold', 'model']):
        q = d.groupby('perturbation', as_index=False).agg(
            transferability=('k562_rpe1_pcc', 'mean'), pcc=('pcc', 'mean'),
            mse_improvement=('mse_improvement_vs_zero', 'mean'),
            relative_improvement=('relative_mse_improvement', 'mean'))
        for outcome in ('pcc', 'mse_improvement', 'relative_improvement'):
            rho, n = safe_spearman(q['transferability'], q[outcome])
            association.append({'threshold': th, 'model': model, 'outcome': outcome,
                                'spearman_rho': rho, 'n_unique_perturbations': n})
    pd.DataFrame(association).to_csv(out / 'transferability_associations.csv', index=False)

    # Quartiles based on biological agreement (not model predictions).
    qrows = []
    for (th, model), d in df.groupby(['threshold', 'model']):
        q = d.groupby('perturbation', as_index=False).agg(
            transferability=('k562_rpe1_pcc', 'mean'), pcc=('pcc', 'mean'),
            improvement=('mse_improvement_vs_zero', 'mean'))
        q = q[np.isfinite(q.transferability)].copy()
        if len(q) < 8:
            continue
        q['quartile'] = pd.qcut(q.transferability.rank(method='first'), 4,
                               labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'])
        for quartile, subset in q.groupby('quartile', observed=True):
            qrows.append({'threshold': th, 'model': model, 'quartile': str(quartile),
                          'n_perturbations': len(subset),
                          'mean_k562_rpe1_pcc': subset.transferability.mean(),
                          'mean_prediction_pcc': subset.pcc.mean(),
                          'mean_mse_improvement': subset.improvement.mean()})
    pd.DataFrame(qrows).to_csv(out / 'transferability_quartiles.csv', index=False)

    # Sensitivity: same seed-specific perturbations across ALL thresholds.
    common = {seed: set.intersection(*(sets[(th, seed)] for th in args.thresholds))
              for seed in args.seeds}
    matched = df[df.apply(lambda r: r['perturbation'] in common[r['seed']], axis=1)].copy()
    matched.to_csv(out / 'matched_threshold_per_perturbation.csv', index=False)
    matched_summary = matched.groupby(['threshold', 'model'], as_index=False).agg(
        n_seed_perturbation_pairs=('pcc', 'size'),
        n_unique_perturbations=('perturbation', 'nunique'),
        mean_pcc=('pcc', 'mean'), mean_mse=('mse', 'mean'),
        mean_mse_improvement=('mse_improvement_vs_zero', 'mean'),
        mean_cross_cell_pcc=('k562_rpe1_pcc', 'mean'))
    matched_summary.to_csv(out / 'matched_threshold_summary.csv', index=False)

    # Aggregate biological agreement independent of the models, once per perturbation.
    bio = df[df.model == MODELS[0]].groupby(['threshold', 'perturbation'], as_index=False).agg(
        n_seeds=('seed', 'nunique'), k562_rpe1_pcc=('k562_rpe1_pcc', 'mean'),
        k562_rpe1_mse=('k562_rpe1_mse', 'mean'),
        k562_response_norm=('k562_response_norm', 'mean'),
        rpe1_response_norm=('rpe1_response_norm', 'mean'))
    bio.to_csv(out / 'observed_cross_cell_response_agreement.csv', index=False)

    if not args.no_plots:
        write_plot(df, out, ('pdf', 'svg'))
    summary = {
        'purpose': 'Post-hoc descriptive paired analysis of frozen K562-to-RPE1 predictions',
        'seeds': args.seeds, 'thresholds': args.thresholds,
        'matched_perturbations_per_seed': {str(k): len(v) for k, v in common.items()},
        'bootstrap_unit': 'perturbation; average across available seeds before resampling',
        'bootstrap_iterations': args.bootstrap,
        'caveats': [
            'K562 and RPE1 response estimators differ: global controls vs batch-matched controls.',
            'Five seeds are overlapping train splits, not independent biological replicates.',
            'Bootstrap intervals are descriptive perturbation-sampling uncertainty, not experimental replicate uncertainty.',
            'Association of observed cross-cell-line agreement with model performance is descriptive, not causal.',
            'batch3/5/10 differ in response estimates and perturbation coverage; matched comparison fixes perturbation IDs.'
        ]
    }
    (out / 'analysis_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('[DONE]', out, flush=True)
    print('[MATCHED SET PER SEED]', summary['matched_perturbations_per_seed'], flush=True)


if __name__ == '__main__':
    main()
