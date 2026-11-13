#!/usr/bin/env python3
"""CellPert-Lite v1.1: paired, nested holdout tuning without test-based selection."""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning

from cellpert.data import load_response_npz, load_gene_embeddings
from cellpert.benchmark import align_embeddings
from cellpert.metrics import response_metrics


def make_model(kind, config, seed):
    if kind == 'ridge_gene':
        est = Ridge(alpha=float(config['alpha']))
    else:
        est = MLPRegressor(hidden_layer_sizes=tuple(config['hidden']),
                           alpha=float(config['alpha']), max_iter=250,
                           learning_rate_init=0.001, early_stopping=False,
                           random_state=seed)
    return make_pipeline(StandardScaler(), est)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--response', default='data/processed/k562/response_matrix_genept_common.npz')
    p.add_argument('--large-embedding', default='data/processed/embeddings/genept_protein_3large.npz')
    p.add_argument('--ada-embedding', default='data/processed/embeddings/genept_ada.npz')
    p.add_argument('--outdir', default='results/k562_tuned_v1')
    p.add_argument('--seeds', type=int, nargs='+', default=[42,123,2025,3407,2026])
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--val-size', type=float, default=0.2)
    p.add_argument('--baseline-dir', default='results/k562_multiseed')
    args = p.parse_args()
    if not 0 < args.test_size < 1 or not 0 < args.val_size < 1:
        p.error('split fractions must be between 0 and 1')
    if len(set(args.seeds)) != len(args.seeds):
        p.error('seeds must be unique')
    out = Path(args.outdir)
    if out.exists() and any(out.iterdir()):
        p.error(f'output directory is nonempty: {out}')
    response = load_response_npz(args.response)
    perts = np.asarray(response['perturbations']).astype(str)
    Y = np.asarray(response['delta'], dtype=np.float32)
    if Y.ndim != 2 or Y.shape[0] != len(perts) or not np.isfinite(Y).all():
        raise ValueError('invalid response matrix')
    if len(set(perts)) != len(perts):
        raise ValueError('duplicate perturbation IDs')
    embeddings = {
        '3large': align_embeddings(perts, load_gene_embeddings(args.large_embedding)),
        'ada': align_embeddings(perts, load_gene_embeddings(args.ada_embedding)),
    }
    configs = {
        'ridge_gene': [{'alpha': a} for a in (1.0, 100.0, 10000.0)],
        'mlp_gene': [{'hidden': list(h), 'alpha': a}
                     for h, a in [((128,64), 1e-3), ((128,64), 0.1), ((64,), 0.01)]],
    }
    out.mkdir(parents=True)
    results, searches = [], []
    for seed in args.seeds:
        # Reuse the exact baseline outer split to support paired comparison.
        manifest_path = Path(args.baseline_dir) / f'seed_{seed}' / 'genept_3large' / 'split_manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'baseline split manifest missing: {manifest_path}')
        manifest = json.loads(manifest_path.read_text())
        if manifest['seed'] != seed:
            raise ValueError('seed mismatch in manifest')
        lookup = {name: i for i, name in enumerate(perts)}
        train_names = manifest['train_perturbations']
        test_names = manifest['test_perturbations']
        if set(train_names) & set(test_names) or set(train_names) | set(test_names) != set(perts):
            raise ValueError('manifest does not partition current response perturbations')
        outer_train = np.asarray([lookup[g] for g in train_names])
        outer_test = np.asarray([lookup[g] for g in test_names])
        fit_idx, val_idx = train_test_split(outer_train, test_size=args.val_size,
                                            random_state=seed+100000, shuffle=True)
        seed_dir = out / f'seed_{seed}'
        seed_dir.mkdir()
        (seed_dir/'split_manifest.json').write_text(json.dumps({
            'seed': seed, 'fit': perts[fit_idx].tolist(),
            'validation': perts[val_idx].tolist(), 'outer_test': perts[outer_test].tolist()
        }, indent=2))
        for emb_name, X in embeddings.items():
            for kind, candidates in configs.items():
                best_score, best_cfg = float('inf'), None
                for cfg in candidates:
                    model = make_model(kind, cfg, seed)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', ConvergenceWarning)
                        model.fit(X[fit_idx], Y[fit_idx])
                    pred_val = model.predict(X[val_idx])
                    score = float(np.mean((pred_val-Y[val_idx])**2))
                    searches.append({'seed': seed, 'embedding': emb_name, 'model': kind,
                                     'config': json.dumps(cfg), 'validation_mse': score})
                    if np.isfinite(score) and score < best_score:
                        best_score, best_cfg = score, cfg
                if best_cfg is None:
                    raise RuntimeError('no valid candidate')
                final = make_model(kind, best_cfg, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    final.fit(X[outer_train], Y[outer_train])
                pred = final.predict(X[outer_test])
                scores = response_metrics(Y[outer_test], pred)
                results.append({'seed': seed, 'embedding': emb_name, 'model': kind,
                                'best_config': json.dumps(best_cfg),
                                'validation_mse': best_score, **scores})
                print(seed, emb_name, kind, best_cfg, 'test MSE:', scores['mse'], flush=True)
    pd.DataFrame(searches).to_csv(out/'validation_search.csv', index=False)
    df = pd.DataFrame(results)
    df.to_csv(out/'test_metrics.csv', index=False)
    metrics = [c for c in df.select_dtypes(include='number').columns if c != 'seed']
    df.groupby(['embedding','model'])[metrics].agg(['mean','std','count']).to_csv(out/'test_summary.csv')
    (out/'protocol.json').write_text(json.dumps(vars(args), indent=2))
    print('Saved to:', out)

if __name__ == '__main__':
    main()

