#!/usr/bin/env python3
"""Step06A: paired K562 GenePT-3large + UCE_input_tokens fusion baselines.

Reuse Step05's outer and inner perturbation-gene splits. Never use outer-test
responses for training or hyperparameter selection. Exploratory holdout results.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def load_features(path, perts):
    with np.load(path, allow_pickle=False) as z:
        ids = z['genes'].astype(str)
        x = z['embeddings'].astype(np.float32)
    if not np.array_equal(ids, perts):
        raise ValueError(f'Feature IDs/order mismatch: {path}')
    if x.ndim != 2 or x.shape[0] != len(perts) or not np.isfinite(x).all():
        raise ValueError(f'Invalid feature array: {path}')
    return x


def make_preprocessor(d1, d2, mode):
    # Each scaler is fit only on the training fold/inner-train/outer-train.
    # 'balanced': each block has similar expected total squared norm.
    if mode == 'balanced':
        weights = {'genept': 1.0 / np.sqrt(d1), 'uce': 1.0 / np.sqrt(d2)}
    elif mode == 'raw':
        weights = {'genept': 1.0, 'uce': 1.0}
    else:
        raise ValueError(mode)
    return ColumnTransformer(
        [('genept', StandardScaler(), slice(0, d1)),
         ('uce', StandardScaler(), slice(d1, d1 + d2))],
        transformer_weights=weights, remainder='drop', sparse_threshold=0,
    )


def train_mlp(x, y, seed, alpha, max_epochs, batch_size, lr,
              validation=None, patience=30, min_delta=1e-7):
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
        alpha=alpha, learning_rate_init=lr, batch_size=batch_size,
        random_state=seed, max_iter=1, shuffle=True, early_stopping=False,
    )
    best_loss = np.inf
    significant_best = np.inf
    best_epoch = None
    wait = 0
    trace = []
    for epoch in range(1, max_epochs + 1):
        model.partial_fit(x, y)
        if not np.isfinite(model.loss_):
            raise RuntimeError('Non-finite MLP loss')
        row = {'epoch': epoch, 'train_loss': float(model.loss_)}
        if validation is not None:
            xv, yv = validation
            val = float(mean_squared_error(yv, model.predict(xv)))
            if not np.isfinite(val):
                raise RuntimeError('Non-finite validation MSE')
            row['validation_mse'] = val
            if val < best_loss:
                best_loss, best_epoch = val, epoch
            if significant_best - val > min_delta:
                significant_best, wait = val, 0
            else:
                wait += 1
        trace.append(row)
        if validation is not None and wait >= patience:
            break
    return model, best_epoch, best_loss, trace


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--outdir', default='results/step06A_fusion_baselines')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--ridge-alphas', nargs='+', type=float,
                   default=[0.01, 0.1, 1, 10, 100, 1000, 10000])
    p.add_argument('--mlp-alphas', nargs='+', type=float,
                   default=[0.0001, 0.01, 1.0])
    p.add_argument('--fusion-modes', nargs='+', choices=['balanced', 'raw'],
                   default=['balanced', 'raw'])
    p.add_argument('--max-epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    out = root / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(root))
    from cellpert.metrics import response_metrics
    import cellpert.metrics as metric_module

    response = root / 'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
    feature_dir = root / 'results/step03_genept_uce_common/aligned_features'
    feature_paths = [feature_dir / 'GenePT_3large.npz', feature_dir / 'UCE_input_tokens.npz']
    split_dir = root / 'results/step05_tuned_genept_uce/inner_splits'
    with np.load(response, allow_pickle=False) as z:
        perts = z['perturbations'].astype(str)
        genes = z['genes'].astype(str)
        y = z['delta'].astype(np.float32)
    if y.shape != (len(perts), len(genes)) or not np.isfinite(y).all():
        raise ValueError('Invalid response matrix')
    if len(set(perts)) != len(perts) or len(set(genes)) != len(genes):
        raise ValueError('Duplicate perturbation or response gene IDs')
    x1, x2 = [load_features(p, perts) for p in feature_paths]
    d1, d2 = x1.shape[1], x2.shape[1]
    x = np.concatenate([x1, x2], axis=1)
    del x1, x2
    lookup = {p: i for i, p in enumerate(perts)}
    config = {
        'stage': 'Step06A exploratory fusion', 'seeds': args.seeds,
        'ridge_alphas': args.ridge_alphas, 'mlp_alphas': args.mlp_alphas,
        'fusion_modes': args.fusion_modes, 'max_epochs': args.max_epochs,
        'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
        'patience': args.patience, 'threads': args.threads,
        'feature_dims': {'GenePT_3large': d1, 'UCE_input_tokens': d2},
        'input_sha256': {str(p.relative_to(root)): digest(p)
                         for p in [response] + feature_paths},
        'split_sha256': {str(seed): digest(split_dir / f'seed_{seed}.json')
                         for seed in args.seeds},
        'script_sha256': digest(Path(__file__).resolve()),
        'metrics_sha256': digest(Path(metric_module.__file__).resolve()),
        'exploratory_refinement': True,
        'warning': 'Previously inspected outer test; not confirmatory independent validation.',
    }
    config_path = out / 'experiment_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError('Existing output has different config/inputs/code. Use a NEW --outdir.')
    else:
        write_json(config_path, config)
    print('[DATA]', len(perts), 'perturbations,', len(genes), 'response genes;',
          'feature dims', d1, '+', d2, flush=True)
    if args.dry_run:
        for seed in args.seeds:
            manifest = json.loads((split_dir / f'seed_{seed}.json').read_text())
            tr, te = manifest['outer_train'], manifest['outer_test']
            assert len(tr) == len(set(tr)) and len(te) == len(set(te))
            assert not (set(tr) & set(te)) and set(tr) | set(te) == set(perts)
            assert set(manifest['mlp_inner_train']) | set(manifest['mlp_inner_validation']) == set(tr)
            assert not (set(manifest['mlp_inner_train']) & set(manifest['mlp_inner_validation']))
            for fold in manifest['ridge_folds']:
                assert set(fold['train']) | set(fold['validation']) == set(tr)
                assert not (set(fold['train']) & set(fold['validation']))
            print('[DRY RUN PASS]', seed, 'train', len(tr), 'test', len(te))
        return

    all_metrics = []
    with threadpool_limits(limits=args.threads):
        for seed in args.seeds:
            seed_dir = out / f'seed_{seed}'
            seed_dir.mkdir(parents=True, exist_ok=True)
            marker = seed_dir / 'STEP06A_COMPLETE.json'
            if marker.exists():
                needed = ['metrics.csv', 'predictions.npz', 'ridge_model.joblib',
                          'mlp_model.joblib', 'selection_summary.json']
                if not all((seed_dir / n).exists() for n in needed):
                    raise RuntimeError(f'Incomplete artifacts for {seed}')
                print('[SKIP COMPLETE]', seed, flush=True)
                all_metrics.append(pd.read_csv(seed_dir / 'metrics.csv'))
                continue
            manifest = json.loads((split_dir / f'seed_{seed}.json').read_text())
            tr, te = manifest['outer_train'], manifest['outer_test']
            if set(tr) & set(te) or set(tr) | set(te) != set(perts):
                raise ValueError(f'Invalid outer split for seed {seed}')
            ti = np.array([lookup[g] for g in tr], dtype=int)
            ei = np.array([lookup[g] for g in te], dtype=int)
            ytr = y[ti]
            local = {g: i for i, g in enumerate(tr)}
            folds = [([local[g] for g in f['train']],
                      [local[g] for g in f['validation']])
                     for f in manifest['ridge_folds']]
            it = np.array([local[g] for g in manifest['mlp_inner_train']], dtype=int)
            iv = np.array([local[g] for g in manifest['mlp_inner_validation']], dtype=int)
            assert set(it).isdisjoint(set(iv)) and len(it) + len(iv) == len(ti)
            print('\n[SEED]', seed, 'outer train', len(ti), 'test', len(ei), flush=True)
            tuning = seed_dir / 'tuning'
            tuning.mkdir(exist_ok=True)
            # Ridge: all preprocessing is refit separately within each inner fold.
            ridge_rows = []
            for candidate in ParameterGrid({'mode': args.fusion_modes,
                                            'alpha': args.ridge_alphas}):
                for fi, (fit_idx, val_idx) in enumerate(folds):
                    model = Pipeline([
                        ('preprocess', make_preprocessor(d1, d2, candidate['mode'])),
                        ('ridge', Ridge(alpha=candidate['alpha'], solver='auto')),
                    ])
                    model.fit(x[ti[fit_idx]], ytr[fit_idx])
                    pred = model.predict(x[ti[val_idx]])
                    ridge_rows.append({**candidate, 'fold': fi,
                                       'validation_mse': float(mean_squared_error(ytr[val_idx], pred))})
            rdf = pd.DataFrame(ridge_rows)
            rdf.to_csv(tuning / 'ridge_cv_results.csv', index=False)
            ranked = rdf.groupby(['mode', 'alpha'], as_index=False)['validation_mse'].mean()
            ranked = ranked.sort_values(['validation_mse', 'mode', 'alpha'])
            rbest = ranked.iloc[0]
            rmode, ralpha = str(rbest['mode']), float(rbest['alpha'])
            print('[RIDGE SELECT]', rmode, ralpha, float(rbest['validation_mse']), flush=True)
            # MLP: choose block weighting, regularization, and epoch on inner validation only.
            mlp_rows = []
            for mode in args.fusion_modes:
                prep = make_preprocessor(d1, d2, mode)
                xi = prep.fit_transform(x[ti[it]])
                xv = prep.transform(x[ti[iv]])
                for alpha in args.mlp_alphas:
                    _, best_epoch, best_loss, trace = train_mlp(
                        xi, ytr[it], seed, alpha, args.max_epochs,
                        args.batch_size, args.learning_rate,
                        validation=(xv, ytr[iv]), patience=args.patience)
                    if best_epoch is None:
                        raise RuntimeError('MLP did not select an epoch')
                    mlp_rows.append({'mode': mode, 'alpha': alpha,
                                     'best_epoch': best_epoch, 'validation_mse': best_loss,
                                     'epochs_run': len(trace),
                                     'hit_epoch_limit': len(trace) == args.max_epochs})
                    pd.DataFrame(trace).to_csv(tuning / f'mlp_{mode}_a{alpha:g}_curve.csv', index=False)
                    print('[MLP VALIDATE]', mode, alpha, best_epoch, best_loss, flush=True)
            mdf = pd.DataFrame(mlp_rows).sort_values(['validation_mse', 'mode', 'alpha'])
            mdf.to_csv(tuning / 'mlp_validation_results.csv', index=False)
            mbest = mdf.iloc[0]
            mmode, malpha, mepoch = str(mbest['mode']), float(mbest['alpha']), int(mbest['best_epoch'])
            print('[MLP SELECT]', mmode, malpha, mepoch, float(mbest['validation_mse']), flush=True)
            selection = {
                'seed': seed,
                'ridge': {'mode': rmode, 'alpha': ralpha,
                          'inner_cv_mse': float(rbest['validation_mse']),
                          'alpha_at_grid_boundary': ralpha in [min(args.ridge_alphas), max(args.ridge_alphas)]},
                'mlp': {'mode': mmode, 'alpha': malpha, 'epochs': mepoch,
                        'inner_validation_mse': float(mbest['validation_mse']),
                        'alpha_at_grid_boundary': malpha in [min(args.mlp_alphas), max(args.mlp_alphas)],
                        'hit_epoch_limit': bool(mbest['hit_epoch_limit'])},
                'outer_test_used_for_selection': False,
            }
            write_json(seed_dir / 'selection_summary.json', selection)
            # Refit both models on all outer-training perturbations.
            start = time.perf_counter()
            ridge = Pipeline([
                ('preprocess', make_preprocessor(d1, d2, rmode)),
                ('ridge', Ridge(alpha=ralpha, solver='auto')),
            ])
            ridge.fit(x[ti], ytr)
            ridge_seconds = time.perf_counter() - start
            start = time.perf_counter()
            prep = make_preprocessor(d1, d2, mmode)
            xtr_scaled = prep.fit_transform(x[ti])
            mlp, _, _, _ = train_mlp(xtr_scaled, ytr, seed, malpha, mepoch,
                                     args.batch_size, args.learning_rate)
            mlp_seconds = time.perf_counter() - start
            joblib.dump({'model': ridge, 'seed': seed, 'selection': selection,
                         'train_perturbations': tr, 'response_genes': genes.tolist()},
                        seed_dir / 'ridge_model.joblib')
            joblib.dump({'preprocessor': prep, 'model': mlp, 'seed': seed,
                         'selection': selection, 'train_perturbations': tr,
                         'response_genes': genes.tolist()}, seed_dir / 'mlp_model.joblib')
            write_json(seed_dir / 'refit_timing.json', {
                'ridge_refit_seconds': ridge_seconds, 'mlp_refit_seconds': mlp_seconds,
                'note': 'Excludes hyperparameter tuning; not end-to-end runtime.'})
            # OUTER TEST LABELS ARE FIRST USED BELOW.
            p_ridge = ridge.predict(x[ei]).astype(np.float32)
            p_mlp = mlp.predict(prep.transform(x[ei])).astype(np.float32)
            if p_ridge.shape != y[ei].shape or p_mlp.shape != y[ei].shape:
                raise ValueError('Prediction shape mismatch')
            if not np.isfinite(p_ridge).all() or not np.isfinite(p_mlp).all():
                raise ValueError('Nonfinite prediction')
            metrics = []
            for name, pred in [('ridge_fusion', p_ridge), ('mlp_fusion', p_mlp)]:
                metrics.append({'seed': seed, 'representation': 'GenePT_3large+UCE_input_tokens',
                                'model': name, **response_metrics(y[ei], pred)})
            df = pd.DataFrame(metrics)
            df.to_csv(seed_dir / 'metrics.csv', index=False)
            np.savez_compressed(seed_dir / 'predictions.npz',
                                test_perturbations=perts[ei], genes=genes,
                                y_true=y[ei], pred_ridge_fusion=p_ridge,
                                pred_mlp_fusion=p_mlp)
            write_json(marker, {'seed': seed, 'n_train': len(ti), 'n_test': len(ei),
                                'outer_split_sha256': digest(split_dir / f'seed_{seed}.json'),
                                'models': ['ridge_fusion', 'mlp_fusion']})
            all_metrics.append(df)
            print('[COMPLETE SEED]', seed, flush=True)
    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(out / 'metrics_all_runs.csv', index=False)
    numeric = [c for c in combined.select_dtypes(include=np.number) if c != 'seed']
    combined.groupby(['representation', 'model'])[numeric].agg(['mean', 'std', 'count']).to_csv(
        out / 'metrics_mean_sd.csv')
    print('[STEP06A COMPLETE]', out, flush=True)
    print('WARNING: exploratory outer test, not a new untouched confirmatory cohort.')


if __name__ == '__main__':
    main()
