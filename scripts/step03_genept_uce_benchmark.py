#!/usr/bin/env python3
"""Align three representations, run the existing CellPert CLI, verify splits.

Uses the user's installed cellpert implementation; does not reimplement models.
UCE means static input gene tokens, not contextual UCE cell embeddings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd


SOURCES = {
    'GenePT_ada': 'data/processed/embeddings/genept_ada.npz',
    'GenePT_3large': 'data/processed/embeddings/genept_protein_3large.npz',
    'UCE_input_tokens': 'data/processed/benchmark_genept_uce_common/uce_gene_tokens_common.npz',
}


def names(a):
    a = np.asarray(a)
    if a.ndim != 1 or a.dtype.kind not in 'US':
        raise ValueError('Expected one-dimensional string labels')
    a = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])
    if len(set(a)) != len(a) or any(not x or x.strip() != x for x in a):
        raise ValueError('Duplicate/empty/whitespace labels; review before alignment')
    return a


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def save_features(path, genes, x):
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if not (np.array_equal(z['genes'], genes)
                    and np.array_equal(z['embeddings'], x)):
                raise RuntimeError(f'Existing aligned features differ: {path}')
    else:
        tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, genes=genes, embeddings=x)
        tmp.replace(path)
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z['genes'], genes):
            raise RuntimeError('Saved label verification failed')


def canonical_split(manifest, genes):
    """Recognize explicit train/test label or index lists; never infer a split."""
    available = set(genes.tolist())
    found = {'train': [], 'test': [], 'val': []}

    def walk(obj, path=()):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, path + (str(k),))
        elif isinstance(obj, list) and obj and all(
                isinstance(x, (str, int)) and not isinstance(x, bool) for x in obj):
            words = set(re.findall(r'[a-z]+', '/'.join(path).lower()))
            roles = []
            for role, aliases in [('train', {'train', 'training'}),
                                  ('test', {'test', 'testing'}),
                                  ('val', {'val', 'valid', 'validation'})]:
                if words & aliases:
                    roles.append(role)
            if len(roles) != 1:
                return
            if all(isinstance(x, int) for x in obj):
                if any(x < 0 or x >= len(genes) for x in obj):
                    return
                labels = genes[np.asarray(obj, dtype=int)].tolist()
            elif all(isinstance(x, str) for x in obj) and set(obj) <= available:
                labels = obj
            else:
                return
            if len(set(labels)) != len(labels):
                raise RuntimeError(f'Duplicate split labels at {path}')
            found[roles[0]].append(set(labels))

    walk(manifest)
    result = {}
    for role, candidates in found.items():
        if candidates:
            if any(s != candidates[0] for s in candidates):
                raise RuntimeError(f'Conflicting {role} split records')
            result[role] = candidates[0]
    if not {'train', 'test'} <= result.keys():
        raise RuntimeError(
            'Cannot recognize explicit train/test lists in split_manifest.json. '
            'Training output is retained. Send that JSON for schema adaptation; '
            'do not treat this run as a verified fair comparison yet.'
        )
    roles = list(result)
    for i, a in enumerate(roles):
        for b in roles[i+1:]:
            if result[a] & result[b]:
                raise RuntimeError(f'{a}/{b} overlap detected')
    if set().union(*result.values()) != available:
        raise RuntimeError('Saved split does not partition the common perturbation set')
    return {k: sorted(v) for k, v in result.items()}


def stream_run(command, root, log, env):
    print('[COMMAND]', ' '.join(command), flush=True)
    with log.open('a') as f:
        proc = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end='', flush=True)
            f.write(line)
            f.flush()
        code = proc.wait()
    if code:
        raise RuntimeError(f'Training exited {code}; see {log}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='.')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--outdir', default='results/step03_genept_uce_common')
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    if not 0 < args.test_size < 1 or args.threads < 1:
        ap.error('Require 0 < test-size < 1 and threads >= 1')
    if len(set(args.seeds)) != len(args.seeds):
        ap.error('Seeds must be unique')
    root = Path(args.root).resolve()
    output = Path(args.outdir)
    output = output.resolve() if output.is_absolute() else root / output
    response = root / 'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
    with np.load(response, allow_pickle=False) as z:
        genes = names(z['perturbations'])
        response_genes = names(z['genes'])
        delta = z['delta']
        if delta.shape != (len(genes), len(response_genes)):
            raise ValueError('delta axes do not match perturbations and response genes')
        if not np.isfinite(delta).all():
            raise ValueError('delta contains NaN/Inf')
    if len(genes) != 1950:
        raise ValueError(f'Expected 1950 common perturbations, got {len(genes)}')
    print('[RESPONSE]', delta.shape, flush=True)
    del delta
    data_out = output / 'aligned_features'
    data_out.mkdir(parents=True, exist_ok=True)
    feature_paths = {}
    hashes = {'response': digest_file(response)}
    for representation, relative in SOURCES.items():
        source = root / relative
        with np.load(source, allow_pickle=False) as z:
            source_names = names(z['genes'])
            x = z['embeddings']
            if x.ndim != 2 or x.shape[0] != len(source_names):
                raise ValueError(f'Invalid feature dimensions: {source}')
            lookup = {g: i for i, g in enumerate(source_names)}
            missing = [g for g in genes if g not in lookup]
            if missing:
                raise ValueError(f'{representation}: missing exact labels: {missing}')
            x = x[[lookup[g] for g in genes]].astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f'{representation}: NaN/Inf')
        path = data_out / f'{representation}.npz'
        save_features(path, genes, x)
        feature_paths[representation] = path
        hashes[representation] = digest_file(path)
        print('[ALIGNED]', representation, x.shape, flush=True)
        del x
    prep = {'response': str(response), 'sha256': hashes,
            'n_perturbations': len(genes),
            'feature_definition_UCE': 'Static UCE input gene tokens, not contextual cell embeddings'}
    prep_path = output / 'input_manifest.json'
    if prep_path.exists() and json.loads(prep_path.read_text()) != prep:
        raise RuntimeError('Existing input manifest differs; choose a fresh --outdir')
    write_json(prep_path, prep)
    if args.prepare_only:
        print('[PREPARED] All three feature files share the response perturbation order.')
        return

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS']:
        env[key] = str(args.threads)
    source_hashes = {
        str(p.relative_to(root)): digest_file(p)
        for p in sorted((root / 'cellpert').rglob('*.py'))
    }
    if not source_hashes:
        raise RuntimeError('Cannot find cellpert Python sources in project root')
    tables = []
    split_checks = []
    for seed in args.seeds:
        reference_split = None
        for representation, feature_path in feature_paths.items():
            run = output / f'seed_{seed}' / representation
            run.mkdir(parents=True, exist_ok=True)
            expected = {'seed': seed, 'test_size': args.test_size,
                        'representation': representation, 'input_sha256': hashes,
                        'cellpert_source_sha256': source_hashes,
                        'python_executable': sys.executable, 'threads': args.threads}
            config_path = run / 'step03_run_config.json'
            if config_path.exists():
                if json.loads(config_path.read_text()) != expected:
                    raise RuntimeError(f'Existing run config differs: {run}')
            else:
                if any(p.name != 'step03_run_config.json' for p in run.iterdir()):
                    raise RuntimeError(f'Unrecognized existing run directory: {run}')
                write_json(config_path, expected)
            done = run / 'STEP03_VERIFIED.json'
            if not done.exists():
                stream_run([
                    sys.executable, '-m', 'cellpert.cli', 'train',
                    '--response', str(response), '--gene-embeddings', str(feature_path),
                    '--outdir', str(run), '--seed', str(seed),
                    '--test-size', str(args.test_size),
                ], root, run / 'step03_train.log', env)
            split_path = run / 'split_manifest.json'
            metric_path = run / 'metrics.csv'
            if not split_path.exists() or not metric_path.exists():
                raise RuntimeError(f'Missing split_manifest.json or metrics.csv in {run}')
            split = canonical_split(json.loads(split_path.read_text()), genes)
            if reference_split is None:
                reference_split = split
                write_json(output / f'verified_split_seed_{seed}.json', split)
            elif split != reference_split:
                raise RuntimeError(f'Split mismatch for {representation}, seed={seed}')
            metrics = pd.read_csv(metric_path)
            if metrics.empty:
                raise RuntimeError(f'Empty metrics.csv: {metric_path}')
            for col in ['representation', 'seed']:
                if col in metrics:
                    metrics = metrics.rename(columns={col: f'original_{col}'})
            metrics.insert(0, 'representation', representation)
            metrics.insert(1, 'seed', seed)
            tables.append(metrics)
            write_json(done, {'config': expected, 'split': split})
            split_checks.append({'seed': seed, 'representation': representation,
                                 **{f'n_{k}': len(v) for k, v in split.items()}})
            print('[VERIFIED]', representation, 'seed=', seed,
                  {k: len(v) for k, v in split.items()}, flush=True)
    pd.concat(tables, ignore_index=True).to_csv(output / 'metrics_all_runs.csv', index=False)
    pd.DataFrame(split_checks).to_csv(output / 'split_verification.csv', index=False)
    print('\n[COMPLETE] Training complete; saved splits match across representations.')
    print('[METRICS]', output / 'metrics_all_runs.csv')
    print('[SPLITS]', output / 'split_verification.csv')
    print('[FIGURES] Existing CLI figure files, if any:')
    for pattern in ['*.png', '*.svg', '*.pdf']:
        for p in sorted(output.rglob(pattern)):
            print(' ', p)
    print('Publication comparison figures are generated separately in Step04.')


if __name__ == '__main__':
    main()
