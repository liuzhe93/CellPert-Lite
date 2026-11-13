#!/usr/bin/env python3
"""Step06B: CCRT exploratory gated dual-embedding residual model.

K562 perturbation-held-out prediction. Reuses Step05 inner/outer splits.
The UCE input tokens are gene-level embeddings, NOT cell-context embeddings.
Outer test labels are accessed only after model selection/refit. Existing outer
holdouts have been inspected in prior stages: results remain exploratory.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def load_features(path, perts):
    with np.load(path, allow_pickle=False) as z:
        ids = z['genes'].astype(str)
        x = z['embeddings'].astype(np.float32)
    if not np.array_equal(ids, perts) or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError(f'Feature mismatch or invalid values: {path}')
    return x


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_network(d1, d2, n_out, width, dropout):
    import torch
    from torch import nn

    class GatedResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.branch_a = nn.Sequential(nn.Linear(d1, width), nn.LayerNorm(width), nn.GELU())
            self.branch_b = nn.Sequential(nn.Linear(d2, width), nn.LayerNorm(width), nn.GELU())
            self.gate = nn.Sequential(nn.Linear(width * 2, width), nn.Sigmoid())
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, width),
                                      nn.GELU(), nn.Dropout(dropout), nn.Linear(width, n_out))
            # Begin near the Ridge prediction, not a random large residual.
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

        def forward(self, a, b):
            za = self.branch_a(a)
            zb = self.branch_b(b)
            g = self.gate(torch.cat((za, zb), dim=-1))
            fused = g * za + (1 - g) * zb
            return self.head(fused), g

    return GatedResidual()


def transform_blocks(x1, x2, fit_idx, eval_idx):
    from sklearn.preprocessing import StandardScaler
    s1 = StandardScaler().fit(x1[fit_idx])
    s2 = StandardScaler().fit(x2[fit_idx])
    a_fit = s1.transform(x1[fit_idx]).astype(np.float32) / np.float32(np.sqrt(x1.shape[1]))
    b_fit = s2.transform(x2[fit_idx]).astype(np.float32) / np.float32(np.sqrt(x2.shape[1]))
    a_eval = s1.transform(x1[eval_idx]).astype(np.float32) / np.float32(np.sqrt(x1.shape[1]))
    b_eval = s2.transform(x2[eval_idx]).astype(np.float32) / np.float32(np.sqrt(x2.shape[1]))
    return (a_fit, b_fit), (a_eval, b_eval), (s1, s2)


def predict_residual(model, a, b, device, batch=128):
    import torch
    model.eval()
    outputs, gates = [], []
    with torch.no_grad():
        for start in range(0, len(a), batch):
            aa = torch.as_tensor(a[start:start + batch], dtype=torch.float32, device=device)
            bb = torch.as_tensor(b[start:start + batch], dtype=torch.float32, device=device)
            pred, gate = model(aa, bb)
            outputs.append(pred.cpu().numpy())
            gates.append(gate.mean(dim=1).cpu().numpy())
    return np.concatenate(outputs), np.concatenate(gates)


def fit_residual(a, b, residual, seed, device, width, dropout, weight_decay,
                 epochs, batch_size, lr, validation=None, patience=25, min_delta=1e-7,
                 residual_penalty=0.0):
    import torch
    seed_everything(seed)
    model = make_network(a.shape[1], b.shape[1], residual.shape[1], width, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ta = torch.as_tensor(a, dtype=torch.float32)
    tb = torch.as_tensor(b, dtype=torch.float32)
    ty = torch.as_tensor(residual, dtype=torch.float32)
    best_mse, best_epoch, wait, significant_best = float('inf'), None, 0, float('inf')
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(a), generator=torch.Generator().manual_seed(seed + epoch))
        losses = []
        for ids in order.split(batch_size):
            xa, xb, yr = ta[ids].to(device), tb[ids].to(device), ty[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            pr, _ = model(xa, xb)
            loss = ((pr - yr) ** 2).mean() + residual_penalty * pr.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite CCRT loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        item = {'epoch': epoch, 'training_loss': float(np.mean(losses))}
        if validation is not None:
            av, bv, base_val, y_val = validation
            pr, _ = predict_residual(model, av, bv, device)
            val_mse = float(np.mean((base_val.astype(np.float64) + pr - y_val) ** 2))
            item['validation_mse'] = val_mse
            if val_mse < best_mse:
                best_mse, best_epoch = val_mse, epoch
            if significant_best - val_mse > min_delta:
                significant_best, wait = val_mse, 0
            else:
                wait += 1
        trace.append(item)
        if validation is not None and wait >= patience:
            break
    return model, best_epoch, best_mse, trace


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--outdir', default='results/step06B_ccrt')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--widths', nargs='+', type=int, default=[128])
    p.add_argument('--weight-decays', nargs='+', type=float, default=[0.001, 0.01])
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--residual-penalty', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    args = args_parser()
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError('Seeds must be unique')
    if any(v <= 0 for v in args.widths) or any(v < 0 for v in args.weight_decays):
        raise ValueError('Invalid hyperparameter grid')
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from cellpert.metrics import response_metrics
    import cellpert.metrics as metrics_module
    import torch

    torch.set_num_threads(args.threads)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    rp = root / 'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
    fd = root / 'results/step03_genept_uce_common/aligned_features'
    f1, f2 = fd / 'GenePT_3large.npz', fd / 'UCE_input_tokens.npz'
    split_dir = root / 'results/step05_tuned_genept_uce/inner_splits'
    selection_dir = root / 'results/step05_tuned_genept_uce'
    with np.load(rp, allow_pickle=False) as z:
        perts, genes = z['perturbations'].astype(str), z['genes'].astype(str)
        y = z['delta'].astype(np.float32)
    if y.shape != (len(perts), len(genes)) or not np.isfinite(y).all():
        raise ValueError('Response matrix invalid')
    if len(set(perts)) != len(perts):
        raise ValueError('Duplicate perturbation IDs')
    x1, x2 = load_features(f1, perts), load_features(f2, perts)
    lookup = {name: i for i, name in enumerate(perts)}
    manifests = {}
    for seed in args.seeds:
        manifest = json.loads((split_dir / f'seed_{seed}.json').read_text())
        tr, te = manifest['outer_train'], manifest['outer_test']
        inner_tr, inner_val = manifest['mlp_inner_train'], manifest['mlp_inner_validation']
        if set(tr) & set(te) or set(tr) | set(te) != set(perts):
            raise ValueError(f'Invalid outer split seed={seed}')
        if set(inner_tr) & set(inner_val) or set(inner_tr) | set(inner_val) != set(tr):
            raise ValueError(f'Invalid inner split seed={seed}')
        for fold in manifest['ridge_folds']:
            if set(fold['train']) & set(fold['validation']) or set(fold['train']) | set(fold['validation']) != set(tr):
                raise ValueError(f'Invalid Ridge fold seed={seed}')
        baseline_selection = json.loads((selection_dir / f'seed_{seed}/GenePT_3large/selection_summary.json').read_text())
        alpha = float(baseline_selection['ridge_alpha'])
        if alpha <= 0:
            raise ValueError('Invalid Ridge alpha')
        manifests[seed] = (manifest, alpha)
        print(f'[SPLIT PASS] seed={seed} train={len(tr)} test={len(te)} inner_train={len(inner_tr)} inner_val={len(inner_val)} ridge_alpha={alpha}', flush=True)
    if args.dry_run:
        print(f'[DRY RUN PASS] perturbations={len(perts)} responses={len(genes)} genept_dim={x1.shape[1]} uce_dim={x2.shape[1]} device={device}')
        return

    out = root / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    config = {
        'stage': 'Step06B exploratory CCRT gated residual', 'seeds': args.seeds,
        'widths': args.widths, 'weight_decays': args.weight_decays,
        'dropout': args.dropout, 'residual_penalty': args.residual_penalty,
        'epochs': args.epochs, 'patience': args.patience,
        'batch_size': args.batch_size, 'lr': args.lr, 'threads': args.threads,
        'input_sha256': {str(p.relative_to(root)): sha256(p) for p in [rp, f1, f2]},
        'inner_split_sha256': {str(s): sha256(split_dir / f'seed_{s}.json') for s in args.seeds},
        'baseline_selection_sha256': {str(s): sha256(selection_dir / f'seed_{s}/GenePT_3large/selection_summary.json') for s in args.seeds},
        'script_sha256': sha256(Path(__file__).resolve()),
        'metrics_sha256': sha256(Path(metrics_module.__file__).resolve()),
        'torch_version': torch.__version__, 'device': device,
        'exploratory_refinement': True,
        'caveat': 'Outer test has been inspected in prior work; not an untouched confirmatory evaluation.',
        'uce_note': 'UCE input tokens are gene-level protein-based embeddings, not UCE cell embeddings.'
    }
    config_path = out / 'experiment_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError('Existing experiment config differs; use a NEW --outdir')
    else:
        save_json(config_path, config)

    records = []
    with threadpool_limits(limits=args.threads):
        for seed in args.seeds:
            sd = out / f'seed_{seed}'
            sd.mkdir(parents=True, exist_ok=True)
            complete = sd / 'STEP06B_COMPLETE.json'
            if complete.exists():
                needed = ['metrics.csv', 'predictions.npz', 'selection_summary.json', 'ccrt_model.pt', 'ridge_base.joblib']
                if not all((sd / f).exists() for f in needed):
                    raise RuntimeError(f'Completed seed {seed} missing files')
                frame = pd.read_csv(sd / 'metrics.csv')
                records.append(frame)
                print(f'[SKIP COMPLETE] seed={seed}', flush=True)
                continue
            manifest, alpha = manifests[seed]
            tr = np.array([lookup[g] for g in manifest['outer_train']], dtype=int)
            te = np.array([lookup[g] for g in manifest['outer_test']], dtype=int)
            it = np.array([lookup[g] for g in manifest['mlp_inner_train']], dtype=int)
            iv = np.array([lookup[g] for g in manifest['mlp_inner_validation']], dtype=int)
            (ai, bi), (av, bv), _ = transform_blocks(x1, x2, it, iv)
            # Baseline for selection uses ONLY inner training labels.
            inner_base = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='auto'))
            inner_base.fit(x1[it], y[it])
            base_it = inner_base.predict(x1[it]).astype(np.float32)
            base_iv = inner_base.predict(x1[iv]).astype(np.float32)
            residual_it = y[it] - base_it
            tuning_rows = []
            best = None
            for width in args.widths:
                for wd in args.weight_decays:
                    model, epoch, val, trace = fit_residual(
                        ai, bi, residual_it, seed, device, width, args.dropout, wd,
                        args.epochs, args.batch_size, args.lr,
                        validation=(av, bv, base_iv, y[iv]), patience=args.patience,
                        residual_penalty=args.residual_penalty)
                    row = {'width': width, 'weight_decay': wd, 'best_epoch': epoch,
                           'validation_mse': val, 'epochs_run': len(trace),
                           'hit_epoch_limit': len(trace) == args.epochs}
                    tuning_rows.append(row)
                    pd.DataFrame(trace).to_csv(sd / f'tuning_width{width}_wd{wd:g}.csv', index=False)
                    print(f'[TUNE] seed={seed} width={width} wd={wd} epoch={epoch} val_mse={val:.8f}', flush=True)
                    if best is None or (val, width, wd) < (best['validation_mse'], best['width'], best['weight_decay']):
                        best = row
                    del model
            pd.DataFrame(tuning_rows).to_csv(sd / 'tuning_summary.csv', index=False)
            if best is None or best['best_epoch'] is None:
                raise RuntimeError('No valid candidate')
            # Final Ridge baseline: ALL outer-training labels only.
            base = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='auto'))
            base.fit(x1[tr], y[tr])
            base_tr = base.predict(x1[tr]).astype(np.float32)
            base_te = base.predict(x1[te]).astype(np.float32)
            (at, bt), (ae, be), (s1, s2) = transform_blocks(x1, x2, tr, te)
            final, _, _, history = fit_residual(
                at, bt, y[tr] - base_tr, seed, device,
                best['width'], args.dropout, best['weight_decay'],
                best['best_epoch'], args.batch_size, args.lr,
                residual_penalty=args.residual_penalty)
            residual_te, gate_means = predict_residual(final, ae, be, device)
            pred = base_te + residual_te
            if not np.isfinite(pred).all():
                raise RuntimeError('Nonfinite test predictions')
            import joblib
            joblib.dump(base, sd / 'ridge_base.joblib')
            joblib.dump({'genept_scaler': s1, 'uce_scaler': s2, 'dims': [x1.shape[1], x2.shape[1]],
                         'block_scaling': '1/sqrt(dim)'}, sd / 'feature_preprocessors.joblib')
            torch.save({'state_dict': final.cpu().state_dict(), 'width': best['width'],
                        'dropout': args.dropout, 'genept_dim': x1.shape[1],
                        'uce_dim': x2.shape[1], 'response_dim': len(genes)}, sd / 'ccrt_model.pt')
            pd.DataFrame(history).to_csv(sd / 'final_refit_history.csv', index=False)
            np.savez_compressed(sd / 'predictions.npz', test_perturbations=perts[te],
                                genes=genes, y_true=y[te], pred_ccrt=pred.astype(np.float32),
                                pred_ridge_base=base_te, residual_ccrt=residual_te,
                                gate_mean_per_perturbation=gate_means)
            scores = response_metrics(y[te], pred)
            frame = pd.DataFrame([{'seed': seed, 'model': 'ccrt_gated_residual', **scores}])
            frame.to_csv(sd / 'metrics.csv', index=False)
            save_json(sd / 'selection_summary.json', {
                'seed': seed, 'base': 'Ridge-GenePT_3large', 'ridge_alpha': alpha,
                'selected': best, 'selection_metric': 'inner validation MSE',
                'outer_test_used_for_selection': False,
                'note': 'Ridge alpha inherited from Step05 outer-training CV; residual selection uses inner validation.'
            })
            save_json(complete, {'seed': seed, 'n_train': len(tr), 'n_test': len(te),
                                 'status': 'complete', 'test_perturbations': perts[te].tolist()})
            records.append(frame)
            print(f'[COMPLETE] seed={seed} MSE={scores["mse"]:.8f} PCC={scores["global_pcc_delta"]:.5f}', flush=True)
            del final
            if device == 'cuda':
                torch.cuda.empty_cache()
    all_df = pd.concat(records, ignore_index=True)
    all_df.to_csv(out / 'metrics_all_runs.csv', index=False)
    num_cols = [c for c in all_df.select_dtypes(include=np.number).columns if c != 'seed']
    all_df.groupby('model')[num_cols].agg(['mean', 'std', 'count']).to_csv(out / 'metrics_mean_sd.csv')
    print('[STEP06B COMPLETE]', out, flush=True)


if __name__ == '__main__':
    main()
