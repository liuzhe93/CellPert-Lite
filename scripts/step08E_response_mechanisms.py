#!/usr/bin/env python3
"""Step08E: frozen K562->RPE1 transfer diagnostics (no fitting or tuning models).

A: signal-strength adjusted rank partial association.
B: exact squared-error decomposition: (prediction-K562)^2 + (K562-RPE1)^2
   + 2*(prediction-K562)*(K562-RPE1).
C: gene-level direction concordance on signal-filtered response pairs.

Descriptive diagnostics only: K562 global controls vs RPE1 batch-matched controls,
and no independent biological replicates for uncertainty quantification.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ('ridge_fusion', 'mlp_fusion', 'ccrt')


def load(path, keys):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        if not set(keys).issubset(z.files):
            raise ValueError(f'{path}: missing {set(keys)-set(z.files)}; available {z.files}')
        return {k: z[k] for k in keys}


def unique(a, name):
    a = np.asarray(a).astype(str)
    if len(set(a)) != len(a):
        raise ValueError(f'Duplicate IDs in {name}')
    return a


def pcc_rows(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aa = a - a.mean(axis=1, keepdims=True)
    bb = b - b.mean(axis=1, keepdims=True)
    den = np.linalg.norm(aa, axis=1)*np.linalg.norm(bb, axis=1)
    return np.divide(np.einsum('ij,ij->i', aa, bb), den,
                     out=np.full(len(a), np.nan), where=den > 1e-12)


def partial_rank(x, y, controls):
    """Spearman partial correlation via rank-residualization (linear rank adjustment)."""
    z = np.column_stack([np.asarray(c, float) for c in controls])
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z).all(axis=1)
    n = int(good.sum())
    if n < z.shape[1] + 8:
        return np.nan, n
    xx = rankdata(np.asarray(x)[good]); yy = rankdata(np.asarray(y)[good])
    zz = np.column_stack([np.ones(n)] + [rankdata(z[good, j]) for j in range(z.shape[1])])
    xr = xx - zz @ np.linalg.lstsq(zz, xx, rcond=None)[0]
    yr = yy - zz @ np.linalg.lstsq(zz, yy, rcond=None)[0]
    if np.std(xr) < 1e-10 or np.std(yr) < 1e-10:
        return np.nan, n
    return float(np.corrcoef(xr, yr)[0, 1]), n


def spearman(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 8 or np.std(np.asarray(x)[good]) < 1e-12 or np.std(np.asarray(y)[good]) < 1e-12:
        return np.nan
    return float(spearmanr(np.asarray(x)[good], np.asarray(y)[good]).statistic)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--step08c', default='results/step08C_k562_to_rpe1_transfer')
    p.add_argument('--outdir', default='results/step08E_response_mechanisms')
    p.add_argument('--thresholds', nargs='+', type=int, default=[3, 5, 10])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--signal-thresholds', nargs='+', type=float, default=[0.05, 0.1, 0.2])
    p.add_argument('--topk', type=int, default=50)
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.thresholds)) != len(args.thresholds):
        raise ValueError('Seeds and thresholds must be unique')
    if any(t <= 0 for t in args.signal_thresholds) or args.topk < 1:
        raise ValueError('Signal thresholds and topk must be positive')
    root = Path(args.root).resolve()
    out = root / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    src = root / args.step08c
    k = load(root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz',
             ('perturbations', 'genes', 'delta'))
    kp = unique(k['perturbations'], 'K562 perturbations')
    kg = unique(k['genes'], 'K562 genes')
    ky = np.asarray(k['delta'], np.float32)
    if ky.shape != (len(kp), len(kg)) or not np.isfinite(ky).all():
        raise ValueError('Invalid K562 response matrix')
    pi = {v:i for i,v in enumerate(kp)}
    gi = {v:i for i,v in enumerate(kg)}
    records = []
    gene_records = []
    seen_biology = set()
    for th in args.thresholds:
        for seed in args.seeds:
            path = src/f'batch{th}'/f'seed_{seed}'/'predictions.npz'
            keys = ('perturbations', 'genes', 'y_true') + tuple('pred_'+m for m in MODELS)
            z = load(path, keys)
            perts = unique(z['perturbations'], str(path)+' perturbations')
            genes = unique(z['genes'], str(path)+' genes')
            if not set(perts).issubset(pi) or not set(genes).issubset(gi):
                raise ValueError(f'{path}: IDs not present in K562')
            split_path = root/'results/step05_tuned_genept_uce/inner_splits'/f'seed_{seed}.json'
            split = json.loads(split_path.read_text())
            if not set(perts).issubset(set(split['outer_train'])):
                raise ValueError(f'{path}: includes unseen K562 perturbations')
            y = np.asarray(z['y_true'], dtype=np.float64)
            if y.shape != (len(perts), len(genes)) or not np.isfinite(y).all():
                raise ValueError(f'{path}: bad RPE1 response')
            kk = ky[np.ix_([pi[q] for q in perts], [gi[g] for g in genes])].astype(np.float64)
            cross = pcc_rows(kk, y)
            k_rms = np.sqrt(np.mean(kk**2, axis=1))
            r_rms = np.sqrt(np.mean(y**2, axis=1))
            k_sd = np.std(kk, axis=1)
            r_sd = np.std(y, axis=1)
            k_nstrong = (np.abs(kk) >= 0.1).sum(axis=1)
            r_nstrong = (np.abs(y) >= 0.1).sum(axis=1)
            biological_shift = np.mean((kk-y)**2, axis=1)
            for i, pert in enumerate(perts):
                # Deduplicate gene-level biological pairs across seeds.
                bio_key = (th, pert)
                if bio_key in seen_biology:
                    continue
                seen_biology.add(bio_key)
                a, b = kk[i], y[i]
                for cutoff in args.signal_thresholds:
                    selected = (np.abs(a) >= cutoff) | (np.abs(b) >= cutoff)
                    if not selected.any():
                        frac = np.nan
                    else:
                        frac = float(np.mean(np.sign(a[selected]) == np.sign(b[selected])))
                    gene_records.append({'threshold':th, 'perturbation':pert,
                        'signal_cutoff':cutoff, 'n_union_signal_genes':int(selected.sum()),
                        'direction_concordance':frac,
                        'n_both_signal_genes':int(((np.abs(a)>=cutoff)&(np.abs(b)>=cutoff)).sum())})
            for model in MODELS:
                pred = np.asarray(z['pred_'+model], dtype=np.float64)
                if pred.shape != y.shape or not np.isfinite(pred).all():
                    raise ValueError(f'{path}: invalid {model} predictions')
                # p-y = (p-k)+(k-y), exact including cross term.
                e = pred-kk
                shift = kk-y
                model_k562_error = np.mean(e**2, axis=1)
                cross_term = 2*np.mean(e*shift, axis=1)
                observed_error = np.mean((pred-y)**2, axis=1)
                recon = model_k562_error + biological_shift + cross_term
                if not np.allclose(recon, observed_error, atol=1e-12, rtol=1e-9):
                    raise RuntimeError(f'Error decomposition failed {path} {model}')
                pp = pcc_rows(pred, y)
                for i, pert in enumerate(perts):
                    records.append({'threshold':th,'seed':seed,'perturbation':pert,'model':model,
                        'cross_cell_pcc':cross[i], 'prediction_pcc':pp[i],
                        'rpe1_mse':observed_error[i],
                        'zero_mse':float(np.mean(y[i]**2)),
                        'mse_improvement':float(np.mean(y[i]**2))-observed_error[i],
                        'k562_rms':k_rms[i], 'rpe1_rms':r_rms[i],
                        'k562_sd':k_sd[i], 'rpe1_sd':r_sd[i],
                        'k562_n_abs_ge_0p1':k_nstrong[i], 'rpe1_n_abs_ge_0p1':r_nstrong[i],
                        'model_vs_k562_mse':model_k562_error[i],
                        'k562_vs_rpe1_mse':biological_shift[i],
                        'cross_term':cross_term[i], 'reconstructed_rpe1_mse':recon[i]})
            print(f'[PASS] batch{th} seed={seed} n={len(perts)}', flush=True)
    df = pd.DataFrame(records)
    df.to_csv(out/'per_perturbation_error_decomposition.csv', index=False)
    gene = pd.DataFrame(gene_records)
    gene.to_csv(out/'direction_concordance_per_perturbation.csv', index=False)
    # One row per perturbation, not one row per seed: seeds are not biological replicates.
    aggcols = ['cross_cell_pcc','prediction_pcc','rpe1_mse','zero_mse','mse_improvement',
               'k562_rms','rpe1_rms','k562_sd','rpe1_sd','k562_n_abs_ge_0p1',
               'rpe1_n_abs_ge_0p1','model_vs_k562_mse','k562_vs_rpe1_mse','cross_term']
    avg = df.groupby(['threshold','model','perturbation'],as_index=False)[aggcols].mean()
    avg.to_csv(out/'per_perturbation_seed_averaged.csv',index=False)
    assoc = []
    for (th, model), d in avg.groupby(['threshold','model']):
        x = d['cross_cell_pcc'].to_numpy()
        controls = [np.log1p(d[c].to_numpy()) for c in
                    ('k562_rms','rpe1_rms','k562_n_abs_ge_0p1','rpe1_n_abs_ge_0p1')]
        for outcome in ('prediction_pcc','mse_improvement'):
            y = d[outcome].to_numpy()
            adjusted, n = partial_rank(x,y,controls)
            assoc.append({'threshold':th,'model':model,'outcome':outcome,
                'n_perturbations':n,'unadjusted_spearman':spearman(x,y),
                'partial_rank_rho':adjusted,
                'controls':'log1p(K562 RMS), log1p(RPE1 RMS), log1p(K562 count |delta|>=0.1), log1p(RPE1 count |delta|>=0.1)'})
    pd.DataFrame(assoc).to_csv(out/'signal_adjusted_associations.csv',index=False)
    decom = avg.groupby(['threshold','model'],as_index=False).agg(
        n_perturbations=('perturbation','size'),
        mean_rpe1_mse=('rpe1_mse','mean'),
        mean_model_vs_k562_mse=('model_vs_k562_mse','mean'),
        mean_k562_vs_rpe1_mse=('k562_vs_rpe1_mse','mean'),
        mean_cross_term=('cross_term','mean'),
        mean_prediction_pcc=('prediction_pcc','mean'))
    decom['reconstruction_error'] = (decom['mean_rpe1_mse'] -
        decom['mean_model_vs_k562_mse'] - decom['mean_k562_vs_rpe1_mse'] - decom['mean_cross_term'])
    decom.to_csv(out/'error_decomposition_summary.csv',index=False)
    direction = gene.groupby(['threshold','signal_cutoff'],as_index=False).agg(
        n_perturbations=('perturbation','size'),
        n_with_signal=('direction_concordance','count'),
        mean_direction_concordance=('direction_concordance','mean'),
        median_direction_concordance=('direction_concordance','median'),
        mean_n_union_signal_genes=('n_union_signal_genes','mean'),
        mean_n_both_signal_genes=('n_both_signal_genes','mean'))
    direction.to_csv(out/'direction_concordance_summary.csv',index=False)
    if not args.no_plots:
        plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
                             'axes.spines.top':False,'axes.spines.right':False})
        # Main batch5; if omitted, first requested threshold.
        th = 5 if 5 in args.thresholds else args.thresholds[0]
        fig, axs = plt.subplots(1,3,figsize=(12,3.6))
        for ax, model in zip(axs, MODELS):
            d = avg[(avg.threshold==th)&(avg.model==model)]
            ax.scatter(d.cross_cell_pcc,d.prediction_pcc,s=9,alpha=.45,rasterized=True)
            q = pd.read_csv(out/'signal_adjusted_associations.csv')
            z = q[(q.threshold==th)&(q.model==model)&(q.outcome=='prediction_pcc')].iloc[0]
            ax.set_title(f'{model}\nraw rho={z.unadjusted_spearman:.2f}; adjusted={z.partial_rank_rho:.2f}')
            ax.set_xlabel('Observed K562–RPE1 PCC')
            ax.axhline(0,color='grey',lw=.6,ls=':'); ax.axvline(0,color='grey',lw=.6,ls=':')
        axs[0].set_ylabel('RPE1 prediction PCC')
        fig.tight_layout()
        for ext in ('pdf','svg'):
            fig.savefig(out/f'Figure3E_signal_adjusted_association.{ext}',bbox_inches='tight')
        plt.close(fig)
        q = decom[decom.threshold==th]
        fig,ax=plt.subplots(figsize=(7,4))
        xs=np.arange(len(q))
        components=['mean_model_vs_k562_mse','mean_k562_vs_rpe1_mse','mean_cross_term']
        labels=['Model vs K562','K562 vs RPE1','Cross term (signed)']
        width=.24
        for j,(col,label) in enumerate(zip(components,labels)):
            ax.bar(xs+(j-1)*width,q[col],width,label=label)
        ax.set_xticks(xs,q.model)
        ax.axhline(0,color='black',lw=.7)
        ax.set_ylabel('Mean squared-error component')
        ax.set_title(f'Exact error decomposition | batch{th}')
        ax.legend(frameon=False,fontsize=8)
        fig.tight_layout()
        for ext in ('pdf','svg'):
            fig.savefig(out/f'Figure3E_error_decomposition.{ext}',bbox_inches='tight')
        plt.close(fig)
        q=direction[direction.threshold==th]
        fig,ax=plt.subplots(figsize=(6,4))
        ax.plot(q.signal_cutoff,q.mean_direction_concordance,marker='o')
        ax.set_ylim(0,1);ax.set_xlabel('Absolute log1p(CP10k) delta cutoff')
        ax.set_ylabel('Mean response-direction concordance')
        ax.set_title(f'Observed cross-cell response directions | batch{th}')
        fig.tight_layout()
        for ext in ('pdf','svg'):
            fig.savefig(out/f'Figure3E_direction_concordance.{ext}',bbox_inches='tight')
        plt.close(fig)
    manifest={'analysis':'Step08E observational transfer diagnostics',
        'seeds':args.seeds,'thresholds':args.thresholds,
        'signal_thresholds':args.signal_thresholds,
        'n_seed_perturbation_model_rows':len(df),
        'n_unique_perturbations_per_threshold':df.groupby('threshold').perturbation.nunique().to_dict(),
        'limitations':[
            'K562 global-control vs RPE1 batch-matched-control response estimates differ.',
            'Signal-strength adjusted rank associations are descriptive, not causal.',
            'Exact error decomposition includes a signed cross term; components are not causal fractions.',
            'Model-vs-K562 error is evaluated on K562 TRAIN perturbations; it is not K562 holdout error.',
            'Direction agreement on union-signal genes can be reduced by near-zero signals in the other cell line.',
            'No pathway enrichment performed: requires gene annotation and a prespecified gene-selection strategy.',
            'Seed-averaging does not create independent biological replicates.'
        ]}
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[DONE]',out,flush=True)

if __name__=='__main__':
    main()
