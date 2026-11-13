#!/usr/bin/env python3
"""Step11B-6: frozen external transfer diagnostics, no fitting or tuning.

A. External response strength versus predictive accuracy.
B. Measured K562-external biological shift versus predictive accuracy.
C. Seen/held-out strata and paired model differences.
D. Rank candidate failures; optionally threshold sensitivity using a conservative response NPZ.

Important: K562 uses global controls while external datasets use batch-matched
controls. Cross-cell-line differences conflate biological and processing effects.
Each perturbation is the descriptive sampling unit; seeds are not biological replicates.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODELS = ('gated_direct', 'no_gate_direct', 'genept_only_direct', 'uce_only_direct')


def load(path, keys=('perturbations', 'genes', 'delta')):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        missing = set(keys) - set(z.files)
        if missing:
            raise ValueError(f'{path}: missing {sorted(missing)}; available {z.files}')
        return tuple(np.asarray(z[k]) for k in keys)


def ids_unique(a, name):
    a = np.asarray(a).astype(str)
    if len(set(a)) != len(a):
        raise ValueError(f'Duplicate {name}')
    return a


def pcc_rows(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    den = np.linalg.norm(ac, axis=1) * np.linalg.norm(bc, axis=1)
    return np.divide(np.einsum('ij,ij->i', ac, bc), den,
                     out=np.full(len(a), np.nan), where=den > 1e-12)


def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return np.nan, int(ok.sum())
    return float(spearmanr(x[ok], y[ok]).statistic), int(ok.sum())


def rank_bins(df, metric, n=4):
    """Quartiles assigned per dataset using true responses only, no model selection."""
    out = df.copy()
    ranks = out[metric].rank(method='first')
    out['strength_quartile'] = pd.qcut(ranks, q=n, labels=[f'Q{i+1}' for i in range(n)])
    return out


def paired_bootstrap(frame, n_boot, rng):
    """Bootstrap target means after averaging seed-level paired differences per target."""
    arr = frame.groupby('perturbation', sort=True)['difference'].mean().to_numpy(float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return (np.nan, np.nan, np.nan, 0)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        estimates[i] = arr[rng.integers(len(arr), size=len(arr))].mean()
    return float(arr.mean()), float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975)), len(arr)


def draw(df, assoc, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    # Aggregate across seeds per target so the figure does not treat seeds as independent replicates.
    for dataset in ('HepG2', 'Jurkat'):
        d = df[df.dataset == dataset]
        if d.empty:
            continue
        for metric, label, filename in [
            ('external_rms', 'External observed response RMS', 'Figure6A_strength_vs_pcc'),
            ('k562_external_pcc', 'Measured K562–external response PCC', 'Figure6B_shift_vs_pcc')]:
            fig, ax = plt.subplots(figsize=(6.2, 4.4))
            for model in MODELS:
                x = d[d.model == model].groupby('perturbation', as_index=False).agg(
                    x=(metric, 'first'), y=('prediction_pcc', 'mean'))
                if len(x):
                    ax.scatter(x.x, x.y, s=7, alpha=.23, rasterized=True, label=model)
            ax.set_xlabel(label); ax.set_ylabel('Mean prediction PCC across seeds')
            ax.axhline(0, color='gray', ls='--', lw=.7)
            ax.legend(frameon=False, markerscale=2, fontsize=7)
            fig.tight_layout()
            for ext in ('pdf', 'svg', 'png'):
                fig.savefig(out/f'{filename}_{dataset}.{ext}', dpi=220, bbox_inches='tight')
            plt.close(fig)
        s = d.groupby(['model', 'strength_quartile'], observed=True, as_index=False).agg(
            mean_pcc=('prediction_pcc', 'mean'), n_rows=('prediction_pcc', 'size'))
        fig, ax = plt.subplots(figsize=(6.4, 4.1))
        for model in MODELS:
            q = s[s.model == model]
            if not q.empty:
                ax.plot(q.strength_quartile.astype(str), q.mean_pcc, marker='o', label=model)
        ax.set_xlabel('External response strength quartile (Q1 weak → Q4 strong)')
        ax.set_ylabel('Mean prediction PCC'); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(out/f'Figure6C_strength_quartiles_{dataset}.{ext}', dpi=220, bbox_inches='tight')
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='.')
    ap.add_argument('--step11b4', default='results/step11B_4_frozen_external_transfer')
    ap.add_argument('--outdir', default='results/step11B_6_response_strength_and_biological_shift')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    ap.add_argument('--signal-thresholds', nargs='+', type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--conservative-hepg2', default=None, help='Optional min_batch=5 response NPZ')
    ap.add_argument('--conservative-jurkat', default=None, help='Optional min_batch=5 response NPZ')
    args = ap.parse_args()
    if args.bootstrap < 100 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('bootstrap >=100 and unique seeds required')
    root = Path(args.root).resolve()
    src = root/args.step11b4
    out = root/args.outdir
    out.mkdir(parents=True, exist_ok=True)
    k_ids, k_genes, k_y = load(root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz')
    k_ids = ids_unique(k_ids, 'K562 targets'); k_genes = ids_unique(k_genes, 'K562 genes')
    if k_y.shape != (len(k_ids), len(k_genes)) or not np.isfinite(k_y).all():
        raise ValueError('Invalid K562 response')
    ki = {p:i for i,p in enumerate(k_ids)}
    kg = {g:i for i,g in enumerate(k_genes)}
    rows=[]; biological=[]; signal_rows=[]; sens=[]
    rng = np.random.default_rng(args.random_state)
    for dataset in ('HepG2', 'Jurkat'):
        ext_path = root/'data/external_validation/step11B_2A_expanded_minbatch1'/dataset/'response_matrix_k562_gene_intersection.npz'
        ext_ids, ext_genes, ext_y = load(ext_path)
        ext_ids = ids_unique(ext_ids, dataset+' targets'); ext_genes = ids_unique(ext_genes, dataset+' genes')
        if ext_y.shape != (len(ext_ids), len(ext_genes)) or not np.isfinite(ext_y).all():
            raise ValueError(f'Invalid {dataset} response')
        if not set(ext_genes).issubset(kg):
            raise ValueError(f'{dataset} genes not a subset of K562')
        ei = {p:i for i,p in enumerate(ext_ids)}
        kgidx = np.array([kg[g] for g in ext_genes], dtype=int)
        common_ids = [p for p in ext_ids if p in ki]
        common_idx = np.array([ei[p] for p in common_ids])
        external = np.asarray(ext_y[common_idx], np.float64)
        k562 = np.asarray(k_y[np.ix_([ki[p] for p in common_ids], kgidx)], np.float64)
        bio_pcc = pcc_rows(k562, external)
        bio_mse = np.mean((k562-external)**2, axis=1)
        ext_rms = np.sqrt(np.mean(external**2, axis=1))
        ext_sd = np.std(external, axis=1)
        k_rms = np.sqrt(np.mean(k562**2, axis=1))
        ext_abs = np.mean(np.abs(external), axis=1)
        bio_lookup={}
        for i,p in enumerate(common_ids):
            rec={'dataset':dataset, 'perturbation':p, 'external_rms':ext_rms[i],
                 'external_sd':ext_sd[i], 'external_mean_abs':ext_abs[i],
                 'k562_rms':k_rms[i], 'k562_external_pcc':bio_pcc[i],
                 'k562_external_mse':bio_mse[i], 'external_zero_baseline_mse':ext_rms[i]**2}
            biological.append(rec); bio_lookup[p]=rec
            for threshold in args.signal_thresholds:
                chosen=(np.abs(external[i]) >= threshold) | (np.abs(k562[i]) >= threshold)
                concord=float(np.mean(np.sign(external[i, chosen]) == np.sign(k562[i, chosen]))) if chosen.any() else np.nan
                signal_rows.append({'dataset':dataset,'perturbation':p,'signal_threshold':threshold,
                                    'n_external_signal_genes':int((np.abs(external[i])>=threshold).sum()),
                                    'n_k562_signal_genes':int((np.abs(k562[i])>=threshold).sum()),
                                    'n_union_signal_genes':int(chosen.sum()),'direction_concordance':concord})
        # Verify every saved prediction against the source response NPZ; never silently realign.
        for seed in args.seeds:
            split=json.loads((root/'results/step10A_unseen_perturbation_transfer'/f'seed_{seed}'/'split.json').read_text())
            train=set(split['train']); test=set(split['test'])
            if train & test or train | test != set(k_ids):
                raise ValueError(f'Invalid split seed {seed}')
            for model in args.models:
                path=src/dataset/f'seed_{seed}'/model/'predictions.npz'
                perts, genes, truth, pred, seen = load(path, ('perturbations','genes','y_true','prediction','k562_train_seen'))
                perts=ids_unique(perts, str(path)+' targets'); genes=ids_unique(genes, str(path)+' genes')
                if not np.array_equal(genes,ext_genes):
                    raise ValueError(f'{path}: response gene order mismatch')
                if not set(perts).issubset(bio_lookup):
                    raise ValueError(f'{path}: target outside K562-external intersection')
                expected=ext_y[[ei[p] for p in perts]]
                if pred.shape != expected.shape or truth.shape != expected.shape or not np.isfinite(pred).all():
                    raise ValueError(f'{path}: invalid prediction')
                if not np.allclose(truth, expected, atol=1e-6, rtol=1e-6):
                    raise ValueError(f'{path}: saved truth differs from external response')
                if not np.array_equal(seen.astype(bool), np.array([p in train for p in perts])):
                    raise ValueError(f'{path}: seen labels mismatch split')
                pcs=pcc_rows(expected, pred)
                mses=np.mean((expected.astype(np.float64)-pred.astype(np.float64))**2, axis=1)
                for i,p in enumerate(perts):
                    rec=dict(bio_lookup[p]); rec.update({'seed':seed,'model':model,
                        'group':'seen' if p in train else 'heldout',
                        'prediction_pcc':pcs[i], 'prediction_mse':mses[i],
                        'mse_improvement_vs_zero':rec['external_zero_baseline_mse']-mses[i]})
                    rows.append(rec)
                print(f'[{dataset}] seed={seed} model={model} targets={len(perts)}', flush=True)
        # Optional conservative sensitivity, evaluated on EXACT SAME targets and genes.
        cons_arg=args.conservative_hepg2 if dataset=='HepG2' else args.conservative_jurkat
        if cons_arg:
            cp = Path(cons_arg); cp=cp if cp.is_absolute() else root/cp
            cids, cgenes, cy=load(cp)
            cids=ids_unique(cids,'conservative targets');cgenes=ids_unique(cgenes,'conservative genes')
            if cy.shape!=(len(cids),len(cgenes)) or not np.isfinite(cy).all():
                raise ValueError('Invalid conservative response')
            cg={g:i for i,g in enumerate(cgenes)}; ci={p:i for i,p in enumerate(cids)}
            for seed in args.seeds:
                for model in args.models:
                    pp,gg,yt,yp=load(src/dataset/f'seed_{seed}'/model/'predictions.npz',
                                     ('perturbations','genes','y_true','prediction'))
                    pp=pp.astype(str);gg=gg.astype(str)
                    keep=[i for i,p in enumerate(pp) if p in ci]
                    shared=[g for g in gg if g in cg]
                    if not keep or len(shared)<2:continue
                    pi=[list(gg).index(g) for g in shared]
                    cc=[cg[g] for g in shared]
                    a=cy[np.ix_([ci[pp[i]] for i in keep],cc)]
                    b=yp[np.ix_(keep,pi)]
                    expanded=yt[np.ix_(keep,pi)]
                    for label,y in [('expanded_minbatch1',expanded),('conservative',a)]:
                        pcs=pcc_rows(y,b)
                        sens.append({'dataset':dataset,'seed':seed,'model':model,'threshold_group':label,
                            'n_matched_targets':len(keep),'n_matched_genes':len(shared),
                            'mean_target_pcc':float(np.nanmean(pcs)) if np.isfinite(pcs).any() else np.nan,
                            'mse':float(np.mean((y-b)**2))})
    df=pd.DataFrame(rows)
    bio=pd.DataFrame(biological)
    if df.empty:raise ValueError('No prediction rows')
    bio= pd.concat([rank_bins(g, 'external_rms') for _,g in bio.groupby('dataset',sort=False)],ignore_index=True)
    df=df.merge(bio[['dataset','perturbation','strength_quartile']],on=['dataset','perturbation'],validate='many_to_one')
    df.to_csv(out/'per_target_diagnostics.csv',index=False)
    bio.to_csv(out/'biological_shift_per_target.csv',index=False)
    pd.DataFrame(signal_rows).to_csv(out/'direction_concordance_per_target.csv',index=False)
    # Summaries: per-seed means, and target-level correlations averaged across seeds.
    summaries=df.groupby(['dataset','model','group','strength_quartile'],observed=True).agg(
        n_rows=('prediction_pcc','size'),mean_pcc=('prediction_pcc','mean'),
        median_pcc=('prediction_pcc','median'),mean_mse=('prediction_mse','mean'),
        mean_mse_improvement_vs_zero=('mse_improvement_vs_zero','mean')).reset_index()
    summaries.to_csv(out/'strength_group_summary.csv',index=False)
    associations=[]
    for (dataset,model,group),d in df.groupby(['dataset','model','group']):
        by_target=d.groupby('perturbation',as_index=False).agg(
            prediction_pcc=('prediction_pcc','mean'),external_rms=('external_rms','first'),
            external_sd=('external_sd','first'),k562_external_pcc=('k562_external_pcc','first'),
            k562_external_mse=('k562_external_mse','first'))
        for x in ('external_rms','external_sd','k562_external_pcc','k562_external_mse'):
            rho,n=corr(by_target[x],by_target.prediction_pcc)
            associations.append({'dataset':dataset,'model':model,'group':group,
                                 'predictor':x,'spearman_rho':rho,'n_targets':n})
    assoc=pd.DataFrame(associations)
    assoc.to_csv(out/'association_summary.csv',index=False)
    # Paired model difference, target-cluster bootstrap; no independent-seed pseudoreplication.
    paired=[]
    for (dataset,group),d in df.groupby(['dataset','group']):
        for other in args.models:
            if other=='gated_direct':continue
            z=d[d.model.isin(('gated_direct',other))].pivot_table(
                index=['seed','perturbation'],columns='model',values='prediction_pcc')
            if not {'gated_direct',other}.issubset(z.columns):continue
            z=z.dropna(subset=['gated_direct',other]);z['difference']=z['gated_direct']-z[other]
            tmp=z.reset_index()[['perturbation','difference']]
            mean,lo,hi,n=paired_bootstrap(tmp,args.bootstrap,rng)
            paired.append({'dataset':dataset,'group':group,'contrast':f'gated_direct - {other}',
                           'mean_pcc_difference':mean,'ci95_low':lo,'ci95_high':hi,
                           'n_distinct_targets':n,'bootstrap_unit':'perturbation'})
    pd.DataFrame(paired).to_csv(out/'paired_pcc_bootstrap.csv',index=False)
    # Candidate failures ranked on mean PCC, not cherry-picked for inferential testing.
    cand=df.groupby(['dataset','model','perturbation','group'],as_index=False).agg(
        mean_pcc=('prediction_pcc','mean'),mean_mse=('prediction_mse','mean'),
        external_rms=('external_rms','first'),k562_external_pcc=('k562_external_pcc','first'),
        k562_external_mse=('k562_external_mse','first'),n_seeds=('seed','nunique'))
    cand=cand.sort_values(['dataset','model','group','mean_pcc'],ascending=[True,True,True,True])
    cand.to_csv(out/'candidate_success_failure_rankings.csv',index=False)
    if sens:pd.DataFrame(sens).to_csv(out/'threshold_sensitivity_matched.csv',index=False)
    if not args.no_plots:draw(df,assoc,out)
    manifest={'input_prediction_dir':str(src),'seeds':args.seeds,'models':args.models,
              'n_diagnostic_rows':len(df),'n_unique_dataset_targets':len(bio),
              'cautions':['No model fitting/tuning or model selection performed',
                          'K562 global controls versus external batch-matched controls confound biological shift',
                          'Perturbation bootstrap is descriptive, not independent biological replication',
                          'PCC may be undefined for near-constant response vectors',
                          'Quartiles defined from external truth; descriptive stratification only',
                          'Threshold sensitivity requires explicitly supplied conservative response NPZs']}
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print('DONE:',out,flush=True)

if __name__=='__main__':
    main()
