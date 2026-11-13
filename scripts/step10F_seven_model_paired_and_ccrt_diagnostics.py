#!/usr/bin/env python3
"""Step10F: seven-model locked-split paired benchmark and CCRT residual diagnostics.

No training. Aligns actual saved prediction NPZs by perturbation and gene IDs,
verifies identical ground truth and Step10A held-out membership, then performs
perturbation-cluster bootstrap (mean repeated-seed differences per unique ID).

CCRT diagnostics reconstruct Ridge-only predictions from the saved Step10E
ridge_base.joblib and GenePT embeddings, then compare CCRT vs Ridge on the
same targets. RPE1 thresholds are distinct cohorts, NOT longitudinal batches.
Confidence intervals are exploratory, unadjusted for multiplicity; seeds are
not independent biological replicates.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ('full_ccrt','gated_direct','mlp_fusion','no_gate_direct',
          'genept_only_direct','ridge_fusion','uce_only_direct')
LABEL = {'full_ccrt':'Full CCRT','gated_direct':'Gated Direct',
         'mlp_fusion':'MLP-Fusion','no_gate_direct':'No-Gate Direct',
         'genept_only_direct':'GenePT-only','ridge_fusion':'Ridge-Fusion',
         'uce_only_direct':'UCE-only'}

def read_prediction(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing prediction: {path}')
    with np.load(path,allow_pickle=False) as z:
        keys=('perturbations','genes','y_true','prediction')
        if not all(k in z.files for k in keys):
            raise ValueError(f'{path}: expected {keys}; found {z.files}')
        ids=z['perturbations'].astype(str)
        genes=z['genes'].astype(str)
        truth=np.asarray(z['y_true'],dtype=np.float64)
        pred=np.asarray(z['prediction'],dtype=np.float64)
    if len(set(ids))!=len(ids) or len(set(genes))!=len(genes):
        raise ValueError(f'Duplicate IDs in {path}')
    if truth.shape!=(len(ids),len(genes)) or pred.shape!=truth.shape:
        raise ValueError(f'Shape mismatch: {path}')
    if not (np.isfinite(truth).all() and np.isfinite(pred).all()):
        raise ValueError(f'Non-finite values: {path}')
    return ids,genes,truth,pred

def pcc_rows(y,p):
    a=y-y.mean(axis=1,keepdims=True)
    b=p-p.mean(axis=1,keepdims=True)
    d=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return np.divide(np.sum(a*b,axis=1),d,out=np.full(len(y),np.nan),where=d>1e-12)

def overall(y,p):
    a=(y-y.mean()).ravel();b=(p-p.mean()).ravel()
    d=np.linalg.norm(a)*np.linalg.norm(b)
    pcs=pcc_rows(y,p)
    return dict(mse=float(np.mean((y-p)**2)),
                global_pcc=float(a@b/d) if d>1e-12 else np.nan,
                mean_perturbation_pcc=float(np.nanmean(pcs)) if np.isfinite(pcs).any() else np.nan)

def bootstrap_unique(differences, nboot, rng):
    """Bootstrap unique perturbations; repeats across seeds already averaged."""
    vals=np.asarray(differences,dtype=float)
    vals=vals[np.isfinite(vals)]
    if len(vals)<3:
        return np.nan,np.nan,np.nan
    draws=np.empty(nboot,dtype=float)
    # Chunk for memory stability on HPC.
    for start in range(0,nboot,100):
        n=min(100,nboot-start)
        ix=rng.integers(0,len(vals),size=(n,len(vals)))
        draws[start:start+n]=vals[ix].mean(axis=1)
    return float(vals.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))

def plot_save(fig,stem,formats):
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(str(stem)+'.'+fmt,bbox_inches='tight',dpi=220)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--root',default='.')
    ap.add_argument('--step10a',default='results/step10A_unseen_perturbation_transfer')
    ap.add_argument('--step10c',default='results/step10C_strong_baseline_unseen')
    ap.add_argument('--step10e',default='results/step10E_full_ccrt_unseen')
    ap.add_argument('--outdir',default='results/step10F_seven_model_paired_and_ccrt_diagnostics')
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    ap.add_argument('--thresholds',nargs='+',type=int,default=[3,5,10])
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--rng-seed',type=int,default=20261110)
    ap.add_argument('--formats',nargs='+',choices=['pdf','svg','png'],default=['pdf','svg'])
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    if a.bootstrap<100 or not a.seeds or len(set(a.seeds))!=len(a.seeds) or len(set(a.thresholds))!=len(a.thresholds):
        ap.error('Bootstrap must be >=100 and seed/threshold lists must be nonempty and unique')
    root=Path(a.root).resolve()
    dirs={'a':root/a.step10a,'c':root/a.step10c,'e':root/a.step10e}
    out=root/a.outdir
    model_dir={m:(dirs['e'] if m=='full_ccrt' else dirs['c'] if m in ('ridge_fusion','mlp_fusion') else dirs['a']) for m in MODELS}
    # Ridge-only diagnostic: load original GenePT embeddings with strict ID mapping.
    features=root/'results/step03_genept_uce_common/aligned_features/GenePT_3large.npz'
    if not features.is_file():raise FileNotFoundError(features)
    with np.load(features,allow_pickle=False) as z:
        ids_emb=z['genes'].astype(str)
        x_emb=np.asarray(z['embeddings'],dtype=np.float32)
    if len(set(ids_emb))!=len(ids_emb):raise ValueError('Duplicate GenePT perturbation IDs')
    index_emb={p:i for i,p in enumerate(ids_emb)}
    audit=[];seed_rows=[];pert_rows=[];diagnostic=[];histories=[]
    for seed in a.seeds:
        split_file=dirs['a']/f'seed_{seed}/split.json'
        split=json.loads(split_file.read_text())
        tr=set(map(str,split['train']));te=set(map(str,split['test']))
        it=set(map(str,split['inner_train']));iv=set(map(str,split['inner_val']))
        if tr&te or it&iv or it|iv!=tr:raise ValueError(f'Split leakage seed {seed}')
        ccrt_dir=dirs['e']/f'seed_{seed}/full_ccrt'
        sel=ccrt_dir/'selection.json'
        if not sel.is_file():raise FileNotFoundError(sel)
        selection=json.loads(sel.read_text())
        import joblib
        ridge_path=ccrt_dir/'ridge_base.joblib'
        ridge=joblib.load(ridge_path)  # Load only your own trusted local checkpoints.
        history_file=ccrt_dir/'final_fit_history.csv'
        if history_file.is_file():
            history=pd.read_csv(history_file)
            for _,row in history.iterrows():
                histories.append({'seed':seed,**row.to_dict()})
        for th in [0]+a.thresholds:
            ds='K562_unseen' if th==0 else 'RPE1_unseen_transfer'
            loaded={}
            for m in MODELS:
                md=model_dir[m]/f'seed_{seed}'/m
                f=md/'K562_holdout_predictions.npz' if th==0 else md/f'RPE1_batch{th}/predictions.npz'
                loaded[m]=read_prediction(f)
            refids,refgenes,refy,_=loaded['gated_direct']
            if not len(refids) or not set(refids).issubset(te) or (th==0 and set(refids)!=te):
                raise ValueError(f'Invalid held-out IDs seed={seed}, threshold={th}')
            aligned={}
            for m,(ids,genes,y,p) in loaded.items():
                if set(ids)!=set(refids) or set(genes)!=set(refgenes):
                    raise ValueError(f'ID/gene mismatch seed={seed} threshold={th} model={m}')
                ri={v:i for i,v in enumerate(ids)};gi={v:i for i,v in enumerate(genes)}
                rr=[ri[v] for v in refids];cc=[gi[v] for v in refgenes]
                yy=y[np.ix_(rr,cc)];pp=p[np.ix_(rr,cc)]
                if not np.allclose(yy,refy,atol=1e-7,rtol=1e-6):
                    raise ValueError(f'Truth mismatch seed={seed} threshold={th} model={m}')
                aligned[m]=pp
                metrics=overall(refy,pp)
                seed_rows.append({'dataset':ds,'threshold':th,'seed':seed,'model':m,
                                  'n_perturbations':len(refids),'n_genes':len(refgenes),**metrics})
                err=np.mean((refy-pp)**2,axis=1);pc=pcc_rows(refy,pp)
                for j,pid in enumerate(refids):
                    pert_rows.append({'dataset':ds,'threshold':th,'seed':seed,'perturbation':pid,
                                      'model':m,'mse':float(err[j]),'pcc':float(pc[j])})
            # Ridge base output is trained on K562; select RPE1 common genes in the original K562 order.
            if not set(refids).issubset(index_emb):raise ValueError('Ridge input missing perturbations')
            base_full=np.asarray(ridge.predict(x_emb[[index_emb[p] for p in refids]]),dtype=np.float64)
            if th==0:
                # K562 ref gene order matches original response_matrix_common.npz.
                base_genes_file=root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
                with np.load(base_genes_file,allow_pickle=False) as z:
                    base_genes=z['genes'].astype(str)
            else:
                base_genes_file=root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
                with np.load(base_genes_file,allow_pickle=False) as z:
                    base_genes=z['genes'].astype(str)
            if base_full.shape[1]!=len(base_genes):raise ValueError('Ridge output gene dimension mismatch')
            gi={g:i for i,g in enumerate(base_genes)}
            if not set(refgenes).issubset(gi):raise ValueError('Ridge output missing response genes')
            ridge_pred=base_full[:,[gi[g] for g in refgenes]]
            ccrt=aligned['full_ccrt'];resid=ccrt-ridge_pred
            rmse_r=np.mean((refy-ridge_pred)**2,axis=1)
            rmse_c=np.mean((refy-ccrt)**2,axis=1)
            per_pcc_r=pcc_rows(refy,ridge_pred)
            per_pcc_c=pcc_rows(refy,ccrt)
            norm=np.sqrt(np.mean(resid**2,axis=1))
            for j,pid in enumerate(refids):
                diagnostic.append({'dataset':ds,'threshold':th,'seed':seed,'perturbation':pid,
                                   'ridge_mse':float(rmse_r[j]),'ccrt_mse':float(rmse_c[j]),
                                   'ccrt_minus_ridge_mse':float(rmse_c[j]-rmse_r[j]),
                                   'ridge_pcc':float(per_pcc_r[j]),'ccrt_pcc':float(per_pcc_c[j]),
                                   'ccrt_minus_ridge_pcc':float(per_pcc_c[j]-per_pcc_r[j]),
                                   'residual_rms':float(norm[j]),
                                   'ridge_prediction_rms':float(np.sqrt(np.mean(ridge_pred[j]**2))),
                                   'truth_rms':float(np.sqrt(np.mean(refy[j]**2)))})
            audit.append({'dataset':ds,'threshold':th,'seed':seed,'n_perturbations':len(refids),
                          'n_genes':len(refgenes),'n_models':len(aligned),'truth_aligned':True,
                          'heldout_verified':True,'ridge_alpha':selection.get('alpha'),
                          'ccrt_selected_epoch':selection.get('best',{}).get('selected_epoch')})
            print(f'[AUDIT] seed={seed} threshold={th} n={len(refids)} genes={len(refgenes)} models=7',flush=True)
    if a.dry_run:
        print('[DRY RUN PASS] All seven models, held-out splits, truths and CCRT Ridge baselines aligned; no files written.')
        return
    out.mkdir(parents=True,exist_ok=True)
    (out/'experiment_config.json').write_text(json.dumps(vars(a),indent=2,ensure_ascii=False))
    pd.DataFrame(audit).to_csv(out/'alignment_audit.csv',index=False)
    per=pd.DataFrame(pert_rows);per.to_csv(out/'per_perturbation_metrics.csv',index=False)
    seeds=pd.DataFrame(seed_rows);seeds.to_csv(out/'metrics_per_seed.csv',index=False)
    summary=seeds.groupby(['dataset','threshold','model'],as_index=False).agg(
        n_seeds=('seed','nunique'),n_perturbations_mean=('n_perturbations','mean'),
        mse_mean=('mse','mean'),mse_sd=('mse','std'),global_pcc_mean=('global_pcc','mean'),
        global_pcc_sd=('global_pcc','std'),perturbation_pcc_mean=('mean_perturbation_pcc','mean'))
    summary.to_csv(out/'metrics_summary.csv',index=False)
    diag=pd.DataFrame(diagnostic);diag.to_csv(out/'ccrt_vs_ridge_per_perturbation.csv',index=False)
    diag_sum=diag.groupby(['dataset','threshold'],as_index=False).agg(
        n_unique_perturbations=('perturbation','nunique'),n_pairs=('perturbation','size'),
        ridge_mse_mean=('ridge_mse','mean'),ccrt_mse_mean=('ccrt_mse','mean'),
        mean_ccrt_minus_ridge_mse=('ccrt_minus_ridge_mse','mean'),
        mean_ccrt_minus_ridge_pcc=('ccrt_minus_ridge_pcc','mean'),
        residual_rms_mean=('residual_rms','mean'),
        ridge_prediction_rms_mean=('ridge_prediction_rms','mean'),
        fraction_ccrt_better_mse=('ccrt_minus_ridge_mse',lambda x:float((x<0).mean())))
    diag_sum.to_csv(out/'ccrt_vs_ridge_summary.csv',index=False)
    if histories:pd.DataFrame(histories).to_csv(out/'ccrt_final_fit_histories.csv',index=False)
    rng=np.random.default_rng(a.rng_seed)
    comparisons=[]
    # All 7x6 directed comparisons against full CCRT and Gated Direct, plus MLP vs Ridge.
    pairs=[(base,m) for base in ('full_ccrt','gated_direct') for m in MODELS if m!=base]
    pairs.append(('mlp_fusion','ridge_fusion'))
    for (ds,th),group in per.groupby(['dataset','threshold'],sort=True):
        for base,competitor in pairs:
            for metric in ('mse','pcc'):
                wide=group.pivot(index=['seed','perturbation'],columns='model',values=metric)
                z=wide[[base,competitor]].dropna()
                if z.empty:continue
                unique=(z[base]-z[competitor]).groupby(level='perturbation').mean()
                mean,lo,hi=bootstrap_unique(unique.to_numpy(),a.bootstrap,rng)
                comparisons.append({'dataset':ds,'threshold':th,'model_a':base,'model_b':competitor,
                                    'metric':metric,'mean_difference':mean,'ci_low':lo,'ci_high':hi,
                                    'n_unique_perturbations':len(unique),'n_seed_perturbation_pairs':len(z),
                                    'fraction_a_better':float((unique<0).mean() if metric=='mse' else (unique>0).mean()),
                                    'ci_excludes_zero':bool(np.isfinite(lo) and np.isfinite(hi) and (lo>0 or hi<0))})
    paired=pd.DataFrame(comparisons)
    paired.to_csv(out/'paired_bootstrap_summary.csv',index=False)
    diag_boot=[]
    for (ds,th),group in diag.groupby(['dataset','threshold'],sort=True):
        for metric in ('ccrt_minus_ridge_mse','ccrt_minus_ridge_pcc'):
            unique=group.groupby('perturbation')[metric].mean().dropna()
            mean,lo,hi=bootstrap_unique(unique.to_numpy(),a.bootstrap,rng)
            diag_boot.append({'dataset':ds,'threshold':th,'metric':metric,'mean_difference':mean,
                              'ci_low':lo,'ci_high':hi,'n_unique_perturbations':len(unique)})
    pd.DataFrame(diag_boot).to_csv(out/'ccrt_vs_ridge_bootstrap.csv',index=False)
    ranks=[]
    for (ds,th,seed),g in seeds.groupby(['dataset','threshold','seed']):
        for _,r in g.iterrows():
            ranks.append({'dataset':ds,'threshold':th,'seed':seed,'model':r['model'],
                          'mse_rank':int(g['mse'].rank(method='min').loc[r.name]),
                          'global_pcc_rank':int(g['global_pcc'].rank(ascending=False,method='min').loc[r.name]),
                          'perturbation_pcc_rank':int(g['mean_perturbation_pcc'].rank(ascending=False,method='min').loc[r.name])})
    pd.DataFrame(ranks).to_csv(out/'model_ranks_per_seed.csv',index=False)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    # Per-cohort seven-model summary (seed mean ± SD).
    for th in [0]+a.thresholds:
        z=seeds[seeds.threshold==th]
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for ax,metric,label in zip(axes,['mse','global_pcc','mean_perturbation_pcc'],
                                    ['MSE ↓','Global PCC ↑','Mean perturbation PCC ↑']):
            for j,m in enumerate(MODELS):
                vals=z[z.model==m][metric].to_numpy()
                ax.scatter(np.full(len(vals),j),vals,s=14,alpha=.45)
                ax.errorbar(j,vals.mean(),yerr=vals.std(ddof=1),fmt='o',capsize=3,color='black')
            ax.set_xticks(range(len(MODELS)),[LABEL[m] for m in MODELS],rotation=48,ha='right')
            ax.set_ylabel(label);ax.grid(axis='y',alpha=.15)
        fig.suptitle('K562 unseen' if th==0 else f'RPE1 threshold {th} (distinct cohort)')
        plot_save(fig,out/f'Figure_SevenModels_{"K562" if th==0 else "RPE1_batch"+str(th)}',a.formats)
    # Paired CCRT comparisons: forest plot of MSE and PCC differences.
    for metric in ('mse','pcc'):
        z=paired[(paired.model_a=='full_ccrt')&(paired.metric==metric)].copy()
        z=z.sort_values(['threshold','model_b'])
        if z.empty:continue
        fig,ax=plt.subplots(figsize=(10,max(5,len(z)*.23)))
        ypos=np.arange(len(z));x=z.mean_difference.to_numpy();lo=z.ci_low.to_numpy();hi=z.ci_high.to_numpy()
        ax.errorbar(x,ypos,xerr=np.vstack([x-lo,hi-x]),fmt='o',capsize=2,markersize=3)
        ax.axvline(0,ls='--',color='black',lw=1)
        ax.set_yticks(ypos,[('K562' if t==0 else f'RPE1 {t}')+' / '+LABEL[m] for t,m in zip(z.threshold,z.model_b)])
        ax.invert_yaxis();ax.set_xlabel(f'Full CCRT − competitor, {metric} (95% cluster-bootstrap CI)')
        ax.grid(axis='x',alpha=.15)
        plot_save(fig,out/f'Figure_CCRT_paired_{metric}',a.formats)
    # Ridge-residual contribution plot.
    fig,ax=plt.subplots(figsize=(9,4.2))
    dd=pd.DataFrame(diag_boot)
    zz=dd[dd.metric=='ccrt_minus_ridge_mse'].sort_values('threshold')
    yloc=np.arange(len(zz));v=zz.mean_difference.to_numpy()
    ax.errorbar(v,yloc,xerr=np.vstack([v-zz.ci_low.to_numpy(),zz.ci_high.to_numpy()-v]),fmt='o',capsize=4)
    ax.axvline(0,color='black',ls='--',lw=1)
    ax.set_yticks(yloc,['K562' if t==0 else f'RPE1 batch {t}' for t in zz.threshold])
    ax.invert_yaxis();ax.set_xlabel('Full CCRT − its own Ridge baseline: per-perturbation MSE')
    plot_save(fig,out/'Figure_CCRT_residual_contribution',a.formats)
    report=['# Step10F seven-model audit','',
            'All seven model predictions were aligned by perturbation and gene IDs and checked against Step10A locked splits.',
            'Pairwise CIs cluster by unique perturbation after averaging repeat appearances across seeds.',
            'No multiplicity correction. RPE1 threshold cohorts have different perturbation sets.',
            'CCRT Ridge-only diagnostics use the saved Step10E ridge_base.joblib and GenePT embeddings.',
            'CCRT residual contribution is prediction minus Ridge prediction; it does not establish mechanistic causality.',
            'Early selected epochs are from inner validation; final_fit_history.csv may be shorter by design.',
            '', '## Key CCRT comparisons (unadjusted exploratory intervals)', '']
    for _,r in paired[(paired.model_a=='full_ccrt') & (paired.model_b.isin(['gated_direct','mlp_fusion']))].iterrows():
        report.append(f"- {r.dataset} th={r.threshold}, CCRT − {r.model_b}, {r.metric}: {r.mean_difference:.6g} [{r.ci_low:.6g}, {r.ci_high:.6g}]")
    (out/'README_interpretation.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n=== FULL CCRT vs GATED/MLP ===')
    print(paired[(paired.model_a=='full_ccrt') & (paired.model_b.isin(['gated_direct','mlp_fusion']))].to_string(index=False))
    print('\n=== CCRT vs ITS OWN RIDGE BASELINE ===')
    print(pd.DataFrame(diag_boot).to_string(index=False))
    print(f'\n[COMPLETE] {out}',flush=True)

if __name__=='__main__':
    main()
