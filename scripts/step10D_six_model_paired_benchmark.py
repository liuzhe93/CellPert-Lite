#!/usr/bin/env python3
"""Step10D: six-model paired benchmark on Step10A locked unseen splits.

No training. Repeated seeds for the same perturbation are averaged before
perturbation-cluster bootstrap. Intervals are exploratory; no multiplicity
correction and no claim of independent biological replicates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ('gated_direct','mlp_fusion','no_gate_direct',
          'genept_only_direct','ridge_fusion','uce_only_direct')
LABELS = {'gated_direct':'Gated Direct','mlp_fusion':'MLP-Fusion',
          'no_gate_direct':'No-Gate Direct','genept_only_direct':'GenePT-only',
          'ridge_fusion':'Ridge-Fusion','uce_only_direct':'UCE-only'}

def read_pred(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path,allow_pickle=False) as z:
        keys=('perturbations','genes','y_true','prediction')
        if not all(k in z.files for k in keys):
            raise ValueError(f'{path}: expected keys {keys}, got {z.files}')
        ids=z['perturbations'].astype(str)
        genes=z['genes'].astype(str)
        y=np.asarray(z['y_true'],dtype=np.float64)
        pred=np.asarray(z['prediction'],dtype=np.float64)
    if len(set(ids))!=len(ids) or len(set(genes))!=len(genes):
        raise ValueError(f'{path}: duplicate perturbations/genes')
    if y.shape!=(len(ids),len(genes)) or pred.shape!=y.shape:
        raise ValueError(f'{path}: shape mismatch')
    if not (np.isfinite(y).all() and np.isfinite(pred).all()):
        raise ValueError(f'{path}: non-finite values')
    return ids,genes,y,pred

def row_pcc(y,p):
    ya=y-y.mean(axis=1,keepdims=True)
    pa=p-p.mean(axis=1,keepdims=True)
    den=np.linalg.norm(ya,axis=1)*np.linalg.norm(pa,axis=1)
    return np.divide((ya*pa).sum(axis=1),den,
                     out=np.full(len(y),np.nan),where=den>1e-12)

def overall_metrics(y,p):
    yc=y.ravel()-y.mean()
    pc=p.ravel()-p.mean()
    den=np.linalg.norm(yc)*np.linalg.norm(pc)
    per=row_pcc(y,p)
    return {'mse':float(np.mean((y-p)**2)),
            'global_pcc':float(np.dot(yc,pc)/den) if den>1e-12 else np.nan,
            'perturbation_pcc':float(np.nanmean(per)) if np.isfinite(per).any() else np.nan}

def bootstrap(values,replicates,rng):
    vals=np.asarray(values,dtype=float)
    vals=vals[np.isfinite(vals)]
    if len(vals)<3:
        return (float(np.mean(vals)) if len(vals) else np.nan,np.nan,np.nan)
    samples=np.empty(replicates,dtype=float)
    for i in range(replicates):
        samples[i]=np.mean(vals[rng.integers(0,len(vals),size=len(vals))])
    return float(vals.mean()),float(np.quantile(samples,.025)),float(np.quantile(samples,.975))

def savefig(fig,stem,formats):
    for ext in formats:
        fig.savefig(str(stem)+'.'+ext,bbox_inches='tight',dpi=240)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--step10a',default='results/step10A_unseen_perturbation_transfer')
    ap.add_argument('--step10c',default='results/step10C_strong_baseline_unseen')
    ap.add_argument('--outdir',default='results/step10D_six_model_paired_benchmark')
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    ap.add_argument('--thresholds',nargs='+',type=int,default=[3,5,10])
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--rng-seed',type=int,default=20261110)
    ap.add_argument('--formats',nargs='+',choices=['pdf','svg','png'],default=['pdf','svg'])
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.bootstrap<100 or not args.seeds or not args.thresholds:
        raise ValueError('Invalid bootstrap/seeds/thresholds')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.thresholds))!=len(args.thresholds):
        raise ValueError('Duplicate seeds or thresholds')
    root=Path(args.root).resolve()
    a=root/args.step10a;c=root/args.step10c;out=root/args.outdir
    model_dir={m:(c if m in ('ridge_fusion','mlp_fusion') else a) for m in MODELS}
    audit=[];details=[];seedmetrics=[]
    for seed in args.seeds:
        split_file=a/f'seed_{seed}'/'split.json'
        if not split_file.is_file():raise FileNotFoundError(split_file)
        split=json.loads(split_file.read_text())
        tr=set(map(str,split['train']));te=set(map(str,split['test']))
        if tr&te:raise RuntimeError(f'Leakage seed={seed}')
        for th in [0]+args.thresholds:
            dataset='K562_unseen' if th==0 else 'RPE1_unseen_transfer'
            loaded={}
            for m in MODELS:
                md=model_dir[m]/f'seed_{seed}'/m
                path=md/'K562_holdout_predictions.npz' if th==0 else md/f'RPE1_batch{th}'/'predictions.npz'
                ids,genes,y,p=read_pred(path)
                if not set(ids).issubset(te):raise RuntimeError(f'Non-heldout perturbations: {path}')
                if th==0 and set(ids)!=te:raise RuntimeError(f'Incomplete K562 holdout: {path}')
                loaded[m]=(ids,genes,y,p)
            # Align by labels, not by arbitrary file row/column order.
            ref_ids,ref_genes,ref_y,_=loaded['gated_direct']
            if not len(ref_ids):raise RuntimeError(f'Empty test set seed={seed} threshold={th}')
            aligned={}
            for m,(ids,genes,y,p) in loaded.items():
                if set(ids)!=set(ref_ids) or set(genes)!=set(ref_genes):
                    raise RuntimeError(f'ID/gene mismatch seed={seed} th={th} model={m}')
                ri={v:i for i,v in enumerate(ids)};gi={v:i for i,v in enumerate(genes)}
                r=np.array([ri[v] for v in ref_ids]);g=np.array([gi[v] for v in ref_genes])
                yy=y[np.ix_(r,g)];pp=p[np.ix_(r,g)]
                if not np.allclose(yy,ref_y,rtol=1e-6,atol=1e-7):
                    raise RuntimeError(f'Truth mismatch seed={seed} th={th} model={m}')
                aligned[m]=pp
                scores=overall_metrics(ref_y,pp)
                seedmetrics.append({'dataset':dataset,'threshold':th,'seed':seed,'model':m,
                                    'n_perturbations':len(ref_ids),'n_genes':len(ref_genes),**scores})
                mses=np.mean((ref_y-pp)**2,axis=1);pccs=row_pcc(ref_y,pp)
                for i,pid in enumerate(ref_ids):
                    details.append({'dataset':dataset,'threshold':th,'seed':seed,'perturbation':pid,
                                    'model':m,'mse':float(mses[i]),'pcc':float(pccs[i])})
            audit.append({'dataset':dataset,'threshold':th,'seed':seed,
                          'n_perturbations':len(ref_ids),'n_genes':len(ref_genes),
                          'n_aligned_models':len(aligned)})
            print(f'[AUDIT] seed={seed} {dataset} threshold={th} n={len(ref_ids)} genes={len(ref_genes)} models={len(aligned)}',flush=True)
    if args.dry_run:
        print('[DRY RUN PASS] All six models aligned, held-out splits verified; no files written.',flush=True)
        return
    out.mkdir(parents=True,exist_ok=True)
    (out/'experiment_config.json').write_text(json.dumps(vars(args),indent=2))
    pd.DataFrame(audit).to_csv(out/'alignment_audit.csv',index=False)
    df=pd.DataFrame(details);df.to_csv(out/'per_perturbation_metrics.csv',index=False)
    seeds=pd.DataFrame(seedmetrics);seeds.to_csv(out/'metrics_per_seed.csv',index=False)
    summ=seeds.groupby(['dataset','threshold','model'],as_index=False).agg(
        n_seeds=('seed','nunique'),n_perturbations_mean=('n_perturbations','mean'),
        mse_mean=('mse','mean'),mse_sd=('mse','std'),
        global_pcc_mean=('global_pcc','mean'),global_pcc_sd=('global_pcc','std'),
        perturbation_pcc_mean=('perturbation_pcc','mean'))
    summ.to_csv(out/'metrics_summary.csv',index=False)
    rng=np.random.default_rng(args.rng_seed)
    pairs=[]
    for (dataset,th),group in df.groupby(['dataset','threshold'],sort=True):
        for competitor in MODELS:
            if competitor=='gated_direct':continue
            for metric in ('mse','pcc'):
                pivot=group.pivot(index=['seed','perturbation'],columns='model',values=metric)
                z=pivot[['gated_direct',competitor]].dropna()
                if z.empty:continue
                d=z['gated_direct']-z[competitor]
                # Average repeats of each perturbation BEFORE resampling.
                unique=d.groupby(level='perturbation').mean()
                mean,lo,hi=bootstrap(unique.to_numpy(),args.bootstrap,rng)
                fraction=float(np.mean(unique<0)) if metric=='mse' else float(np.mean(unique>0))
                pairs.append({'dataset':dataset,'threshold':th,'model_a':'gated_direct',
                              'model_b':competitor,'metric':metric,'mean_difference':mean,
                              'ci_low':lo,'ci_high':hi,'n_unique_perturbations':len(unique),
                              'n_seed_perturbation_pairs':len(d),
                              'fraction_gated_better':fraction,
                              'ci_excludes_zero':bool(np.isfinite(lo) and np.isfinite(hi) and (lo>0 or hi<0))})
    paired=pd.DataFrame(pairs);paired.to_csv(out/'paired_bootstrap_summary.csv',index=False)
    # Seed-level descriptive ranks (not independent-sample statistical tests).
    ranking=[]
    for (dataset,th,seed),g in seeds.groupby(['dataset','threshold','seed']):
        for _,r in g.iterrows():
            ranking.append({'dataset':dataset,'threshold':th,'seed':seed,'model':r['model'],
                            'mse_rank':int(g['mse'].rank(method='min').loc[r.name]),
                            'global_pcc_rank':int(g['global_pcc'].rank(ascending=False,method='min').loc[r.name])})
    pd.DataFrame(ranking).to_csv(out/'model_ranks_per_seed.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    # Figure 5B: all six models on K562, seed-level values + mean/SD.
    fig,axs=plt.subplots(1,2,figsize=(12,4.6))
    k=seeds[seeds.threshold==0]
    for ax,metric,title in zip(axs,['mse','global_pcc'],['MSE (lower is better)','Global PCC (higher is better)']):
        for i,m in enumerate(MODELS):
            z=k[k.model==m][metric].to_numpy()
            ax.scatter(np.full(len(z),i),z,s=18,alpha=.55,color='#557d9f')
            ax.errorbar(i,np.mean(z),yerr=np.std(z,ddof=1),fmt='o',color='#243b53',capsize=4)
        ax.set_xticks(range(len(MODELS)),[LABELS[m] for m in MODELS],rotation=35,ha='right')
        ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('K562 | unseen perturbations | five locked splits')
    fig.tight_layout();savefig(fig,out/'Figure5B_six_models_K562',args.formats)
    # Figure 5C: six-model RPE1 results. Thresholds are distinct cohorts.
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,metric,title in zip(axs,['mse','global_pcc'],['MSE (lower is better)','Global PCC (higher is better)']):
        for m in MODELS:
            z=seeds[(seeds.threshold>0)&(seeds.model==m)].groupby('threshold')[metric].agg(['mean','std']).reindex(args.thresholds)
            ax.errorbar(args.thresholds,z['mean'],yerr=z['std'],marker='o',capsize=3,label=LABELS[m])
        ax.set_xticks(args.thresholds,[f'Batch {t}' for t in args.thresholds]);ax.set_ylabel(title);ax.grid(alpha=.2)
    axs[1].legend(frameon=False,fontsize=8,loc='best')
    fig.suptitle('RPE1 | unseen-perturbation transfer (different threshold cohorts)')
    fig.tight_layout();savefig(fig,out/'Figure5C_six_models_RPE1',args.formats)
    # Figure 5D: primary paired Gated-vs-MLP MSE comparison; cluster CIs.
    z=paired[(paired.model_b=='mlp_fusion')&(paired.metric=='mse')].sort_values(['threshold'])
    fig,ax=plt.subplots(figsize=(9,4.3))
    yy=np.arange(len(z));xx=z.mean_difference.to_numpy();lo=z.ci_low.to_numpy();hi=z.ci_high.to_numpy()
    ax.errorbar(xx,yy,xerr=np.vstack([xx-lo,hi-xx]),fmt='o',capsize=4,color='#345d7d')
    ax.axvline(0,color='black',ls='--',lw=1)
    ax.set_yticks(yy,[('K562' if t==0 else f'RPE1 batch {t}') for t in z.threshold]);ax.invert_yaxis()
    ax.set_xlabel('Gated Direct − MLP-Fusion: per-perturbation MSE (95% bootstrap CI)')
    ax.grid(axis='x',alpha=.2);fig.tight_layout()
    savefig(fig,out/'Figure5D_gated_vs_mlp_paired_MSE',args.formats)
    # Supplement: all competitors paired to Gated Direct, MSE.
    z=paired[paired.metric=='mse'].copy()
    z['cohort']=z.apply(lambda r:'K562' if r.threshold==0 else f'RPE1 batch{int(r.threshold)}',axis=1)
    cohorts=['K562']+[f'RPE1 batch{t}' for t in args.thresholds]
    fig,axs=plt.subplots(1,len(cohorts),figsize=(4.2*len(cohorts),5),sharey=True)
    if len(cohorts)==1:axs=[axs]
    comp=[m for m in MODELS if m!='gated_direct']
    for ax,cohort in zip(axs,cohorts):
        d=z[z.cohort==cohort].set_index('model_b').reindex(comp)
        x=d.mean_difference.to_numpy();l=d.ci_low.to_numpy();h=d.ci_high.to_numpy()
        ax.errorbar(x,np.arange(len(comp)),xerr=np.vstack([x-l,h-x]),fmt='o',capsize=3,color='#345d7d')
        ax.axvline(0,color='black',ls='--',lw=1);ax.set_title(cohort)
        ax.set_yticks(np.arange(len(comp)),[LABELS[m] for m in comp]);ax.invert_yaxis()
        ax.set_xlabel('Gated − competitor MSE');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();savefig(fig,out/'Supplement_six_model_paired_MSE',args.formats)
    print('\n=== GATED VS MLP-FUSION (MSE) ===')
    print(paired[(paired.model_b=='mlp_fusion')&(paired.metric=='mse')].to_string(index=False))
    print('\n=== GATED VS RIDGE-FUSION (MSE) ===')
    print(paired[(paired.model_b=='ridge_fusion')&(paired.metric=='mse')].to_string(index=False))
    print(f'\n[COMPLETE] {out}',flush=True)

if __name__=='__main__':
    main()
