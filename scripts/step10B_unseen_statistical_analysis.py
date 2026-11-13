#!/usr/bin/env python3
"""Step10B: paired, perturbation-clustered inference and Figure 5 for Step10A.
No training. Repeated seeds are averaged within perturbation before bootstrap.
Exploratory CIs, not confirmatory significance tests. Requires numpy pandas matplotlib.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['gated_direct','no_gate_direct','genept_only_direct','uce_only_direct']
LABEL={'gated_direct':'Gated','no_gate_direct':'No-Gate','genept_only_direct':'GenePT-only','uce_only_direct':'UCE-only'}

def pcc_rows(y,p):
    a=y.astype('float64')-y.mean(axis=1,keepdims=True)
    b=p.astype('float64')-p.mean(axis=1,keepdims=True)
    d=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return np.divide(np.sum(a*b,axis=1),d,out=np.full(len(y),np.nan),where=d>1e-12)

def load(path):
    if not path.is_file():raise FileNotFoundError(path)
    with np.load(path,allow_pickle=False) as z:
        for k in ('perturbations','genes','y_true','prediction'):
            if k not in z:raise ValueError(f'{path}: missing {k}')
        ids=z['perturbations'].astype(str);genes=z['genes'].astype(str)
        y=z['y_true'].astype('float64');p=z['prediction'].astype('float64')
    if len(set(ids))!=len(ids) or len(set(genes))!=len(genes):raise ValueError(f'Duplicate labels: {path}')
    if y.shape!=p.shape or y.shape!=(len(ids),len(genes)) or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError(f'Bad matrices: {path}')
    return ids,genes,y,p

def bootstrap_cluster(d, rng, reps):
    # One observation per unique perturbation. Seeds are repeated measurements.
    v=d.to_numpy(dtype=float);v=v[np.isfinite(v)]
    if len(v)<3:return (float(np.mean(v)) if len(v) else np.nan,np.nan,np.nan)
    vals=np.empty(reps)
    for k in range(reps):vals[k]=np.mean(v[rng.integers(0,len(v),size=len(v))])
    return float(np.mean(v)),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def savefig(fig,base,formats):
    for fmt in formats:fig.savefig(str(base)+'.'+fmt,bbox_inches='tight',dpi=250)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--step10a',default='results/step10A_unseen_perturbation_transfer')
    ap.add_argument('--outdir',default='results/step10B_unseen_statistical_analysis')
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    ap.add_argument('--thresholds',nargs='+',type=int,default=[3,5,10])
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--rng-seed',type=int,default=20261109)
    ap.add_argument('--formats',nargs='+',choices=['svg','pdf','png'],default=['svg','pdf'])
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.bootstrap<100 or len(set(args.seeds))!=len(args.seeds) or len(set(args.thresholds))!=len(args.thresholds):raise ValueError('Invalid args')
    root=Path(args.root).resolve();base=root/args.step10a;out=root/args.outdir
    if not base.exists():raise FileNotFoundError(base)
    metrics=[];audit=[];true_map={};cross=[]
    for seed in args.seeds:
        sd=base/f'seed_{seed}'
        sf=sd/'split.json'
        if not sf.exists():raise FileNotFoundError(sf)
        split=json.loads(sf.read_text());held=set(map(str,split['test']));train=set(map(str,split['train']))
        if held&train:raise RuntimeError(f'Leakage seed {seed}')
        for th in [0]+args.thresholds:
            dataset='K562_unseen' if th==0 else 'RPE1_unseen_transfer'
            loaded={}
            for model in MODELS:
                path=(sd/model/'K562_holdout_predictions.npz') if th==0 else (sd/model/f'RPE1_batch{th}'/'predictions.npz')
                ids,genes,y,p=load(path)
                if not set(ids).issubset(held):raise RuntimeError(f'Non-heldout IDs: {path}')
                if th==0 and set(ids)!=held:raise RuntimeError(f'K562 incomplete holdout: {path}')
                if loaded:
                    ri,rg,ry,_=loaded[MODELS[0]]
                    if not np.array_equal(ids,ri) or not np.array_equal(genes,rg) or not np.array_equal(y,ry):raise RuntimeError(f'Alignment/truth mismatch: {path}')
                loaded[model]=(ids,genes,y,p)
                mse=np.mean((y-p)**2,axis=1);pc=pcc_rows(y,p)
                for j,pid in enumerate(ids):
                    metrics.append({'dataset':dataset,'threshold':th,'seed':seed,'model':model,'perturbation':pid,'mse':float(mse[j]),'pcc':float(pc[j]),'true_rms':float(np.sqrt(np.mean(y[j]**2))), 'n_active_01':int(np.sum(np.abs(y[j])>.1))})
            ids,genes,y,_=loaded[MODELS[0]]
            true_map[(seed,th)]={str(pid):(genes,y[j]) for j,pid in enumerate(ids)}
            audit.append({'seed':seed,'dataset':dataset,'threshold':th,'n_perturbations':len(ids),'n_genes':len(genes),'aligned_models':len(loaded)})
            print(f'[AUDIT] seed={seed} {dataset} batch={th} n={len(ids)} genes={len(genes)}',flush=True)
        for th in args.thresholds:
            km=true_map[(seed,0)];rm=true_map[(seed,th)]
            for pid in sorted(set(km)&set(rm)):
                kg,ky=km[pid];rg,ry=rm[pid];lookup={g:i for i,g in enumerate(kg)}
                if not set(rg).issubset(lookup):raise RuntimeError(f'Gene mismatch seed {seed} batch{th}')
                kk=ky[np.array([lookup[g] for g in rg])]
                r=float(pcc_rows(kk[None,:],ry[None,:])[0])
                cross.append({'seed':seed,'threshold':th,'perturbation':pid,'cross_cell_true_pcc':r,
                              'k562_true_rms_shared':float(np.sqrt(np.mean(kk**2))),
                              'rpe1_true_rms':float(np.sqrt(np.mean(ry**2)))})
    if args.dry_run:
        print('[DRY RUN PASS] All predictions aligned, no files written');return
    out.mkdir(parents=True,exist_ok=True)
    m=pd.DataFrame(metrics);m.to_csv(out/'per_perturbation_metrics.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'alignment_audit.csv',index=False)
    c=pd.DataFrame(cross);c.to_csv(out/'cross_cell_truth_metrics.csv',index=False)
    # Difference A-B. For MSE negative favors A; for PCC positive favors A.
    pairs=[];rng=np.random.default_rng(args.rng_seed)
    for (dataset,th),grp in m.groupby(['dataset','threshold'],sort=True):
        for competitor in MODELS[1:]:
            for metric in ('mse','pcc'):
                pivot=grp.pivot(index=['seed','perturbation'],columns='model',values=metric)
                paired=pivot[['gated_direct',competitor]].dropna()
                if len(paired)==0:continue
                delta=paired['gated_direct']-paired[competitor]
                unique=delta.groupby(level='perturbation').mean()
                mean,lo,hi=bootstrap_cluster(unique,rng,args.bootstrap)
                fraction=float(np.mean(unique<0)) if metric=='mse' else float(np.mean(unique>0))
                pairs.append({'dataset':dataset,'threshold':th,'model_a':'gated_direct','model_b':competitor,'metric':metric,
                              'mean_difference':mean,'ci_low':lo,'ci_high':hi,'n_unique_perturbations':len(unique),
                              'n_seed_perturbation_pairs':len(delta),'fraction_gate_better':fraction})
    pair=pd.DataFrame(pairs);pair.to_csv(out/'paired_bootstrap_summary.csv',index=False)
    # K562 and RPE1: compare MSE only on common response genes, not raw K562 8563 vs RPE1 7226.
    shift=[]
    for seed in args.seeds:
        for th in args.thresholds:
            for model in MODELS:
                kd=base/f'seed_{seed}'/model/'K562_holdout_predictions.npz'
                rd=base/f'seed_{seed}'/model/f'RPE1_batch{th}'/'predictions.npz'
                ki,kg,ky,kp=load(kd);ri,rg,ry,rp=load(rd)
                ki_map={v:i for i,v in enumerate(ki)};kg_map={v:i for i,v in enumerate(kg)}
                jj=np.array([kg_map[v] for v in rg]);
                for j,pid in enumerate(ri):
                    k=ki_map[pid];errk=float(np.mean((ky[k,jj]-kp[k,jj])**2));errr=float(np.mean((ry[j]-rp[j])**2))
                    shift.append({'seed':seed,'threshold':th,'model':model,'perturbation':pid,'k562_mse_shared_genes':errk,'rpe1_mse':errr,'mse_shift_rpe1_minus_k562':errr-errk})
    sh=pd.DataFrame(shift);sh.to_csv(out/'cross_cell_error_shift.csv',index=False)
    shiftrows=[]
    for (th,model),g in sh.groupby(['threshold','model']):
        unique=g.groupby('perturbation')['mse_shift_rpe1_minus_k562'].mean()
        mean,lo,hi=bootstrap_cluster(unique,rng,args.bootstrap)
        shiftrows.append({'threshold':th,'model':model,'mean_shift':mean,'ci_low':lo,'ci_high':hi,'n_unique_perturbations':len(unique)})
    pd.DataFrame(shiftrows).to_csv(out/'cross_cell_error_shift_summary.csv',index=False)
    # Exploratory stratification using RPE1 truth, not a deployable input feature.
    sub=[]
    for th in args.thresholds:
        g=m[(m.dataset=='RPE1_unseen_transfer')&(m.threshold==th)]
        w=g.pivot(index=['seed','perturbation'],columns='model',values='mse')
        dif=(w['gated_direct']-w['no_gate_direct']).groupby(level='perturbation').mean()
        info=g[g.model=='gated_direct'].groupby('perturbation')[['true_rms','n_active_01']].mean()
        cx=c[c.threshold==th].groupby('perturbation')['cross_cell_true_pcc'].mean()
        info=info.join(cx)
        for feat in ('true_rms','n_active_01','cross_cell_true_pcc'):
            vals=info[feat].dropna()
            if vals.nunique()<3:continue
            # rank-based terciles guarantee near-balanced groups even with ties
            ranks=vals.rank(method='first');bins=pd.qcut(ranks,3,labels=['low','mid','high'])
            for label in ('low','mid','high'):
                ids=bins.index[bins==label];z=dif.reindex(ids).dropna()
                mean,lo,hi=bootstrap_cluster(z,rng,args.bootstrap)
                sub.append({'threshold':th,'feature':feat,'bin':label,'n_unique_perturbations':len(z),
                            'gate_minus_no_gate_mse':mean,'ci_low':lo,'ci_high':hi})
    subdf=pd.DataFrame(sub);subdf.to_csv(out/'subgroup_bootstrap_summary.csv',index=False)
    # Figure 5B: K562 performance across seeds
    summ=pd.read_csv(base/'metrics_per_seed.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,col,lab in zip(axes,['mse','global_pcc'],['MSE (lower is better)','Global PCC (higher is better)']):
        d=summ[summ.dataset=='K562_unseen']
        for i,model in enumerate(MODELS):
            z=d[d.model==model][col].to_numpy()
            ax.scatter(np.full(len(z),i),z,alpha=.65,s=23)
            ax.errorbar(i,z.mean(),yerr=z.std(ddof=1),fmt='o',capsize=4,color='black')
        ax.set_xticks(range(len(MODELS)),[LABEL[x] for x in MODELS],rotation=20,ha='right');ax.set_ylabel(lab);ax.grid(axis='y',alpha=.2)
    fig.suptitle('K562: unseen perturbations (five splits)');fig.tight_layout();savefig(fig,out/'Figure5B_K562_unseen',args.formats)
    # Figure 5C: RPE1 performance by threshold
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,col,lab in zip(axes,['mse','global_pcc'],['MSE','Global PCC']):
        for model in MODELS:
            z=summ[(summ.dataset=='RPE1_unseen_transfer')&(summ.model==model)].groupby('threshold')[col].agg(['mean','std']).reindex(args.thresholds)
            ax.errorbar(args.thresholds,z['mean'],yerr=z['std'],marker='o',capsize=3,label=LABEL[model])
        ax.set_xlabel('RPE1 batch threshold');ax.set_ylabel(lab);ax.set_xticks(args.thresholds);ax.grid(alpha=.2)
    axes[1].legend(frameon=False,fontsize=8);fig.suptitle('RPE1: unseen-perturbation transfer');fig.tight_layout();savefig(fig,out/'Figure5C_RPE1_unseen',args.formats)
    # Figure 5D: paired bootstrap confidence intervals
    d=pair[(pair.model_b=='no_gate_direct')&(pair.metric=='mse')].sort_values(['dataset','threshold'])
    fig,ax=plt.subplots(figsize=(7,4));ypos=np.arange(len(d))
    x=d.mean_difference.to_numpy();lo=d.ci_low.to_numpy();hi=d.ci_high.to_numpy()
    ax.errorbar(x,ypos,xerr=np.vstack([x-lo,hi-x]),fmt='o',capsize=4)
    ax.axvline(0,color='black',lw=1,ls='--')
    ax.set_yticks(ypos,[('K562' if t==0 else f'RPE1 batch{t}') for t in d.threshold]);ax.invert_yaxis()
    ax.set_xlabel('Gated - No-Gate per-perturbation MSE (95% cluster bootstrap CI)');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();savefig(fig,out/'Figure5D_paired_gate_benefit',args.formats)
    # Figure 5E: response-strength stratification (batch3 and batch5 emphasized)
    d=subdf[(subdf.feature=='true_rms')&(subdf.threshold.isin([3,5]))]
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,th in zip(axes,[3,5]):
        t=d[d.threshold==th].set_index('bin').reindex(['low','mid','high'])
        x=np.arange(3);v=t.gate_minus_no_gate_mse.to_numpy();lo=t.ci_low.to_numpy();hi=t.ci_high.to_numpy()
        ax.errorbar(x,v,yerr=np.vstack([v-lo,hi-v]),fmt='o',capsize=4)
        ax.axhline(0,color='black',lw=1,ls='--');ax.set_xticks(x,['Low','Mid','High']);ax.set_title(f'RPE1 batch{th}')
        ax.set_xlabel('True response RMS tertile');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Gated - No-Gate MSE (95% CI)');fig.tight_layout();savefig(fig,out/'Figure5E_response_strength',args.formats)
    print('\n=== PRIMARY PAIRED GATE VS NO-GATE ===')
    print(pair[(pair.model_b=='no_gate_direct')].to_string(index=False))
    print(f'\n[COMPLETE] {out}',flush=True)

if __name__=='__main__':main()
