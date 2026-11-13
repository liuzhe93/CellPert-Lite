#!/usr/bin/env python3
"""Publication Figure 3: frozen K562 -> RPE1 transfer, no retraining.
Step10A (four direct models) supplies A-C; Step08D supplies an optional
independent descriptive biological-agreement panel D (different protocol).
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODELS=['gated_direct','no_gate_direct','genept_only_direct','uce_only_direct']
LABELS=['Gated fusion','No-gate fusion','GenePT only','UCE only']
COLORS=['#176B82','#667D99','#CA8D38','#7A65A5']

def pcc_rows(a,b):
    x=a.astype('float64'); y=b.astype('float64')
    x=x-x.mean(axis=1,keepdims=True); y=y-y.mean(axis=1,keepdims=True)
    den=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
    return np.divide(np.sum(x*y,axis=1),den,out=np.full(len(x),np.nan),where=den>1e-12)

def read_npz(p):
    if not p.is_file(): raise FileNotFoundError(f'Missing predictions: {p}')
    with np.load(p,allow_pickle=False) as z:
        for k in ('perturbations','genes','y_true','prediction'):
            if k not in z: raise ValueError(f'{p}: missing {k}')
        return {k:z[k] for k in ('perturbations','genes','y_true','prediction')}

def bootstrap_target(df, boot, rng):
    # Equal weight to unique target; seed repetitions stay clustered.
    x=df.groupby('perturbation')['diff'].mean().dropna().to_numpy()
    if not len(x):return (np.nan,np.nan,np.nan,0)
    draws=np.empty(boot)
    for i in range(boot):draws[i]=np.mean(x[rng.integers(0,len(x),len(x))])
    return float(x.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975)),len(x)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--step10a',default='results/step10A_unseen_perturbation_transfer')
    ap.add_argument('--step08d',default='results/step08D_transfer_analysis')
    ap.add_argument('--outdir',default='results/step13H_figure3_rpe1_publication')
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    ap.add_argument('--thresholds',type=int,nargs='+',default=[3,5,10])
    ap.add_argument('--bootstrap',type=int,default=3000)
    ap.add_argument('--comparison-threshold',type=int,default=3)
    args=ap.parse_args()
    if args.comparison_threshold not in args.thresholds:ap.error('comparison-threshold must be in thresholds')
    root=Path(args.root).resolve(); src=root/args.step10a; out=root/args.outdir
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; audit=[]; target_sets={}
    for seed in args.seeds:
        for threshold in args.thresholds:
            baseline=None
            for model in MODELS:
                p=src/f'seed_{seed}'/model/f'RPE1_batch{threshold}'/'predictions.npz'
                z=read_npz(p)
                perts=z['perturbations'].astype(str); genes=z['genes'].astype(str)
                y=np.asarray(z['y_true']); pred=np.asarray(z['prediction'])
                if len(perts)!=len(set(perts)):raise ValueError(f'Duplicate perturbations: {p}')
                if len(genes)!=len(set(genes)):raise ValueError(f'Duplicate genes: {p}')
                if y.shape!=pred.shape or y.shape!=(len(perts),len(genes)):raise ValueError(f'Shape mismatch: {p}')
                if not np.isfinite(y).all() or not np.isfinite(pred).all():raise ValueError(f'Nonfinite data: {p}')
                if baseline is None:baseline=(perts,genes,y)
                else:
                    bp,bg,by=baseline
                    if not np.array_equal(perts,bp) or not np.array_equal(genes,bg) or not np.allclose(y,by,rtol=0,atol=1e-7):
                        raise ValueError(f'Unmatched test set or truth across models: seed={seed}, threshold={threshold}')
                pcc=pcc_rows(y,pred); mse=np.mean((y.astype('float64')-pred.astype('float64'))**2,axis=1)
                for t,r,e in zip(perts,pcc,mse):
                    rows.append({'seed':seed,'threshold':threshold,'model':model,'perturbation':t,'pcc':r,'mse':e})
                audit.append({'seed':seed,'threshold':threshold,'model':model,'n_targets':len(perts),'n_genes':len(genes),'source':str(p)})
            target_sets[(seed,threshold)]=set(baseline[0])
    df=pd.DataFrame(rows)
    df.to_csv(out/'per_target_metrics.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'input_audit.csv',index=False)
    per_seed=df.groupby(['seed','threshold','model'],as_index=False).agg(mean_pcc=('pcc','mean'),mean_mse=('mse','mean'),n_targets=('perturbation','nunique'))
    per_seed.to_csv(out/'per_seed_metrics.csv',index=False)
    coverage=per_seed[per_seed.model==MODELS[0]][['seed','threshold','n_targets']]
    coverage.to_csv(out/'retained_targets_by_seed.csv',index=False)
    common={seed:set.intersection(*(target_sets[(seed,th)] for th in args.thresholds)) for seed in args.seeds}
    matched=df[df.apply(lambda r:r.perturbation in common[r.seed],axis=1)].copy()
    matched.to_csv(out/'matched_targets_per_target_metrics.csv',index=False)
    matched_seed=matched.groupby(['seed','threshold','model'],as_index=False).agg(mean_pcc=('pcc','mean'),n_targets=('perturbation','nunique'))
    matched_seed.to_csv(out/'matched_targets_per_seed.csv',index=False)
    rng=np.random.default_rng(42); paired=[]
    q=df[df.threshold==args.comparison_threshold]
    for other in MODELS[1:]:
        for metric in ('pcc','mse'):
            wide=q.pivot(index=['seed','perturbation'],columns='model',values=metric)
            if metric=='pcc':d=(wide['gated_direct']-wide[other]).dropna()
            else:d=(wide[other]-wide['gated_direct']).dropna()
            z=d.rename('diff').reset_index()
            est,lo,hi,n=bootstrap_target(z,args.bootstrap,rng)
            paired.append({'threshold':args.comparison_threshold,'comparison':f'Gated vs {other}',
                           'metric':metric,'direction':'positive_favors_gated','estimate':est,
                           'ci95_low':lo,'ci95_high':hi,'n_unique_targets':n,
                           'n_seed_target_pairs':len(z)})
    paired=pd.DataFrame(paired);paired.to_csv(out/'paired_target_bootstrap.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.linewidth':0.8,'savefig.facecolor':'white'})
    fig,axs=plt.subplots(2,2,figsize=(12.6,8.4))
    fig.subplots_adjust(left=.075,right=.975,top=.91,bottom=.12,wspace=.31,hspace=.43)
    a,b,c,d=axs.flat
    offsets=np.linspace(-.17,.17,len(MODELS))
    # A: all eligible targets, seed-level means
    for j,(m,label,col) in enumerate(zip(MODELS,LABELS,COLORS)):
        z=per_seed[per_seed.model==m]
        for th in args.thresholds:
            x=z[z.threshold==th].mean_pcc.to_numpy()
            a.scatter(np.full(len(x),th+offsets[j]),x,color=col,s=18,alpha=.48,zorder=3,edgecolor='none')
        means=z.groupby('threshold').mean_pcc.mean().reindex(args.thresholds)
        a.plot(np.array(args.thresholds)+offsets[j],means,'-o',color=col,lw=1.8,ms=5,label=label)
    a.set_xticks(args.thresholds);a.set_xlabel('Minimum perturbed cells per batch')
    a.set_ylabel('Mean per-target Pearson r');a.set_title('A   External transfer performance',loc='left',weight='bold',pad=12)
    a.grid(axis='y',alpha=.13);a.legend(frameon=False,fontsize=8,ncol=2,loc='upper right')
    # B: counts and matched set
    for seed in args.seeds:
        z=coverage[coverage.seed==seed].sort_values('threshold')
        b.plot(z.threshold,z.n_targets,'o-',color='#A2ACB5',lw=.85,alpha=.55,ms=3)
    avg=coverage.groupby('threshold').n_targets.mean().reindex(args.thresholds)
    b.plot(args.thresholds,avg,'-o',color=COLORS[0],lw=2.5,ms=6,label='Mean retained targets')
    matched_counts=np.array([len(common[s]) for s in args.seeds])
    b.axhline(matched_counts.mean(),ls='--',color=COLORS[2],lw=1.5,label='Common targets across thresholds')
    b.set_xticks(args.thresholds);b.set_xlabel('Minimum perturbed cells per batch')
    b.set_ylabel('Number of held-out RPE1 targets');b.set_title('B   Evaluation-set coverage',loc='left',weight='bold',pad=12)
    b.grid(axis='y',alpha=.13);b.legend(frameon=False,fontsize=8,loc='best')
    # C: paired PCC effects
    z=paired[paired.metric=='pcc'].set_index('comparison')
    for i,(m,col) in enumerate(zip(MODELS[1:],COLORS[1:])):
        row=z.loc[f'Gated vs {m}'];y=2-i
        c.plot([row.ci95_low,row.ci95_high],[y,y],color=col,lw=2.4)
        c.plot(row.estimate,y,'o',color=col,ms=6)
    c.axvline(0,color='#666666',lw=1,ls='--')
    c.set_yticks([2,1,0],LABELS[1:]);c.set_xlabel('Δ mean target PCC (gated − comparator)')
    c.set_title(f'C   Paired differences · threshold {args.comparison_threshold}',loc='left',weight='bold',pad=12)
    c.grid(axis='x',alpha=.12)
    # D: matched-set sensitivity, not mixed with changing target compositions
    for j,(m,label,col) in enumerate(zip(MODELS,LABELS,COLORS)):
        z=matched_seed[matched_seed.model==m]
        for th in args.thresholds:
            zz=z[z.threshold==th]
            d.scatter(np.full(len(zz),th+offsets[j]),zz.mean_pcc,color=col,s=14,alpha=.34,edgecolor='none')
        d.plot(np.array(args.thresholds)+offsets[j],z.groupby('threshold').mean_pcc.mean().reindex(args.thresholds),
               '-o',color=col,lw=1.8,ms=4.5,label=label)
    d.set_xticks(args.thresholds);d.set_xlabel('Minimum perturbed cells per batch')
    d.set_ylabel('Mean per-target Pearson r')
    d.set_title('D   Matched-target threshold sensitivity',loc='left',weight='bold',pad=12)
    d.grid(axis='y',alpha=.13)
    fig.suptitle('K562-trained models | Frozen transfer to RPE1',x=.075,y=.98,ha='left',fontsize=17,weight='bold')
    fig.text(.075,.054,'Dots = individual random splits; lines = split means. Panels A and B use threshold-specific target sets.',fontsize=8,color='#555')
    fig.text(.075,.034,'Panel D restricts each seed to targets present at all thresholds. Panel C: target-cluster bootstrap (95% CI).',fontsize=8,color='#555')
    fig.text(.075,.014,'Repeated random splits are not independent biological replicates. RPE1 transfer uses frozen K562-trained weights.',fontsize=8,color='#555')
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'Figure3_RPE1_publication.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    legend=f'''# Figure 3 | Frozen K562-to-RPE1 transfer\n\n(A) Mean per-perturbation Pearson correlation across threshold-specific held-out RPE1 targets for four models; points show individual random seeds. (B) Number of eligible RPE1 held-out perturbations per seed and minimum per-batch cell threshold. Dashed line indicates the average size of each seed's target intersection across thresholds. (C) Paired difference in per-target Pearson correlation between gated fusion and each comparator at threshold {args.comparison_threshold}; positive values favor gated fusion. Each difference is paired by seed and perturbation, then averaged by unique target for 95% percentile bootstrap confidence intervals ({args.bootstrap} resamples). (D) Threshold sensitivity on a fixed target intersection within each seed; this reduces target-composition confounding but does not isolate measurement-noise effects. No external refitting. Five random splits are not independent biological replicates.\n\nSource: Step10A frozen prediction NPZ files, recomputed directly. Step08D uses different models and a different evaluation protocol and is deliberately not pooled into this figure.\n'''
    (out/'Figure3_legend.md').write_text(legend,encoding='utf-8')
    (out/'audit.json').write_text(json.dumps({'seeds':args.seeds,'thresholds':args.thresholds,'comparison_threshold':args.comparison_threshold,'bootstrap':args.bootstrap,'models':MODELS,'source':str(src),'matched_target_counts':{str(k):len(v) for k,v in common.items()},'note':'Step08D not pooled due to protocol/model mismatch'},indent=2),encoding='utf-8')
    print('[DONE]',out)
    print('[PAIRWISE]');print(paired.to_string(index=False))

if __name__=='__main__':main()
