#!/usr/bin/env python3
"""Step11B-5: paired descriptive analysis of frozen external-transfer predictions.
No training, model selection or biological-replicate significance claims.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['gated_direct','no_gate_direct','genept_only_direct','uce_only_direct']
LABELS={'gated_direct':'Gated fusion','no_gate_direct':'No-gate fusion','genept_only_direct':'GenePT only','uce_only_direct':'UCE only'}

def bootstrap_diff(paired,nboot,seed):
    # A perturbation is the cluster; seeds stay together within perturbation.
    v=paired.groupby('perturbation',sort=True)['difference'].mean().dropna().to_numpy(float)
    if len(v)==0:return (np.nan,np.nan,np.nan,0)
    rng=np.random.default_rng(seed)
    means=np.empty(nboot)
    for j in range(nboot):means[j]=v[rng.integers(0,len(v),len(v))].mean()
    return float(v.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975)),len(v)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='results/step11B_4_frozen_external_transfer/per_perturbation_metrics.csv')
    ap.add_argument('--root',default='.')
    ap.add_argument('--outdir',default='results/step11B_5_external_transfer_analysis')
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    if args.bootstrap<100:raise ValueError('--bootstrap must be >=100')
    root=Path(args.root).resolve(); src=root/args.input;out=root/args.outdir;out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(src)
    cols=['dataset','seed','model','perturbation','K562_train_seen','K562_test_heldout','pcc','mse']
    missing=set(cols)-set(d.columns)
    if missing:raise ValueError(f'Missing columns: {missing}')
    if d[cols[:4]].isna().any().any():raise ValueError('Missing row identifiers')
    if d.duplicated(['dataset','seed','model','perturbation']).any():raise ValueError('Duplicate dataset/seed/model/perturbation')
    if (~d.model.isin(MODELS)).any():raise ValueError('Unknown model')
    for k in ['K562_train_seen','K562_test_heldout']:
        if d[k].dtype!='bool':
            if not d[k].astype(str).str.lower().isin(['true','false']).all():raise ValueError(f'Invalid {k}')
            d[k]=d[k].astype(str).str.lower().eq('true')
    if (d.K562_train_seen==d.K562_test_heldout).any():raise ValueError('Seen/heldout flags are not complementary')
    if not np.isfinite(d.mse).all() or (d.mse<0).any():raise ValueError('Invalid MSE')
    if ((d.pcc.dropna().abs()>1+1e-6)).any():raise ValueError('PCC outside [-1,1]')
    d['group']=np.where(d.K562_train_seen,'seen','heldout')
    d['model']=pd.Categorical(d.model,categories=MODELS,ordered=True)
    summary=[]
    for (ds,grp,model,seed),x in d.groupby(['dataset','group','model','seed'],observed=True):
        summary.append(dict(dataset=ds,group=grp,model=model,seed=int(seed),n=len(x),valid_pcc=int(x.pcc.notna().sum()),mean_pcc=x.pcc.mean(),median_pcc=x.pcc.median(),mean_mse=x.mse.mean(),median_mse=x.mse.median()))
    s=pd.DataFrame(summary)
    s.to_csv(out/'metrics_per_seed_recomputed.csv',index=False)
    agg=s.groupby(['dataset','group','model'],observed=True).agg(n_seeds=('seed','nunique'),mean_pcc=('mean_pcc','mean'),sd_seed_pcc=('mean_pcc','std'),mean_mse=('mean_mse','mean'),sd_seed_mse=('mean_mse','std'),mean_n=('n','mean')).reset_index()
    agg.to_csv(out/'model_group_summary.csv',index=False)
    # Exact paired rows, then cluster bootstrap over perturbations, retaining all seeds.
    records=[];paired_all=[]
    for ds in sorted(d.dataset.unique()):
      for group in ['seen','heldout']:
        q=d[(d.dataset==ds)&(d.group==group)]
        for metric in ['pcc','mse']:
          piv=q.pivot(index=['seed','perturbation'],columns='model',values=metric)
          for competitor in MODELS[1:]:
            pair=piv[['gated_direct',competitor]].dropna().reset_index()
            pair['difference']=pair['gated_direct']-pair[competitor]
            est,lo,hi,n=bootstrap_diff(pair,args.bootstrap,args.seed)
            records.append(dict(dataset=ds,group=group,metric=metric,model_a='gated_direct',model_b=competitor,
                                difference_definition='a_minus_b',mean_paired_difference=est,ci95_low=lo,ci95_high=hi,
                                n_unique_perturbations=n,n_seed_perturbation_pairs=len(pair),
                                descriptive_only=True))
            if metric=='pcc':
                pair['dataset']=ds;pair['group']=group;pair['competitor']=competitor
                paired_all.append(pair[['dataset','group','seed','perturbation','competitor','difference']])
    pd.DataFrame(records).to_csv(out/'paired_model_differences.csv',index=False)
    pd.concat(paired_all,ignore_index=True).to_csv(out/'paired_pcc_per_perturbation.csv',index=False)
    # Same seed and perturbation across seen and heldout is impossible; report differences descriptively.
    gap=[]
    for (ds,model,seed),q in s.groupby(['dataset','model','seed'],observed=True):
        if set(q.group)=={'seen','heldout'}:
            a=q.set_index('group')
            gap.append(dict(dataset=ds,model=model,seed=int(seed),seen_mean_pcc=float(a.loc['seen','mean_pcc']),heldout_mean_pcc=float(a.loc['heldout','mean_pcc']),heldout_minus_seen_pcc=float(a.loc['heldout','mean_pcc']-a.loc['seen','mean_pcc'])))
    pd.DataFrame(gap).to_csv(out/'seen_heldout_gap_per_seed.csv',index=False)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    for ds in sorted(d.dataset.unique()):
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for ax,group in zip(axes,['seen','heldout']):
            q=s[(s.dataset==ds)&(s.group==group)]
            for i,m in enumerate(MODELS):
                a=q[q.model==m].sort_values('seed')
                ax.scatter(np.full(len(a),i)+np.linspace(-.08,.08,len(a)),a.mean_pcc,s=18,alpha=.8)
                ax.hlines(a.mean_pcc.mean(),i-.25,i+.25,linewidth=2)
            ax.set_xticks(range(4),[LABELS[x] for x in MODELS],rotation=30,ha='right')
            ax.set_title(f'{ds} | {group} | per-seed mean PCC')
            ax.axhline(0,color='gray',lw=.7,ls='--')
        axes[0].set_ylabel('Mean per-perturbation PCC')
        fig.tight_layout()
        for ext in ['svg','pdf','png']:fig.savefig(out/f'Figure4_{ds}_seen_heldout_models.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,3.7))
        for ax,group in zip(axes,['seen','heldout']):
            q=pd.DataFrame(records)
            q=q[(q.dataset==ds)&(q.group==group)&(q.metric=='pcc')]
            ys=np.arange(len(q))
            for i,(_,r) in enumerate(q.iterrows()):
                ax.plot([r.ci95_low,r.ci95_high],[i,i],lw=2)
                ax.scatter([r.mean_paired_difference],[i],s=32)
            ax.axvline(0,color='gray',ls='--',lw=1)
            ax.set_yticks(ys,[f'Gated - {LABELS[m]}' for m in q.model_b])
            ax.set_title(f'{ds} | {group}')
            ax.set_xlabel('Paired PCC difference (perturbation-cluster 95% CI)')
        fig.tight_layout()
        for ext in ['svg','pdf','png']:fig.savefig(out/f'Figure5_{ds}_paired_pcc.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    manifest={'input':str(src),'n_rows':len(d),'datasets':sorted(d.dataset.unique().tolist()),'seeds':sorted(d.seed.unique().astype(int).tolist()),
              'models':MODELS,'bootstrap':args.bootstrap,'cluster_unit':'perturbation; seeds kept together',
              'caveats':['Intervals are descriptive perturbation-cluster bootstrap, not biological replicate uncertainty.',
                         'Seen vs heldout are different perturbation sets; gap is not a paired causal comparison.',
                         'No response-strength stratification or zero-baseline evaluation without raw response matrices.',
                         'No model was retrained or tuned.']}
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(agg.to_string(index=False));print('\n[PAIRED]');print(pd.DataFrame(records).to_string(index=False));print('\nDONE:',out)
if __name__=='__main__':main()
