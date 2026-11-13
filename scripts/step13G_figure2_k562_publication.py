#!/usr/bin/env python3
"""Publication Figure 2: K562 held-out gene perturbation prediction.
Recompute metrics from Step10A frozen prediction NPZs, no model fitting.
Panel D: cluster bootstrap over unique perturbation IDs, preserving all seeds.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODELS = ['gated_direct', 'no_gate_direct', 'genept_only_direct', 'uce_only_direct']
LABEL = {'gated_direct':'Gated fusion','no_gate_direct':'No-gate fusion',
         'genept_only_direct':'GenePT only','uce_only_direct':'UCE only'}
COLORS = {'gated_direct':'#187B83','no_gate_direct':'#7A83A7',
          'genept_only_direct':'#C2864B','uce_only_direct':'#8A6D9F'}

def correlation_rows(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(a*a, axis=1) * np.sum(b*b, axis=1))
    result = np.full(a.shape[0], np.nan)
    np.divide(np.sum(a*b, axis=1), denom, out=result, where=denom>1e-14)
    return result

def global_pcc(a, b):
    x = a.ravel(); y = b.ravel()
    x = x-x.mean(); y = y-y.mean()
    den = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x.dot(y)/den) if den > 1e-14 else np.nan

def load_predictions(path):
    with np.load(path, allow_pickle=False) as z:
        required={'perturbations','genes','y_true','prediction'}
        if not required.issubset(z.files):
            raise ValueError(f'{path}: missing {required-set(z.files)}')
        ids=z['perturbations'].astype(str)
        genes=z['genes'].astype(str)
        y=np.asarray(z['y_true'], dtype=np.float64)
        pred=np.asarray(z['prediction'], dtype=np.float64)
    if y.ndim!=2 or y.shape!=pred.shape or y.shape!=(len(ids),len(genes)):
        raise ValueError(f'{path}: inconsistent array shapes')
    if len(set(ids))!=len(ids) or len(set(genes))!=len(genes):
        raise ValueError(f'{path}: duplicate perturbation/gene IDs')
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError(f'{path}: non-finite predictions/truth')
    return ids,genes,y,pred

def bootstrap_target_cluster(per, nboot, seed):
    # Pair on seed and perturbation. Cluster at perturbation ID so repeats
    # across seeds are never treated as independent biological observations.
    rng=np.random.default_rng(seed)
    rows=[]
    base=per[per.model==MODELS[0]][['seed','perturbation','mse']].rename(columns={'mse':'base_mse'})
    for other in MODELS[1:]:
        cmp=per[per.model==other][['seed','perturbation','mse']].rename(columns={'mse':'other_mse'})
        matched=base.merge(cmp,on=['seed','perturbation'],how='inner',validate='one_to_one')
        if len(matched)!=len(base) or len(matched)!=len(cmp):
            raise ValueError(f'Incomplete paired comparison for {other}')
        matched['improvement']=matched.other_mse-matched.base_mse  # >0: gated lower MSE
        by_target=matched.groupby('perturbation',sort=True).improvement.agg(['sum','count'])
        totals=by_target['sum'].to_numpy(); counts=by_target['count'].to_numpy()
        n=len(by_target)
        boot=np.empty(nboot)
        for k in range(nboot):
            draw=rng.integers(0,n,n)
            boot[k]=totals[draw].sum()/counts[draw].sum()
        lo,hi=np.quantile(boot,[0.025,0.975])
        rows.append({'comparison':f'Gated vs {LABEL[other]}','other_model':other,
                     'mse_improvement':matched.improvement.mean(),
                     'ci_low':lo,'ci_high':hi,'n_unique_targets':n,
                     'n_seed_target_pairs':len(matched),'n_bootstrap':nboot,
                     'ci_excludes_zero':bool(lo>0 or hi<0)})
    return pd.DataFrame(rows)

def make_figure(per, seeds, paired, out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.titlesize':12,'axes.labelsize':10,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(13.2,9.3),constrained_layout=True)
    fig.suptitle('K562 | Generalization to held-out gene perturbations',
                 fontsize=16,fontweight='bold',x=.05,ha='left')
    specs=[('mse','A','Mean squared error','Lower is better'),
           ('global_pcc','B','Global Pearson correlation','Higher is better'),
           ('mean_perturbation_pcc','C','Mean perturbation Pearson correlation','Higher is better')]
    for ax,(metric,panel,title,subtitle) in zip([axs[0,0],axs[0,1],axs[1,0]],specs):
        for i,m in enumerate(MODELS):
            vals=seeds.loc[seeds.model==m].sort_values('seed')[metric].to_numpy(dtype=float)
            ax.scatter(np.full(len(vals),i),vals,s=42,color=COLORS[m],alpha=.72,zorder=3,
                       edgecolor='white',linewidth=.6)
            mu=np.nanmean(vals)
            sd=np.nanstd(vals,ddof=1) if len(vals)>1 else 0
            ax.errorbar(i,mu,yerr=sd,fmt='_',markersize=16,capsize=5,
                        color='#182A3B',linewidth=1.7,zorder=4)
        ax.set_xticks(range(4),[LABEL[m] for m in MODELS],rotation=20,ha='right')
        ax.set_ylabel(title)
        ax.set_title(f'{panel}   {title}',loc='left',fontweight='bold',pad=13)
        ax.text(1,1.01,subtitle,transform=ax.transAxes,ha='right',va='bottom',fontsize=9,color='#667483')
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
        if metric=='mse':ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3))
    ax=axs[1,1]
    y=np.arange(len(paired))[::-1]
    for i,(_,r) in enumerate(paired.iterrows()):
        ax.errorbar(r.mse_improvement,y[i],xerr=[[r.mse_improvement-r.ci_low],[r.ci_high-r.mse_improvement]],
                    fmt='o',markersize=7,color=COLORS[r.other_model],ecolor=COLORS[r.other_model],
                    elinewidth=2,capsize=4)
    ax.axvline(0,color='#7E8790',linestyle='--',linewidth=1)
    ax.set_yticks(y,[f'vs {LABEL[m]}' for m in paired.other_model])
    ax.set_xlabel('MSE(other) − MSE(gated)')
    ax.set_title('D   Paired improvement of gated fusion',loc='left',fontweight='bold',pad=13)
    ax.text(1,1.01,'Positive favors gated fusion',transform=ax.transAxes,
            ha='right',va='bottom',fontsize=9,color='#667483')
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,-3))
    fig.text(.05,.012,'Dots: individual random seeds; dark bars: mean ± SD. '
             'Panel D: 95% percentile bootstrap CI clustered by unique perturbation target. '
             'Repeated splits are not biological replicates.',fontsize=9,color='#576675')
    for ext in ['svg','pdf','png']:
        fig.savefig(out/f'Figure2_K562_publication.{ext}',dpi=350,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.')
    p.add_argument('--step10a',default='results/step10A_unseen_perturbation_transfer')
    p.add_argument('--outdir',default='results/step13G_figure2_k562_publication')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    p.add_argument('--bootstrap',type=int,default=3000)
    p.add_argument('--bootstrap-seed',type=int,default=2026)
    args=p.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or args.bootstrap<100:
        p.error('Seeds must be unique; bootstrap >=100')
    root=Path(args.root).resolve()
    inp=Path(args.step10a);inp=inp if inp.is_absolute() else root/inp
    out=Path(args.outdir);out=out if out.is_absolute() else root/out
    out.mkdir(parents=True,exist_ok=True)
    missing=[str(inp/f'seed_{s}'/m/'K562_holdout_predictions.npz') for s in args.seeds for m in MODELS
             if not (inp/f'seed_{s}'/m/'K562_holdout_predictions.npz').is_file()]
    if missing:raise FileNotFoundError('Missing prediction NPZs:\n'+'\n'.join(missing))
    seed_rows=[];target_rows=[];audit=[];reference_genes=None
    for s in args.seeds:
        reference=None
        for m in MODELS:
            path=inp/f'seed_{s}'/m/'K562_holdout_predictions.npz'
            ids,genes,y,pred=load_predictions(path)
            if reference_genes is None:reference_genes=genes
            if not np.array_equal(genes,reference_genes):
                raise ValueError(f'{path}: response gene order differs')
            if reference is None:reference=(ids,y)
            elif not np.array_equal(ids,reference[0]) or not np.array_equal(y,reference[1]):
                raise ValueError(f'{path}: held-out target IDs/order or ground truth differs')
            row_mse=np.mean((pred-y)**2,axis=1)
            row_pcc=correlation_rows(y,pred)
            seed_rows.append({'seed':s,'model':m,'n_perturbations':len(ids),'n_genes':len(genes),
                              'mse':float(row_mse.mean()),'global_pcc':global_pcc(y,pred),
                              'mean_perturbation_pcc':float(np.nanmean(row_pcc)),
                              'n_valid_target_pcc':int(np.isfinite(row_pcc).sum())})
            target_rows.extend({'seed':s,'model':m,'perturbation':target,'mse':float(error),
                                'pcc':float(corr)} for target,error,corr in zip(ids,row_mse,row_pcc))
            audit.append({'seed':s,'model':m,'file':str(path),'n_perturbations':len(ids),'n_genes':len(genes)})
    per=pd.DataFrame(target_rows);seeds=pd.DataFrame(seed_rows)
    paired=bootstrap_target_cluster(per,args.bootstrap,args.bootstrap_seed)
    per.to_csv(out/'per_target_metrics.csv',index=False)
    seeds.to_csv(out/'metrics_per_seed_recomputed.csv',index=False)
    summary=seeds.groupby('model',sort=False).agg(n_seeds=('seed','nunique'),
             mse_mean=('mse','mean'),mse_sd=('mse','std'),
             global_pcc_mean=('global_pcc','mean'),global_pcc_sd=('global_pcc','std'),
             mean_target_pcc_mean=('mean_perturbation_pcc','mean'),
             mean_target_pcc_sd=('mean_perturbation_pcc','std')).reset_index()
    summary.to_csv(out/'metrics_summary.csv',index=False)
    paired.to_csv(out/'paired_target_cluster_bootstrap.csv',index=False)
    (out/'input_audit.json').write_text(json.dumps({'input':audit,'seeds':args.seeds,
      'bootstrap_unit':'unique perturbation target (all seed observations retained)',
      'pairing':'exact seed + perturbation ID','bootstrap_replicates':args.bootstrap},indent=2))
    make_figure(per,seeds,paired,out)
    legend='''# Figure 2 | Prediction performance on held-out K562 perturbations\n\n'''+\
      '''**A–C**, MSE, global Pearson correlation and mean per-perturbation Pearson correlation across five repeated held-out splits (seeds 42–46). Dots show individual splits and dark bars indicate mean ± standard deviation. **D**, Paired difference in per-target MSE between each comparator and gated fusion, calculated on the same perturbation targets within the same split. Positive differences favor gated fusion. Error bars represent 95% percentile bootstrap intervals from resampling unique perturbation targets with replacement while retaining all available seed observations for each target. These intervals describe target-sampling uncertainty conditional on this dataset and analysis; they are not confidence intervals across independent biological replicates. All results were recomputed from saved Step10A prediction arrays; no models were retrained.\n'''
    (out/'Figure2_legend.md').write_text(legend)
    print('[DONE]',out)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
if __name__=='__main__':main()
