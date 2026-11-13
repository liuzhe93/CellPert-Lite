#!/usr/bin/env python3
"""Figure 2 benchmark, strictly from five-seed metrics. CPU only, no model training."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS=[42,43,44,45,46]
METRICS=['mse','global_pcc_delta','mean_perturbation_pcc','deg_recovery_top20','deg_recovery_top50','deg_recovery_top100']
LABELS={
 'No-change':'No-change','Train-mean':'Train mean',
 'Ridge | GenePT-ada':'Ridge · GenePT-ada',
 'MLP | GenePT-ada':'MLP · GenePT-ada',
 'Ridge | GenePT-3large':'Ridge · GenePT-3large',
 'MLP | GenePT-3large':'MLP · GenePT-3large',
 'Ridge | UCE tokens':'Ridge · UCE tokens',
 'MLP | UCE tokens':'MLP · UCE tokens',
 'Ridge fusion':'Ridge fusion','MLP fusion':'MLP fusion',
 'CCRT OOF':'CCRT (OOF)','CCRT Ridge base':'CCRT Ridge base'}
ORDER=list(LABELS)
COLORS={k:('#D55E00' if k=='CCRT OOF' else '#0072B2' if k=='MLP fusion' else '#666666' if k=='CCRT Ridge base' else '#7A9E9F') for k in ORDER}

def read(path,tag):
 if not path.is_file(): raise FileNotFoundError(f'{tag}: {path}')
 df=pd.read_csv(path)
 needed={'seed','model',*METRICS}
 if tag!='06B': needed.add('representation')
 missing=needed-set(df.columns)
 if missing: raise ValueError(f'{tag} missing columns {sorted(missing)}')
 df['seed']=pd.to_numeric(df.seed,errors='raise').astype(int)
 if set(df.seed)!=set(SEEDS): raise ValueError(f'{tag} seeds {sorted(set(df.seed))} != {SEEDS}')
 return df

def prepare(root):
 a=read(root/'step05_tuned_genept_uce/metrics_all_runs.csv','05')
 b=read(root/'step06A_fusion_baselines/metrics_all_runs.csv','06A')
 c=read(root/'step06B_ccrt_oof_cpu_v3/metrics_all_runs.csv','06B')
 if len(a)!=60 or len(b)!=10 or len(c)!=10: raise ValueError(f'Unexpected row counts Step05={len(a)} Step06A={len(b)} Step06B={len(c)}')
 for tag,df,keys in [('05',a,['seed','representation','model']),('06A',b,['seed','representation','model']),('06B',c,['seed','model'])]:
  if df.duplicated(keys).any(): raise ValueError(f'{tag} genuine duplicated keys: {df.loc[df.duplicated(keys,keep=False),keys].to_string(index=False)}')
  print(f'[PASS] Step{tag}: {len(df)} rows, unique experiment keys')
 # Baseline repeated for 3 representations: verify equality across representations, do not silently discard.
 out=[]
 for (seed,model),g in a.groupby(['seed','model']):
  if model in ('no_change','train_mean'):
   if len(g)!=3: raise ValueError(f'Missing baseline representations for {seed}/{model}')
   values=g[METRICS].astype(float).to_numpy()
   if not np.allclose(values,values[0],rtol=1e-7,atol=1e-10,equal_nan=True):
    raise ValueError(f'Baseline {seed}/{model} differs by representation; cannot collapse safely')
   row=g.iloc[0].copy(); row['label']='No-change' if model=='no_change' else 'Train-mean'; out.append(row)
  else:
   for _,row in g.iterrows():
    rep={'GenePT_ada':'GenePT-ada','GenePT_3large':'GenePT-3large','UCE_input_tokens':'UCE tokens'}.get(row['representation'])
    if rep is None: raise ValueError(f'Unknown Step05 representation {row["representation"]}')
    row=row.copy();row['label']=('Ridge' if model=='ridge_gene' else 'MLP' if model=='mlp_gene' else 'UNKNOWN')+' | '+rep
    if row['label'].startswith('UNKNOWN'): raise ValueError(f'Unknown Step05 model {model}')
    out.append(row)
 for _,row in b.iterrows():
  if row['representation']!='GenePT_3large+UCE_input_tokens': raise ValueError('Unexpected fusion representation')
  row=row.copy();row['label']={'ridge_fusion':'Ridge fusion','mlp_fusion':'MLP fusion'}.get(row['model'])
  if row['label'] is None: raise ValueError('Unexpected Step06A model')
  out.append(row)
 for _,row in c.iterrows():
  row=row.copy();row['label']={'ridge_base':'CCRT Ridge base','ccrt_oof_gated_residual':'CCRT OOF'}.get(row['model'])
  if row['label'] is None: raise ValueError('Unexpected Step06B model')
  out.append(row)
 data=pd.DataFrame(out).reset_index(drop=True)
 if data.duplicated(['seed','label']).any(): raise ValueError('Duplicate seed/label after mapping')
 expected=set(ORDER)
 if set(data.label)!=expected: raise ValueError(f'Model mismatch missing={expected-set(data.label)} extra={set(data.label)-expected}')
 if not (data.groupby('label').seed.nunique()==5).all(): raise ValueError('Some models lack 5 seeds')
 for m in METRICS:
  data[m]=pd.to_numeric(data[m],errors='raise')
  if not np.isfinite(data[m]).all():
   # no-change correlations may be NaN by definition; keep as missing and skip chart points
   if data.loc[~np.isfinite(data[m]),'label'].isin(['No-change']).all():
    print(f'[NOTICE] {m}: undefined for no-change; omitted from metric chart')
   else: raise ValueError(f'Nonfinite {m} in non-baseline model')
 print(f'[PASS] merged {len(data)} rows across {data.label.nunique()} models')
 return data

def draw(data,outdir):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 summary=data.groupby('label')[METRICS].agg(['mean','std','count'])
 summary.columns=['_'.join(x) for x in summary.columns]
 summary=summary.reindex(ORDER)
 summary.to_csv(outdir/'metrics_summary.csv',float_format='%.10g')
 pairs=[]
 for baseline in ['Ridge | GenePT-3large','MLP fusion','CCRT Ridge base']:
  for metric in METRICS:
   pivot=data.pivot(index='seed',columns='label',values=metric)
   for seed in SEEDS:
    x=pivot.loc[seed,'CCRT OOF']; y=pivot.loc[seed,baseline]
    pairs.append({'baseline':baseline,'seed':seed,'metric':metric,'ccrt':x,'baseline_value':y,'delta_ccrt_minus_baseline':x-y})
 pd.DataFrame(pairs).to_csv(outdir/'paired_deltas.csv',index=False)
 fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
 fig.suptitle('Prediction of unseen K562 perturbation responses',fontsize=15,fontweight='bold')
 def metric_panel(ax,metric,title,letter):
  sub=data.pivot(index='seed',columns='label',values=metric).reindex(columns=ORDER)
  means=sub.mean(axis=0,skipna=True); sds=sub.std(axis=0,ddof=1,skipna=True)
  yy=np.arange(len(ORDER))
  for i,key in enumerate(ORDER):
   if not np.isfinite(means[key]): continue
   ax.barh(i,means[key],color=COLORS[key],alpha=.88,height=.68)
   if np.isfinite(sds[key]): ax.errorbar(means[key],i,xerr=sds[key],fmt='none',ecolor='#333333',capsize=2,lw=.8)
   vals=sub[key].dropna().values
   ax.scatter(vals,np.full(len(vals),i)+np.linspace(-.12,.12,len(vals)),s=9,c='#222222',zorder=4)
  ax.set_yticks(yy,[LABELS[k] for k in ORDER],fontsize=7)
  ax.invert_yaxis();ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold')
  ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
  if metric=='mse':ax.set_xlabel('MSE (lower is better)')
  else:ax.set_xlabel('Higher is better')
 metric_panel(axs[0,0],'mse','Mean squared error','A')
 metric_panel(axs[0,1],'global_pcc_delta','Global Pearson correlation','B')
 metric_panel(axs[0,2],'mean_perturbation_pcc','Mean per-perturbation PCC','C')
 ax=axs[1,0]; piv=data.pivot(index='seed',columns='label',values='mse')
 for baseline,style in [('Ridge | GenePT-3large','o-'),('MLP fusion','s-'),('CCRT Ridge base','^-')]:
  delta=(piv['CCRT OOF']-piv[baseline])*1e6
  ax.plot(SEEDS,delta.values,style,label=f'vs {LABELS[baseline]}',lw=1.3,ms=5)
 ax.axhline(0,color='#333333',lw=.8);ax.set_xticks(SEEDS)
 ax.set_title('D  Paired MSE differences',loc='left',fontweight='bold')
 ax.set_ylabel('CCRT − baseline (×10⁻⁶)');ax.set_xlabel('Split seed');ax.legend(fontsize=7,frameon=False);ax.grid(axis='y',alpha=.15)
 ax=axs[1,1]; selected=['Ridge | GenePT-3large','MLP fusion','CCRT OOF']; xs=np.arange(3); w=.24
 for i,m in enumerate(['deg_recovery_top20','deg_recovery_top50','deg_recovery_top100']):
  vals=[data.loc[data.label==k,m].mean() for k in selected]
  errs=[data.loc[data.label==k,m].std(ddof=1) for k in selected]
  ax.bar(xs+(i-1)*w,vals,width=w,yerr=errs,capsize=2,label=f'Top { [20,50,100][i]}',alpha=.85)
 ax.set_xticks(xs,[LABELS[k] for k in selected],rotation=14,ha='right',fontsize=8)
 ax.set_title('E  Top-K absolute-response overlap',loc='left',fontweight='bold');ax.set_ylabel('Overlap fraction');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15)
 ax=axs[1,2]
 selected=['Ridge | GenePT-3large','MLP fusion','CCRT OOF']
 piv=data.pivot(index='seed',columns='label',values='mse')
 for seed in SEEDS:
  ax.plot(np.arange(3),[piv.loc[seed,k] for k in selected],marker='o',lw=1,alpha=.75,label=f'Seed {seed}')
 ax.set_xticks(np.arange(3),[LABELS[k] for k in selected],rotation=14,ha='right',fontsize=8)
 ax.set_title('F  Matched-split MSE',loc='left',fontweight='bold');ax.set_ylabel('MSE');ax.legend(frameon=False,fontsize=7);ax.grid(axis='y',alpha=.15)
 for ext in ('pdf','svg','png'):
  fig.savefig(outdir/f'Figure2_Benchmark.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
 # Print quantitative comparisons
 for base in ['Ridge | GenePT-3large','MLP fusion','CCRT Ridge base']:
  diff=piv['CCRT OOF']-piv[base]
  print(f'[PAIRED MSE] CCRT vs {base}: mean={diff.mean():+.9f}, better={int((diff<0).sum())}/5')
 print('[DONE] PDF SVG PNG + metrics_summary.csv + paired_deltas.csv')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,default=Path('.'))
 p.add_argument('--outdir',type=Path,default=Path('results/step07_figure2_all'))
 args=p.parse_args(); root=args.root.resolve(); output=args.outdir if args.outdir.is_absolute() else root/args.outdir
 output.mkdir(parents=True,exist_ok=True)
 data=prepare(root/'results');data.to_csv(output/'metrics_normalized_all_runs.csv',index=False)
 draw(data,output)
if __name__=='__main__':main()
