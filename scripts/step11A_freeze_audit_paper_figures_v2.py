#!/usr/bin/env python3
"""Step11A: freeze/audit Step10H predictions and assemble exploratory paper figures.

No model fitting, no test-set model selection, no claim of independent validation.
Requires Step10H CSVs, optionally Step10G/F summaries. Figures are descriptive.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ['gated_direct','mlp_fusion','no_gate_direct','genept_only_direct',
          'uce_only_direct','ridge_fusion','full_ccrt','ccrt_oof',
          'ccrt_gamma_calibrated','ccrt_internal_ridge','k562_train_mean','zero_change']
LABELS = {'gated_direct':'Gated Direct','mlp_fusion':'MLP Fusion','no_gate_direct':'No Gate',
          'genept_only_direct':'GenePT only','uce_only_direct':'UCE only','ridge_fusion':'Ridge Fusion',
          'full_ccrt':'Full CCRT','ccrt_oof':'CCRT OOF','ccrt_gamma_calibrated':'CCRT calibrated',
          'ccrt_internal_ridge':'CCRT internal Ridge','k562_train_mean':'K562 train mean','zero_change':'Zero change'}
PRIMARY = ['gated_direct','mlp_fusion','full_ccrt','ccrt_internal_ridge','ccrt_gamma_calibrated','k562_train_mean']
SEVEN = ['gated_direct','no_gate_direct','genept_only_direct','uce_only_direct','ridge_fusion','mlp_fusion','full_ccrt']

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def savefig(fig, base, formats):
    for ext in formats:fig.savefig(str(base)+'.'+ext,bbox_inches='tight',dpi=220)
    plt.close(fig)

def load_csv(folder,name,required=True):
    path=folder/name
    if not path.is_file():
        if required:raise FileNotFoundError(f'Required input missing: {path}')
        return None
    return pd.read_csv(path)

def require(df,cols,name):
    missing=set(cols)-set(df.columns)
    if missing:raise ValueError(f'{name}: missing columns {sorted(missing)}')

def check_unique(df,cols,name):
    d=df.duplicated(cols,keep=False)
    if d.any():raise ValueError(f'{name}: duplicate keys {cols}, examples: {df.loc[d,cols].head(4).to_dict("records")}')

def finite_nonnegative(df,col,name):
    v=pd.to_numeric(df[col],errors='coerce')
    if v.isna().any() or not np.isfinite(v).all() or (v<0).any():
        raise ValueError(f'{name}: invalid {col}')

def mean_sd(df,group,cols):
    return df.groupby(group,dropna=False)[cols].agg(['mean','std','count'])

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
                         'figure.facecolor':'white','savefig.facecolor':'white'})

def panel_label(ax,letter):ax.text(-.12,1.07,letter,transform=ax.transAxes,fontweight='bold',fontsize=13)

def fig1_protocol(out,formats,stats):
    fig,ax=plt.subplots(figsize=(11.4,4.6));ax.axis('off')
    boxes=[('Input\nK562 perturbation responses','Input'),('Locked splits\n5 seeds (42–46)','Split'),
           ('Representations\nGenePT + UCE','Features'),('Predictors\n7 models + baselines','Models'),
           ('Evaluation\nK562 holdout + RPE1 transfer','Evaluation')]
    xs=np.linspace(.10,.90,len(boxes))
    for i,((t,_),x) in enumerate(zip(boxes,xs)):
        ax.text(x,.67,t,ha='center',va='center',transform=ax.transAxes,fontsize=9,
                bbox=dict(boxstyle='round,pad=.7',facecolor='#F2F5F8',edgecolor='#9EADB8'))
        if i<len(boxes)-1:ax.annotate('',xy=(xs[i+1]-.095,.67),xytext=(x+.095,.67),
                                     xycoords='axes fraction',textcoords='axes fraction',
                                     arrowprops=dict(arrowstyle='->',color='#5E6A75',lw=1.5))
    ax.text(.5,.28,'K562: unseen target perturbations   |   RPE1: external cell-line transfer',
            ha='center',transform=ax.transAxes,fontsize=11,fontweight='bold')
    ax.text(.5,.16,'Post hoc: biological response recovery, response-strength strata, residual calibration',
            ha='center',transform=ax.transAxes,fontsize=9)
    ax.text(.5,.055,'Schematic only — not an independent prospective validation',ha='center',
            transform=ax.transAxes,fontsize=8,color='#555')
    savefig(fig,out/'Figure1_Benchmark_Design',formats)

def _seed_points(ax, data, models, metric, title, xlabel, lower=False):
    """Show each seed rather than hiding variation behind zero-based bars."""
    for i,m in enumerate(models):
        q=data[data.model==m].sort_values('seed')
        if q.empty:continue
        v=q[metric].dropna().to_numpy(dtype=float)
        if not len(v):continue
        yy=np.full(len(v),i,dtype=float)+np.linspace(-.12,.12,len(v))
        ax.scatter(v,yy,s=19,color='#9CAAB5',alpha=.8,zorder=2)
        ax.scatter([np.mean(v)],[i],marker='D',s=48,color=('#1F678C' if m=='gated_direct' else '#334155'),zorder=3)
        if len(v)>1:ax.plot([np.min(v),np.max(v)],[i,i],color='#C5CDD4',lw=1,zorder=1)
    ax.set_yticks(range(len(models)),[LABELS.get(m,m) for m in models]);ax.invert_yaxis()
    ax.set_title(title,fontsize=10);ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.15)
    ax.ticklabel_format(axis='x',style='plain',useOffset=False)


def fig2_k562(pooled,per,paired,out,formats):
    z=pooled[(pooled.dataset=='K562_unseen')&(pooled.threshold==0)&pooled.model.isin(SEVEN)]
    order=z.groupby('model').mse.mean().sort_values().index.tolist()
    fig,axs=plt.subplots(2,2,figsize=(13.6,9.4))
    _seed_points(axs[0,0],z,order,'mse','A | Prediction error','MSE (lower is better)')
    _seed_points(axs[0,1],z,order,'global_pcc','B | Global agreement','Global Pearson r (higher is better)')
    # Mean per-perturbation PCC per seed, without treating repeated seeds as independent samples.
    q=per[(per.dataset=='K562_unseen')&(per.threshold==0)&per.model.isin(SEVEN)]
    q=q.groupby(['model','seed'],as_index=False).pcc.mean()
    _seed_points(axs[1,0],q,order,'pcc','C | Perturbation-level agreement','Mean per-perturbation Pearson r')
    ax=axs[1,1]
    comparisons=[('gated_direct','full_ccrt'),('gated_direct','k562_train_mean'),('full_ccrt','ccrt_internal_ridge')]
    # Only display comparisons actually present in Step10H bootstrap output.
    available=[]
    for aa,bb in comparisons:
        x=paired[(paired.dataset=='K562_unseen')&(paired.threshold==0)&(paired.metric=='mse')&
                 (paired.model_a==aa)&(paired.model_b==bb)]
        sign=1
        if x.empty:
            x=paired[(paired.dataset=='K562_unseen')&(paired.threshold==0)&(paired.metric=='mse')&
                     (paired.model_a==bb)&(paired.model_b==aa)]
            sign=-1
        if len(x):
            r=x.iloc[0];available.append((f'{LABELS[aa]} − {LABELS[bb]}',sign*float(r.difference),
                                         min(sign*float(r.ci_low),sign*float(r.ci_high)),
                                         max(sign*float(r.ci_low),sign*float(r.ci_high))))
    if available:
        for i,(label,d,lo,hi) in enumerate(available):
            ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',capsize=3,color='#1F678C')
        ax.set_yticks(range(len(available)),[x[0] for x in available]);ax.invert_yaxis()
        ax.axvline(0,color='#444',lw=1);ax.set_xlabel('Paired MSE difference (95% exploratory bootstrap CI)')
        ax.set_title('D | Paired comparisons',fontsize=10)
    else:
        ax.text(.5,.5,'Paired MSE comparisons unavailable',ha='center',va='center',transform=ax.transAxes)
        ax.set_axis_off()
    fig.suptitle('Figure 2 | K562: unseen perturbation prediction across five locked seeds',fontweight='bold',fontsize=13)
    fig.text(.5,.008,'Dots: individual seeds; diamonds: seed means. Bootstrap intervals are exploratory, not independent confirmation.',ha='center',fontsize=8,color='#555')
    fig.tight_layout(rect=[0,.03,1,.95]);savefig(fig,out/'Figure2_K562_Seven_Model_Benchmark',formats)


def fig3_rpe1(pooled,per,paired,out,formats):
    """Within-batch paired comparisons; never interpret thresholds as a continuous dose."""
    z=pooled[(pooled.dataset=='RPE1_unseen_transfer')&pooled.model.isin(PRIMARY)]
    fig,axs=plt.subplots(2,3,figsize=(15.8,8.8))
    for j,th in enumerate([3,5,10]):
        q=z[z.threshold==th]
        order=PRIMARY
        _seed_points(axs[0,j],q,order,'pooled_skill_vs_train_mean',
                     f'Batch{th} | Skill vs train mean','1 − MSE(model)/MSE(train mean)')
        axs[0,j].axvline(0,color='#555',lw=.8)
        _seed_points(axs[1,j],q,order,'global_pcc',f'Batch{th} | Global PCC','Global Pearson r')
        axs[0,j].text(.98,.98,f'n={int(q.n_perturbations.iloc[0])} per seed' if len(q) else '',
                      transform=axs[0,j].transAxes,ha='right',va='top',fontsize=8,color='#555')
    fig.suptitle('Figure 3 | RPE1 cross-cell-line transfer: within-batch comparisons',fontweight='bold',fontsize=13)
    fig.text(.5,.008,'Batch3/5/10 use different perturbation subsets; points are seeds, not independent biological replicates.',ha='center',fontsize=8,color='#555')
    fig.tight_layout(rect=[0,.03,1,.95]);savefig(fig,out/'Figure3_RPE1_Transfer',formats)
    fig,axs=plt.subplots(1,3,figsize=(14,4.4))
    for ax,th in zip(axs,[3,5,10]):
        s=paired[(paired.dataset=='RPE1_unseen_transfer')&(paired.threshold==th)&
                 (paired.metric=='mse')]
        comps=[('full_ccrt','ccrt_internal_ridge'),('ccrt_gamma_calibrated','ccrt_internal_ridge'),
               ('ccrt_gamma_calibrated','full_ccrt'),('gated_direct','full_ccrt'),
               ('gated_direct','k562_train_mean')]
        vals=[]
        for aa,bb in comps:
            x=s[(s.model_a==aa)&(s.model_b==bb)];sgn=1
            if x.empty:x=s[(s.model_a==bb)&(s.model_b==aa)];sgn=-1
            if len(x):
                r=x.iloc[0];d=sgn*float(r.difference)
                lo=min(sgn*float(r.ci_low),sgn*float(r.ci_high));hi=max(sgn*float(r.ci_low),sgn*float(r.ci_high))
                vals.append((f'{LABELS[aa]} − {LABELS[bb]}',d,lo,hi))
        for i,(_,d,lo,hi) in enumerate(vals):ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',capsize=3,color='#1F678C')
        ax.set_yticks(range(len(vals)),[v[0] for v in vals]);ax.invert_yaxis()
        ax.axvline(0,color='#555',lw=.8);ax.set_title(f'Batch{th}')
        ax.set_xlabel('Paired MSE difference (95% bootstrap CI)')
        ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
    fig.suptitle('Figure 3B | Exploratory paired model effects in RPE1',fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.93]);savefig(fig,out/'Figure3B_RPE1_Baseline_Relative_Skill',formats)


def fig4_biology(per,strength,pooled,out,formats):
    targets=['gated_direct','full_ccrt','ccrt_internal_ridge','k562_train_mean']
    sub=per[(per.dataset=='K562_unseen')&per.model.isin(targets)]
    ks=[k for k in (20,50,100) if f'top{k}_overlap' in sub.columns]
    fig,axs=plt.subplots(1,3,figsize=(15.4,4.6))
    for m in targets:
        q=sub[sub.model==m]
        axs[0].plot(ks,[q[f'top{k}_overlap'].mean() for k in ks],marker='o',label=LABELS[m])
    # Random expected fraction K/G, where G is the number of evaluated genes.
    genes=pooled[(pooled.dataset=='K562_unseen')&(pooled.threshold==0)].n_genes
    if len(genes) and genes.nunique()==1:
        G=int(genes.iloc[0]);axs[0].plot(ks,[min(k,G)/G for k in ks],linestyle='--',color='#555',marker='x',label=f'Random ranking (G={G})')
    axs[0].set_xlabel('Top-K genes');axs[0].set_ylabel('Magnitude-based Top-K overlap')
    axs[0].legend(fontsize=7);axs[0].set_title('A | Response-gene ranking')
    ss=strength[(strength.dataset=='K562_unseen')&(strength.model.isin(targets))]
    x=np.arange(3);width=.18
    for i,m in enumerate(targets):
        v=ss[ss.model==m].set_index('response_stratum').reindex(['low','medium','high'])
        axs[1].bar(x+(i-1.5)*width,v.skill_vs_zero,width,label=LABELS[m])
    axs[1].set_xticks(x,['Low','Medium','High']);axs[1].axhline(0,color='#444',lw=.8)
    axs[1].set_ylabel('Mean per-perturbation skill vs zero');axs[1].set_title('B | Post hoc response strata')
    for m in targets:
        q=sub[sub.model==m]
        if 'better_than_zero' in q:
            vals=[q[q.seed==seed].better_than_zero.mean() for seed in sorted(q.seed.unique())]
            axs[2].scatter([LABELS[m]]*len(vals),vals,s=16,alpha=.55)
            axs[2].plot([LABELS[m]],[np.mean(vals)],marker='D',color='#222',markersize=5)
    axs[2].set_ylim(0,1);axs[2].tick_params(axis='x',labelrotation=40)
    axs[2].set_ylabel('Fraction of perturbations outperforming zero')
    axs[2].set_title('C | Per-perturbation success')
    fig.suptitle('Figure 4 | Biological response recovery and heterogeneity (K562)',fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.93]);savefig(fig,out/'Figure4_Biological_Response_Recovery',formats)
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.2),sharey=True)
    for ax,th in zip(axs,[3,5,10]):
        q=strength[(strength.dataset=='RPE1_unseen_transfer')&(strength.threshold==th)&strength.model.isin(targets)]
        for i,m in enumerate(targets):
            v=q[q.model==m].set_index('response_stratum').reindex(['low','medium','high'])
            ax.bar(x+(i-1.5)*width,v.skill_vs_zero,width,label=LABELS[m])
        ax.set_xticks(x,['Low','Medium','High']);ax.axhline(0,color='#444',lw=.8);ax.set_title(f'Batch{th}')
    axs[0].set_ylabel('Mean per-perturbation skill vs zero')
    axs[2].legend(fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left')
    fig.suptitle('Supplement | RPE1 response strata defined using test truth (descriptive only)')
    fig.tight_layout();savefig(fig,out/'Supplement_RPE1_Response_Strata',formats)

def fig5_residual(pooled,choices,out,formats):
    sub=pooled[pooled.model.isin(['ccrt_internal_ridge','full_ccrt','ccrt_oof','ccrt_gamma_calibrated'])]
    fig,axs=plt.subplots(1,2,figsize=(12,4.7))
    labels=['K562','RPE1 B3','RPE1 B5','RPE1 B10'];datasets=[('K562_unseen',0)]+[('RPE1_unseen_transfer',t) for t in (3,5,10)]
    mods=['ccrt_internal_ridge','full_ccrt','ccrt_oof','ccrt_gamma_calibrated']
    x=np.arange(4);width=.19
    for i,m in enumerate(mods):
        v=[]
        for d,t in datasets:
            z=sub[(sub.dataset==d)&(sub.threshold==t)&(sub.model==m)]
            v.append(z.pooled_skill_vs_train_mean.mean())
        axs[0].bar(x+(i-1.5)*width,v,width,label=LABELS[m])
    axs[0].set_xticks(x,labels);axs[0].axhline(0,color='black',lw=.8)
    axs[0].set_ylabel('Pooled skill vs K562 train mean');axs[0].legend(fontsize=8);panel_label(axs[0],'A')
    choices=choices.sort_values('seed')
    axs[1].bar(choices.seed.astype(str),choices.gamma,color='#7496A8')
    axs[1].set_ylim(0,1.05);axs[1].set_ylabel('Selected residual multiplier γ')
    axs[1].set_xlabel('K562 split seed');panel_label(axs[1],'B')
    fig.suptitle('Figure 5 | Residual refinement and exploratory K562-only calibration',fontweight='bold')
    fig.tight_layout();savefig(fig,out/'Figure5_CCRT_Residual_Diagnostics',formats)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root',default='.')
    p.add_argument('--step10h',default='results/step10H_metric_validation_residual_calibration')
    p.add_argument('--step10g',default='results/step10G_ccrt_rescue_biological_evaluation')
    p.add_argument('--step10f',default='results/step10F_seven_model_paired_and_ccrt_diagnostics')
    p.add_argument('--outdir',default='results/step11A_paper_assembly')
    p.add_argument('--formats',nargs='+',choices=['pdf','svg','png'],default=['pdf','svg'])
    p.add_argument('--force',action='store_true',help='Allow replacement of an existing Step11A output directory')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();root=Path(a.root).resolve();h=root/a.step10h;g=root/a.step10g;f=root/a.step10f;out=root/a.outdir
    required=['pooled_metrics_per_seed.csv','pooled_metrics_summary.csv','mean_per_perturbation_metrics.csv',
              'per_perturbation_metrics_validated.csv','response_strength_summary.csv',
              'response_strength_per_perturbation.csv','gamma_choices.csv','alignment_audit.csv',
              'paired_bootstrap_BH_exploratory.csv']
    inputs=[h/name for name in required]
    optional=[g/'biological_metrics_summary.csv',f/'paired_bootstrap_summary.csv']
    for x in inputs:
        if not x.is_file():raise FileNotFoundError(x)
    pooled=load_csv(h,'pooled_metrics_per_seed.csv');summary=load_csv(h,'pooled_metrics_summary.csv')
    per=load_csv(h,'per_perturbation_metrics_validated.csv');strength=load_csv(h,'response_strength_summary.csv')
    audit=load_csv(h,'alignment_audit.csv');choices=load_csv(h,'gamma_choices.csv')
    paired=load_csv(h,'paired_bootstrap_BH_exploratory.csv')
    require(pooled,['dataset','threshold','seed','model','mse','global_pcc','pooled_r2_vs_zero','pooled_skill_vs_train_mean','n_perturbations','n_genes'],'pooled')
    require(per,['dataset','threshold','seed','perturbation','model','mse','pcc','per_pert_r2_vs_zero','better_than_zero','top20_overlap'],'per perturbation')
    require(strength,['dataset','threshold','response_stratum','model','mse','pcc','skill_vs_zero'],'strata')
    require(choices,['seed','gamma','source'],'gamma choices')
    require(audit,['dataset','threshold','seed','n_perturbations','n_genes','n_models','gamma'],'alignment audit')
    require(paired,['dataset','threshold','model_a','model_b','metric','difference','ci_low','ci_high','q_bh_exploratory'],'paired bootstrap')
    check_unique(pooled,['dataset','threshold','seed','model'],'pooled')
    check_unique(per,['dataset','threshold','seed','perturbation','model'],'per perturbation')
    check_unique(choices,['seed'],'gamma choices')
    check_unique(audit,['dataset','threshold','seed'],'alignment audit')
    finite_nonnegative(pooled,'mse','pooled');finite_nonnegative(per,'mse','per perturbation')
    if not (choices.gamma.between(0,1)).all():raise ValueError('gamma outside [0,1]')
    expected=set(MODELS)
    errors=[];warnings=[]
    for (d,t,s),part in pooled.groupby(['dataset','threshold','seed']):
        missing=expected-set(part.model)
        if missing:errors.append(f'{d}/{t}/seed{s} missing {sorted(missing)}')
        arow=audit[(audit.dataset==d)&(audit.threshold==t)&(audit.seed==s)]
        if len(arow)!=1:errors.append(f'Missing audit row {d}/{t}/seed{s}')
        elif int(arow.iloc[0].n_perturbations)!=int(part.n_perturbations.iloc[0]):errors.append(f'Perturbation count mismatch {d}/{t}/seed{s}')
        for m in set(part.model):
            n=len(per[(per.dataset==d)&(per.threshold==t)&(per.seed==s)&(per.model==m)])
            if n!=int(part[part.model==m].n_perturbations.iloc[0]):errors.append(f'Per-pert count mismatch {d}/{t}/seed{s}/{m}: {n}')
    if not set(pooled.seed)==set(choices.seed):errors.append('Gamma seed set differs from pooled metrics seed set')
    for (d,t,s),part in pooled.groupby(['dataset','threshold','seed']):
        z=part.set_index('model')
        if 'zero_change' in z.index and 'gated_direct' in z.index:
            expected_r2=1-float(z.loc['gated_direct','mse'])/float(z.loc['zero_change','mse'])
            actual=float(z.loc['gated_direct','pooled_r2_vs_zero'])
            if not np.isclose(expected_r2,actual,rtol=1e-4,atol=1e-5):errors.append(f'R2 identity failed {d}/{t}/seed{s}')
        if 'k562_train_mean' in z.index:
            val=float(z.loc['k562_train_mean','pooled_skill_vs_train_mean'])
            if abs(val)>1e-6:errors.append(f'Train mean skill not zero {d}/{t}/seed{s}')
    if errors:raise ValueError('AUDIT FAILED:\n'+'\n'.join(errors[:30]))
    if len(set(pooled.seed))<5:warnings.append('Fewer than five seeds available')
    if len(paired):warnings.append('Bootstrap p-values/BH q-values are exploratory, not confirmatory')
    warnings += ['RPE1 batch3/5/10 contain different perturbation subsets; avoid direct threshold trends.',
                 'Response-strength strata use observed test responses (post hoc).',
                 'Gamma selected on reused K562 inner validation; test outcomes have been inspected repeatedly.',
                 'No runtime/parameter-count, independent cohort, or external-method benchmark is generated here.',
                 'Figure 1 is a schematic based on workflow labels, not measured data.']
    print('[AUDIT PASS] seeds=',sorted(pooled.seed.unique().tolist()),'rows=',len(pooled),'per-pert rows=',len(per))
    print('[DATASETS]',pooled.groupby(['dataset','threshold']).seed.nunique().to_dict())
    if a.dry_run:
        print('[DRY RUN PASS] no files written');return
    if out.exists() and any(out.iterdir()) and not a.force:
        raise FileExistsError(f'{out} already contains files. Use a new --outdir or --force.')
    out.mkdir(parents=True,exist_ok=True);figdir=out/'figures';figdir.mkdir(exist_ok=True)
    style();fig1_protocol(figdir,a.formats,audit);fig2_k562(pooled,per,paired,figdir,a.formats)
    fig3_rpe1(pooled,per,paired,figdir,a.formats);fig4_biology(per,strength,pooled,figdir,a.formats)
    fig5_residual(pooled,choices,figdir,a.formats)
    # Freeze all consumed input bytes, including optional summaries if present.
    consumed=inputs+[x for x in optional if x.exists()]
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'Exploratory figure assembly, not a preregistered/independent test',
              'script':Path(__file__).name,'files':[{'path':str(x.relative_to(root)) if x.is_relative_to(root) else str(x),
                'bytes':x.stat().st_size,'sha256':sha256(x)} for x in consumed],
              'warnings':warnings,'seeds':sorted(int(v) for v in pooled.seed.unique())}
    (out/'frozen_inputs_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (out/'audit_report.json').write_text(json.dumps({'status':'PASS','errors':errors,'warnings':warnings,
                 'n_pooled_rows':len(pooled),'n_per_perturbation_rows':len(per),'n_audit_rows':len(audit)},indent=2))
    summary.to_csv(out/'Table1_Overall_Pooled_Metrics.csv',index=False)
    pd.read_csv(h/'mean_per_perturbation_metrics.csv').to_csv(out/'Table2_Mean_Per_Perturbation.csv',index=False)
    strength.to_csv(out/'Table3_Response_Strata.csv',index=False)
    paired.to_csv(out/'TableS1_Exploratory_Paired_Comparisons.csv',index=False)
    pd.read_csv(h/'alignment_audit.csv').to_csv(out/'TableS2_Alignment_Audit.csv',index=False)
    def val(d,t,m,col):
        v=summary[(summary.dataset==d)&(summary.threshold==t)&(summary.model==m)]
        return float(v.iloc[0][col]) if len(v) else float('nan')
    k='K562_unseen';r='RPE1_unseen_transfer'
    lines=['# Step11A — frozen exploratory manuscript figure package','',
           '## Data-integrity status','PASS: required files, unique keys, perturbation counts and pooled R² identities checked.','',
           '## Descriptive results (not independent confirmation)',
           f'- K562 Gated Direct: MSE {val(k,0,"gated_direct","mse"):.7f}; global PCC {val(k,0,"gated_direct","global_pcc"):.4f}; pooled R² vs zero {val(k,0,"gated_direct","pooled_r2_vs_zero"):.4f}.',
           f'- K562 Full CCRT: MSE {val(k,0,"full_ccrt","mse"):.7f}; calibrated CCRT {val(k,0,"ccrt_gamma_calibrated","mse"):.7f}.',
           f'- RPE1 batch3 internal Ridge MSE {val(r,3,"ccrt_internal_ridge","mse"):.7f}; calibrated CCRT MSE {val(r,3,"ccrt_gamma_calibrated","mse"):.7f}.',
           f'- RPE1 batch5 internal Ridge MSE {val(r,5,"ccrt_internal_ridge","mse"):.7f}; train-mean MSE {val(r,5,"k562_train_mean","mse"):.7f}.',
           f'- RPE1 batch10 train-mean MSE {val(r,10,"k562_train_mean","mse"):.7f}; Gated Direct MSE {val(r,10,"gated_direct","mse"):.7f}.',
           '', '## Figure captions (draft)',
           'Figure 1. Analysis design and held-out perturbation benchmark workflow (schematic).',
           'Figure 2. K562 seven-model benchmark: seed-level scatter, per-perturbation PCC and exploratory paired bootstrap differences.',
           'Figure 3. RPE1 external transfer with within-batch skill and PCC; Figure 3B shows paired bootstrap effects. Batches contain different perturbation sets.',
           'Figure 4. Response-gene magnitude ranking with random ranking expectation, post hoc strata, and perturbation-level success; not DEG recovery.',
           'Figure 5. Ridge and residual-model comparisons; gamma chosen on previously used K562 inner validation.',
           '', '## Required safeguards']+[f'- {w}' for w in warnings]+['',
           '## Next additions','- Independent third-cell-line evaluation with a locked, prospective protocol.',
           '- Fair external published-model comparisons with matching targets and gene sets.',
           '- Gene-level biological case studies with curated perturbation IDs.',
           '- Model efficiency/compute cost and parameter counts.',
           '', 'All plotted values are read from Step10H outputs; no synthetic model-performance measurements are used.']
    (out/'README_manuscript_results.md').write_text('\n'.join(lines)+'\n')
    print('[DONE]',out)
    print('[FIGURES]',len(list(figdir.glob('*.pdf'))),'PDF files; formats:',','.join(a.formats))

if __name__=='__main__':main()
