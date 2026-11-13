#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step13C: integrate K562/RPE1 and HepG2/Jurkat manuscript validation panels.
No model fitting, no cross-cohort pooling, no inferred missing metrics.
Dependencies: pandas numpy matplotlib.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['gated_direct','no_gate_direct','genept_only_direct','uce_only_direct']
LABELS={'gated_direct':'Gated fusion','no_gate_direct':'No-gate fusion','genept_only_direct':'GenePT-only','uce_only_direct':'UCE-only'}
COLORS=dict(zip(MODELS,['#2B6CB0','#2F855A','#D69E2E','#805AD5']))
GROUPS={'K562_train_seen':'Seen','K562_test_heldout':'Held-out','all_embedding_covered':'All'}

def resolve(root, explicit, candidates, required=True):
    paths=[Path(explicit)] if explicit else [root/x for x in candidates]
    for p in paths:
        if p.exists(): return p.resolve()
    if required: raise FileNotFoundError('Missing required input. Tried: '+', '.join(map(str,paths)))
    return None

def validate(df, cols, name):
    missing=set(cols)-set(df.columns)
    if missing: raise ValueError(f'{name}: missing columns {sorted(missing)}; found {list(df.columns)}')
    if df.duplicated(cols[:3]).any():
        pass  # uniqueness checked separately for each input

def save(fig, out, stem):
    fig.tight_layout()
    for ext in ['png','pdf','svg']:
        fig.savefig(out/f'{stem}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)

def styled(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)

def grouped_bars(ax, df, category, metric, title, ylabel='Mean perturbation PCC', error=None):
    cats=list(dict.fromkeys(df[category].tolist()))
    x=np.arange(len(cats)); width=.19
    for j,model in enumerate(MODELS):
        sub=df[df.model==model].set_index(category)
        ys=[float(sub.loc[c,metric]) if c in sub.index else np.nan for c in cats]
        errs=[float(sub.loc[c,error]) if error and c in sub.index and pd.notna(sub.loc[c,error]) else 0 for c in cats]
        ax.bar(x+(j-1.5)*width,ys,width,label=LABELS[model],color=COLORS[model],yerr=errs if error else None,capsize=2)
    ax.set_xticks(x,cats);ax.set_ylabel(ylabel);ax.set_title(title);styled(ax)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.')
    p.add_argument('--k562-rpe1-file',default=None)
    p.add_argument('--external-file',default=None)
    p.add_argument('--paired-file',default=None)
    p.add_argument('--gap-file',default=None)
    p.add_argument('--output',default=None)
    args=p.parse_args();root=Path(args.root).resolve()
    out=Path(args.output).resolve() if args.output else root/'results/step13C_four_cell_line_manuscript_figures'
    out.mkdir(parents=True,exist_ok=True)
    kpath=resolve(root,args.k562_rpe1_file,['results/step13B_internal_external_validation/K562_input_copy.csv','results/step10A_unseen_perturbation_transfer/metrics_summary.csv'])
    epath=resolve(root,args.external_file,['results/step13B_internal_external_validation/Table_external_seed_mean_sd.csv','results/step12E_metric_reconciliation/paper_summary.csv'])
    ppath=resolve(root,args.paired_file,['results/step13A_core_results_integration/Table2_paired_model_comparisons.csv','results/step12E_metric_reconciliation/paired_model_target_bootstrap.csv'],False)
    gpath=resolve(root,args.gap_file,['results/step13B_internal_external_validation/Table_seen_heldout_gap_summary.csv'],False)
    k=pd.read_csv(kpath);e=pd.read_csv(epath)
    validate(k,['dataset','threshold','model','n_seeds','n_perturbations_mean','mse_mean','global_pcc_mean','perturbation_pcc_mean'],'K562/RPE1')
    if 'mean_target_pcc_mean' in e.columns:
        validate(e,['dataset','model','group','mean_target_pcc_mean','mean_seed_micro_relative_skill_mean','mean_target_skill_mean'],'external')
        e=e.rename(columns={'mean_target_pcc_mean':'pcc','mean_target_pcc_sd':'pcc_sd','mean_seed_micro_relative_skill_mean':'micro_skill','mean_target_skill_mean':'macro_skill'})
    else:
        validate(e,['dataset','model','group','mean_target_pcc','mean_micro_relative_skill','mean_macro_target_skill'],'external')
        e=e.rename(columns={'mean_target_pcc':'pcc','mean_micro_relative_skill':'micro_skill','mean_macro_target_skill':'macro_skill'})
        e['pcc_sd']=np.nan
    for name,d,keys in [('K562/RPE1',k,['dataset','threshold','model']),('external',e,['dataset','model','group'])]:
        if d.duplicated(keys).any(): raise ValueError(f'{name}: duplicated keys {keys}')
        unknown=set(d.model)-set(MODELS)
        if unknown: raise ValueError(f'{name}: unexpected models {unknown}')
    k['context']=np.where(k.dataset.eq('K562_unseen'),'K562 internal held-out','RPE1 external transfer')
    k['metric_definition']='perturbation_pcc_mean (reported source summary)'
    k.to_csv(out/'Table_K562_RPE1_source_metrics.csv',index=False)
    e['group_label']=e.group.map(GROUPS)
    e.to_csv(out/'Table_HepG2_Jurkat_source_metrics.csv',index=False)
    audit=[]
    for f in [kpath,epath,ppath,gpath]:
        if f:audit.append({'source':str(f),'sha256':sha(f),'rows':len(pd.read_csv(f))})
    pd.DataFrame(audit).to_csv(out/'input_manifest.csv',index=False)
    # Unified descriptive table: do not merge incompatible metric definitions.
    rows=[]
    for _,r in k.iterrows():
        rows.append(dict(dataset='K562' if r.dataset=='K562_unseen' else 'RPE1',setting='Internal held-out' if r.dataset=='K562_unseen' else f'External threshold {int(r.threshold)}',model=r.model,mean_perturbation_pcc=r.perturbation_pcc_mean,pcc_sd=np.nan,global_pcc=r.global_pcc_mean,micro_relative_skill=np.nan,macro_target_skill=np.nan,n_perturbations_mean=r.n_perturbations_mean,source='Step10A summary',comparability='descriptive only; gene set/response processing may differ'))
    for _,r in e.iterrows():
        if r.dataset not in ('HepG2','Jurkat'):continue
        rows.append(dict(dataset=r.dataset,setting=r.group_label or r.group,model=r.model,mean_perturbation_pcc=r.pcc,pcc_sd=r.pcc_sd,global_pcc=np.nan,micro_relative_skill=r.micro_skill,macro_target_skill=r.macro_skill,n_perturbations_mean=r.get('mean_n_targets_per_seed',r.get('mean_n_targets',np.nan)),source='Step12E summary',comparability='external matched-gene response; do not infer global PCC'))
    pd.DataFrame(rows).to_csv(out/'Table_four_cell_line_descriptive.csv',index=False)
    # Panel A K562
    fig,ax=plt.subplots(figsize=(9,5));t=k[k.dataset=='K562_unseen'].copy();t['setting']='Held-out';grouped_bars(ax,t,'setting','perturbation_pcc_mean','A | K562 internal unseen perturbations');ax.legend(ncol=2,fontsize=9);save(fig,out,'Figure13C_A_K562_internal')
    # Panel B RPE1 threshold sensitivity (not matched perturbation subsets)
    fig,ax=plt.subplots(figsize=(9,5));t=k[k.dataset=='RPE1_unseen_transfer'].copy();t['setting']=t.threshold.map(lambda x:f'Threshold {int(x)}');grouped_bars(ax,t,'setting','perturbation_pcc_mean','B | RPE1 transfer: different perturbation subsets');ax.legend(ncol=2,fontsize=9);save(fig,out,'Figure13C_B_RPE1_thresholds')
    # Panel C external Heldout
    fig,ax=plt.subplots(figsize=(9,5));t=e[(e.dataset.isin(['HepG2','Jurkat']))&(e.group=='K562_test_heldout')];grouped_bars(ax,t,'dataset','pcc','C | External held-out perturbations',error='pcc_sd');ax.legend(ncol=2,fontsize=9);save(fig,out,'Figure13C_C_external_heldout')
    # Panel D seen and held-out separate panels
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,d in zip(axes,['HepG2','Jurkat']):
        t=e[(e.dataset==d)&(e.group.isin(['K562_train_seen','K562_test_heldout']))].copy()
        t['setting']=pd.Categorical(t.group.map(GROUPS),['Seen','Held-out'],ordered=True)
        t=t.sort_values('setting');grouped_bars(ax,t,'setting','pcc',f'D | {d}: seen vs held-out',error='pcc_sd')
    axes[1].legend(ncol=2,fontsize=8);save(fig,out,'Figure13C_D_seen_heldout')
    # Panel E skill is only available in HepG2/Jurkat
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,metric,title in zip(axes,['micro_skill','macro_skill'],['Micro relative skill','Macro target skill']):
        t=e[(e.dataset.isin(['HepG2','Jurkat']))&(e.group=='K562_test_heldout')]
        grouped_bars(ax,t,'dataset',metric,'E | '+title,ylabel=title)
        ax.axhline(0,color='black',linewidth=.8)
    axes[1].legend(ncol=2,fontsize=8);save(fig,out,'Figure13C_E_external_skills')
    # Panel F paired PCC: filter strictly to exact metric, comparator, and seen/heldout
    if ppath:
        q=pd.read_csv(ppath)
        validate(q,['dataset','group','comparison','metric','mean_difference','ci95_low','ci95_high'],'paired')
        q.to_csv(out/'Table_paired_source.csv',index=False)
        q=q[(q.metric.astype(str).str.lower()=='pcc') & q.comparison.astype(str).str.contains('gated_direct - genept_only_direct',regex=False) & q.dataset.isin(['HepG2','Jurkat']) & q.group.isin(['seen','heldout'])].copy()
        if len(q):
            q=q.sort_values(['dataset','group']);fig,ax=plt.subplots(figsize=(9,5));ys=np.arange(len(q));vals=q.mean_difference.to_numpy(float)
            lo=q.ci95_low.to_numpy(float);hi=q.ci95_high.to_numpy(float)
            ax.errorbar(vals,ys,xerr=np.array([vals-lo,hi-vals]),fmt='o',color='#2B6CB0',capsize=4)
            ax.axvline(0,color='black',linewidth=.9);ax.set_yticks(ys,[f'{r.dataset} {r.group}' for _,r in q.iterrows()]);ax.invert_yaxis();ax.set_xlabel('Δ PCC (Gated − GenePT-only), bootstrap 95% CI');ax.set_title('F | Paired perturbation comparison');styled(ax);save(fig,out,'Figure13C_F_paired_PCC')
        else:audit.append({'warning':'paired file present but no exact gated vs genept PCC rows found'})
    if gpath:
        gap=pd.read_csv(gpath);gap.to_csv(out/'Table_seen_heldout_gap.csv',index=False)
    report={
      'input_rows':{'K562_RPE1':len(k),'HepG2_Jurkat':len(e)},
      'models':MODELS,
      'warnings':['K562/RPE1 perturbation PCC and HepG2/Jurkat mean-target PCC are shown descriptively, not treated as directly exchangeable.','RPE1 thresholds have different perturbation counts: no paired threshold effect is inferred.','Global PCC is only displayed when provided in original source.','RPE1 threshold subsets and K562 held-out data are not independent biological replicates.','Source summaries alone do not establish statistical significance across cell lines.']}
    (out/'validation_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'README_manuscript_panels.md').write_text('''# Step13C: four-cell-line validation panels\n\n- A: K562 internal unseen perturbation test (mean per-perturbation PCC).\n- B: RPE1 transfer at thresholds 3, 5, 10; perturbation sets differ, so this is descriptive sensitivity analysis.\n- C: HepG2/Jurkat K562-held-out transfer, mean per-target PCC (error bars: across-seed SD when available).\n- D: HepG2/Jurkat K562-seen vs K562-held-out performance; seed splits are not independent biological replicates.\n- E: HepG2/Jurkat micro-relative and macro-target skill. Negative macro skill is compatible with positive micro skill.\n- F: Gated vs GenePT-only paired target-level bootstrap PCC difference, when provided.\n\n**Caution:** The source datasets differ in gene coverage, response processing and/or target selection. Do not interpret visual cross-cell-line height differences as calibrated head-to-head comparisons. No significance tests are performed across cell lines. No retraining or external-test calibration.\n''',encoding='utf-8')
    print('[DONE]',out)
    print('K562/RPE1 rows:',len(k),'External rows:',len(e))
    print('Generated figures:',len(list(out.glob('Figure*.png'))))

if __name__=='__main__':main()
