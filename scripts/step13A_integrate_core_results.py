#!/usr/bin/env python3
"""CellPert-Lite Step13A: evidence-checked manuscript results integration.
Requires pandas, numpy, matplotlib. No model fitting or web access.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['gated_direct','no_gate_direct','genept_only_direct','uce_only_direct']
LABEL={'gated_direct':'Gated fusion','no_gate_direct':'No-gate fusion','genept_only_direct':'GenePT only','uce_only_direct':'UCE only'}
GROUP={'all_embedding_covered':'All covered','K562_train_seen':'Seen','K562_test_heldout':'Held-out'}

def get(root, name, candidates, required=False):
    for c in candidates:
        p=root/c
        if p.is_file(): return p
    if required: raise FileNotFoundError(f'{name} not found. Tried: '+', '.join(map(str,candidates)))
    return None

def load(root,name,paths,required=False):
    p=get(root,name,paths,required)
    return (pd.read_csv(p),p) if p else (None,None)

def save_fig(fig,out,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'{name}.{ext}',bbox_inches='tight',dpi=220)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--outdir',default='results/step13A_core_results_integration')
    ap.add_argument('--include-optional',action='store_true',help='Include Step11B-7/12B supplementary summaries if present')
    a=ap.parse_args();root=Path(a.root).resolve();out=Path(a.outdir);out=out if out.is_absolute() else root/out;out.mkdir(parents=True,exist_ok=True)
    specs={
      'paper_summary':['results/step12E_metric_reconciliation/paper_summary.csv','paper_summary.csv'],
      'per_seed':['results/step12E_metric_reconciliation/recomputed_metrics_per_seed.csv','recomputed_metrics_per_seed.csv'],
      'paired':['results/step12E_metric_reconciliation/paired_model_target_bootstrap.csv','paired_model_target_bootstrap.csv'],
      'reconciliation':['results/step12E_metric_reconciliation/metric_reconciliation.csv','metric_reconciliation.csv'],
      'threshold':['results/step11B_7_threshold_sensitivity/model_threshold_summary.csv','model_threshold_summary.csv'],
      'enrichment':['results/step12B_functional_enrichment/enrichment_significant.csv','enrichment_significant.csv'],
      'k562':['results/step10A_unseen_perturbation_transfer/summary.csv','results/step10A_unseen_perturbation_transfer/metrics_summary.csv'],
      'rpe1':['results/step08D_transfer_analysis/metrics_summary.csv']}
    data={};manifest=[]
    for name,paths in specs.items():
        df,p=load(root,name,paths,name in ('paper_summary','per_seed','paired','reconciliation'))
        data[name]=df
        manifest.append({'source':name,'found':p is not None,'path':str(p) if p else '', 'rows':len(df) if df is not None else 0,'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p else ''})
    pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False)
    summary=data['paper_summary']; per=data['per_seed']; paired=data['paired']; recon=data['reconciliation']
    req={'dataset','model','group','mean_target_pcc','mean_micro_relative_skill','mean_macro_target_skill','median_amplitude_ratio','mean_fraction_positive_skill'}
    if not req.issubset(summary.columns):raise ValueError('paper_summary missing columns: '+str(req-set(summary.columns)))
    if not {'source','status'}.issubset(recon.columns):raise ValueError('metric_reconciliation missing source/status')
    recon.groupby(['source','status']).size().rename('count').reset_index().to_csv(out/'metric_reconciliation_status.csv',index=False)
    if (recon.loc[recon.source=='Step11B-4','status']!='PASS').any():
        raise RuntimeError('Step11B-4 numerical reconciliation not fully PASS; investigate before manuscript figures')
    summary=summary.copy();summary['model_label']=summary.model.map(LABEL).fillna(summary.model)
    summary['group_label']=summary.group.map(GROUP).fillna(summary.group)
    summary['micro_relative_skill_pct']=summary.mean_micro_relative_skill*100
    summary['macro_target_skill_pct']=summary.mean_macro_target_skill*100
    summary['positive_skill_pct']=summary.mean_fraction_positive_skill*100
    summary.to_csv(out/'Table1_external_model_performance.csv',index=False)
    per.to_csv(out/'Supplementary_per_seed_metrics.csv',index=False)
    paired.to_csv(out/'Table2_paired_model_comparisons.csv',index=False)
    # Only display contrasts with genuinely paired uncertainty, no significance assertions beyond CI excluding zero.
    pp=paired.copy();pp['ci_excludes_zero']=((pp.ci95_low>0)|(pp.ci95_high<0))
    pp.to_csv(out/'paired_effect_interpretation.csv',index=False)
    datasets=sorted(summary.dataset.dropna().unique())
    groups=['K562_train_seen','K562_test_heldout']
    for metric,label,suffix in [('mean_target_pcc','Mean target PCC','PCC'),('micro_relative_skill_pct','Micro relative skill (%)','micro_skill'),('macro_target_skill_pct','Macro mean target skill (%)','macro_skill')]:
        fig,axs=plt.subplots(1,len(datasets),figsize=(6*len(datasets),4.4),squeeze=False,sharey=True)
        for ax,ds in zip(axs[0],datasets):
            sub=summary[summary.dataset==ds];x=np.arange(len(groups));width=.19
            for j,m in enumerate(MODELS):
                vals=[]
                for g in groups:
                    rows=sub[(sub.group==g)&(sub.model==m)]
                    vals.append(float(rows.iloc[0][metric]) if len(rows)==1 else np.nan)
                ax.bar(x+(j-1.5)*width,vals,width,label=LABEL[m])
            ax.set_xticks(x,[GROUP[g] for g in groups]);ax.set_title(ds);ax.set_ylabel(label);ax.axhline(0,color='gray',lw=.7)
        axs[0][-1].legend(fontsize=8,loc='best');fig.suptitle(f'External frozen-model evaluation: {label}');fig.tight_layout()
        save_fig(fig,out,'Figure_core_'+suffix)
    # Matched model effect CI plots, selected metric pcc
    p=paired[paired.metric.astype(str).str.lower().eq('pcc')].copy()
    if len(p):
        fig,axs=plt.subplots(1,len(datasets),figsize=(6*len(datasets),4.7),squeeze=False)
        for ax,ds in zip(axs[0],datasets):
            s=p[(p.dataset==ds)&p.comparison.astype(str).str.contains('gated_direct - genept_only_direct',regex=False)].copy()
            if len(s):
                s=s.sort_values('group');x=np.arange(len(s))
                y=s.mean_difference.to_numpy(float);lo=s.ci95_low.to_numpy(float);hi=s.ci95_high.to_numpy(float)
                ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o',capsize=4)
                ax.set_xticks(x,s.group.tolist(),rotation=15);ax.axhline(0,color='gray',ls='--');ax.set_title(ds)
            ax.set_ylabel('Gated − GenePT PCC (95% target bootstrap CI)')
        fig.tight_layout();save_fig(fig,out,'Figure_core_paired_Gated_vs_GenePT')
    if a.include_optional:
        if data['threshold'] is not None:data['threshold'].to_csv(out/'Supplementary_threshold_sensitivity.csv',index=False)
        if data['enrichment'] is not None:data['enrichment'].to_csv(out/'Supplementary_exploratory_enrichment.csv',index=False)
    # K562 and RPE1 are not inferred from external data; explicit missing-evidence notice.
    missing=[x for x in ('k562','rpe1') if data[x] is None]
    (out/'evidence_gaps.txt').write_text('Missing independently verified inputs (not inferred): '+(', '.join(missing) if missing else 'none')+'\nK562/RPE1 results require explicit validated summary schemas and are NOT included in external-performance plots.\n',encoding='utf-8')
    lines=['# Step13A: core results integration','',
       'All numerical values below originate from Step12E paper_summary.csv; no K562/RPE1 scores are inferred.','',
       '## External held-out perturbation performance','',
       '| Dataset | Model | Mean PCC | Micro skill (%) | Macro skill (%) | Median RMS ratio | Targets with positive skill (%) |','|---|---|---:|---:|---:|---:|---:|']
    for _,r in summary[summary.group=='K562_test_heldout'].sort_values(['dataset','model']).iterrows():
        lines.append(f"| {r.dataset} | {r.model_label} | {r.mean_target_pcc:.4f} | {r.micro_relative_skill_pct:.2f} | {r.macro_target_skill_pct:.2f} | {r.median_amplitude_ratio:.3f} | {r.positive_skill_pct:.1f} |")
    lines+=['','## Interpretive guardrails',
        '- Micro skill is a ratio of pooled MSEs; macro skill averages per-target ratios. They can have opposite signs.',
        '- The held-out label is relative to each K562 seed split, not necessarily unseen in every seed.',
        '- Bootstrap intervals use target-level sampling after seed averaging; seeds are not independent biological replicates.',
        '- Cross-line response differences may reflect technical and processing effects; do not infer causal mechanisms.',
        '- No calibration is fitted to external test responses.',
        '- GO/Reactome findings remain exploratory and subject to redundancy and selection bias.',
        '', '## Source integrity',
        'See source_manifest.csv and metric_reconciliation_status.csv.','',
        '## Missing results', ', '.join(missing) if missing else 'No missing optional files; verify K562/RPE1 schemas before integration.']
    (out/'manuscript_results_outline.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[DONE]',out)
    print('[INPUTS]',[(r['source'],r['found']) for r in manifest]);print('[FIGURES] PCC, micro skill, macro skill, paired Gated vs GenePT')

if __name__=='__main__':main()
