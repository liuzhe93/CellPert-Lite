#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CellPert-Lite Step13E: submission-oriented figures, source audits, legends.

Requires pandas, numpy, matplotlib. Optional Step13D source script for inherited panels.
No model weights or external validation data are modified.
"""
import argparse, json, hashlib, textwrap, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

MODELS=['gated_direct','no_gate_direct','genept_only_direct','uce_only_direct']
NAMES=['Gated fusion','No-gate fusion','GenePT-only','UCE-only']
COLORS=['#2C659B','#25836A','#D39A31','#7954C8']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
                     'savefig.dpi':240})

def path(root, supplied, candidates, required=False):
    if supplied:
        p=Path(supplied); p=p if p.is_absolute() else root/p
        if not p.exists():raise FileNotFoundError(p)
        return p
    for x in candidates:
        p=root/x
        if p.exists():return p
    if required:raise FileNotFoundError('Cannot locate any of: '+str(candidates))
    return None

def save(fig, dest, name, bundle):
    fig.tight_layout(pad=1.4)
    for ext in ['png','pdf','svg']:
        fig.savefig(dest/f'{name}.{ext}',bbox_inches='tight',dpi=240)
    bundle.savefig(fig,bbox_inches='tight')
    plt.close(fig)

def valid_cols(df,keys,filename):
    missing=[x for x in keys if x not in df]
    if missing:raise ValueError(f'{filename}: missing {missing}; columns={list(df.columns)}')

def metric_figure(df, out, pdf):
    valid_cols(df,['dataset','model','group','mean_target_pcc_mean','mean_target_pcc_sd','mean_n_targets_per_seed'],'external')
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.9),sharey=True)
    for ax,ds in zip(axes,['HepG2','Jurkat']):
        for i,(m,n,c) in enumerate(zip(MODELS,NAMES,COLORS)):
            s=df[(df.dataset==ds)&(df.model==m)]
            for j,g in enumerate(['K562_train_seen','K562_test_heldout']):
                q=s[s.group==g]
                if q.empty:continue
                r=q.iloc[0]
                ax.errorbar(j+i*.08-.12,float(r.mean_target_pcc_mean),yerr=float(r.mean_target_pcc_sd),
                            fmt='o',ms=6,capsize=2.5,color=c,label=n if j==0 else None)
        ax.set_xticks([0,1],['Seen','Held-out']);ax.set_xlim(-.3,1.3)
        ax.set_title(ds);ax.grid(axis='y',alpha=.18)
        ax.set_ylabel('Mean perturbation PCC' if ds=='HepG2' else '')
        if ds=='Jurkat':ax.legend(frameon=False,fontsize=8,loc='upper right')
    fig.suptitle('Figure 4 (revised) | External transfer by target familiarity')
    save(fig,out,'Figure4_revised_seen_heldout_with_seed_SD',pdf)

def skill_figure(df,out,pdf):
    keys=['mean_seed_micro_relative_skill_mean','mean_target_skill_mean']
    valid_cols(df,['dataset','model','group']+keys,'external')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharex=True)
    for ax,metric,title in zip(axes,keys,['Micro relative skill','Macro target skill']):
        for ds,mark in [('HepG2','o'),('Jurkat','s')]:
            for i,(m,c) in enumerate(zip(MODELS,COLORS)):
                q=df[(df.dataset==ds)&(df.group=='K562_test_heldout')&(df.model==m)]
                if len(q):
                    x=float(q.iloc[0][metric]);y=i+(-.12 if ds=='HepG2' else .12)
                    ax.scatter(x,y,color=c,marker=mark,s=65,zorder=3,label=ds if i==0 else None)
        ax.axvline(0,color='black',ls='--',lw=.8)
        ax.set_yticks(range(4),NAMES);ax.invert_yaxis();ax.set_title(title)
        ax.set_xlabel('Skill vs zero-response baseline');ax.grid(axis='x',alpha=.2)
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Figure 6A | Held-out: distinct aggregation of relative skill')
    save(fig,out,'Figure6A_revised_micro_macro_skill',pdf)

def paired_figure(df,out,pdf):
    valid_cols(df,['dataset','group','comparison','metric','mean_difference','ci95_low','ci95_high','n_targets'],'paired')
    q=df[(df.metric=='pcc')&df.comparison.astype(str).str.contains('gated_direct - genept_only_direct',regex=False)].copy()
    if q.empty:return False
    q=q.sort_values(['dataset','group']); fig,ax=plt.subplots(figsize=(10,4.6))
    yy=np.arange(len(q));v=q.mean_difference.to_numpy(float)
    ax.errorbar(v,yy,xerr=[v-q.ci95_low.to_numpy(float),q.ci95_high.to_numpy(float)-v],
                fmt='o',capsize=3,color=COLORS[0],lw=1.5)
    ax.axvline(0,color='#555',ls='--')
    ax.set_yticks(yy,[f'{r.dataset} · {r.group} (n={r.n_targets})' for r in q.itertuples()])
    ax.invert_yaxis();ax.set_xlabel('Gated − GenePT mean PCC (target-bootstrap 95% CI)')
    ax.set_title('Figure 5 (revised) | Paired target-level model contrast')
    ax.grid(axis='x',alpha=.2)
    save(fig,out,'Figure5_revised_paired_effects',pdf);return True

def response_figure(df,out,pdf):
    valid_cols(df,['dataset','perturbation','mean_seed_pcc','observed_rms','predicted_rms'],'representative')
    fig,axes=plt.subplots(1,2,figsize=(12,5.1))
    piv=df.pivot_table(index='perturbation',columns='dataset',values='mean_seed_pcc').sort_index()
    x=np.arange(len(piv));w=.36
    for i,(ds,c) in enumerate([('HepG2',COLORS[0]),('Jurkat',COLORS[2])]):
        if ds in piv:axes[0].bar(x+(i-.5)*w,piv[ds],width=w,label=ds,color=c)
    axes[0].axhline(0,color='#555',lw=.8)
    axes[0].set_xticks(x,piv.index,rotation=35,ha='right');axes[0].set_ylabel('Mean seed PCC')
    axes[0].legend(frameon=False)
    for ds,c in [('HepG2',COLORS[0]),('Jurkat',COLORS[2])]:
        q=df[df.dataset==ds]
        axes[1].scatter(q.observed_rms,q.predicted_rms,label=ds,color=c,s=58)
    lim=max(df.observed_rms.max(),df.predicted_rms.max())*1.08
    axes[1].plot([0,lim],[0,lim],'--',color='#666',lw=1)
    axes[1].set(xlabel='Observed response RMS',ylabel='Predicted response RMS')
    axes[1].legend(frameon=False)
    fig.suptitle('Figure 7 (revised) | Selected target examples (descriptive)')
    save(fig,out,'Figure7_revised_representative',pdf)

def enrich_figure(df,out,pdf):
    valid_cols(df,['dataset','group','category','term','fdr_bh','overlap','overlap_genes','universe_size','query_size'],'enrichment')
    d=df[(df.fdr_bh>0)&(df.fdr_bh<=.05)].sort_values('fdr_bh')
    d=d.drop_duplicates(['dataset','group','category']).head(9).iloc[::-1]
    if d.empty:return False
    fig,ax=plt.subplots(figsize=(12,max(5,.6*len(d)+1.6)))
    labels=[f'{r.dataset} | {r.group} | {str(r.term)[:48]}' for r in d.itertuples()]
    xx=-np.log10(d.fdr_bh.clip(lower=1e-300).to_numpy(float))
    ax.barh(range(len(d)),xx,color='#528C99')
    ax.set_yticks(range(len(d)),labels,fontsize=8)
    for i,r in enumerate(d.itertuples()):
        ax.text(xx[i]+.05,i,f'overlap={r.overlap}; q={r.query_size}; U={r.universe_size}',va='center',fontsize=8)
    ax.set_xlim(0,max(xx)*1.65)
    ax.set_xlabel('−log10(BH FDR), within-group exploratory enrichment')
    ax.set_title('Figure 8 (revised) | Selected enrichment with set sizes')
    save(fig,out,'Figure8_revised_enrichment',pdf)
    d.to_csv(out/'Figure8_selected_enrichment_terms.csv',index=False)
    return True

def architecture_check(root,out,source):
    # Explicit architecture sources are mandatory for an architecture diagram.
    if source is None:
        status='NOT_VERIFIED: model source was not provided; Figure 1 remains evaluation schematic.'
    else:
        text=source.read_text(errors='replace')
        markers=['GenePT','UCE','gate','forward','nn.Module']
        status=('SOURCE_PROVIDED_FOR_MANUAL_REVIEW: '+str(source)+'; marker counts '+
                ', '.join(f'{k}={text.lower().count(k.lower())}' for k in markers)+
                '. Do not infer the network from these counts; inspect forward(), tensor shapes, loss and checkpoint.')
    (out/'architecture_source_audit.txt').write_text(status+'\n',encoding='utf-8')
    return status

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--out',default=None)
    ap.add_argument('--external-file');ap.add_argument('--paired-file');ap.add_argument('--representative-file')
    ap.add_argument('--enrichment-file');ap.add_argument('--model-source')
    ap.add_argument('--step13d-dir');ap.add_argument('--include-gene-response-images',action='store_true')
    a=ap.parse_args();root=Path(a.root).resolve()
    out=Path(a.out).resolve() if a.out else root/'results/step13E_publication_figures'
    out.mkdir(parents=True,exist_ok=True)
    inputs={
      'external':path(root,a.external_file,['results/step13B_internal_external_validation/Table_external_seed_mean_sd.csv'],True),
      'paired':path(root,a.paired_file,['results/step13A_core_results_integration/Table2_paired_model_comparisons.csv','results/step12E_metric_reconciliation/paired_model_target_bootstrap.csv']),
      'representative':path(root,a.representative_file,['results/step12C_representative_response_visualization/representative_prediction_metrics.csv']),
      'enrichment':path(root,a.enrichment_file,['results/step12B_functional_enrichment/enrichment_significant.csv','results/step12B_functional_enrichment_fixed/enrichment_significant.csv'])}
    model_source=path(root,a.model_source,[]) if a.model_source else None
    d13=Path(a.step13d_dir).resolve() if a.step13d_dir else root/'results/step13D_manuscript_figure_assembly'
    manifest=[]
    for k,v in inputs.items():
        manifest.append({'source':k,'path':str(v) if v else '', 'exists':bool(v),
                         'sha256':hashlib.sha256(v.read_bytes()).hexdigest() if v else ''})
    pd.DataFrame(manifest).to_csv(out/'input_manifest.csv',index=False)
    status=architecture_check(root,out,model_source)
    done=[];missing=[k for k,v in inputs.items() if v is None]
    with PdfPages(out/'Step13E_revised_panels.pdf') as pdf:
        e=pd.read_csv(inputs['external']);metric_figure(e,out,pdf);done.append('Figure4')
        skill_figure(e,out,pdf);done.append('Figure6A')
        if inputs['paired'] is not None and paired_figure(pd.read_csv(inputs['paired']),out,pdf):done.append('Figure5')
        if inputs['representative'] is not None:
            response_figure(pd.read_csv(inputs['representative']),out,pdf);done.append('Figure7')
        if inputs['enrichment'] is not None and enrich_figure(pd.read_csv(inputs['enrichment']),out,pdf):done.append('Figure8')
    # Preserve original validated Figure1-3, Figure6 as separate source panels; do not rasterize them.
    copied=[]
    for num,base in [(1,'Figure1_evaluation_design'),(2,'Figure2_K562_internal'),
                     (3,'Figure3_RPE1_threshold_sensitivity'),(6,'Figure6_signal_strength_and_amplitude')]:
        for ext in ['pdf','svg']:
            f=d13/f'{base}.{ext}'
            if f.exists():
                shutil.copy2(f,out/f'{base}_source.{ext}');copied.append(f.name)
    # Preserve original gene-response panels; PNG is never described as editable vector.
    image_list=[]
    if a.include_gene_response_images:
        roots=[root/'results/step12C_representative_response_visualization',root]
        for folder in roots:
            if folder.exists():
                for f in sorted(folder.rglob('Figure8_*png')):
                    if len(image_list)>=24:break
                    target=out/'gene_response_png'/f.name;target.parent.mkdir(exist_ok=True)
                    if not target.exists():shutil.copy2(f,target)
                    image_list.append(str(target))
    legends='''# Step13E | Draft manuscript figure legends (verify before submission)

**Figure 1.** Evaluation schematic showing K562 model development and internal held-out assessment followed by external transfer to RPE1, HepG2, and Jurkat. This schematic does not depict the neural network architecture; network details must be extracted from the original model implementation.

**Figure 2.** Internal evaluation on K562 held-out perturbations. Global PCC is calculated across the response matrix, whereas mean perturbation PCC is averaged across perturbations; the metrics are not interchangeable. Error bars should be labeled with their actual seed-level definition in the final panel.

**Figure 3.** RPE1 external transfer at minimum batch-perturbation cell thresholds of 3, 5 and 10. Threshold-specific perturbation sets differ, so the trends are descriptive and do not estimate a paired filtering effect.

**Figure 4.** Mean perturbation PCC in HepG2 and Jurkat for targets seen versus held out during K562 training. Points show five-seed means and error bars indicate between-seed SD. Seen and held-out groups contain different perturbations; differences are associations under this evaluation protocol, not causal effects.

**Figure 5.** Paired Gated-fusion minus GenePT-only mean perturbation PCC differences with 95% target-bootstrap confidence intervals. Independent resampling unit is the unique perturbation after seed aggregation, not a biological replicate. Labels give the number of matched targets used by each comparison.

**Figure 6.** Retrospective response-strength analysis and comparison of micro versus macro relative skill against the zero-response baseline. Response filters require the observed response and cannot be used for deployment-time target selection. Micro and macro skill weight targets differently.

**Figure 7.** Six preselected representative perturbations in HepG2 and Jurkat. Left: perturbation-level mean seed PCC. Right: observed and predicted response RMS; dashed identity line. Examples are descriptive and are not an unbiased estimate of model performance.

**Figure 8.** Selected exploratory gene-set enrichments with BH-adjusted FDR, query size, universe size and overlap. Terms overlap in membership and do not represent independent pathways. Biological mechanism claims require independent evidence.
'''
    (out/'Figure_legends_draft.md').write_text(legends,encoding='utf-8')
    audit={'generated_revised_panels':done,'missing_optional_inputs':missing,'preserved_source_panels':copied,
           'architecture':status,'copied_gene_response_png':len(image_list),
           'warnings':['Original network structure is not inferred from evaluation summaries.',
                       'The revised bundle is a collection of panels, not a finished journal-specific composite.',
                       'Gene-response PNG files are raster images, not editable vector figures.',
                       'Mean target PCC and global PCC must not be conflated.',
                       'No significance is inferred solely from overlapping seed SD bars.']}
    (out/'validation_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('[DONE]',out,'revised panels:',done,'missing optional:',missing)
    print('[ARCHITECTURE]',status)

if __name__=='__main__':main()
