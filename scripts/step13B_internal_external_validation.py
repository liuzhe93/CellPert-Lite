#!/usr/bin/env python3
"""CellPert-Lite Step13B: evidence-safe K562 internal + HepG2/Jurkat external figures.
No model training. Do not equate global PCC with mean target PCC or mix datasets.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['gated_direct','no_gate_direct','genept_only_direct','uce_only_direct']
NAMES={'gated_direct':'Gated fusion','no_gate_direct':'No-gate fusion','genept_only_direct':'GenePT only','uce_only_direct':'UCE only'}
GROUPS={'K562_train_seen':'Seen in K562 training','K562_test_heldout':'Held out in K562','all_embedding_covered':'All covered'}

def locate(root,override,options):
    if override:
        p=Path(override);p=p if p.is_absolute() else root/p
        if not p.is_file():raise FileNotFoundError(p)
        return p
    for o in options:
        p=root/o
        if p.is_file():return p
    return None

def save(fig,out,name):
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'{name}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)

def require(df,cols,label):
    missing=set(cols)-set(df.columns)
    if missing:raise ValueError(f'{label}: missing columns {sorted(missing)}; available {list(df.columns)}')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--outdir',default='results/step13B_internal_external_validation')
    ap.add_argument('--k562-file',default=None,help='Explicit K562 metrics_summary.csv or seed-level CSV')
    ap.add_argument('--external-summary',default=None)
    ap.add_argument('--external-per-seed',default=None)
    ap.add_argument('--strict-k562',action='store_true',help='Fail if K562 input absent or incompatible')
    args=ap.parse_args();root=Path(args.root).resolve();out=Path(args.outdir);out=out if out.is_absolute() else root/out;out.mkdir(parents=True,exist_ok=True)
    paths={
      'K562':locate(root,args.k562_file,['results/step10A_unseen_perturbation_transfer/metrics_summary.csv','results/step10A_unseen_perturbation_transfer/summary.csv']),
      'external_summary':locate(root,args.external_summary,['results/step13A_core_results_integration/Table1_external_model_performance.csv','results/step12E_metric_reconciliation/paper_summary.csv']),
      'external_per_seed':locate(root,args.external_per_seed,['results/step12E_metric_reconciliation/recomputed_metrics_per_seed.csv','results/step13A_core_results_integration/Supplementary_per_seed_metrics.csv'])}
    if paths['external_summary'] is None:raise FileNotFoundError('Missing Step13A Table1 or Step12E paper_summary')
    if paths['external_per_seed'] is None:raise FileNotFoundError('Missing Step12E recomputed_metrics_per_seed')
    manifest=[]
    for key,p in paths.items():
        manifest.append({'source':key,'found':p is not None,'path':str(p) if p else '', 'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p else ''})
    pd.DataFrame(manifest).to_csv(out/'input_manifest.csv',index=False)
    summary=pd.read_csv(paths['external_summary']); per=pd.read_csv(paths['external_per_seed'])
    require(summary,['dataset','model','group','mean_target_pcc','mean_micro_relative_skill','mean_macro_target_skill','median_amplitude_ratio'], 'external_summary')
    require(per,['dataset','seed','model','group','n_targets','macro_mean_target_pcc','micro_relative_skill','macro_mean_target_skill'], 'external_per_seed')
    summary=summary[summary.dataset.isin(['HepG2','Jurkat']) & summary.model.isin(MODELS)].copy()
    per=per[per.dataset.isin(['HepG2','Jurkat']) & per.model.isin(MODELS)].copy()
    if summary.empty or per.empty:raise ValueError('No HepG2/Jurkat rows found')
    if summary.duplicated(['dataset','model','group']).any():raise ValueError('Duplicate summary dataset/model/group keys')
    if per.duplicated(['dataset','seed','model','group']).any():raise ValueError('Duplicate per-seed keys')
    # Derive paper table from seed-level metrics; micro relative skill is mean of seed-level micro skills, NOT pooled across seeds.
    measures={'macro_mean_target_pcc':'mean_target_pcc','micro_relative_skill':'mean_seed_micro_relative_skill','macro_mean_target_skill':'mean_target_skill'}
    rows=[]
    for (ds,model,grp),sub in per.groupby(['dataset','model','group'],sort=True):
        rec={'dataset':ds,'model':model,'group':grp,'n_seeds':sub.seed.nunique(), 'mean_n_targets_per_seed':sub.n_targets.mean()}
        for col,name in measures.items():
            v=pd.to_numeric(sub[col],errors='coerce').dropna()
            rec[name+'_mean']=v.mean() if len(v) else np.nan
            rec[name+'_sd']=v.std(ddof=1) if len(v)>1 else np.nan
        rows.append(rec)
    ext=pd.DataFrame(rows)
    ext.to_csv(out/'Table_external_seed_mean_sd.csv',index=False)
    summary.to_csv(out/'Table_external_original_summary.csv',index=False)
    # Audit Step12E summary against per-seed aggregation, do not silently reconcile.
    audit=[]
    for _,r in summary.iterrows():
        m=ext[(ext.dataset==r.dataset)&(ext.model==r.model)&(ext.group==r.group)]
        if len(m)!=1:continue
        m=m.iloc[0]
        for a,b in [('mean_target_pcc','mean_target_pcc_mean'),('mean_micro_relative_skill','mean_seed_micro_relative_skill_mean'),('mean_macro_target_skill','mean_target_skill_mean')]:
            delta=float(r[a])-float(m[b]);audit.append({'dataset':r.dataset,'model':r.model,'group':r.group,'metric':a,'difference':delta,'match_1e_8':abs(delta)<1e-8})
    audit=pd.DataFrame(audit);audit.to_csv(out/'external_summary_consistency_audit.csv',index=False)
    if len(audit) and not audit.match_1e_8.all():raise ValueError('External summary does not match seed aggregation; see audit CSV')
    # K562 is only accepted with explicit mean target PCC field and model mapping.
    kstatus={'found':paths['K562'] is not None,'integrated':False,'reason':'not found'}
    if paths['K562']:
        k=pd.read_csv(paths['K562']);k.to_csv(out/'K562_input_copy.csv',index=False)
        name_col=next((x for x in ['model','label'] if x in k.columns),None)
        pcc_col=next((x for x in ['mean_perturbation_pcc_mean','mean_target_pcc','macro_mean_target_pcc','mean_perturbation_pcc'] if x in k.columns),None)
        mse_col=next((x for x in ['mse_mean','mse','micro_mse'] if x in k.columns),None)
        if name_col is None or pcc_col is None:
            kstatus['reason']='K562 CSV lacks model/label or mean target PCC; cannot harmonize safely'
        else:
            k=k.copy();k['model']=k[name_col].astype(str).str.strip()
            # Only exact model names are comparable; avoid conflating unrelated K562 experiments.
            k=k[k.model.isin(MODELS)]
            if k.empty:
                kstatus['reason']='K562 summary model names do not exactly match frozen Step10A models; preserved as separate unharmonized evidence'
            else:
                k['mean_target_pcc']=pd.to_numeric(k[pcc_col],errors='coerce')
                if mse_col:k['mse']=pd.to_numeric(k[mse_col],errors='coerce')
                k[['model','mean_target_pcc']+(['mse'] if mse_col else [])].to_csv(out/'Table_K562_internal_validated.csv',index=False)
                kstatus={'found':True,'integrated':True,'reason':'exact model IDs and target-level PCC field verified','models':sorted(k.model.unique().tolist())}
                fig,ax=plt.subplots(figsize=(7,4.5));k=k.sort_values('mean_target_pcc',ascending=False)
                ax.bar(k.model.map(NAMES),k.mean_target_pcc);ax.set_ylabel('Mean perturbation PCC (K562 internal)');ax.set_title('K562 internal held-out evaluation');ax.tick_params(axis='x',rotation=18)
                save(fig,out,'Figure13B_K562_internal_only')
    if args.strict_k562 and not kstatus['integrated']:raise ValueError(kstatus['reason'])
    # Plot external data with separate cells and groups; SD across seeds (not independent biological replicates).
    for metric,axis,name in [('mean_target_pcc','Mean target PCC','PCC'),('mean_seed_micro_relative_skill','Mean seed-level micro relative skill','micro_skill'),('mean_target_skill','Mean target-level skill','macro_skill')]:
        fig,axs=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
        for ax,ds in zip(axs,['HepG2','Jurkat']):
            x=np.arange(2);width=.19
            for j,model in enumerate(MODELS):
                means=[];sds=[]
                for group in ['K562_train_seen','K562_test_heldout']:
                    z=ext[(ext.dataset==ds)&(ext.model==model)&(ext.group==group)]
                    means.append(float(z.iloc[0][metric+'_mean']) if len(z)==1 else np.nan)
                    sds.append(float(z.iloc[0][metric+'_sd']) if len(z)==1 else np.nan)
                ax.bar(x+(j-1.5)*width,means,width,label=NAMES[model],yerr=sds,capsize=2,error_kw={'elinewidth':.8})
            ax.set_xticks(x,['Seen','Held-out']);ax.set_title(ds);ax.set_ylabel(axis);ax.axhline(0,color='gray',lw=.7)
        axs[1].legend(fontsize=8,loc='best');fig.suptitle('Frozen K562 model transferred to external cell lines (mean ± SD across seeds)')
        save(fig,out,'Figure13B_external_'+name)
    # Performance gap is paired within seed, not across independent datasets.
    gap=[]
    for (ds,model,seed),z in per.groupby(['dataset','model','seed']):
        seen=z[z.group=='K562_train_seen'];held=z[z.group=='K562_test_heldout']
        if len(seen)!=1 or len(held)!=1:continue
        gap.append({'dataset':ds,'model':model,'seed':seed,'seen_pcc':float(seen.iloc[0].macro_mean_target_pcc),'heldout_pcc':float(held.iloc[0].macro_mean_target_pcc),'seen_minus_heldout_pcc':float(seen.iloc[0].macro_mean_target_pcc-held.iloc[0].macro_mean_target_pcc)})
    gaps=pd.DataFrame(gap);gaps.to_csv(out/'Table_seen_heldout_seed_paired_gap.csv',index=False)
    if len(gaps):gaps.groupby(['dataset','model']).seen_minus_heldout_pcc.agg(['mean','std','count']).reset_index().to_csv(out/'Table_seen_heldout_gap_summary.csv',index=False)
    # Keep cross-context juxtaposition as separate panels only if validated K562 metrics exist.
    if kstatus['integrated']:
        k=pd.read_csv(out/'Table_K562_internal_validated.csv')
        fig,axs=plt.subplots(1,3,figsize=(16,4.5))
        axs[0].bar(k.model.map(NAMES),k.mean_target_pcc);axs[0].set_title('K562 internal test')
        for ax,ds in zip(axs[1:],['HepG2','Jurkat']):
            z=ext[(ext.dataset==ds)&(ext.group=='K562_test_heldout')]
            zz=z.set_index('model').reindex(MODELS)
            ax.bar([NAMES[m] for m in MODELS],zz.mean_target_pcc_mean);ax.set_title(ds+' transfer: held-out targets')
        for ax in axs:ax.set_ylabel('Mean target PCC');ax.tick_params(axis='x',rotation=22)
        fig.suptitle('Different evaluation contexts; do not interpret as matched performance degradation')
        save(fig,out,'Figure13B_internal_external_contexts')
    with open(out/'K562_schema_audit.json','w') as f:json.dump(kstatus,f,indent=2)
    lines=['# Step13B evidence and manuscript guidance','',
      'External HepG2/Jurkat: frozen K562 checkpoints; Seen/Held-out defined by each K562 seed split.',
      'Bar errors are SD across five seeds, NOT biological replicate confidence intervals.',
      'External micro relative skill is the mean of per-seed micro skills; do not call it pooled micro skill.',
      'K562 internal and external contexts are separate evaluation settings; no matched cross-context significance is claimed.',
      'RPE1 is not included unless independently validated.','',
      'K562 integration: '+str(kstatus),
      '', 'Do not conflate global PCC with mean per-target PCC.']
    (out/'README_interpretation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[DONE]',out,'K562:',kstatus['reason'],'external records:',len(ext),'seed gaps:',len(gaps))

if __name__=='__main__':main()
