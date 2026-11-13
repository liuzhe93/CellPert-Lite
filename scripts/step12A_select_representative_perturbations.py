#!/usr/bin/env python3
"""Step12A: reproducible representative-perturbation selection (NO model fitting).

Inputs: Step11B-6 per_target_diagnostics.csv and biological_shift_per_target.csv.
Selection uses Gated Fusion as the predeclared primary model; other models are
included for context. Categories are descriptive, not independent test sets.
A held-out target may occur in only one seed: report n_heldout_seeds and do not
interpret that as cross-seed stability. Strength-matched ranks avoid simply
calling strong-response targets 'successful'. K562/external response PCC shares
external truth with prediction PCC; it is descriptive, not causal.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PRIMARY='gated_direct'
REQUIRED={'dataset','seed','model','perturbation','group','prediction_pcc','prediction_mse','external_rms','k562_external_pcc','mse_improvement_vs_zero'}

def unique_one(s):
    a=s.dropna().unique()
    return a[0] if len(a)==1 else np.nan

def save_fig(df, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.size':9})
    for dataset, d in df.groupby('dataset'):
        fig,ax=plt.subplots(figsize=(6.4,4.5))
        for grp, marker in [('seen','o'),('heldout','^')]:
            x=d[d.group==grp]
            ax.scatter(x.external_rms,x.mean_pcc,alpha=.28,s=12,marker=marker,label=grp)
        chosen=d[d.selected_categories!='']
        for _,r in chosen.sort_values('selection_priority').head(12).iterrows():
            ax.annotate(r.perturbation,(r.external_rms,r.mean_pcc),fontsize=7,xytext=(3,3),textcoords='offset points')
        ax.axhline(0,color='gray',ls='--',lw=.7)
        ax.set(xlabel='Observed external response RMS',ylabel='Gated mean PCC (within group/seed)')
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ('png','pdf','svg'):
            fig.savefig(out/f'Figure8A_candidates_{dataset}.{ext}',dpi=230,bbox_inches='tight')
        plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root',default='.')
    p.add_argument('--input',default='results/step11B_6_response_strength_and_biological_shift/per_target_diagnostics.csv')
    p.add_argument('--outdir',default='results/step12A_representative_perturbations')
    p.add_argument('--top-n',type=int,default=12)
    p.add_argument('--min-rms-quantile',type=float,default=.25)
    p.add_argument('--min-seeds-seen',type=int,default=3)
    p.add_argument('--min-seeds-heldout',type=int,default=1)
    p.add_argument('--min-perturbed-cells',type=int,default=30)
    p.add_argument('--min-batches',type=int,default=2)
    p.add_argument('--no-plots',action='store_true')
    args=p.parse_args()
    if not 0<=args.min_rms_quantile<1 or args.top_n<1: raise ValueError('Invalid quantile/top-n')
    root=Path(args.root).resolve(); src=root/args.input; out=root/args.outdir;out.mkdir(parents=True,exist_ok=True)
    if not src.is_file():raise FileNotFoundError(f'Missing {src}; run Step11B-6 first')
    raw=pd.read_csv(src)
    missing=REQUIRED-set(raw.columns)
    if missing:raise ValueError(f'Missing columns {sorted(missing)} in {src}')
    raw=raw[raw.model==PRIMARY].copy()
    raw['perturbation']=raw.perturbation.astype(str)
    if raw.empty:raise ValueError('No gated_direct rows')
    if raw.duplicated(['dataset','seed','perturbation']).any():
        raise ValueError('Duplicate dataset/seed/perturbation for gated_direct')
    if not raw.group.isin(['seen','heldout']).all():raise ValueError('Unexpected group labels')
    raw['seed']=pd.to_numeric(raw.seed,errors='raise').astype(int)
    for c in ('prediction_pcc','prediction_mse','external_rms','k562_external_pcc','mse_improvement_vs_zero'):
        raw[c]=pd.to_numeric(raw[c],errors='coerce')
    raw=raw[np.isfinite(raw.external_rms)&(raw.external_rms>=0)].copy()
    # QC: optional, sourced from external data. Missing QC is NOT assumed to pass.
    qc_frames=[]
    for ds in ('HepG2','Jurkat'):
        candidates=[root/'data/external_validation/step11B_2A_expanded_minbatch1'/ds/f'{ds}_perturbation_qc.csv',
                    root/'data/external_validation/step11B_2A_expanded_minbatch1'/ds/'perturbation_qc.csv',
                    root/'data/external_validation/step11B_2_responses'/ds/f'{ds}_perturbation_qc.csv']
        qfile=next((f for f in candidates if f.is_file()),None)
        if qfile is not None:
            q=pd.read_csv(qfile)
            if {'perturbation','n_perturbed','n_batches'}.issubset(q.columns):
                q=q[['perturbation','n_perturbed','n_batches']].copy()
                q['dataset']=ds;q['qc_source']=str(qfile.relative_to(root))
                q=q.groupby(['dataset','perturbation'],as_index=False).agg(n_perturbed=('n_perturbed','max'),n_batches=('n_batches','max'),qc_source=('qc_source','first'))
                qc_frames.append(q)
    # Optional explicit QC CSVs passed as files for other layouts.
    allqc=pd.concat(qc_frames,ignore_index=True) if qc_frames else pd.DataFrame(columns=['dataset','perturbation','n_perturbed','n_batches','qc_source'])
    # Check that the true-response strength is consistent across seeds.
    chk=raw.groupby(['dataset','perturbation']).external_rms.agg(['min','max'])
    if ((chk['max']-chk['min'])>1e-7).any():raise ValueError('External RMS differs between seeds for same target')
    # Across seed means per group, not across mixed seen/heldout states.
    agg=raw.groupby(['dataset','perturbation','group'],as_index=False).agg(
        n_seeds=('seed','nunique'),mean_pcc=('prediction_pcc','mean'),sd_pcc=('prediction_pcc','std'),
        median_pcc=('prediction_pcc','median'),mean_mse=('prediction_mse','mean'),
        mean_skill_vs_zero=('mse_improvement_vs_zero','mean'),external_rms=('external_rms','first'),
        k562_external_pcc=('k562_external_pcc','first'))
    agg=agg.merge(allqc,on=['dataset','perturbation'],how='left',validate='many_to_one')
    agg['qc_available']=agg[['n_perturbed','n_batches']].notna().all(axis=1)
    agg['qc_pass']=agg.qc_available & (agg.n_perturbed>=args.min_perturbed_cells) & (agg.n_batches>=args.min_batches)
    agg['seed_pass']=np.where(agg.group.eq('seen'),agg.n_seeds>=args.min_seeds_seen,agg.n_seeds>=args.min_seeds_heldout)
    # RMS threshold is computed from full feature-covered universe per dataset,
    # not independently from the candidate groups.
    thresholds=raw.groupby('dataset').external_rms.quantile(args.min_rms_quantile).to_dict()
    agg['rms_cutoff']=agg.dataset.map(thresholds)
    agg['signal_pass']=agg.external_rms>=agg.rms_cutoff
    agg['eligible']=agg.seed_pass & agg.signal_pass & np.isfinite(agg.mean_pcc)
    # If QC is found, enforce it; if absent, leave eligible but explicitly flag unverified.
    agg['eligible']=agg.eligible & (~agg.qc_available | agg.qc_pass)
    agg['selection_priority']=np.nan
    agg['selected_categories']=''
    # Residual-like within-strength-bin percentile: rank PCC only among comparable
    # observed signal strengths, and never compare heldout to seen directly.
    for (ds,grp), ix in agg.groupby(['dataset','group']).groups.items():
        a=agg.loc[ix]
        ranks=a.external_rms.rank(method='first')
        bins=min(5,max(1,len(a)//20))
        agg.loc[ix,'strength_bin']=pd.qcut(ranks,q=bins,labels=False,duplicates='drop').to_numpy()+1
    agg['within_strength_pcc_percentile']=agg.groupby(['dataset','group','strength_bin']).mean_pcc.rank(pct=True,method='average')
    selected=[]
    def add_category(frame,category,sortcol,ascending):
        nonlocal selected
        for (ds,grp),d in frame.groupby(['dataset','group']):
            d=d.sort_values([sortcol,'perturbation'],ascending=[ascending,True]).head(args.top_n)
            for pos,(_,r) in enumerate(d.iterrows(),1):
                selected.append({'dataset':ds,'perturbation':r.perturbation,'group':grp,'category':category,'rank_in_category':pos,
                                 'mean_pcc':r.mean_pcc,'external_rms':r.external_rms,'n_seeds':int(r.n_seeds),
                                 'qc_available':bool(r.qc_available),'n_perturbed':r.n_perturbed,'n_batches':r.n_batches})
    valid=agg[agg.eligible].copy()
    add_category(valid,'high_pcc_strength_matched','within_strength_pcc_percentile',False)
    add_category(valid,'low_pcc_strength_matched','within_strength_pcc_percentile',True)
    # For biologically shifted responses, compare high vs low similarity; negative
    # K562-external PCC can indicate reversal but also measurement effects.
    shift=valid[np.isfinite(valid.k562_external_pcc)].copy()
    add_category(shift,'high_K562_external_similarity','k562_external_pcc',False)
    add_category(shift,'low_K562_external_similarity','k562_external_pcc',True)
    # Cross-cell-line contrasts: require same perturbation, same group, both lines.
    cross=agg.pivot_table(index=['perturbation','group'],columns='dataset',values=['mean_pcc','external_rms','k562_external_pcc','eligible'],aggfunc='first')
    cross.columns=['_'.join(x) for x in cross.columns];cross=cross.reset_index()
    need=['mean_pcc_HepG2','mean_pcc_Jurkat','eligible_HepG2','eligible_Jurkat']
    if set(need).issubset(cross):
        cross=cross[cross.eligible_HepG2.eq(True)&cross.eligible_Jurkat.eq(True)].copy()
        cross['delta_Jurkat_minus_HepG2']=cross.mean_pcc_Jurkat-cross.mean_pcc_HepG2
        cross['abs_delta_pcc']=cross.delta_Jurkat_minus_HepG2.abs()
        cross=cross.sort_values('abs_delta_pcc',ascending=False)
        for grp,d in cross.groupby('group'):
            for rank,(_,r) in enumerate(d.head(args.top_n).iterrows(),1):
                selected.append({'dataset':'cross_cell_line','perturbation':r.perturbation,'group':grp,
                                 'category':'largest_cross_cell_line_prediction_gap','rank_in_category':rank,
                                 'mean_pcc':np.nan,'external_rms':np.nan,'n_seeds':np.nan,'qc_available':True,
                                 'n_perturbed':np.nan,'n_batches':np.nan})
    else:cross=pd.DataFrame()
    chosen=pd.DataFrame(selected)
    if chosen.empty:raise ValueError('No eligible targets: lower --min-rms-quantile or --min-seeds-seen; inspect QC.')
    chosen.to_csv(out/'selected_candidates_by_category.csv',index=False)
    # Build distinct union for enrichment. This is a biased, exploratory gene list;
    # compare to the eligible, embedding-covered perturbation universe.
    agg['selected_categories']=''
    agg['selection_priority']=np.nan
    for (ds,p,g),group in chosen[chosen.dataset!='cross_cell_line'].groupby(['dataset','perturbation','group']):
        mask=(agg.dataset==ds)&(agg.perturbation==p)&(agg.group==g)
        agg.loc[mask,'selected_categories']=';'.join(sorted(group.category.unique()))
        agg.loc[mask,'selection_priority']=int(group.rank_in_category.min())
    agg.to_csv(out/'all_target_group_diagnostics.csv',index=False)
    agg[agg.selected_categories!=''].to_csv(out/'selected_candidates_unique.csv',index=False)
    valid[['dataset','perturbation','group']].drop_duplicates().to_csv(out/'enrichment_background_by_dataset_group.csv',index=False)
    # Background includes only QC-/signal-eligible genes for a like-for-like comparison.
    # Target-level background may be reused for Step12B but do not pool cell lines blindly.
    if not cross.empty:cross.to_csv(out/'cross_cell_line_matched_targets.csv',index=False)
    pd.DataFrame([{'dataset':ds,'n_rows':len(a),'n_eligible':int(a.eligible.sum()),
                   'n_qc_available':int(a.qc_available.sum()),'n_qc_pass':int(a.qc_pass.sum()),
                   'rms_cutoff':thresholds[ds]} for ds,a in agg.groupby('dataset')]).to_csv(out/'selection_audit.csv',index=False)
    if not args.no_plots:save_fig(agg,out)
    print('DONE',out)
    print('Category selections:',len(chosen),'unique dataset/target/group:',len(agg[agg.selected_categories!='']))
    if not allqc.empty:print('QC sources:',sorted(allqc.qc_source.unique()))
    else:print('WARNING: QC files not found; candidates are QC-UNVERIFIED. Review before biological claims.')

if __name__=='__main__':main()
