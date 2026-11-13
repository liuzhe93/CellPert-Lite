#!/usr/bin/env python3
"""Step12D: audit response ties, prediction amplitude/calibration and signal filtering.
No model fitting. Reads Step11B-4 frozen predictions and K562 observations.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MODELS=['gated_direct','no_gate_direct','genept_only_direct','uce_only_direct']
CELLS=['HepG2','Jurkat']

def load(path,keys):
    if not path.is_file(): raise FileNotFoundError(path)
    with np.load(path,allow_pickle=False) as z:
        miss=set(keys)-set(z.files)
        if miss:raise ValueError(f'{path}: missing {miss}; available {z.files}')
        return [np.asarray(z[k]) for k in keys]

def pcc(a,b):
    if len(a)<3:return np.nan
    x=a-a.mean();y=b-b.mean();d=np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.dot(x,y)/d) if d>1e-12 else np.nan

def metrics(y,p,mask):
    y=np.asarray(y,dtype=np.float64)[mask];p=np.asarray(p,dtype=np.float64)[mask]
    if len(y)==0:return dict(n_genes=0,pcc=np.nan,mse=np.nan,skill_vs_zero=np.nan,amplitude_ratio=np.nan,calibration_slope=np.nan,sign_accuracy=np.nan)
    yy=float(np.mean(y*y));mse=float(np.mean((y-p)**2))
    # Slope of predicted response on observed response, through the origin.
    slope=float(np.dot(y,p)/np.dot(y,y)) if np.dot(y,y)>1e-16 else np.nan
    return dict(n_genes=len(y),pcc=pcc(y,p),mse=mse,skill_vs_zero=1-mse/yy if yy>1e-16 else np.nan,
                amplitude_ratio=float(np.sqrt(np.mean(p*p))/np.sqrt(yy)) if yy>1e-16 else np.nan,
                calibration_slope=slope,sign_accuracy=float(np.mean(np.sign(y)==np.sign(p))))

def tie_stats(y,round_digits):
    y=np.asarray(y,dtype=np.float64);n=len(y)
    vals,cnts=np.unique(y,return_counts=True)
    most=int(cnts.max()); mode=float(vals[cnts.argmax()]); zero=int(np.count_nonzero(y==0))
    rounded=np.round(y,round_digits);_,rc=np.unique(rounded,return_counts=True)
    return dict(n_genes=n,n_unique_exact=len(vals),unique_exact_fraction=len(vals)/n,
                largest_tie_count=most,largest_tie_fraction=most/n,most_common_value=mode,
                zero_fraction=zero/n,n_unique_rounded=len(rc),largest_rounded_tie_fraction=float(rc.max()/n),
                rms=float(np.sqrt(np.mean(y*y))),std=float(np.std(y)),median_abs=float(np.median(np.abs(y))))

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--root',default='.')
    ap.add_argument('--predictions-dir',default='results/step11B_4_frozen_external_transfer')
    ap.add_argument('--k562',default='data/processed/benchmark_genept_uce_common/response_matrix_common.npz')
    ap.add_argument('--outdir',default='results/step12D_response_matrix_quality_and_calibration')
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    ap.add_argument('--models',nargs='+',default=MODELS)
    ap.add_argument('--genes',nargs='+',default=['COPB1','DTL','PSMB3','GTF2H3','ATP6V1D','PGD'])
    ap.add_argument('--round-digits',type=int,default=6)
    ap.add_argument('--abs-thresholds',nargs='+',type=float,default=[0,0.05,0.1,0.2])
    ap.add_argument('--top-fractions',nargs='+',type=float,default=[0.05,0.1,0.2])
    ap.add_argument('--min-filtered-genes',type=int,default=20)
    ap.add_argument('--max-tie-fraction-warning',type=float,default=0.1)
    ap.add_argument('--plot-top',type=int,default=15)
    args=ap.parse_args()
    root=Path(args.root).resolve();out=root/args.outdir;out.mkdir(parents=True,exist_ok=True)
    if any(x<0 for x in args.abs_thresholds):raise ValueError('Absolute thresholds must be >=0')
    if any(not 0<x<=1 for x in args.top_fractions):raise ValueError('Top fractions must be in (0,1]')
    if args.min_filtered_genes<3:raise ValueError('min-filtered-genes must be >=3')
    for model in args.models:
        if model not in MODELS:raise ValueError('Unknown model: '+model)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    k_ids,k_genes,k_y=load(root/args.k562,['perturbations','genes','delta'])
    k_ids=k_ids.astype(str);k_genes=k_genes.astype(str)
    if len(set(k_ids))!=len(k_ids) or len(set(k_genes))!=len(k_genes):raise ValueError('Duplicate K562 IDs')
    if k_y.shape!=(len(k_ids),len(k_genes)):raise ValueError('K562 matrix shape mismatch')
    ki={g:i for i,g in enumerate(k_ids)};kg={g:i for i,g in enumerate(k_genes)}
    tie_rows=[]; perf_rows=[]; audits=[]; examples=[]; seen_truth={}; nfiles=0
    for cell in CELLS:
      for seed in args.seeds:
       for model in args.models:
        path=root/args.predictions_dir/cell/f'seed_{seed}'/model/'predictions.npz'
        if not path.exists():
            audits.append(dict(dataset=cell,seed=seed,model=model,status='missing',path=str(path)))
            print('[WARN] Missing',path,flush=True);continue
        ids,genes,truth,pred,seen=load(path,['perturbations','genes','y_true','prediction','k562_train_seen'])
        ids=ids.astype(str);genes=genes.astype(str)
        if len(set(ids))!=len(ids) or len(set(genes))!=len(genes):raise ValueError('Duplicate IDs: '+str(path))
        if truth.shape!=pred.shape or truth.shape!=(len(ids),len(genes)) or len(seen)!=len(ids):raise ValueError('Shape mismatch '+str(path))
        if not (np.isfinite(truth).all() and np.isfinite(pred).all()):raise ValueError('Nonfinite matrix '+str(path))
        if not set(genes).issubset(kg):raise ValueError('External response genes not in K562 '+str(path))
        # Exact truth consistency across models/seeds, including target order.
        if cell not in seen_truth:
            seen_truth[cell]=(ids.copy(),genes.copy(),truth.copy())
        else:
            ri,rg,ry=seen_truth[cell]
            if not np.array_equal(ids,ri) or not np.array_equal(genes,rg) or not np.allclose(truth,ry,atol=1e-7,rtol=1e-6):
                raise ValueError('External truth inconsistent between prediction files: '+str(path))
        nfiles+=1
        audits.append(dict(dataset=cell,seed=seed,model=model,status='ok',n_targets=len(ids),n_genes=len(genes),path=str(path)))
        # Truth ties computed once per cell, including K562 comparison on same genes.
        if seed==args.seeds[0] and model==args.models[0]:
            gi=np.array([kg[g] for g in genes],int)
            for i,target in enumerate(ids):
                group='seen' if bool(seen[i]) else 'heldout'
                t=tie_stats(truth[i],args.round_digits)
                tie_rows.append(dict(dataset=cell,perturbation=target,group=group,source='external',**t))
                if target in ki:
                    t=tie_stats(k_y[ki[target],gi],args.round_digits)
                    tie_rows.append(dict(dataset=cell,perturbation=target,group=group,source='K562_on_external_genes',**t))
        for i,target in enumerate(ids):
            y=np.asarray(truth[i],np.float64);p=np.asarray(pred[i],np.float64)
            group='seen' if bool(seen[i]) else 'heldout'
            masks=[('all',np.ones(len(y),bool))]
            for thr in args.abs_thresholds:
                masks.append((f'abs_observed_ge_{thr:g}',np.abs(y)>=thr))
            for frac in args.top_fractions:
                count=max(1,int(np.ceil(len(y)*frac)))
                sel=np.argpartition(np.abs(y),-count)[-count:]
                mask=np.zeros(len(y),bool);mask[sel]=True
                masks.append((f'top_abs_observed_{frac:g}',mask))
            for name,mask in masks:
                if mask.sum()<args.min_filtered_genes:
                    perf_rows.append(dict(dataset=cell,seed=seed,model=model,perturbation=target,group=group,
                                          filter=name,n_genes=int(mask.sum()),pcc=np.nan,mse=np.nan,skill_vs_zero=np.nan,
                                          amplitude_ratio=np.nan,calibration_slope=np.nan,sign_accuracy=np.nan,status='too_few_genes'))
                    continue
                perf_rows.append(dict(dataset=cell,seed=seed,model=model,perturbation=target,group=group,
                                      filter=name,**metrics(y,p,mask),status='ok'))
            if target in args.genes and model==args.models[0] and seed==args.seeds[0]:
                values,inv,cnts=np.unique(y,return_inverse=True,return_counts=True)
                tieidx=np.argsort(-cnts)[:10]
                for j in tieidx:
                    examples.append(dict(dataset=cell,perturbation=target,value=float(values[j]),count=int(cnts[j]),
                                         fraction=float(cnts[j]/len(y)),example_genes=';'.join(genes[np.where(inv==j)[0][:12]])))
        print(f'[READ] {cell} seed={seed} model={model} targets={len(ids)} genes={len(genes)}',flush=True)
    if not nfiles:raise RuntimeError('No prediction NPZ found')
    ties=pd.DataFrame(tie_rows);perf=pd.DataFrame(perf_rows);aud=pd.DataFrame(audits)
    ties.to_csv(out/'response_tie_audit.csv',index=False)
    perf.to_csv(out/'per_target_calibration_and_filters.csv',index=False)
    aud.to_csv(out/'input_consistency_audit.csv',index=False)
    pd.DataFrame(examples,columns=['dataset','perturbation','value','count','fraction','example_genes']).to_csv(out/'repeated_value_examples.csv',index=False)
    good=perf[perf.status=='ok'].copy()
    # Target-level descriptive summaries; seeds are repeated measurements, not biological replicates.
    summary=good.groupby(['dataset','model','group','filter'],as_index=False).agg(
        n_rows=('pcc','size'),n_unique_targets=('perturbation','nunique'),n_seeds=('seed','nunique'),
        mean_pcc=('pcc','mean'),median_pcc=('pcc','median'),mean_skill_vs_zero=('skill_vs_zero','mean'),
        median_amplitude_ratio=('amplitude_ratio','median'),median_calibration_slope=('calibration_slope','median'),
        median_sign_accuracy=('sign_accuracy','median'),median_n_genes=('n_genes','median'))
    summary.to_csv(out/'filter_sensitivity_summary.csv',index=False)
    flags=ties[(ties.source=='external')&(ties.largest_tie_fraction>=args.max_tie_fraction_warning)].copy()
    flags.to_csv(out/'high_tie_targets_for_review.csv',index=False)
    if len(flags):print(f'[CAUTION] {len(flags)} external targets have largest exact tie fraction >= {args.max_tie_fraction_warning:.2f}',flush=True)
    # Two compact figures: model calibration and filtering, separated by cell line.
    for cell in CELLS:
        subset=good[(good.dataset==cell)&(good.model==args.models[0])&(good['filter']=='all')]
        if len(subset):
            fig,ax=plt.subplots(figsize=(6,4.5))
            for group,grp in subset.groupby('group'):
                x=grp.amplitude_ratio.to_numpy(float);y=grp.pcc.to_numpy(float)
                ok=np.isfinite(x)&np.isfinite(y)
                ax.scatter(x[ok],y[ok],s=9,alpha=.22,label=f'{group} (n={grp.perturbation.nunique()} targets)',rasterized=True)
            ax.axvline(1,color='gray',ls='--',lw=1)
            ax.set(xlabel='Predicted RMS / observed RMS',ylabel='Per-target PCC',title=f'{cell}: amplitude calibration vs PCC')
            ax.legend(frameon=False,fontsize=8);fig.tight_layout()
            for ext in ('png','pdf','svg'):fig.savefig(out/f'Figure9_{cell}_amplitude_vs_pcc.{ext}',dpi=220,bbox_inches='tight')
            plt.close(fig)
        s=summary[(summary.dataset==cell)&(summary.model==args.models[0])]
        filters=['all']+[f'abs_observed_ge_{t:g}' for t in args.abs_thresholds if t>0]+[f'top_abs_observed_{f:g}' for f in args.top_fractions]
        fig,ax=plt.subplots(figsize=(max(7,len(filters)*1.25),4.5))
        for group,grp in s.groupby('group'):
            lookup=grp.set_index('filter').mean_pcc.to_dict()
            ax.plot(range(len(filters)),[lookup.get(k,np.nan) for k in filters],marker='o',label=group)
        ax.set_xticks(range(len(filters)));ax.set_xticklabels(filters,rotation=35,ha='right')
        ax.set(ylabel='Mean per-target PCC',title=f'{cell}: response-signal filter sensitivity')
        ax.legend(frameon=False);fig.tight_layout()
        for ext in ('png','pdf','svg'):fig.savefig(out/f'Figure9_{cell}_signal_filter_sensitivity.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    print(f'[DONE] {out} | {nfiles} prediction files, {len(perf)} per-target/filter records, {len(flags)} tie flags',flush=True)
    print('NOTE: exact ties are diagnostic, not proof of processing errors. Filtering on observed response is post-hoc/descriptive.',flush=True)

if __name__=='__main__':main()
