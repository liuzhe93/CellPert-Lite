#!/usr/bin/env python3
"""Step12E: reproducible frozen-external evaluation reconciliation.

No fitting or external calibration. Reads the SAME frozen Step11B-4 predictions
for all four models. Recomputes micro MSE, macro target PCC, mean/median
per-target relative skill and amplitude ratios. Compares with Step11B-4 and
Step12D summaries, performs paired model comparisons with target-cluster
bootstrap (seed-averaged within target), and creates publication-ready plots.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ['gated_direct','no_gate_direct','genept_only_direct','uce_only_direct']
GROUPS = ['all_embedding_covered','K562_train_seen','K562_test_heldout']
LABEL = {'all_embedding_covered':'all','K562_train_seen':'seen','K562_test_heldout':'heldout'}

def load(path):
    with np.load(path, allow_pickle=False) as z:
        required = ['perturbations','genes','y_true','prediction','k562_train_seen']
        if any(k not in z for k in required):
            raise ValueError(f'{path}: missing keys; available={z.files}')
        return tuple(np.asarray(z[k]) for k in required)

def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = y.shape[0]
    mse = np.mean((y-p)**2, axis=1)
    zero = np.mean(y*y, axis=1)
    y0 = y-y.mean(axis=1,keepdims=True)
    p0 = p-p.mean(axis=1,keepdims=True)
    den = np.sqrt(np.sum(y0*y0,axis=1)*np.sum(p0*p0,axis=1))
    pcc = np.divide(np.sum(y0*p0,axis=1),den,out=np.full(n,np.nan),where=den>1e-14)
    skill = np.divide(zero-mse,zero,out=np.full(n,np.nan),where=zero>1e-16)
    amp = np.divide(np.sqrt(np.mean(p*p,axis=1)),np.sqrt(zero),out=np.full(n,np.nan),where=zero>1e-16)
    slope = np.divide(np.sum(y*p,axis=1),np.sum(y*y,axis=1),out=np.full(n,np.nan),where=np.sum(y*y,axis=1)>1e-14)
    return dict(mse=mse,zero_mse=zero,pcc=pcc,skill=skill,amplitude_ratio=amp,slope_pred_on_observed=slope)

def safe_mean(x):
    a=np.asarray(x,dtype=float); a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan

def safe_median(x):
    a=np.asarray(x,dtype=float); a=a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else np.nan

def bootstrap(x, repeats, rng):
    a=np.asarray(x,dtype=float);a=a[np.isfinite(a)]
    if len(a)<2:return (np.nan,np.nan)
    means=np.empty(repeats)
    for i in range(repeats):means[i]=a[rng.integers(0,len(a),size=len(a))].mean()
    return tuple(np.quantile(means,[.025,.975]))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--predictions-dir',default='results/step11B_4_frozen_external_transfer')
    ap.add_argument('--step11b-summary',default='results/step11B_4_frozen_external_transfer/metrics_per_seed.csv')
    ap.add_argument('--step12d-summary',default='results/step12D_response_matrix_quality_and_calibration/filter_sensitivity_summary.csv')
    ap.add_argument('--outdir',default='results/step12E_metric_reconciliation')
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    ap.add_argument('--models',nargs='+',default=MODELS)
    ap.add_argument('--datasets',nargs='+',default=['HepG2','Jurkat'])
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--random-state',type=int,default=42)
    ap.add_argument('--tolerance',type=float,default=1e-6)
    ap.add_argument('--allow-missing',action='store_true')
    args=ap.parse_args()
    root=Path(args.root).resolve(); out=root/args.outdir;out.mkdir(parents=True,exist_ok=True)
    for model in args.models:
        if model not in MODELS:raise ValueError('Unknown model '+model)
    rng=np.random.default_rng(args.random_state)
    rows=[]; per=[]; audit=[]; reference={}; missing=[]
    for ds in args.datasets:
      for seed in args.seeds:
        for model in args.models:
            path=root/args.predictions_dir/ds/f'seed_{seed}'/model/'predictions.npz'
            if not path.exists():
                missing.append(str(path));audit.append(dict(dataset=ds,seed=seed,model=model,status='MISSING',path=str(path)))
                continue
            ids,genes,y,p,seen=load(path)
            ids=ids.astype(str);genes=genes.astype(str);seen=seen.astype(bool)
            if len(set(ids))!=len(ids) or len(set(genes))!=len(genes):raise ValueError('Duplicate IDs '+str(path))
            if y.shape!=p.shape or y.shape!=(len(ids),len(genes)) or len(seen)!=len(ids):raise ValueError('Shape mismatch '+str(path))
            if not (np.isfinite(y).all() and np.isfinite(p).all()):raise ValueError('Nonfinite '+str(path))
            # Verify the SAME measured response across models/seeds, regardless of target order.
            key=ds
            if key not in reference:
                reference[key]=(genes.copy(),{v:i for i,v in enumerate(ids)},y.copy())
            else:
                refgenes,refids,refy=reference[key]
                if not np.array_equal(genes,refgenes) or set(ids)!=set(refids):raise ValueError('Target/gene universe differs '+str(path))
                order=[refids[v] for v in ids]
                if not np.allclose(y,refy[order],atol=1e-7,rtol=1e-6):raise ValueError('Truth differs '+str(path))
            m=metrics(y,p)
            audit.append(dict(dataset=ds,seed=seed,model=model,status='OK',n_targets=len(ids),n_genes=len(genes),path=str(path)))
            groups={GROUPS[0]:np.ones(len(ids),bool),GROUPS[1]:seen,GROUPS[2]:~seen}
            for group,mask in groups.items():
                if not mask.any():continue
                a=m['mse'][mask];z=m['zero_mse'][mask]
                micro_mse=float(a.mean());micro_zero=float(z.mean())
                rows.append(dict(dataset=ds,seed=seed,model=model,group=group,n_targets=int(mask.sum()),n_genes=len(genes),
                    micro_mse=micro_mse,micro_zero_mse=micro_zero,
                    micro_absolute_improvement=micro_zero-micro_mse,
                    micro_relative_skill=1-micro_mse/micro_zero if micro_zero>1e-16 else np.nan,
                    macro_mean_target_pcc=safe_mean(m['pcc'][mask]),macro_median_target_pcc=safe_median(m['pcc'][mask]),
                    macro_mean_target_skill=safe_mean(m['skill'][mask]),macro_median_target_skill=safe_median(m['skill'][mask]),
                    median_amplitude_ratio=safe_median(m['amplitude_ratio'][mask]),
                    median_slope_pred_on_observed=safe_median(m['slope_pred_on_observed'][mask]),
                    fraction_targets_positive_skill=safe_mean(m['skill'][mask]>0)))
            for i,t in enumerate(ids):
                per.append(dict(dataset=ds,seed=seed,model=model,perturbation=t,group='seen' if seen[i] else 'heldout',
                    n_genes=len(genes),pcc=float(m['pcc'][i]),mse=float(m['mse'][i]),zero_mse=float(m['zero_mse'][i]),
                    skill=float(m['skill'][i]),amplitude_ratio=float(m['amplitude_ratio'][i]),
                    slope_pred_on_observed=float(m['slope_pred_on_observed'][i])))
            print(f'[OK] {ds} seed={seed} {model} n={len(ids)}',flush=True)
    if missing and not args.allow_missing:
        pd.DataFrame(audit).to_csv(out/'input_audit.csv',index=False)
        raise FileNotFoundError(f'{len(missing)} frozen prediction files missing; see input_audit.csv. Pass --allow-missing to continue.')
    if not rows:raise RuntimeError('No valid prediction files')
    df=pd.DataFrame(rows);pt=pd.DataFrame(per)
    df.to_csv(out/'recomputed_metrics_per_seed.csv',index=False)
    pt.to_csv(out/'per_target_metrics.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'input_audit.csv',index=False)
    summary=df.groupby(['dataset','model','group'],as_index=False).agg(n_seeds=('seed','nunique'),mean_n_targets=('n_targets','mean'),
        mean_micro_relative_skill=('micro_relative_skill','mean'),mean_macro_target_skill=('macro_mean_target_skill','mean'),
        mean_target_pcc=('macro_mean_target_pcc','mean'),median_amplitude_ratio=('median_amplitude_ratio','median'),
        mean_fraction_positive_skill=('fraction_targets_positive_skill','mean'))
    summary.to_csv(out/'paper_summary.csv',index=False)
    # Reconciliation: original Step11B uses absolute MSE difference, NOT relative skill.
    rec=[]
    path=root/args.step11b_summary
    if path.exists():
        old=pd.read_csv(path)
        required=['dataset','seed','model','group','mse','zero_baseline_mse','mse_improvement_vs_zero','mean_target_pcc']
        if all(c in old for c in required):
            z=df.merge(old,on=['dataset','seed','model','group'],suffixes=('_new','_old'),how='left',indicator=True)
            for _,r in z.iterrows():
                checks={'mse':('micro_mse','mse'), 'zero_baseline_mse':('micro_zero_mse','zero_baseline_mse'),
                    'absolute_improvement':('micro_absolute_improvement','mse_improvement_vs_zero'),
                    'mean_pcc':('macro_mean_target_pcc','mean_target_pcc')}
                for metric,(a,b) in checks.items():
                    new=float(r[a]);oldv=float(r[b]) if pd.notna(r[b]) else np.nan
                    rec.append(dict(source='Step11B-4',dataset=r.dataset,seed=r.seed,model=r.model,group=r.group,metric=metric,
                        recomputed=new,previous=oldv,absolute_difference=abs(new-oldv) if np.isfinite(oldv) else np.nan,
                        status='PASS' if np.isfinite(oldv) and abs(new-oldv)<=args.tolerance else 'CHECK'))
        else:print('[WARN] Step11B summary columns differ; skipping reconciliation')
    else:print('[WARN] Step11B summary not found:',path)
    path=root/args.step12d_summary
    if path.exists():
        old=pd.read_csv(path)
        old=old[old['filter'].isin(['all','abs_observed_ge_0'])].copy()
        old=old.sort_values('filter').drop_duplicates(['dataset','model','group'],keep='first')
        for _,r in summary.iterrows():
            v=old[(old.dataset==r.dataset)&(old.model==r.model)&(old.group==LABEL[r.group])]
            if len(v)!=1:continue
            q=v.iloc[0]
            for metric,a,b in [('mean_pcc','mean_target_pcc','mean_pcc'),('mean_target_skill','mean_macro_target_skill','mean_skill_vs_zero'),
                                ('median_amplitude_ratio','median_amplitude_ratio','median_amplitude_ratio')]:
                new=float(r[a]);oldv=float(q[b]);delta=abs(new-oldv)
                # Aggregation may differ: Step12D pooled target-seed rows; Step12E averages seed means.
                rec.append(dict(source='Step12D',dataset=r.dataset,seed='ALL',model=r.model,group=r.group,metric=metric,
                    recomputed=new,previous=oldv,absolute_difference=delta,
                    status='DESCRIPTIVE_DIFFERENT_AGGREGATION' if delta>args.tolerance else 'PASS'))
    else:print('[WARN] Step12D summary not found:',path)
    pd.DataFrame(rec,columns=['source','dataset','seed','model','group','metric','recomputed','previous','absolute_difference','status']).to_csv(out/'metric_reconciliation.csv',index=False)
    # Paired differences: aggregate per target across seeds FIRST, bootstrap independent targets.
    targetmean=pt.groupby(['dataset','group','model','perturbation'],as_index=False).agg(
        n_seeds=('seed','nunique'),pcc=('pcc','mean'),skill=('skill','mean'),mse=('mse','mean'),amplitude_ratio=('amplitude_ratio','mean'))
    targetmean.to_csv(out/'target_averaged_across_seeds.csv',index=False)
    contrasts=[]
    for ds in args.datasets:
      for group in ['seen','heldout']:
        base=targetmean[(targetmean.dataset==ds)&(targetmean.group==group)&(targetmean.model=='gated_direct')]
        for model in args.models:
            if model=='gated_direct':continue
            other=targetmean[(targetmean.dataset==ds)&(targetmean.group==group)&(targetmean.model==model)]
            pair=base.merge(other,on=['dataset','group','perturbation'],suffixes=('_gated','_other'))
            for metric in ['pcc','skill','mse']:
                delta=(pair[f'{metric}_gated']-pair[f'{metric}_other']).to_numpy(dtype=float)
                lo,hi=bootstrap(delta,args.bootstrap,rng)
                contrasts.append(dict(dataset=ds,group=group,comparison='gated_direct - '+model,metric=metric,
                    n_targets=len(pair),mean_difference=safe_mean(delta),ci95_low=lo,ci95_high=hi,
                    bootstrap_unit='unique perturbation (seed-averaged)',
                    interpretation='higher_better' if metric!='mse' else 'lower_better'))
    pd.DataFrame(contrasts).to_csv(out/'paired_model_target_bootstrap.csv',index=False)
    # Visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    for ds in args.datasets:
        sub=summary[(summary.dataset==ds)&(summary.group.isin(['K562_train_seen','K562_test_heldout']))]
        fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        x=np.arange(len(args.models));w=.36
        for j,(group,label) in enumerate([('K562_train_seen','Seen'),('K562_test_heldout','Held-out')]):
            data=sub[sub.group==group].set_index('model').reindex(args.models)
            ax[0].bar(x+(j-.5)*w,data['mean_micro_relative_skill'],width=w,label=label)
            ax[1].bar(x+(j-.5)*w,data['mean_macro_target_skill'],width=w,label=label)
        for a,t in zip(ax,['Micro relative skill (ratio of pooled MSEs)','Mean per-target relative skill']):
            a.axhline(0,color='gray',linewidth=.8);a.set_xticks(x,args.models,rotation=25,ha='right');a.set_title(t);a.legend(frameon=False)
        fig.suptitle(ds+' | different zero-baseline skill aggregations')
        for ext in ['png','pdf','svg']:fig.savefig(out/f'{ds}_skill_aggregation.{ext}',dpi=200)
        plt.close(fig)
    with open(out/'README_metrics.txt','w') as f:
        f.write('Step12E definitions\n'
                'micro_relative_skill = 1 - mean_over_all_target_genes((prediction-observed)^2) / mean_over_all_target_genes(observed^2).\n'
                'macro_mean_target_skill = mean_over_targets[1 - target_MSE / target_zero_MSE].\n'
                'These are NOT algebraically equivalent. A negative macro skill can coexist with a positive micro skill.\n'
                'Step11B-4 mse_improvement_vs_zero is an ABSOLUTE difference of MSEs, not relative skill.\n'
                'Step12D summaries pool all seed-target rows; Step12E paper_summary averages per-seed values.\n'
                'Target bootstrap averages seed-level metrics for each unique target before resampling; descriptive CI only.\n'
                'No calibration is fitted: external observed responses must not be used to calibrate the frozen model.\n'
                'A calibration model requires K562-only out-of-fold predictions and a separately specified calibration protocol.\n')
    print('\n[DONE]',out)
    print(summary[['dataset','model','group','mean_micro_relative_skill','mean_macro_target_skill','mean_target_pcc']].to_string(index=False))

if __name__=='__main__':main()
