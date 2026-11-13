#!/usr/bin/env python3
"""Step09D: exploratory paired gate analyses using frozen predictions only.

Primary comparison: gated_no_residual vs no_gate_direct.
Secondary: gated_no_residual vs genept_only_direct and mlp_fusion (RPE1).
Bootstrap unit: unique perturbation; repeated seeds averaged within perturbation.
No model training, parameter tuning, or outcome-based selection.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DIRECT=('gated_no_residual','no_gate_direct','genept_only_direct','uce_only_direct')
COMPARISONS=(('gated_no_residual','no_gate_direct'),('gated_no_residual','genept_only_direct'),('gated_no_residual','uce_only_direct'),('gated_no_residual','mlp_fusion'))

def load(path,keys):
    if not path.is_file():raise FileNotFoundError(path)
    with np.load(path,allow_pickle=False) as z:
        missing=set(keys)-set(z.files)
        if missing:raise ValueError(f'{path}: missing {sorted(missing)}; available={z.files}')
        return [z[k] for k in keys]

def pcc_rows(y,p):
    y=y.astype(np.float64);p=p.astype(np.float64)
    a=y-y.mean(axis=1,keepdims=True);b=p-p.mean(axis=1,keepdims=True)
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return np.divide(np.einsum('ij,ij->i',a,b),den,out=np.full(len(y),np.nan),where=den>1e-12)

def corr(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<10 or np.std(x[ok])<1e-12 or np.std(y[ok])<1e-12:return np.nan,int(ok.sum())
    return float(spearmanr(x[ok],y[ok]).statistic),int(ok.sum())

def bootstrap(vals,nboot,rng):
    vals=np.asarray(vals,dtype=float);vals=vals[np.isfinite(vals)]
    if len(vals)<2:return (np.nan,np.nan,np.nan,len(vals))
    means=np.empty(nboot)
    for start in range(0,nboot,200):
        end=min(nboot,start+200)
        idx=rng.integers(0,len(vals),size=(end-start,len(vals)))
        means[start:end]=vals[idx].mean(axis=1)
    return float(vals.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975)),len(vals)

def verify_ids(ids,genes,truth,path):
    ids=np.asarray(ids).astype(str);genes=np.asarray(genes).astype(str)
    if len(set(ids))!=len(ids) or len(set(genes))!=len(genes):raise ValueError(f'Duplicate IDs: {path}')
    if truth.shape!=(len(ids),len(genes)) or not np.isfinite(truth).all():raise ValueError(f'Bad truth: {path}')
    return ids,genes

def aligned(reference_ids,reference_genes,reference_truth,ids,genes,truth,pred,path,tol=2e-6):
    ids,genes=verify_ids(ids,genes,truth,path)
    if set(ids)!=set(reference_ids) or not np.array_equal(genes,reference_genes):raise ValueError(f'ID/gene mismatch: {path}')
    ix={k:i for i,k in enumerate(ids)}
    ind=[ix[k] for k in reference_ids]
    if pred.shape!=truth.shape or not np.isfinite(pred).all():raise ValueError(f'Bad predictions: {path}')
    if not np.allclose(truth[ind],reference_truth,rtol=0,atol=tol):raise ValueError(f'Truth mismatch: {path}')
    return pred[ind].astype(np.float32)

def draw(paired,segments,consistency,out,formats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(10,3.8))
    for i,(dataset,threshold) in enumerate((('K562_outer_test',0),('RPE1_seen_transfer',5))):
        d=paired[(paired.dataset==dataset)&(paired.threshold==threshold)&(paired.model_a=='gated_no_residual')&(paired.model_b=='no_gate_direct')&(paired.metric=='mse')]
        if len(d):
            v=d['difference'].to_numpy();ax[i].hist(v,bins=35,edgecolor='white',linewidth=.4)
            ax[i].axvline(0,color='black',ls='--',lw=1);ax[i].axvline(np.mean(v),color='tab:red',lw=1.3)
        ax[i].set_title('K562 outer-test' if i==0 else 'RPE1 batch5')
        ax[i].set_xlabel('Per-perturbation MSE: Gated − No-Gate');ax[i].set_ylabel('Perturbations')
    fig.tight_layout()
    for f in formats:fig.savefig(out/f'Figure4B_gate_paired_mse.{f}',bbox_inches='tight')
    plt.close(fig)
    d=segments[(segments.model_a=='gated_no_residual')&(segments.model_b=='no_gate_direct')&(segments.metric=='mse')&(segments.dataset=='RPE1_seen_transfer')&(segments.threshold==5)]
    if len(d):
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for ax,(feature,part) in zip(axes,d.groupby('feature',sort=True)):
            part=part.sort_values('bin_index');xx=np.arange(len(part))
            y=part['mean_difference'].to_numpy();lo=part['ci_low'].to_numpy();hi=part['ci_high'].to_numpy()
            ax.bar(xx,y);ax.errorbar(xx,y,yerr=np.vstack([np.maximum(0,y-lo),np.maximum(0,hi-y)]),fmt='none',color='black',capsize=3)
            ax.axhline(0,color='black',lw=.8,ls='--');ax.set_xticks(xx,part['bin'].to_list());ax.set_title(feature.replace('_',' '));ax.set_ylabel('Gated − No-Gate MSE')
        fig.tight_layout()
        for f in formats:fig.savefig(out/f'Figure4C_RPE1_gate_subgroups.{f}',bbox_inches='tight')
        plt.close(fig)
    d=consistency[(consistency.threshold==5)&(consistency.model_a=='gated_no_residual')&(consistency.model_b=='no_gate_direct')]
    if len(d):
        fig,ax=plt.subplots(figsize=(5,4.2));ax.scatter(d.k562_gain,d.rpe1_gain,s=10,alpha=.4,rasterized=True)
        ax.axhline(0,color='gray',ls='--',lw=.8);ax.axvline(0,color='gray',ls='--',lw=.8)
        rho,n=corr(d.k562_gain,d.rpe1_gain);ax.set_title(f'Gate benefit concordance | rho={rho:.3f}, n={n}')
        ax.set_xlabel('K562 MSE improvement from Gate');ax.set_ylabel('RPE1 MSE improvement from Gate')
        fig.tight_layout()
        for f in formats:fig.savefig(out/f'Figure4D_gate_benefit_consistency.{f}',bbox_inches='tight')
        plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--step09c',default='results/step09C_direct_ablation_transfer')
    ap.add_argument('--step08c',default='results/step08C_k562_to_rpe1_transfer')
    ap.add_argument('--outdir',default='results/step09D_paired_gate_mechanism')
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    ap.add_argument('--thresholds',type=int,nargs='+',default=[3,5,10])
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--rng-seed',type=int,default=20261109)
    ap.add_argument('--formats',nargs='+',choices=['pdf','svg','png'],default=['pdf','svg'])
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.bootstrap<100 or len(set(args.seeds))!=len(args.seeds) or len(set(args.thresholds))!=len(args.thresholds):raise ValueError('Invalid parameters')
    root=Path(args.root).resolve();src=root/args.step09c;ref=root/args.step08c;out=root/args.outdir
    kpath=root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
    k_ids,k_genes,k_delta=load(kpath,['perturbations','genes','delta'])
    k_ids,k_genes=verify_ids(k_ids,k_genes,k_delta,kpath)
    k_ix={k:i for i,k in enumerate(k_ids)};kg_ix={g:i for i,g in enumerate(k_genes)}
    split_dir=root/'results/step05_tuned_genept_uce/inner_splits'
    records=[];audit=[]
    for seed in args.seeds:
        split=json.loads((split_dir/f'seed_{seed}.json').read_text());te=np.array(split['outer_test']).astype(str)
        if not set(te).issubset(k_ix):raise ValueError('Unknown K562 test IDs')
        scenarios=[('K562_outer_test',0)]+[('RPE1_seen_transfer',t) for t in args.thresholds]
        for dataset,th in scenarios:
            if th==0:
                # Step09B frozen gated model: K562 test predictions, and Step09C no-gate/single-modality predictions.
                gp=root/'results/step09B_ccrt_formal_ablation'/f'seed_{seed}'/'gated_no_residual'/'predictions.npz'
                ids,genes,truth,p=load(gp,['test_perturbations','genes','y_true','prediction'])
                ids,genes=verify_ids(ids,genes,truth,gp)
                if set(ids)!=set(te) or not np.array_equal(genes,k_genes):raise ValueError(f'K562 split/gene mismatch seed={seed}')
                pred={'gated_no_residual':p}
                for model in DIRECT[1:]:
                    path=src/f'seed_{seed}'/model/'predictions.npz'
                    a,b,c,d=load(path,['test_perturbations','genes','y_true','prediction'])
                    pred[model]=aligned(ids,genes,truth,a,b,c,d,path)
            else:
                gp=src/f'seed_{seed}'/'gated_no_residual'/f'RPE1_batch{th}'/'predictions.npz'
                ids,genes,truth,p=load(gp,['perturbations','genes','y_true','prediction'])
                ids,genes=verify_ids(ids,genes,truth,gp)
                if not set(ids).issubset(set(split['outer_train'])):raise ValueError('RPE1 perturbation not seen in K562 outer train')
                pred={'gated_no_residual':p}
                for model in DIRECT[1:]:
                    path=src/f'seed_{seed}'/model/f'RPE1_batch{th}'/'predictions.npz'
                    a,b,c,d=load(path,['perturbations','genes','y_true','prediction'])
                    pred[model]=aligned(ids,genes,truth,a,b,c,d,path)
                path=ref/f'batch{th}'/f'seed_{seed}'/'predictions.npz'
                a,b,c,d=load(path,['perturbations','genes','y_true','pred_mlp_fusion'])
                pred['mlp_fusion']=aligned(ids,genes,truth,a,b,c,d,path)
            if not np.isfinite(truth).all():raise ValueError('Nonfinite true response')
            if th==0:cons=np.full(len(ids),np.nan)
            else:
                common=np.array([kg_ix[g] for g in genes]);ki=np.array([k_ix[k] for k in ids]);cons=pcc_rows(k_delta[np.ix_(ki,common)],truth)
            rms=np.sqrt(np.mean(truth.astype(np.float64)**2,axis=1))
            active=(np.abs(truth)>=0.1).sum(axis=1)
            for model,p in pred.items():
                if p.shape!=truth.shape or not np.isfinite(p).all():raise ValueError(f'Invalid {model} predictions')
                mse=np.mean((p.astype(np.float64)-truth.astype(np.float64))**2,axis=1)
                pc=pcc_rows(truth,p)
                for i,pert in enumerate(ids):
                    records.append({'dataset':dataset,'threshold':th,'seed':seed,'perturbation':pert,'model':model,
                        'mse':float(mse[i]),'pcc':float(pc[i]),'true_rms':float(rms[i]),
                        'n_active_01':int(active[i]),'cross_cell_true_pcc':float(cons[i])})
            audit.append({'dataset':dataset,'threshold':th,'seed':seed,'n_perturbations':len(ids),'n_genes':len(genes),'models':','.join(pred)})
            print(f'[AUDIT] {dataset} batch={th} seed={seed} n={len(ids)} genes={len(genes)}',flush=True)
    if args.dry_run:
        print('[DRY RUN PASS] All predictions aligned; no analysis files written');return
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(audit).to_csv(out/'alignment_audit.csv',index=False)
    df=pd.DataFrame(records);df.to_csv(out/'per_perturbation_metrics.csv',index=False)
    rng=np.random.default_rng(args.rng_seed)
    paired=[];summary=[]
    for (dataset,th),sub in df.groupby(['dataset','threshold'],sort=True):
        for a,b in COMPARISONS:
            if a not in set(sub.model) or b not in set(sub.model):continue
            for metric in ('mse','pcc'):
                piv=sub.pivot(index=['seed','perturbation'],columns='model',values=metric)
                diff=(piv[a]-piv[b]).dropna()
                if diff.empty:continue
                per=diff.groupby(level='perturbation').mean()
                mean,lo,hi,n=bootstrap(per.to_numpy(),args.bootstrap,rng)
                summary.append({'dataset':dataset,'threshold':th,'model_a':a,'model_b':b,'metric':metric,
                                'mean_difference':mean,'ci_low':lo,'ci_high':hi,'n_unique_perturbations':n,
                                'fraction_a_better':float((per<0).mean() if metric=='mse' else (per>0).mean()),
                                'n_seed_perturbation_pairs':len(diff)})
                for pert,v in per.items():paired.append({'dataset':dataset,'threshold':th,'model_a':a,'model_b':b,'metric':metric,'perturbation':pert,'difference':float(v)})
    pd.DataFrame(summary).to_csv(out/'paired_bootstrap_summary.csv',index=False)
    pairs=pd.DataFrame(paired);pairs.to_csv(out/'paired_per_perturbation.csv',index=False)
    # Pre-specified thirds of RPE1 response magnitude, response density and observed conservation.
    segment=[];bins_table=[]
    for th in args.thresholds:
        sub=df[(df.dataset=='RPE1_seen_transfer')&(df.threshold==th)]
        # Average repeated seed measurements for each perturbation; avoid seed-row pseudoreplication.
        features=sub[sub.model=='gated_no_residual'].groupby('perturbation')[['true_rms','n_active_01','cross_cell_true_pcc']].mean()
        for feature in features.columns:
            v=features[feature].dropna()
            if len(v)<9:continue
            ranks=v.rank(method='first'); labels=pd.qcut(ranks,q=3,labels=['low','mid','high'])
            for bi,lab in enumerate(('low','mid','high')):
                ids=set(labels[labels==lab].index)
                bins_table.append({'threshold':th,'feature':feature,'bin':lab,'n_perturbations':len(ids),'min_value':float(v.loc[list(ids)].min()),'max_value':float(v.loc[list(ids)].max())})
                for a,b in COMPARISONS:
                    for metric in ('mse','pcc'):
                        dd=pairs[(pairs.dataset=='RPE1_seen_transfer')&(pairs.threshold==th)&(pairs.model_a==a)&(pairs.model_b==b)&(pairs.metric==metric)&(pairs.perturbation.isin(ids))]
                        if len(dd)<3:continue
                        mean,lo,hi,n=bootstrap(dd.difference.to_numpy(),args.bootstrap,rng)
                        segment.append({'dataset':'RPE1_seen_transfer','threshold':th,'feature':feature,'bin':lab,'bin_index':bi,'model_a':a,'model_b':b,'metric':metric,'mean_difference':mean,'ci_low':lo,'ci_high':hi,'n_unique_perturbations':n})
    seg=pd.DataFrame(segment);seg.to_csv(out/'subgroup_bootstrap_summary.csv',index=False)
    pd.DataFrame(bins_table).to_csv(out/'subgroup_boundaries.csv',index=False)
    consistency=[];con_summary=[]
    for th in args.thresholds:
        for a,b in COMPARISONS:
            if b=='mlp_fusion':continue # MLP K562 predictions not part of this step's matched K562 set.
            k=pairs[(pairs.dataset=='K562_outer_test')&(pairs.threshold==0)&(pairs.model_a==a)&(pairs.model_b==b)&(pairs.metric=='mse')][['perturbation','difference']].rename(columns={'difference':'k_diff'})
            r=pairs[(pairs.dataset=='RPE1_seen_transfer')&(pairs.threshold==th)&(pairs.model_a==a)&(pairs.model_b==b)&(pairs.metric=='mse')][['perturbation','difference']].rename(columns={'difference':'r_diff'})
            d=k.merge(r,on='perturbation',how='inner',validate='one_to_one')
            if d.empty:continue
            rho,n=corr(d.k_diff,d.r_diff)
            con_summary.append({'threshold':th,'model_a':a,'model_b':b,'n_shared_test_perturbations':n,'spearman_rho':rho,
                                'note':'K562 outer-test vs RPE1 seen-transfer sets are disjoint within each seed; overlap may occur across different seeds. Interpret descriptively.'})
            for row in d.itertuples(index=False):consistency.append({'threshold':th,'model_a':a,'model_b':b,'perturbation':row.perturbation,'k562_gain':-row.k_diff,'rpe1_gain':-row.r_diff})
    consdf=pd.DataFrame(consistency);consdf.to_csv(out/'cross_cell_gate_consistency.csv',index=False)
    pd.DataFrame(con_summary).to_csv(out/'cross_cell_consistency_summary.csv',index=False)
    draw(pairs,seg,consdf,out,args.formats)
    (out/'analysis_config.json').write_text(json.dumps({'args':vars(args),'caveats':['Exploratory: test results previously inspected','Bootstrap resamples perturbations, not independent biological replicates','Subgroups based on RPE1 truth are explanatory, not deployable predictors','Within a seed, K562 test and RPE1 seen-transfer perturbations are disjoint; cross-cell overlap across seeds is descriptive']},indent=2))
    print('\n=== PRIMARY PAIRED RESULTS ===',flush=True)
    print(pd.DataFrame(summary).query("model_a == 'gated_no_residual' and model_b == 'no_gate_direct'").to_string(index=False),flush=True)
    print('[COMPLETE]',out,flush=True)
if __name__=='__main__':main()
