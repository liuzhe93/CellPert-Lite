#!/usr/bin/env python3
"""Step10H: audit prediction metrics, baselines, response strata, and CCRT gamma calibration.

Uses frozen Step10A/C/E/G predictions. No RPE1 labels are used for calibration.
Gamma is calibrated on K562 inner_val using a model fitted only on inner_train,
with Step10E-selected hyperparameters. Because those hyperparameters were already
selected using inner_val, gamma selection is exploratory (NOT a fully nested,
independent validation). Existing outer tests have also been inspected; all
new performance comparisons are exploratory, not unbiased confirmation.
"""
import argparse, importlib.util, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

SOURCES={'full_ccrt':('e','full_ccrt'),'ccrt_oof':('o','full_ccrt'),
 'gated_direct':('a','gated_direct'),'no_gate_direct':('a','no_gate_direct'),
 'genept_only_direct':('a','genept_only_direct'),'uce_only_direct':('a','uce_only_direct'),
 'mlp_fusion':('c','mlp_fusion'),'ridge_fusion':('c','ridge_fusion')}

def npz(path,keys):
    with np.load(path,allow_pickle=False) as f:
        return [f[k] for k in keys]

def pred_path(d,seed,model,th):
    b=d/f'seed_{seed}'/model
    return b/'K562_holdout_predictions.npz' if th==0 else b/f'RPE1_batch{th}'/'predictions.npz'

def corr(x,y,rank=False):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if rank:x=rankdata(x);y=rankdata(y)
    x-=x.mean();y-=y.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den>1e-12 else np.nan

def safe_ratio(a,b):return float(a/b) if b>1e-14 else np.nan

def metrics(y,p,zero,mean,kvals,eps=1e-8):
    yy=y.astype(np.float64);pp=p.astype(np.float64);zz=zero.astype(np.float64);mm=mean.astype(np.float64)
    mse=np.mean((yy-pp)**2);zerr=np.mean((yy-zz)**2);merr=np.mean((yy-mm)**2)
    d={'mse':mse,'zero_mse':zerr,'train_mean_mse':merr,
       'per_pert_r2_vs_zero':1-safe_ratio(mse,zerr),
       'per_pert_skill_vs_train_mean':1-safe_ratio(mse,merr),
       'pcc':corr(yy,pp),'spearman':corr(yy,pp,True),
       'true_rms':float(np.sqrt(np.mean(yy**2))),'pred_rms':float(np.sqrt(np.mean(pp**2))),
       'better_than_zero':float(mse<zerr),'better_than_train_mean':float(mse<merr)}
    n=len(y)
    for k in kvals:
        kk=min(k,n); top=np.argsort(-np.abs(yy),kind='stable')[:kk]
        if np.ptp(np.abs(pp))<eps:
            overlap=np.nan
        else:
            pt=np.argsort(-np.abs(pp),kind='stable')[:kk]
            overlap=len(set(top)&set(pt))/kk
        d[f'top{k}_overlap']=overlap
        valid=np.abs(pp[top])>eps
        d[f'top{k}_direction_coverage']=float(valid.mean())
        d[f'top{k}_direction_accuracy']=float(np.mean(np.sign(pp[top][valid])==np.sign(yy[top][valid]))) if valid.any() else np.nan
        d[f'top{k}_direction_accuracy_all']=float(np.mean((np.sign(pp[top])==np.sign(yy[top])) & valid))
    return d

def bh(p):
    p=np.asarray(p,dtype=float);out=np.full(len(p),np.nan);valid=np.where(np.isfinite(p))[0]
    if not len(valid):return out
    ix=valid[np.argsort(p[valid])];vals=p[ix];q=np.minimum.accumulate((vals*len(vals)/np.arange(1,len(vals)+1))[::-1])[::-1]
    out[ix]=np.minimum(q,1);return out

def bootstrap_pairs(frame,a,b,metric,nboot,rng):
    sub=frame[frame.model.isin((a,b))]
    tab=sub.pivot_table(index=['seed','perturbation'],columns='model',values=metric,aggfunc='first')
    if a not in tab or b not in tab:return None
    tab=tab[[a,b]].dropna();group=(tab[a]-tab[b]).groupby(level='perturbation').mean()
    vals=group.to_numpy();n=len(vals)
    if n<5:return None
    obs=float(vals.mean());samples=[]
    for start in range(0,nboot,100):
        batch=min(100,nboot-start);idx=rng.integers(0,n,size=(batch,n));samples.extend(vals[idx].mean(axis=1))
    lo,hi=np.quantile(samples,[.025,.975]);# two-sided bootstrap sign-tail estimate; exploratory
    pv=min(1.,2*min((np.sum(np.asarray(samples)<=0)+1)/(nboot+1),(np.sum(np.asarray(samples)>=0)+1)/(nboot+1)))
    return {'model_a':a,'model_b':b,'metric':metric,'difference':obs,'ci_low':lo,'ci_high':hi,'p_boot_approx':pv,
            'n_unique_perturbations':n,'n_seed_pairs':len(tab)}

def load_module(path):
    spec=importlib.util.spec_from_file_location('step06b_for_step10h',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def calibrated_gammas(root,dirs,seeds,grid,device,threads,bs,lr,penalty,dropout):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(threads)
    mod=load_module(root/'scripts/step06B_ccrt_model.py')
    perts,genes,y=npz(root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz',['perturbations','genes','delta'])
    ids1,x1=npz(root/'results/step03_genept_uce_common/aligned_features/GenePT_3large.npz',['genes','embeddings'])
    ids2,x2=npz(root/'results/step03_genept_uce_common/aligned_features/UCE_input_tokens.npz',['genes','embeddings'])
    perts=perts.astype(str)
    if not (np.array_equal(perts,ids1.astype(str)) and np.array_equal(perts,ids2.astype(str))):raise ValueError('Embedding alignment mismatch')
    lookup={v:i for i,v in enumerate(perts)};selected={};rows=[]
    with threadpool_limits(limits=threads):
        for seed in seeds:
            split=json.loads((dirs['a']/f'seed_{seed}/split.json').read_text())
            it=np.array([lookup[v] for v in split['inner_train']]);iv=np.array([lookup[v] for v in split['inner_val']])
            sel=json.loads((dirs['e']/f'seed_{seed}/full_ccrt/selection.json').read_text())
            best=sel['best'];alpha=float(sel['alpha']);ep=int(best['selected_epoch'])
            base=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(x1[it],y[it]);rtr=(y[it]-base.predict(x1[it])).astype(np.float32)
            (at,bt),(av,bv),_=mod.transform_blocks(x1,x2,it,iv)
            model,_,_,_=mod.fit_residual(at,bt,rtr,seed,device,int(best['width']),dropout,float(best['weight_decay']),ep,bs,lr,
                                         residual_penalty=penalty)
            resid,_=mod.predict_residual(model,av,bv,device,batch=bs)
            bpred=base.predict(x1[iv]);truth=y[iv];losses=[]
            for g in grid:
                loss=float(np.mean((truth-bpred-g*resid)**2));losses.append((loss,g))
                rows.append({'seed':seed,'gamma':g,'inner_val_mse':loss,'best_epoch':ep,'alpha':alpha})
            selected[seed]=min(losses,key=lambda t:(t[0],t[1]))[1]
            print(f'[CALIBRATION] seed={seed} gamma={selected[seed]} inner_val_mse={min(losses)[0]:.8f}',flush=True)
    return selected,pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root',default='.');p.add_argument('--step10a',default='results/step10A_unseen_perturbation_transfer')
    p.add_argument('--step10c',default='results/step10C_strong_baseline_unseen');p.add_argument('--step10e',default='results/step10E_full_ccrt_unseen')
    p.add_argument('--step10g-oof',default='results/step10G_ccrt_oof');p.add_argument('--outdir',default='results/step10H_metric_validation_residual_calibration')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46]);p.add_argument('--thresholds',nargs='+',type=int,default=[3,5,10])
    p.add_argument('--topk',nargs='+',type=int,default=[20,50,100]);p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--rng-seed',type=int,default=20261111);p.add_argument('--calibrate',action='store_true')
    p.add_argument('--gamma-grid',nargs='+',type=float,default=[0.,.25,.5,.75,1.]);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=8);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--residual-penalty',type=float,default=.1);p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.bootstrap<100 or any(k<1 for k in a.topk) or len(set(a.seeds))!=len(a.seeds) or len(set(a.thresholds))!=len(a.thresholds):p.error('Invalid arguments')
    root=Path(a.root).resolve();dirs={'a':root/a.step10a,'c':root/a.step10c,'e':root/a.step10e,'o':root/a.step10g_oof}
    out=root/a.outdir;out.mkdir(parents=True,exist_ok=True) if not a.dry_run else None
    for seed in a.seeds:
        split=json.loads((dirs['a']/f'seed_{seed}/split.json').read_text())
        assert not (set(split['train'])&set(split['test']))
        for model,(src,folder) in SOURCES.items():
            if not pred_path(dirs[src],seed,folder,0).exists():raise FileNotFoundError(pred_path(dirs[src],seed,folder,0))
        if not (dirs['e']/f'seed_{seed}/full_ccrt/ridge_base.joblib').exists():raise FileNotFoundError('Missing internal Ridge')
        print(f'[CHECK] seed={seed} train={len(split["train"])} test={len(split["test"])}',flush=True)
    if a.calibrate and not (root/'scripts/step06B_ccrt_model.py').is_file():raise FileNotFoundError('Need scripts/step06B_ccrt_model.py')
    if a.dry_run:print('[DRY RUN PASS] Core input files exist; no training or evaluation started');return
    from joblib import load as joblib_load
    ids,x1=npz(root/'results/step03_genept_uce_common/aligned_features/GenePT_3large.npz',['genes','embeddings']);emap={v:i for i,v in enumerate(ids.astype(str))}
    ids,genes,y=npz(root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz',['perturbations','genes','delta'])
    pmap={v:i for i,v in enumerate(ids.astype(str))};gmap={v:i for i,v in enumerate(genes.astype(str))}
    chosen={s:1. for s in a.seeds}
    if a.calibrate:
        chosen,calib=calibrated_gammas(root,dirs,a.seeds,a.gamma_grid,a.device,a.threads,a.batch_size,a.lr,a.residual_penalty,a.dropout)
        calib.to_csv(out/'k562_inner_validation_gamma_curve.csv',index=False)
    pd.DataFrame([{'seed':s,'gamma':chosen[s],'source':'K562_inner_val_exploratory' if a.calibrate else 'fixed_1_no_calibration'} for s in a.seeds]).to_csv(out/'gamma_choices.csv',index=False)
    rows=[];pooled=[];audit=[];strata=[];random_rows=[]
    rng=np.random.default_rng(a.rng_seed)
    for seed in a.seeds:
        split=json.loads((dirs['a']/f'seed_{seed}/split.json').read_text());train=np.array([pmap[v] for v in split['train']]);trainmean=y[train].mean(axis=0)
        ridge=joblib_load(dirs['e']/f'seed_{seed}/full_ccrt/ridge_base.joblib')
        for th in [0]+a.thresholds:
            ref=pred_path(dirs['a'],seed,'gated_direct',th)
            if not ref.exists():raise FileNotFoundError(ref)
            pp,gg,yy,_=npz(ref,['perturbations','genes','y_true','prediction']);pp=pp.astype(str);gg=gg.astype(str)
            if len(set(pp))!=len(pp) or len(set(gg))!=len(gg):raise ValueError('Duplicate IDs')
            if not set(pp).issubset(set(split['test'])):raise ValueError('Test split mismatch')
            if not np.isfinite(yy).all():raise ValueError('Nonfinite truth')
            gj=np.array([gmap[v] for v in gg]);mean=np.broadcast_to(trainmean[gj],yy.shape)
            base=ridge.predict(x1[[emap[v] for v in pp]])[:,gj]
            preds={'zero_change':np.zeros_like(yy),'k562_train_mean':mean,'ccrt_internal_ridge':base}
            for model,(src,folder) in SOURCES.items():
                path=pred_path(dirs[src],seed,folder,th)
                if not path.exists():raise FileNotFoundError(path)
                ii,jj,truth,pred=npz(path,['perturbations','genes','y_true','prediction']);ii=ii.astype(str);jj=jj.astype(str)
                if set(ii)!=set(pp) or set(jj)!=set(gg):raise ValueError(f'Prediction IDs/genes differ: {path}')
                ri={v:i for i,v in enumerate(ii)};ci={v:i for i,v in enumerate(jj)}
                pred=pred[np.ix_([ri[v] for v in pp],[ci[v] for v in gg])]
                truth=truth[np.ix_([ri[v] for v in pp],[ci[v] for v in gg])]
                if not np.allclose(truth,yy,rtol=1e-5,atol=1e-6):raise ValueError(f'Truth mismatch: {path}')
                if not np.isfinite(pred).all():raise ValueError(f'Nonfinite prediction: {path}')
                preds[model]=pred
            preds['ccrt_gamma_calibrated']=base+chosen[seed]*(preds['full_ccrt']-base)
            label='K562_unseen' if th==0 else 'RPE1_unseen_transfer'
            for model,pred in preds.items():
                err=(yy.astype(np.float64)-pred.astype(np.float64))**2
                zerr=np.mean(yy.astype(np.float64)**2);merr=np.mean((yy.astype(np.float64)-mean.astype(np.float64))**2)
                pmse=float(err.mean());pooled.append({'dataset':label,'threshold':th,'seed':seed,'model':model,'mse':pmse,
                    'pooled_r2_vs_zero':1-pmse/zerr if zerr>1e-14 else np.nan,
                    'pooled_skill_vs_train_mean':1-pmse/merr if merr>1e-14 else np.nan,
                    'global_pcc':corr(yy.ravel(),pred.ravel()),'n_perturbations':len(pp),'n_genes':len(gg)})
                for i,pid in enumerate(pp):
                    rec={'dataset':label,'threshold':th,'seed':seed,'perturbation':pid,'model':model}
                    rec.update(metrics(yy[i],pred[i],np.zeros_like(yy[i]),mean[i],a.topk));rows.append(rec)
                    for k in a.topk:
                        if model in ('full_ccrt','gated_direct','ccrt_internal_ridge','k562_train_mean'):
                            top_true=set(np.argsort(-np.abs(yy[i]),kind='stable')[:k]);random_rows.append({'dataset':label,'threshold':th,'seed':seed,'perturbation':pid,'model':model,'topk':k,
                             'random_expected_overlap':min(k,len(gg))/len(gg),'observed_overlap':rec[f'top{k}_overlap']})
                    if model in ('full_ccrt','gated_direct','ccrt_internal_ridge','ccrt_gamma_calibrated','k562_train_mean'):
                        # Strata defined using true response RMS; descriptive test-only subgrouping, not selection.
                        strata.append({'dataset':label,'threshold':th,'seed':seed,'perturbation':pid,'model':model,
                            'true_rms':rec['true_rms'],'mse':rec['mse'],'pcc':rec['pcc'],
                            'skill_vs_zero':rec['per_pert_r2_vs_zero']})
            audit.append({'dataset':label,'threshold':th,'seed':seed,'n_perturbations':len(pp),'n_genes':len(gg),'n_models':len(preds),'gamma':chosen[seed]})
            print(f'[EVAL] seed={seed} threshold={th} perturbations={len(pp)} models={len(preds)}',flush=True)
    df=pd.DataFrame(rows);pf=pd.DataFrame(pooled);pd.DataFrame(audit).to_csv(out/'alignment_audit.csv',index=False)
    df.to_csv(out/'per_perturbation_metrics_validated.csv',index=False);pf.to_csv(out/'pooled_metrics_per_seed.csv',index=False)
    sumcols=['mse','per_pert_r2_vs_zero','per_pert_skill_vs_train_mean','pcc','spearman','better_than_zero','better_than_train_mean']+[f'top{k}_overlap' for k in a.topk]
    df.groupby(['dataset','threshold','model'],as_index=False)[sumcols].mean().to_csv(out/'mean_per_perturbation_metrics.csv',index=False)
    pf.groupby(['dataset','threshold','model'],as_index=False)[['mse','pooled_r2_vs_zero','pooled_skill_vs_train_mean','global_pcc']].mean().to_csv(out/'pooled_metrics_summary.csv',index=False)
    pd.DataFrame(random_rows).to_csv(out/'random_ranking_expectations.csv',index=False)
    st=pd.DataFrame(strata);groups=st[['dataset','threshold','seed','perturbation','true_rms']].drop_duplicates().copy()
    # quantile bins per dataset/threshold on unique perturbation IDs, not repeated seed observations
    for (dataset,th),g in groups.groupby(['dataset','threshold']):
        means=g.groupby('perturbation')['true_rms'].mean();qs=np.quantile(means,[1/3,2/3])
        if qs[0]==qs[1]:warnings.warn(f'Strata quantiles tied: {dataset}/{th}')
        mapping={pid:('low' if v<=qs[0] else 'medium' if v<=qs[1] else 'high') for pid,v in means.items()}
        ix=(st.dataset==dataset)&(st.threshold==th);st.loc[ix,'response_stratum']=st.loc[ix,'perturbation'].map(mapping)
    st.to_csv(out/'response_strength_per_perturbation.csv',index=False)
    st.groupby(['dataset','threshold','response_stratum','model'],as_index=False)[['mse','pcc','skill_vs_zero']].mean().to_csv(out/'response_strength_summary.csv',index=False)
    comps=[('full_ccrt','ccrt_internal_ridge'),('ccrt_gamma_calibrated','ccrt_internal_ridge'),('ccrt_gamma_calibrated','full_ccrt'),
           ('gated_direct','full_ccrt'),('gated_direct','k562_train_mean'),('full_ccrt','k562_train_mean')]
    boot=[]
    for (dataset,th),group in df.groupby(['dataset','threshold']):
        for aa,bb in comps:
            for met in ['mse','pcc','spearman','per_pert_r2_vs_zero','per_pert_skill_vs_train_mean']+[f'top{k}_overlap' for k in a.topk]:
                v=bootstrap_pairs(group,aa,bb,met,a.bootstrap,rng)
                if v:boot.append({'dataset':dataset,'threshold':th,**v})
    bf=pd.DataFrame(boot)
    if not bf.empty:
        # BH within each dataset/threshold and metric family, NOT across all possible analyses.
        bf['q_bh_exploratory']=np.nan
        for _,inds in bf.groupby(['dataset','threshold','metric']).groups.items():bf.loc[list(inds),'q_bh_exploratory']=bh(bf.loc[list(inds),'p_boot_approx'])
    bf.to_csv(out/'paired_bootstrap_BH_exploratory.csv',index=False)
    (out/'run_config.json').write_text(json.dumps(vars(a),indent=2,ensure_ascii=False))
    (out/'README_interpretation.md').write_text('''# Step10H: interpretation and safeguards\n\n1. `pooled_r2_vs_zero` = 1 - sum squared model errors / sum squared true deltas, computed within each seed/dataset. `per_pert_r2_vs_zero` is a separate per-perturbation ratio, averaged across rows in `mean_per_perturbation_metrics.csv`. They are NOT interchangeable.\n2. `pooled_skill_vs_train_mean` compares against the K562 outer-training mean on the SAME gene universe. `better_than_*` reports the fraction of perturbation-seed pairs with lower MSE.\n3. Top-K response recovery uses magnitude rankings, NOT statistically significant DEGs. Random expected overlap is K/N for K selected genes among N genes. Train-mean ranking is also provided.\n4. Direction accuracy is conditional on predictions exceeding 1e-8, with coverage and unconditional direction accuracy separately. Direction accuracy must not be interpreted without a naive sign baseline.\n5. Response strength strata are constructed POST HOC using TEST truth; useful only for descriptive error analysis, never for training/model selection.\n6. `--calibrate` fits a fresh CCRT on K562 inner_train and selects gamma on K562 inner_val, using hyperparameters already selected on inner_val in Step10E. This is EXPLORATORY and not fully nested. Gamma is NEVER tuned on RPE1. Full CCRT prediction is adjusted using the frozen final full-train Ridge and CCRT residual.\n7. The same outer test datasets have been inspected repeatedly in earlier Steps. Neither gamma-adjusted performance nor new model comparisons are independent confirmatory tests. Obtain a new held-out cohort or fully nested evaluation for publication-grade claims.\n8. Bootstrap resamples unique perturbation IDs after averaging seed-level differences. Approximate sign-tail bootstrap p-values and BH q-values are EXPLORATORY; they do not fully account for model development and multiple analyses.\n9. Batch3/5/10 have different perturbation sets. Do not compare their means as if sample composition were fixed.\n''')
    print('[COMPLETE]',out,flush=True)
    print(pf.groupby(['dataset','threshold','model'])[['pooled_r2_vs_zero','pooled_skill_vs_train_mean']].mean().round(4).to_string(),flush=True)

if __name__=='__main__':main()
