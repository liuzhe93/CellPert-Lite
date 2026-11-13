#!/usr/bin/env python3
"""Step10G: CCRT OOF-residual sensitivity and biological prediction evaluation.

Phase 1 optionally runs the original Step10E with --residual-target oof into a
NEW directory. Never reuses the in-sample output directory. Model selection
remains entirely on K562 inner train/validation as implemented by Step10E.
Phase 2 aligns locked Step10A split predictions and evaluates 7 original models,
CCRT-OOF (when available), CCRT's internal Ridge-only, zero-change and
K562-train-mean baselines. RPE1 is test-only. Optional residual-scale curves
are POST-HOC DIAGNOSTICS, not a legitimate way to select a final model.

No inference of biological DE significance from delta values; top-k recovery
uses absolute response magnitude, NOT FDR-qualified DE genes.
"""
import argparse, json, subprocess, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

MODELS={'full_ccrt':('e','full_ccrt'),'gated_direct':('a','gated_direct'),
 'no_gate_direct':('a','no_gate_direct'),'genept_only_direct':('a','genept_only_direct'),
 'uce_only_direct':('a','uce_only_direct'),'mlp_fusion':('c','mlp_fusion'),
 'ridge_fusion':('c','ridge_fusion'),'ccrt_oof':('o','full_ccrt')}

def load_npz(path):
    if not path.is_file(): raise FileNotFoundError(path)
    with np.load(path,allow_pickle=False) as z:
        for k in ('perturbations','genes','y_true','prediction'):
            if k not in z.files: raise KeyError(f'{path}: missing {k}, got {z.files}')
        ids=z['perturbations'].astype(str);genes=z['genes'].astype(str)
        y=z['y_true'].astype(np.float32);p=z['prediction'].astype(np.float32)
    if len(set(ids))!=len(ids) or len(set(genes))!=len(genes) or y.shape!=(len(ids),len(genes)) or p.shape!=y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError(f'Invalid prediction file {path}')
    return ids,genes,y,p

def pred_path(d,seed,model,threshold):
    stem=d/f'seed_{seed}'/model
    return stem/'K562_holdout_predictions.npz' if threshold==0 else stem/f'RPE1_batch{threshold}'/'predictions.npz'

def pcc(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64)
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-12 else np.nan

def spearman(a,b):
    if np.std(a)<1e-12 or np.std(b)<1e-12:return np.nan
    return pcc(rankdata(a),rankdata(b))

def row_metrics(y,p,baseline,k_values):
    # All per-perturbation. Top-K evaluated by magnitude; tied zero predictions
    # are not assigned arbitrary high recovery scores.
    n=len(y); err=np.mean((y-p)**2);baseerr=np.mean((y-baseline)**2)
    d={'mse':float(err),'pcc':pcc(y,p),'spearman':spearman(y,p),
       'delta_r2_zero':float(1-err/np.mean(y*y)) if np.mean(y*y)>1e-15 else np.nan,
       'improvement_vs_train_mean':float(1-err/baseerr) if baseerr>1e-15 else np.nan,
       'true_rms':float(np.sqrt(np.mean(y*y))),'pred_rms':float(np.sqrt(np.mean(p*p)))}
    order=np.argsort(-np.abs(y),kind='stable')
    for k in k_values:
        kk=min(k,n)
        idx=order[:kk]
        nonzero=np.abs(p)>1e-12
        if np.count_nonzero(nonzero)<kk:
            overlap=np.nan  # Constant-zero predictions have no meaningful ranking
        else:
            predidx=np.argpartition(np.abs(p),-kk)[-kk:]
            overlap=float(np.intersect1d(idx,predidx).size/kk)
        d[f'top{str(k)}_overlap']=overlap
        valid=(np.abs(y[idx])>1e-12)&(np.abs(p[idx])>1e-12)
        d[f'top{str(k)}_direction_accuracy']=float(np.mean(np.sign(y[idx][valid])==np.sign(p[idx][valid]))) if valid.any() else np.nan
        d[f'top{str(k)}_direction_coverage']=float(valid.sum()/kk)
    return d

def bootstrap_diff(frame,a,b,metric,nboot,rng):
    tmp=frame[frame.model.isin([a,b])].pivot_table(index=['seed','perturbation'],columns='model',values=metric,aggfunc='first')
    if a not in tmp or b not in tmp:return None
    tmp=tmp[[a,b]].dropna();tmp['difference']=tmp[a]-tmp[b]
    per=tmp.reset_index().groupby('perturbation')['difference'].mean().values
    if len(per)<3:return None
    draws=[]
    for _ in range(0,nboot,100):
        sz=min(100,nboot-len(draws)); ix=rng.integers(0,len(per),(sz,len(per)))
        draws.extend(per[ix].mean(axis=1).tolist())
    return {'model_a':a,'model_b':b,'metric':metric,'mean_difference':float(per.mean()),
            'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975)),
            'n_unique_perturbations':len(per),'n_pairs':len(tmp),
            'ci_excludes_zero':bool(np.quantile(draws,.025)>0 or np.quantile(draws,.975)<0)}

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--root',default='.')
    ap.add_argument('--step10a',default='results/step10A_unseen_perturbation_transfer')
    ap.add_argument('--step10c',default='results/step10C_strong_baseline_unseen')
    ap.add_argument('--step10e',default='results/step10E_full_ccrt_unseen')
    ap.add_argument('--step10e-oof',default='results/step10G_ccrt_oof')
    ap.add_argument('--outdir',default='results/step10G_ccrt_rescue_biological_evaluation')
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    ap.add_argument('--thresholds',nargs='+',type=int,default=[3,5,10])
    ap.add_argument('--topk',nargs='+',type=int,default=[20,50,100])
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--rng-seed',type=int,default=20261111)
    ap.add_argument('--run-oof',action='store_true',help='Train OOF CCRT via the original Step10E in a new output directory')
    ap.add_argument('--oof-folds',type=int,default=5)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or len(set(a.thresholds))!=len(a.thresholds) or any(k<1 for k in a.topk) or a.bootstrap<100:ap.error('Invalid seed, threshold, topk or bootstrap')
    root=Path(a.root).resolve(); dirs={'a':root/a.step10a,'c':root/a.step10c,'e':root/a.step10e,'o':root/a.step10e_oof}
    if dirs['o'].resolve()==dirs['e'].resolve():ap.error('OOF output must differ from original Step10E output')
    script=root/'scripts/step10E_full_ccrt_unseen_benchmark.py'
    if a.run_oof and not script.is_file():raise FileNotFoundError(script)
    # For dry-run, validate original predictions before launching expensive training.
    for seed in a.seeds:
        split=json.loads((dirs['a']/f'seed_{seed}/split.json').read_text())
        if set(split['train']) & set(split['test']):raise ValueError('Split leakage')
        for model,(key,folder) in MODELS.items():
            if model=='ccrt_oof' and (a.run_oof or not pred_path(dirs['o'],seed,folder,0).is_file()):continue
            path=pred_path(dirs[key],seed,folder,0)
            if not path.is_file():raise FileNotFoundError(path)
        print(f'[CHECK] seed={seed} split={len(split["train"])} train / {len(split["test"])} test',flush=True)
    if a.dry_run:
        print('[DRY RUN PASS] Existing model paths and splits checked. OOF training not started.');return
    if a.run_oof:
        cmd=[sys.executable,str(script),'--root',str(root),'--outdir',str(dirs['o']),
             '--residual-target','oof','--oof-folds',str(a.oof_folds),'--seeds',*[str(x) for x in a.seeds],
             '--thresholds',*[str(x) for x in a.thresholds],'--ridge-alphas','1','100','10000',
             '--widths','128','--weight-decays','0.001','0.01','--epochs','100',
             '--patience','20','--batch-size','128','--lr','0.001',
             '--threads',str(a.threads),'--device',a.device]
        print('[OOF TRAIN] Running original Step10E with training-only OOF residuals',flush=True)
        subprocess.run(cmd,check=True)
    available={k:v for k,v in MODELS.items() if k!='ccrt_oof' or all(pred_path(dirs['o'],s,'full_ccrt',0).is_file() for s in a.seeds)}
    from joblib import load as joblib_load
    with np.load(root/'results/step03_genept_uce_common/aligned_features/GenePT_3large.npz',allow_pickle=False) as z:
        embids=z['genes'].astype(str);emb=z['embeddings'].astype(np.float32)
    emap={x:i for i,x in enumerate(embids)}
    with np.load(root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz',allow_pickle=False) as z:
        allids=z['perturbations'].astype(str);allgenes=z['genes'].astype(str);alldelta=z['delta'].astype(np.float32)
    ymap={x:i for i,x in enumerate(allids)};gmap={x:i for i,x in enumerate(allgenes)}
    out=root/a.outdir;out.mkdir(parents=True,exist_ok=True)
    records=[];globalrows=[];alignment=[]
    for seed in a.seeds:
        split=json.loads((dirs['a']/f'seed_{seed}/split.json').read_text())
        trainids=split['train']; trainmean=alldelta[[ymap[x] for x in trainids]].mean(axis=0)
        base=joblib_load(dirs['e']/f'seed_{seed}/full_ccrt/ridge_base.joblib')
        for threshold in [0]+a.thresholds:
            ref=pred_path(dirs['a'],seed,'gated_direct',threshold)
            if not ref.is_file():
                if threshold>0:continue
                raise FileNotFoundError(ref)
            ids,genes,y,_=load_npz(ref)
            if not set(ids).issubset(set(split['test'])):raise ValueError(f'Not locked test perturbations {seed}/{threshold}')
            geneidx=np.array([gmap[g] for g in genes]);trainpred=np.broadcast_to(trainmean[geneidx],y.shape)
            # Ridge-only uses original CCRT base estimator (not Ridge-Fusion).
            ridge=base.predict(emb[[emap[x] for x in ids]])[:,geneidx].astype(np.float32)
            predictions={'zero_change':np.zeros_like(y),'k562_train_mean':trainpred,'ccrt_internal_ridge':ridge}
            for model,(key,folder) in available.items():
                path=pred_path(dirs[key],seed,folder,threshold)
                if not path.is_file():raise FileNotFoundError(path)
                ii,gg,yy,pp=load_npz(path)
                if set(ii)!=set(ids) or set(gg)!=set(genes):raise ValueError(f'ID/gene mismatch {path}')
                pi={x:i for i,x in enumerate(ii)};gi={x:i for i,x in enumerate(gg)}
                ix=np.array([pi[x] for x in ids]);gx=np.array([gi[x] for x in genes])
                if not np.allclose(yy[np.ix_(ix,gx)],y,rtol=1e-5,atol=1e-6):raise ValueError(f'Truth mismatch {path}')
                predictions[model]=pp[np.ix_(ix,gx)]
            dataset='K562_unseen' if threshold==0 else 'RPE1_unseen_transfer'
            for model,pp in predictions.items():
                mse=np.mean((y-pp)**2);globalrows.append({'dataset':dataset,'threshold':threshold,'seed':seed,'model':model,
                    'mse':float(mse),'global_pcc':pcc(y.ravel(),pp.ravel()),'n_perturbations':len(ids),'n_genes':len(genes)})
                for j,pid in enumerate(ids):
                    row={'dataset':dataset,'threshold':threshold,'seed':seed,'model':model,'perturbation':pid}
                    row.update(row_metrics(y[j],pp[j],trainpred[j],a.topk));records.append(row)
            alignment.append({'dataset':dataset,'threshold':threshold,'seed':seed,'n_perturbations':len(ids),
                              'n_genes':len(genes),'n_models':len(predictions)})
            print(f'[EVAL] seed={seed} threshold={threshold} perturbations={len(ids)} models={len(predictions)}',flush=True)
    df=pd.DataFrame(records);globaldf=pd.DataFrame(globalrows)
    df.to_csv(out/'per_perturbation_biological_metrics.csv',index=False)
    globaldf.to_csv(out/'global_metrics_per_seed.csv',index=False)
    pd.DataFrame(alignment).to_csv(out/'alignment_audit.csv',index=False)
    aggcols=['mse','pcc','spearman','delta_r2_zero','improvement_vs_train_mean','true_rms','pred_rms']+[f'top{k}_{z}' for k in a.topk for z in ('overlap','direction_accuracy','direction_coverage')]
    summary=df.groupby(['dataset','threshold','model'],as_index=False)[aggcols].mean()
    summary.to_csv(out/'biological_metrics_summary.csv',index=False)
    globaldf.groupby(['dataset','threshold','model'],as_index=False)[['mse','global_pcc']].agg(['mean','std']).to_csv(out/'global_metrics_summary.csv')
    comparisons=[('full_ccrt','gated_direct'),('full_ccrt','mlp_fusion'),('full_ccrt','ccrt_internal_ridge'),('full_ccrt','ccrt_oof'),('ccrt_oof','ccrt_internal_ridge'),('gated_direct','zero_change'),('full_ccrt','zero_change')]
    rng=np.random.default_rng(a.rng_seed);boots=[]
    for (dataset,threshold),group in df.groupby(['dataset','threshold']):
        for m1,m2 in comparisons:
            if m1 not in available and m1 not in ('ccrt_internal_ridge','zero_change','k562_train_mean'):continue
            if m2 not in available and m2 not in ('ccrt_internal_ridge','zero_change','k562_train_mean'):continue
            for metric in ['mse','pcc','spearman','delta_r2_zero','improvement_vs_train_mean']+[f'top{k}_overlap' for k in a.topk]:
                v=bootstrap_diff(group,m1,m2,metric,a.bootstrap,rng)
                if v:boots.append({'dataset':dataset,'threshold':threshold,**v})
    pd.DataFrame(boots).to_csv(out/'paired_bootstrap_biological.csv',index=False)
    # A purely descriptive residual-scaling curve on TEST: NOT tuned or selected.
    # Used to diagnose whether original residual direction is beneficial.
    scale_rows=[]
    for seed in a.seeds:
        split=json.loads((dirs['a']/f'seed_{seed}/split.json').read_text())
        base=joblib_load(dirs['e']/f'seed_{seed}/full_ccrt/ridge_base.joblib')
        for th in [0]+a.thresholds:
            path=pred_path(dirs['e'],seed,'full_ccrt',th)
            if not path.is_file():continue
            ids,genes,y,ccrt=load_npz(path)
            ridge=base.predict(emb[[emap[x] for x in ids]])[:,[gmap[g] for g in genes]]
            resid=ccrt-ridge
            for gamma in (0.,.25,.5,1.):
                pp=ridge+gamma*resid
                scale_rows.append({'dataset':'K562_unseen' if th==0 else 'RPE1_unseen_transfer',
                  'threshold':th,'seed':seed,'gamma':gamma,'mse':float(np.mean((y-pp)**2)),
                  'mean_pcc':float(np.nanmean([pcc(yy,pr) for yy,pr in zip(y,pp)])),
                  'residual_rms':float(np.sqrt(np.mean(resid**2))),
                  'ridge_rms':float(np.sqrt(np.mean(ridge**2)))})
    pd.DataFrame(scale_rows).to_csv(out/'POSTHOC_TEST_ONLY_residual_scaling_diagnostic.csv',index=False)
    (out/'README_interpretation.md').write_text('''# Step10G interpretation and limitations\n\n- All models evaluated on locked Step10A unseen perturbation IDs; RPE1 is never used for training/tuning.\n- CCRT OOF retraining uses Step10E K562-only inner validation and OOF residual targets.\n- The OOF and in-sample versions are separate candidate methods. Comparing on the existing test sets is exploratory; do not choose a winner on RPE1 and claim unbiased external validation.\n- Top-K overlap is magnitude-based, NOT statistical DEG recovery (no replicate-level FDR).\n- Direction accuracy is conditioned on nonzero true AND predicted responses among the true Top-K; report direction coverage alongside accuracy.\n- Constant-zero predictions have undefined PCC, Spearman, and Top-K overlap; they remain valid MSE/delta-R2 baselines.\n- Delta R2 zero = 1 - MSE(model)/mean(true_delta^2), computed per perturbation.\n- Training mean baseline is computed from K562 train perturbations only, restricted to matching response genes.\n- The POSTHOC_TEST_ONLY residual-scale table is a diagnostic curve and MUST NOT be used to tune gamma or select a model.\n- Paired bootstrap resamples unique perturbation IDs after averaging repeated-seed paired differences. CIs are exploratory and uncorrected for multiple comparisons.\n- RPE1 threshold groups may have different perturbation compositions; do not treat them as longitudinal measurements.\n- No biological replicates, DE FDR or gene set annotation are inferred by this script.\n''')
    (out/'run_config.json').write_text(json.dumps(vars(a),indent=2))
    print('[COMPLETE]',out,flush=True)
    print(summary[['dataset','threshold','model','mse','pcc','delta_r2_zero']].to_string(index=False),flush=True)

if __name__=='__main__':main()
