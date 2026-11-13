#!/usr/bin/env python3
"""Step10C: Ridge-Fusion and MLP-Fusion on EXACT Step10A locked splits.

Only K562 train/inner-validation targets enter fitting/model selection. RPE1
responses are used solely for held-out evaluation. No Step06A checkpoints reused.
Ridge/MLP architectures and balanced/raw feature preprocessing follow Step06A.
Outputs match Step10A NPZ schema for subsequent paired analysis.
"""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits

MODELS=('ridge_fusion','mlp_fusion')

def module(path):
    spec=importlib.util.spec_from_file_location('step06a_for_step10c',path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def read_npz(path,keys):
    with np.load(path,allow_pickle=False) as z:
        for k in keys:
            if k not in z.files:raise KeyError(f'{path}: missing {k}; keys={z.files}')
        return [z[k] for k in keys]

def pcc_rows(y,p):
    a=y.astype(np.float64);b=p.astype(np.float64)
    a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return np.divide((a*b).sum(1),den,out=np.full(len(a),np.nan),where=den>1e-12)

def score(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    if y.shape!=p.shape or not np.isfinite(p).all():raise ValueError('Bad predictions')
    a=y.ravel()-y.mean();b=p.ravel()-p.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    pc=pcc_rows(y,p)
    return {'mse':float(np.mean((y-p)**2)),
            'global_pcc':float(a@b/den) if den>1e-12 else np.nan,
            'mean_perturbation_pcc':float(np.nanmean(pc)) if np.isfinite(pc).any() else np.nan,
            'n_perturbations':len(y),'n_response_genes':y.shape[1]}

def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(path)

def validate_split(path,all_ids,seed):
    s=json.loads(path.read_text())
    for key in ('train','test','inner_train','inner_val'):
        if key not in s or len(s[key])!=len(set(s[key])):raise ValueError(f'Invalid {key} in {path}')
    tr,te,it,iv=[set(s[k]) for k in ('train','test','inner_train','inner_val')]
    if tr&te or tr|te!=all_ids or it&iv or it|iv!=tr:raise ValueError(f'Split leakage seed={seed}')
    return s

def save_predictions(path,ids,genes,y,p):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,perturbations=np.asarray(ids,dtype=str),genes=np.asarray(genes,dtype=str),
                        y_true=np.asarray(y,dtype=np.float32),prediction=np.asarray(p,dtype=np.float32))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--step10a',default='results/step10A_unseen_perturbation_transfer')
    ap.add_argument('--rpe1-dir',default='results/step08_figure3_rpe1_v2')
    ap.add_argument('--outdir',default='results/step10C_strong_baseline_unseen')
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    ap.add_argument('--thresholds',nargs='+',type=int,default=[3,5,10])
    ap.add_argument('--ridge-alphas',nargs='+',type=float,default=[1.,100.,10000.])
    ap.add_argument('--mlp-alphas',nargs='+',type=float,default=[.001,.01])
    ap.add_argument('--feature-modes',nargs='+',choices=['balanced','raw'],default=['balanced','raw'])
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=20)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--lr',type=float,default=.0003)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.thresholds))!=len(args.thresholds):raise ValueError('Duplicate seeds or thresholds')
    if min(args.ridge_alphas)<=0 or min(args.mlp_alphas)<0 or args.epochs<1 or args.patience<1:raise ValueError('Invalid hyperparameters')
    root=Path(args.root).resolve();step10a=root/args.step10a;out=root/args.outdir
    modpath=root/'scripts/step06A_fusion_baselines.py'
    if not modpath.is_file():raise FileNotFoundError(f'Required original Step06A script: {modpath}')
    m=module(modpath)
    rp=root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
    fd=root/'results/step03_genept_uce_common/aligned_features'
    f1=fd/'GenePT_3large.npz';f2=fd/'UCE_input_tokens.npz'
    perts,genes,y=read_npz(rp,['perturbations','genes','delta'])
    p1,x1=read_npz(f1,['genes','embeddings']);p2,x2=read_npz(f2,['genes','embeddings'])
    perts=perts.astype(str);genes=genes.astype(str)
    if len(set(perts))!=len(perts) or len(set(genes))!=len(genes):raise ValueError('Duplicate IDs')
    if not np.array_equal(perts,p1.astype(str)) or not np.array_equal(perts,p2.astype(str)):raise ValueError('Embedding IDs mismatch')
    y=np.asarray(y,dtype=np.float32);x1=np.asarray(x1,dtype=np.float32);x2=np.asarray(x2,dtype=np.float32)
    if y.shape!=(len(perts),len(genes)) or not all(np.isfinite(q).all() for q in (y,x1,x2)):raise ValueError('Bad K562 arrays')
    X=np.concatenate([x1,x2],axis=1);d1=x1.shape[1];d2=x2.shape[1]
    lookup={p:i for i,p in enumerate(perts)};gl={g:i for i,g in enumerate(genes)}
    rpe={};files=[rp,f1,f2,modpath]
    for th in args.thresholds:
        f=root/args.rpe1_dir/f'batch{th}/rpe1_response_common_genes.npz'
        ids,gg,delta=read_npz(f,['perturbations','genes','delta'])
        ids=ids.astype(str);gg=gg.astype(str);delta=np.asarray(delta,dtype=np.float32)
        if len(set(ids))!=len(ids) or len(set(gg))!=len(gg):raise ValueError(f'Duplicate RPE1 labels {th}')
        if not set(ids).issubset(lookup) or not set(gg).issubset(gl):raise ValueError(f'RPE1 ID mismatch {th}')
        if delta.shape!=(len(ids),len(gg)) or not np.isfinite(delta).all():raise ValueError(f'Bad RPE1 response {th}')
        rpe[th]=(ids,gg,delta,np.array([gl[g] for g in gg],dtype=int),{p:i for i,p in enumerate(ids)})
        files.append(f)
        print(f'[RPE1] batch={th} n={len(ids)} genes={len(gg)}',flush=True)
    splits={}
    for seed in args.seeds:
        sf=step10a/f'seed_{seed}/split.json'
        if not sf.is_file():raise FileNotFoundError(sf)
        s=validate_split(sf,set(perts),seed);splits[seed]=s;files.append(sf)
        # Also verify saved Step10A test order/content, preventing accidental mismatch.
        ref=step10a/f'seed_{seed}/gated_direct/K562_holdout_predictions.npz'
        ids,gg=read_npz(ref,['perturbations','genes'])
        if set(ids.astype(str))!=set(s['test']) or not np.array_equal(gg.astype(str),genes):raise ValueError(f'Step10A test mismatch seed={seed}')
        for th in args.thresholds:
            expected=set(s['test'])&set(rpe[th][0])
            rf=step10a/f'seed_{seed}/gated_direct/RPE1_batch{th}/predictions.npz'
            if expected:
                a,b=read_npz(rf,['perturbations','genes'])
                if set(a.astype(str))!=expected or not np.array_equal(b.astype(str),rpe[th][1]):raise ValueError(f'Step10A RPE1 test mismatch seed={seed} th={th}')
        print(f'[AUDIT] seed={seed} train={len(s["train"])} unseen={len(s["test"])}',flush=True)
    if args.dry_run:
        print('[DRY RUN PASS] All input files, splits, and alignment checks passed. No outputs written.',flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    cfg={'args':vars(args),'input_sha256':{str(f.relative_to(root)):sha(f) for f in files},
         'note':'Step10A exact locked splits; all tuning uses K562 inner validation; RPE1 evaluation only.'}
    cf=out/'experiment_config.json'
    if cf.exists() and json.loads(cf.read_text())!=cfg:raise RuntimeError('Output directory has different config; choose new --outdir')
    if not cf.exists():atomic_json(cf,cfg)
    metrics=[];details=[]
    with threadpool_limits(limits=args.threads):
      for seed in args.seeds:
        s=splits[seed];tr,te,it,iv=[np.array([lookup[p] for p in s[k]],dtype=int) for k in ('train','test','inner_train','inner_val')]
        sd=out/f'seed_{seed}';sd.mkdir(exist_ok=True)
        for name in MODELS:
          md=sd/name;md.mkdir(exist_ok=True)
          model_file=md/'model.joblib';sel_file=md/'selection.json'
          if model_file.exists() and sel_file.exists():
            artifact=joblib.load(model_file);selection=json.loads(sel_file.read_text())
            if selection['seed']!=seed or selection['model']!=name:raise RuntimeError('Cached model mismatch')
            print(f'[SKIP TRAIN] seed={seed} model={name}',flush=True)
          else:
            if model_file.exists() or sel_file.exists():raise RuntimeError(f'Partial model directory: {md}')
            candidates=[]
            for mode in args.feature_modes:
              pp=m.make_preprocessor(d1,d2,mode)
              xi=pp.fit_transform(X[it]).astype(np.float32)
              xv=pp.transform(X[iv]).astype(np.float32)
              if name=='ridge_fusion':
                for alpha in args.ridge_alphas:
                  model=Ridge(alpha=alpha,solver='auto').fit(xi,y[it]);val=float(mean_squared_error(y[iv],model.predict(xv)))
                  candidates.append({'mode':mode,'alpha':float(alpha),'selected_epoch':0,'validation_mse':val})
                  print(f'[TUNE] seed={seed} {name} {mode} alpha={alpha} mse={val:.8f}',flush=True)
              else:
                for alpha in args.mlp_alphas:
                  _,ep,val,trace=m.train_mlp(xi,y[it],seed,alpha,args.epochs,args.batch_size,args.lr,
                                             validation=(xv,y[iv]),patience=args.patience)
                  if ep is None or ep<1 or not np.isfinite(val):raise RuntimeError('MLP tuning failed')
                  pd.DataFrame(trace).to_csv(md/f'tuning_{mode}_alpha{alpha:g}.csv',index=False)
                  candidates.append({'mode':mode,'alpha':float(alpha),'selected_epoch':int(ep),'validation_mse':float(val)})
                  print(f'[TUNE] seed={seed} {name} {mode} alpha={alpha} epoch={ep} mse={val:.8f}',flush=True)
            best=min(candidates,key=lambda v:(v['validation_mse'],v['mode'],v['alpha']))
            pd.DataFrame(candidates).to_csv(md/'tuning_summary.csv',index=False)
            pp=m.make_preprocessor(d1,d2,best['mode'])
            xt=pp.fit_transform(X[tr]).astype(np.float32)
            if name=='ridge_fusion':model=Ridge(alpha=best['alpha'],solver='auto').fit(xt,y[tr])
            else:
              model,_,_,trace=m.train_mlp(xt,y[tr],seed,best['alpha'],best['selected_epoch'],args.batch_size,args.lr)
              pd.DataFrame(trace).to_csv(md/'final_refit_history.csv',index=False)
            artifact={'model':model,'preprocessor':pp,'seed':seed,'model_name':name,'best':best}
            joblib.dump(artifact,model_file,compress=3)
            selection={'seed':seed,'model':name,'best':best,'selection_scope':'K562 inner-train/validation only'}
            atomic_json(sel_file,selection)
          def predict(indices):
            xx=artifact['preprocessor'].transform(X[indices]).astype(np.float32)
            p=np.asarray(artifact['model'].predict(xx),dtype=np.float32)
            if p.shape!=(len(indices),len(genes)) or not np.isfinite(p).all():raise ValueError('Invalid prediction')
            return p
          p=predict(te)
          save_predictions(md/'K562_holdout_predictions.npz',perts[te],genes,y[te],p)
          metrics.append({'dataset':'K562_unseen','threshold':0,'seed':seed,'model':name,**score(y[te],p)})
          for id_,err,pc in zip(perts[te],np.mean((y[te]-p).astype(np.float64)**2,axis=1),pcc_rows(y[te],p)):
            details.append({'dataset':'K562_unseen','threshold':0,'seed':seed,'model':name,'perturbation':id_,'mse':float(err),'pcc':float(pc) if np.isfinite(pc) else np.nan})
          for th in args.thresholds:
            ids,gg,truth,gidx,rowmap=rpe[th]
            chosen=np.array([id_ for id_ in s['test'] if id_ in rowmap],dtype=str)
            if not len(chosen):continue
            ii=np.array([lookup[id_] for id_ in chosen]);ri=np.array([rowmap[id_] for id_ in chosen])
            yy=truth[ri];pr=predict(ii)[:,gidx]
            save_predictions(md/f'RPE1_batch{th}/predictions.npz',chosen,gg,yy,pr)
            metrics.append({'dataset':'RPE1_unseen_transfer','threshold':th,'seed':seed,'model':name,**score(yy,pr)})
            for id_,err,pc in zip(chosen,np.mean((yy-pr).astype(np.float64)**2,axis=1),pcc_rows(yy,pr)):
              details.append({'dataset':'RPE1_unseen_transfer','threshold':th,'seed':seed,'model':name,'perturbation':id_,'mse':float(err),'pcc':float(pc) if np.isfinite(pc) else np.nan})
          print(f'[COMPLETE] seed={seed} model={name}',flush=True)
    df=pd.DataFrame(metrics);df.to_csv(out/'metrics_per_seed.csv',index=False)
    pd.DataFrame(details).to_csv(out/'per_perturbation_metrics.csv',index=False)
    summary=df.groupby(['dataset','threshold','model'],as_index=False).agg(
      n_seeds=('seed','nunique'),n_perturbations_mean=('n_perturbations','mean'),
      mse_mean=('mse','mean'),mse_sd=('mse','std'),global_pcc_mean=('global_pcc','mean'),
      global_pcc_sd=('global_pcc','std'),perturbation_pcc_mean=('mean_perturbation_pcc','mean'))
    summary.to_csv(out/'metrics_summary.csv',index=False)
    print('\n=== STEP10C SUMMARY ===\n'+summary.to_string(index=False),flush=True)
    print('[COMPLETE]',out,flush=True)

if __name__=='__main__':main()
