#!/usr/bin/env python3
"""Step10E: original full CCRT (GenePT Ridge + gated residual network), locked Step10A splits.

Reuses the original Step06B architecture/trainer, but retrains on Step10A splits.
Default in_sample residual target reproduces original Step06B design. Optional
--residual-target oof provides a stronger out-of-fold residual sensitivity check.
All model selection uses K562 inner train/validation; RPE1 is test-only.
Outputs Step10A/10C-compatible prediction NPZs for later paired bootstrap.
"""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

def load_module(path):
    spec=importlib.util.spec_from_file_location('ccrt_step06b_for_step10e',path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    spec.loader.exec_module(mod);return mod

def read(path,keys):
    with np.load(path,allow_pickle=False) as z:
        if any(k not in z.files for k in keys):raise KeyError(f'{path}: expected {keys}, got {z.files}')
        return [z[k] for k in keys]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def save_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)

def row_pcc(y,p):
    a=np.asarray(y,dtype=np.float64);b=np.asarray(p,dtype=np.float64)
    a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return np.divide((a*b).sum(1),den,out=np.full(len(a),np.nan),where=den>1e-12)

def score(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    if y.shape!=p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Bad target/prediction')
    a=y.ravel()-y.mean();b=p.ravel()-p.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    pc=row_pcc(y,p)
    return dict(mse=float(np.mean((y-p)**2)),global_pcc=float(a@b/den) if den>1e-12 else np.nan,
                mean_perturbation_pcc=float(np.nanmean(pc)) if np.isfinite(pc).any() else np.nan,
                n_perturbations=len(y),n_response_genes=y.shape[1])

def save_prediction(path,ids,genes,truth,pred):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,perturbations=np.asarray(ids,dtype=str),genes=np.asarray(genes,dtype=str),
                        y_true=np.asarray(truth,dtype=np.float32),prediction=np.asarray(pred,dtype=np.float32))

def oof_ridge(x,y,alpha,folds,seed):
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    result=np.full(y.shape,np.nan,dtype=np.float32)
    for tr,va in KFold(n_splits=folds,shuffle=True,random_state=seed).split(x):
        reg=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='auto'))
        reg.fit(x[tr],y[tr]);result[va]=reg.predict(x[va]).astype(np.float32)
    if not np.isfinite(result).all():raise RuntimeError('Invalid OOF predictions')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.')
    p.add_argument('--step10a',default='results/step10A_unseen_perturbation_transfer')
    p.add_argument('--rpe1-dir',default='results/step08_figure3_rpe1_v2')
    p.add_argument('--outdir',default='results/step10E_full_ccrt_unseen')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    p.add_argument('--thresholds',nargs='+',type=int,default=[3,5,10])
    p.add_argument('--ridge-alphas',nargs='+',type=float,default=[1.,100.,10000.])
    p.add_argument('--widths',nargs='+',type=int,default=[128])
    p.add_argument('--weight-decays',nargs='+',type=float,default=[.001,.01])
    p.add_argument('--residual-target',choices=['in_sample','oof'],default='in_sample')
    p.add_argument('--oof-folds',type=int,default=5)
    p.add_argument('--residual-penalty',type=float,default=.1)
    p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or len(set(a.thresholds))!=len(a.thresholds):raise ValueError('Duplicate seed/threshold')
    if min(a.ridge_alphas)<=0 or min(a.widths)<1 or min(a.weight_decays)<0 or a.oof_folds<2:raise ValueError('Invalid hyperparameter')
    root=Path(a.root).resolve();s10=root/a.step10a;out=root/a.outdir
    script=root/'scripts/step06B_ccrt_model.py'
    if not script.is_file():raise FileNotFoundError(f'Original CCRT implementation missing: {script}. Copy original Step06B script there; do not substitute Step09B.')
    m=load_module(script)
    import torch,joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(a.threads)
    device='cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    rp=root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
    fd=root/'results/step03_genept_uce_common/aligned_features'
    f1=fd/'GenePT_3large.npz';f2=fd/'UCE_input_tokens.npz'
    perts,genes,y=read(rp,['perturbations','genes','delta'])
    g1,x1=read(f1,['genes','embeddings']);g2,x2=read(f2,['genes','embeddings'])
    perts=perts.astype(str);genes=genes.astype(str)
    y=np.asarray(y,dtype=np.float32);x1=np.asarray(x1,dtype=np.float32);x2=np.asarray(x2,dtype=np.float32)
    if not np.array_equal(perts,g1.astype(str)) or not np.array_equal(perts,g2.astype(str)):raise ValueError('Embedding IDs/order mismatch')
    if len(set(perts))!=len(perts) or len(set(genes))!=len(genes):raise ValueError('Duplicate K562 IDs')
    if y.shape!=(len(perts),len(genes)) or any(not np.isfinite(v).all() for v in (y,x1,x2)):raise ValueError('Invalid arrays')
    pi={v:i for i,v in enumerate(perts)};gi={v:i for i,v in enumerate(genes)}
    rpe={};inputs=[rp,f1,f2,script]
    for th in a.thresholds:
        path=root/a.rpe1_dir/f'batch{th}/rpe1_response_common_genes.npz'
        ids,gg,yy=read(path,['perturbations','genes','delta'])
        ids=ids.astype(str);gg=gg.astype(str);yy=np.asarray(yy,dtype=np.float32)
        if len(set(ids))!=len(ids) or len(set(gg))!=len(gg) or not set(ids).issubset(pi) or not set(gg).issubset(gi):raise ValueError(f'RPE1 IDs invalid batch{th}')
        if yy.shape!=(len(ids),len(gg)) or not np.isfinite(yy).all():raise ValueError(f'RPE1 truth invalid batch{th}')
        rpe[th]=(ids,gg,yy,np.array([gi[g] for g in gg]),{v:i for i,v in enumerate(ids)})
        inputs.append(path)
    splits={}
    for seed in a.seeds:
        sf=s10/f'seed_{seed}/split.json';sp=json.loads(sf.read_text());inputs.append(sf)
        for key in ('train','test','inner_train','inner_val'):
            if key not in sp or len(sp[key])!=len(set(sp[key])):raise ValueError(f'Invalid split {seed}/{key}')
        tr,te,it,iv=(set(sp[k]) for k in ('train','test','inner_train','inner_val'))
        if tr&te or tr|te!=set(perts) or it&iv or it|iv!=tr:raise ValueError(f'Leakage in split {seed}')
        ref=s10/f'seed_{seed}/gated_direct/K562_holdout_predictions.npz'
        refids,refgenes=read(ref,['perturbations','genes'])
        if set(refids.astype(str))!=te or not np.array_equal(refgenes.astype(str),genes):raise ValueError(f'Step10A reference mismatch seed {seed}')
        for th in a.thresholds:
            expected=te&set(rpe[th][0]);pred=s10/f'seed_{seed}/gated_direct/RPE1_batch{th}/predictions.npz'
            if expected:
                rid,rg=read(pred,['perturbations','genes'])
                if set(rid.astype(str))!=expected or not np.array_equal(rg.astype(str),rpe[th][1]):raise ValueError(f'RPE1 reference mismatch seed {seed} batch{th}')
        splits[seed]=sp
        print(f'[CHECK] seed={seed} train={len(tr)} test={len(te)} inner_train={len(it)} inner_val={len(iv)}',flush=True)
    if a.dry_run:
        print('[DRY RUN PASS] CCRT source, features, splits, and RPE1 alignment verified; no output written.');return
    config={'args':vars(a),'source_sha256':{str(f.relative_to(root)):sha(f) for f in inputs},
            'method':'Original Step06B gated Ridge residual architecture, retrained on locked Step10A splits',
            'residual_target_note':'in_sample matches original Step06B; oof is sensitivity analysis; no RPE1 tuning'}
    out.mkdir(parents=True,exist_ok=True);cfg=out/'experiment_config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config:raise RuntimeError('Config changed: choose another --outdir')
    if not cfg.exists():save_json(cfg,config)
    metrics=[];perrows=[]
    with threadpool_limits(limits=a.threads):
        for seed in a.seeds:
            sp=splits[seed];sd=out/f'seed_{seed}'/'full_ccrt';sd.mkdir(parents=True,exist_ok=True)
            done=sd/'COMPLETE.json'
            if done.exists():
                if not all((sd/f).is_file() for f in ('metrics.csv','per_perturbation_metrics.csv','K562_holdout_predictions.npz','model.pt','ridge_base.joblib','selection.json')):raise RuntimeError(f'Incomplete seed {seed}')
                metrics.extend(pd.read_csv(sd/'metrics.csv').to_dict('records'))
                perrows.extend(pd.read_csv(sd/'per_perturbation_metrics.csv').to_dict('records'))
                print(f'[SKIP COMPLETE] seed={seed}',flush=True);continue
            tr,te,it,iv=[np.array([pi[v] for v in sp[k]],dtype=int) for k in ('train','test','inner_train','inner_val')]
            # Ridge alpha selection: inner train -> inner validation ONLY.
            alpha_scores=[]
            for alpha in a.ridge_alphas:
                rg=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='auto'))
                rg.fit(x1[it],y[it]);pred=rg.predict(x1[iv])
                alpha_scores.append({'alpha':float(alpha),'validation_mse':float(np.mean((pred-y[iv])**2))})
            pd.DataFrame(alpha_scores).to_csv(sd/'ridge_alpha_tuning.csv',index=False)
            alpha=min(alpha_scores,key=lambda z:(z['validation_mse'],z['alpha']))['alpha']
            base_i=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='auto')).fit(x1[it],y[it])
            base_val=base_i.predict(x1[iv]).astype(np.float32)
            if a.residual_target=='oof':pred_i=oof_ridge(x1[it],y[it],alpha,a.oof_folds,seed+3000)
            else:pred_i=base_i.predict(x1[it]).astype(np.float32)
            (ai,bi),(av,bv),_=m.transform_blocks(x1,x2,it,iv)
            candidates=[]
            for width in a.widths:
                for wd in a.weight_decays:
                    model,epoch,val,trace=m.fit_residual(ai,bi,(y[it]-pred_i).astype(np.float32),seed,device,
                        width,a.dropout,wd,a.epochs,a.batch_size,a.lr,
                        validation=(av,bv,base_val,y[iv]),patience=a.patience,residual_penalty=a.residual_penalty)
                    pd.DataFrame(trace).to_csv(sd/f'tuning_width{width}_wd{wd:g}.csv',index=False)
                    candidates.append(dict(width=width,weight_decay=wd,selected_epoch=epoch,validation_mse=val))
                    del model
                    print(f'[TUNE] seed={seed} alpha={alpha:g} width={width} wd={wd:g} epoch={epoch} val={val:.8f}',flush=True)
            pd.DataFrame(candidates).to_csv(sd/'tuning_summary.csv',index=False)
            best=min(candidates,key=lambda z:(z['validation_mse'],z['width'],z['weight_decay']))
            if best['selected_epoch'] is None or best['selected_epoch']<1:raise RuntimeError('No valid CCRT epoch')
            base=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='auto')).fit(x1[tr],y[tr])
            if a.residual_target=='oof':pred_tr=oof_ridge(x1[tr],y[tr],alpha,a.oof_folds,seed+4000)
            else:pred_tr=base.predict(x1[tr]).astype(np.float32)
            (at,bt),(ae,be),(sc1,sc2)=m.transform_blocks(x1,x2,tr,te)
            final,_,_,history=m.fit_residual(at,bt,(y[tr]-pred_tr).astype(np.float32),seed,device,
                best['width'],a.dropout,best['weight_decay'],best['selected_epoch'],a.batch_size,a.lr,
                residual_penalty=a.residual_penalty)
            pd.DataFrame(history).to_csv(sd/'final_fit_history.csv',index=False)
            raw,gate=m.predict_residual(final,ae,be,device,batch=a.batch_size)
            pred=(base.predict(x1[te])+raw).astype(np.float32)
            save_prediction(sd/'K562_holdout_predictions.npz',perts[te],genes,y[te],pred)
            joblib.dump(base,sd/'ridge_base.joblib')
            torch.save({'seed':seed,'width':best['width'],'dropout':a.dropout,'state_dict':{k:v.detach().cpu() for k,v in final.state_dict().items()},
                        'ridge_alpha':alpha,'residual_target':a.residual_target},sd/'model.pt')
            save_json(sd/'selection.json',{'seed':seed,'alpha':alpha,'best':best,'ridge_alpha_tuning':alpha_scores,
                                           'selection_scope':'K562 inner_train/inner_val only','residual_target':a.residual_target})
            rows=[];prs=[]
            def record(dataset,th,ids,yy,pp):
                rows.append({'dataset':dataset,'threshold':th,'seed':seed,'model':'full_ccrt',**score(yy,pp)})
                mse=np.mean((np.asarray(yy,dtype=np.float64)-np.asarray(pp,dtype=np.float64))**2,axis=1)
                for name,err,pc in zip(ids,mse,row_pcc(yy,pp)):
                    prs.append({'dataset':dataset,'threshold':th,'seed':seed,'model':'full_ccrt',
                                'perturbation':str(name),'mse':float(err),'pcc':float(pc) if np.isfinite(pc) else np.nan})
            record('K562_unseen',0,perts[te],y[te],pred)
            for th in a.thresholds:
                ids,gg,yy,gidx,lookup=rpe[th]
                selected=[v for v in sp['test'] if v in lookup]
                if not selected:continue
                rids=np.array([pi[v] for v in selected],dtype=int)
                rtruth=yy[[lookup[v] for v in selected]]
                xa=sc1.transform(x1[rids]).astype(np.float32)/np.float32(np.sqrt(x1.shape[1]))
                xb=sc2.transform(x2[rids]).astype(np.float32)/np.float32(np.sqrt(x2.shape[1]))
                resid,_=m.predict_residual(final,xa,xb,device,batch=a.batch_size)
                rpred=(base.predict(x1[rids])[:,gidx]+resid[:,gidx]).astype(np.float32)
                save_prediction(sd/f'RPE1_batch{th}/predictions.npz',selected,gg,rtruth,rpred)
                record('RPE1_unseen_transfer',th,selected,rtruth,rpred)
            pd.DataFrame(rows).to_csv(sd/'metrics.csv',index=False)
            pd.DataFrame(prs).to_csv(sd/'per_perturbation_metrics.csv',index=False)
            save_json(done,{'seed':seed,'status':'complete','model':'full_ccrt'})
            metrics.extend(rows);perrows.extend(prs)
            print(f'[COMPLETE] seed={seed} K562_MSE={rows[0]["mse"]:.8f}',flush=True)
    frame=pd.DataFrame(metrics);frame.to_csv(out/'metrics_per_seed.csv',index=False)
    pd.DataFrame(perrows).to_csv(out/'per_perturbation_metrics.csv',index=False)
    summary=frame.groupby(['dataset','threshold','model'],as_index=False).agg(
        n_seeds=('seed','nunique'),n_perturbations_mean=('n_perturbations','mean'),
        mse_mean=('mse','mean'),mse_sd=('mse','std'),global_pcc_mean=('global_pcc','mean'),
        global_pcc_sd=('global_pcc','std'),perturbation_pcc_mean=('mean_perturbation_pcc','mean'))
    summary.to_csv(out/'metrics_summary.csv',index=False)
    print('\n'+summary.to_string(index=False));print('[COMPLETE]',out)

if __name__=='__main__':main()
