#!/usr/bin/env python3
"""Step06B v3: CPU-only OOF-residual CCRT, exploratory K562 evaluation.

Fixes (1) in-sample Ridge residual targets -> 3-fold OOF residuals,
(2) early-best-epoch -> explicit epoch-0 Ridge-only option, monitored
training/validation curves, conservative residual scaling and early stopping.
Inner tuning and outer refit each build their OWN OOF targets. All fitting
and model selection use outer-training labels only. Outer test was inspected
in prior project stages; this remains exploratory, not confirmatory.
UCE input tokens are gene-level representations, NOT UCE cell embeddings.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
    os.replace(tmp,path)


def load_features(path,per):
    with np.load(path,allow_pickle=False) as z:
        ids=z['genes'].astype(str); x=z['embeddings'].astype(np.float32)
    if not np.array_equal(ids,per) or x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError(f'Invalid feature alignment: {path}')
    return x


def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def net(d1,d2,n,width,dropout):
    from torch import nn
    class CCRT(nn.Module):
        def __init__(self):
            super().__init__()
            self.a=nn.Sequential(nn.Linear(d1,width),nn.LayerNorm(width),nn.GELU())
            self.b=nn.Sequential(nn.Linear(d2,width),nn.LayerNorm(width),nn.GELU())
            self.gate=nn.Sequential(nn.Linear(width*2,width),nn.Sigmoid())
            self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(width,width),nn.GELU(),
                                    nn.Dropout(dropout),nn.Linear(width,n))
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        def forward(self,x,y):
            a=self.a(x);b=self.b(y)
            g=self.gate(__import__('torch').cat([a,b],dim=1))
            return self.head(g*a+(1-g)*b),g
    return CCRT()


def scaled_features(x1,x2,fit,other):
    from sklearn.preprocessing import StandardScaler
    s1=StandardScaler().fit(x1[fit]);s2=StandardScaler().fit(x2[fit])
    def transform(indices):
        a=np.asarray(s1.transform(x1[indices]),dtype=np.float32)/np.float32(np.sqrt(x1.shape[1]))
        b=np.asarray(s2.transform(x2[indices]),dtype=np.float32)/np.float32(np.sqrt(x2.shape[1]))
        assert a.dtype==np.float32 and b.dtype==np.float32
        return a,b
    return transform(fit),transform(other),(s1,s2)


def predict(model,a,b,batch=128):
    import torch
    model.eval(); ps=[];gs=[]
    with torch.no_grad():
        for i in range(0,len(a),batch):
            xa=torch.as_tensor(a[i:i+batch],dtype=torch.float32)
            xb=torch.as_tensor(b[i:i+batch],dtype=torch.float32)
            p,g=model(xa,xb)
            ps.append(p.numpy());gs.append(g.mean(1).numpy())
    return np.concatenate(ps),np.concatenate(gs)


def fit_network(a,b,target,seed,width,dropout,wd,lr,epochs,batch_size,
                val=None,patience=20,min_delta=1e-7,residual_penalty=0.1,
                residual_scale=0.5):
    import torch
    seed_all(seed)
    model=net(a.shape[1],b.shape[1],target.shape[1],width,dropout).float()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    ta=torch.as_tensor(a,dtype=torch.float32)
    tb=torch.as_tensor(b,dtype=torch.float32)
    ty=torch.as_tensor(target,dtype=torch.float32)
    if val is not None:
        av,bv,base_val,yv=val
        baseline_mse=float(np.mean((base_val.astype(np.float64)-yv.astype(np.float64))**2))
        best_mse=baseline_mse;best_epoch=0
    else:
        best_mse=float('nan');best_epoch=None
    trace=[{'epoch':0,'training_loss':None,'training_target_mse':float(np.mean(target.astype(np.float64)**2)),
            'validation_mse':best_mse if val is not None else None}]
    wait=0
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        order=torch.randperm(len(ta),generator=torch.Generator().manual_seed(seed+epoch))
        for ids in order.split(batch_size):
            xa,xb,yr=ta[ids],tb[ids],ty[ids]
            opt.zero_grad(set_to_none=True)
            r,_=model(xa,xb)
            r=r*residual_scale
            loss=((r-yr)**2).mean()+residual_penalty*r.square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();losses.append(float(loss.detach()))
        item={'epoch':epoch,'training_loss':float(np.mean(losses))}
        if val is not None:
            r,_=predict(model,av,bv,batch_size)
            v=float(np.mean((base_val.astype(np.float64)+residual_scale*r.astype(np.float64)-yv.astype(np.float64))**2))
            if not np.isfinite(v):raise RuntimeError('Nonfinite validation MSE')
            item['validation_mse']=v
            if v < best_mse-min_delta:
                best_mse=v;best_epoch=epoch;wait=0
            else:wait+=1
        trace.append(item)
        if val is not None and wait>=patience:break
    return model,best_epoch,best_mse,trace


def ridge_fit(x,y,alpha):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    m=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='cholesky'))
    m.fit(x,y)
    return m


def oof_ridge(x,y,alpha,folds,tag):
    pred=np.full(y.shape,np.nan,dtype=np.float32)
    hits=np.zeros(len(x),dtype=int)
    for k,(tr,va) in enumerate(folds):
        m=ridge_fit(x[tr],y[tr],alpha)
        pred[va]=m.predict(x[va]).astype(np.float32)
        hits[va]+=1
        print(f'[OOF] {tag} fold={k+1}/{len(folds)} fit={len(tr)} heldout={len(va)}',flush=True)
    if not np.all(hits==1) or not np.isfinite(pred).all():raise RuntimeError('OOF integrity failed')
    return pred


def inner_folds(n,k,seed):
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=k,shuffle=True,random_state=seed).split(np.arange(n)))


def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.')
    p.add_argument('--outdir',default='results/step06B_ccrt_oof_cpu_v3')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    p.add_argument('--widths',nargs='+',type=int,default=[128])
    p.add_argument('--weight-decays',nargs='+',type=float,default=[0.001,0.01])
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--min-delta',type=float,default=1e-7)
    p.add_argument('--dropout',type=float,default=0.1)
    p.add_argument('--residual-penalty',type=float,default=0.1)
    p.add_argument('--residual-scale',type=float,default=0.5)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=0.0003)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--oof-folds',type=int,default=3)
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args()


def main():
    args=args_parser()
    if not args.seeds or len(set(args.seeds))!=len(args.seeds):raise ValueError('Seeds must be unique')
    if args.oof_folds<2 or args.epochs<1 or args.threads<1 or args.patience<1:raise ValueError('Invalid training settings')
    if args.residual_scale<=0 or args.residual_penalty<0 or args.lr<=0 or not 0<=args.dropout<1:raise ValueError('Invalid optimizer settings')
    if not args.widths or min(args.widths)<1 or not args.weight_decays or min(args.weight_decays)<0:raise ValueError('Invalid grid')
    import torch
    from threadpoolctl import threadpool_limits
    import joblib
    root=Path(args.root).resolve();sys.path.insert(0,str(root))
    from cellpert.metrics import response_metrics
    import cellpert.metrics as metric_module
    torch.set_num_threads(args.threads)
    rp=root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
    fd=root/'results/step03_genept_uce_common/aligned_features'
    f1=fd/'GenePT_3large.npz';f2=fd/'UCE_input_tokens.npz'
    split_dir=root/'results/step05_tuned_genept_uce/inner_splits'
    selection_dir=root/'results/step05_tuned_genept_uce'
    with np.load(rp,allow_pickle=False) as z:
        perts=z['perturbations'].astype(str);genes=z['genes'].astype(str);y=z['delta'].astype(np.float32)
    if y.shape!=(len(perts),len(genes)) or not np.isfinite(y).all():raise ValueError('Invalid response')
    if len(set(perts))!=len(perts) or len(set(genes))!=len(genes):raise ValueError('Duplicate IDs')
    x1=load_features(f1,perts);x2=load_features(f2,perts)
    lookup={g:i for i,g in enumerate(perts)}; splits={}
    for seed in args.seeds:
        m=json.loads((split_dir/f'seed_{seed}.json').read_text())
        tr,te=m['outer_train'],m['outer_test'];it,iv=m['mlp_inner_train'],m['mlp_inner_validation']
        if len(tr)!=len(set(tr)) or len(te)!=len(set(te)) or set(tr)&set(te) or set(tr)|set(te)!=set(perts):raise ValueError('Bad outer split')
        if len(it)!=len(set(it)) or len(iv)!=len(set(iv)) or set(it)&set(iv) or set(it)|set(iv)!=set(tr):raise ValueError('Bad inner split')
        baseline=json.loads((selection_dir/f'seed_{seed}/GenePT_3large/selection_summary.json').read_text())
        alpha=float(baseline['ridge_alpha'])
        if alpha<=0:raise ValueError('Bad alpha')
        splits[seed]=(m,alpha)
        print(f'[SPLIT PASS] seed={seed} train={len(tr)} test={len(te)} inner_train={len(it)} inner_val={len(iv)} ridge_alpha={alpha}',flush=True)
    if args.dry_run:
        print(f'[DRY RUN PASS] CPU ONLY perts={len(perts)} responses={len(genes)} genept={x1.shape[1]} uce={x2.shape[1]}');return
    out=(root/args.outdir).resolve();out.mkdir(parents=True,exist_ok=True)
    config={'stage':'Step06B_v3_oof_cpu','args':vars(args),
            'inputs_sha256':{str(p):sha(p) for p in [rp,f1,f2]},
            'split_sha256':{str(s):sha(split_dir/f'seed_{s}.json') for s in args.seeds},
            'ridge_alpha_source_sha256':{str(s):sha(selection_dir/f'seed_{s}/GenePT_3large/selection_summary.json') for s in args.seeds},
            'script_sha256':sha(Path(__file__).resolve()),'metrics_sha256':sha(Path(metric_module.__file__).resolve()),
            'torch_version':torch.__version__,'device':'cpu','exploratory_refinement':True,
            'notes':'OOF residual targets for both inner tuning and outer refit; outer test not used for selection. Alpha inherited from Step05 outer-train CV, which includes inner validation labels; interpret inner-validation MSE as exploratory. Outer test previously inspected.'}
    cp=out/'experiment_config.json'
    if cp.exists():
        if json.loads(cp.read_text())!=config:raise RuntimeError('Config changed; use a NEW --outdir')
    else:atomic_json(cp,config)
    import fcntl
    lock=(out/'run.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another process is writing this output')
    records=[]
    with threadpool_limits(limits=args.threads):
        for seed in args.seeds:
            sd=out/f'seed_{seed}';sd.mkdir(exist_ok=True)
            complete=sd/'STEP06B_COMPLETE.json'
            if complete.exists():
                for name in ['metrics.csv','predictions.npz','ccrt_model.pt','ridge_base.joblib','selection_summary.json']:
                    if not (sd/name).exists():raise RuntimeError(f'Missing completed artifact {sd/name}')
                records.append(pd.read_csv(sd/'metrics.csv'));print(f'[SKIP COMPLETE] seed={seed}',flush=True);continue
            m,alpha=splits[seed]
            tr=np.array([lookup[g] for g in m['outer_train']]);te=np.array([lookup[g] for g in m['outer_test']])
            it=np.array([lookup[g] for g in m['mlp_inner_train']]);iv=np.array([lookup[g] for g in m['mlp_inner_validation']])
            # Inner tuning: OOF residuals built ONLY from inner-training labels.
            fi=inner_folds(len(it),args.oof_folds,seed+3000)
            oof_i=oof_ridge(x1[it],y[it],alpha,fi,f'seed{seed}-inner')
            inner_base=ridge_fit(x1[it],y[it],alpha)
            base_iv=inner_base.predict(x1[iv]).astype(np.float32)
            (ai,bi),(av,bv),_=scaled_features(x1,x2,it,iv)
            target_i=(y[it]-oof_i).astype(np.float32)
            rows=[];best=None
            for width in args.widths:
                for wd in args.weight_decays:
                    model,epoch,val,trace=fit_network(ai,bi,target_i,seed,width,args.dropout,wd,args.lr,
                        args.epochs,args.batch_size,val=(av,bv,base_iv,y[iv]),patience=args.patience,
                        min_delta=args.min_delta,residual_penalty=args.residual_penalty,residual_scale=args.residual_scale)
                    row={'width':width,'weight_decay':wd,'best_epoch':epoch,'validation_mse':val,
                         'ridge_only_validation_mse':float(np.mean((base_iv.astype(np.float64)-y[iv])**2)),
                         'epochs_run':len(trace)-1,'hit_epoch_limit':len(trace)-1==args.epochs}
                    rows.append(row)
                    pd.DataFrame(trace).to_csv(sd/f'tuning_width{width}_wd{wd:g}.csv',index=False)
                    print(f'[TUNE] seed={seed} width={width} wd={wd} best_epoch={epoch} val_mse={val:.8f} ridge_val={row["ridge_only_validation_mse"]:.8f}',flush=True)
                    if best is None or (val,width,wd)<(best['validation_mse'],best['width'],best['weight_decay']):best=row
                    del model
            pd.DataFrame(rows).to_csv(sd/'tuning_summary.csv',index=False)
            # Final: fresh OOF residuals on entire outer train, never in-sample residuals.
            fo=inner_folds(len(tr),args.oof_folds,seed+4000)
            oof_t=oof_ridge(x1[tr],y[tr],alpha,fo,f'seed{seed}-outer')
            base=ridge_fit(x1[tr],y[tr],alpha)
            base_te=base.predict(x1[te]).astype(np.float32)
            (at,bt),(ae,be),(s1,s2)=scaled_features(x1,x2,tr,te)
            final_epochs=int(best['best_epoch'])
            if final_epochs==0:
                # Explicit no-change residual is preferable to forced overfitting.
                seed_all(seed)
                final=net(x1.shape[1],x2.shape[1],len(genes),int(best['width']),args.dropout).float()
                r_te=np.zeros_like(base_te);gate=np.full(len(te),np.nan,dtype=np.float32)
                history=[{'epoch':0,'training_loss':None,'note':'Ridge-only selected on inner validation'}]
            else:
                final,_,_,history=fit_network(at,bt,(y[tr]-oof_t).astype(np.float32),seed,int(best['width']),
                    args.dropout,float(best['weight_decay']),args.lr,final_epochs,args.batch_size,
                    residual_penalty=args.residual_penalty,residual_scale=args.residual_scale)
                r_raw,gate=predict(final,ae,be,args.batch_size)
                r_te=(args.residual_scale*r_raw).astype(np.float32)
            pred=(base_te+r_te).astype(np.float32)
            if not np.isfinite(pred).all():raise RuntimeError('Nonfinite predictions')
            joblib.dump(base,sd/'ridge_base.joblib')
            joblib.dump({'genept_scaler':s1,'uce_scaler':s2,'dims':[x1.shape[1],x2.shape[1]],'block_scaling':'1/sqrt(dim)'},sd/'feature_preprocessors.joblib')
            torch.save({'state_dict':final.state_dict(),'width':int(best['width']),'dropout':args.dropout,
                'genept_dim':x1.shape[1],'uce_dim':x2.shape[1],'response_dim':len(genes),
                'residual_scale':args.residual_scale,'selected_epoch':final_epochs,'cpu_only':True},sd/'ccrt_model.pt')
            pd.DataFrame(history).to_csv(sd/'final_refit_history.csv',index=False)
            np.savez_compressed(sd/'predictions.npz',test_perturbations=perts[te],genes=genes,y_true=y[te],
                pred_ccrt=pred,pred_ridge_base=base_te,residual_ccrt=r_te,gate_mean_per_perturbation=gate,
                oof_train_mse=np.array([np.mean((oof_t.astype(np.float64)-y[tr])**2)]))
            scores=response_metrics(y[te],pred)
            base_scores=response_metrics(y[te],base_te)
            frame=pd.DataFrame([{'seed':seed,'model':'ridge_base',**base_scores},
                                {'seed':seed,'model':'ccrt_oof_gated_residual',**scores}])
            frame.to_csv(sd/'metrics.csv',index=False)
            atomic_json(sd/'selection_summary.json',{'seed':seed,'ridge_alpha':alpha,'selected':best,
                'oof_folds':args.oof_folds,'inner_oof_mse':float(np.mean((oof_i.astype(np.float64)-y[it])**2)),
                'outer_train_oof_mse':float(np.mean((oof_t.astype(np.float64)-y[tr])**2)),
                'residual_scale':args.residual_scale,'final_epochs':final_epochs,
                'outer_test_used_for_selection':False,'exploratory_refinement':True})
            atomic_json(complete,{'seed':seed,'status':'complete','n_train':len(tr),'n_test':len(te),
                'test_perturbations':perts[te].tolist()})
            records.append(frame)
            print(f'[COMPLETE] seed={seed} CCRT_MSE={scores["mse"]:.8f} Ridge_MSE={base_scores["mse"]:.8f} best_epoch={final_epochs}',flush=True)
    df=pd.concat(records,ignore_index=True)
    df.to_csv(out/'metrics_all_runs.csv',index=False)
    nums=[c for c in df.select_dtypes(include=np.number).columns if c!='seed']
    df.groupby('model')[nums].agg(['mean','std','count']).to_csv(out/'metrics_mean_sd.csv')
    print('[STEP06B V3 COMPLETE]',out,flush=True)


if __name__=='__main__':main()
