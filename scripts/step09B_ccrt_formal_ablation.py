#!/usr/bin/env python3
"""Step09B | CCRT architectural ablations; CPU, outer-train-only selection.

Variants: ridge_base, genept_only_residual, uce_only_residual,
          no_gate_residual, gated_no_residual, full_ccrt_frozen.
The original full CCRT is NOT retuned. Each new variant has independent inner
validation selection and fresh outer-train refit. Exploratory, not confirmatory:
prior outer-test results were inspected during project development.
"""
import argparse, hashlib, json, os, random
from pathlib import Path
import numpy as np
import pandas as pd

VARIANTS=('genept_only_residual','uce_only_residual','no_gate_residual','gated_no_residual')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(p,obj):
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))

def load_npz(p,keys):
    with np.load(p,allow_pickle=False) as z:
        if not set(keys).issubset(z.files):raise KeyError(f'{p}: expected {keys}, got {z.files}')
        return [z[k] for k in keys]

def seed_everything(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def make_model(variant,d1,d2,n,width,dropout):
    import torch
    from torch import nn
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            if variant != 'uce_only_residual':
                self.a=nn.Sequential(nn.Linear(d1,width),nn.LayerNorm(width),nn.GELU())
            if variant != 'genept_only_residual':
                self.b=nn.Sequential(nn.Linear(d2,width),nn.LayerNorm(width),nn.GELU())
            if variant in ('gated_no_residual',):
                self.gate=nn.Sequential(nn.Linear(2*width,width),nn.Sigmoid())
            self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(width,width),nn.GELU(),
                                    nn.Dropout(dropout),nn.Linear(width,n))
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        def forward(self,x,y):
            if variant=='genept_only_residual':z=self.a(x)
            elif variant=='uce_only_residual':z=self.b(y)
            else:
                a,b=self.a(x),self.b(y)
                if variant=='no_gate_residual':z=0.5*(a+b)
                else:
                    g=self.gate(torch.cat([a,b],dim=1))
                    z=g*a+(1-g)*b
            return self.head(z)
    return Model()

def inference(model,a,b,batch):
    import torch
    model.eval();out=[]
    with torch.no_grad():
        for i in range(0,len(a),batch):
            out.append(model(torch.from_numpy(a[i:i+batch]),torch.from_numpy(b[i:i+batch])).numpy())
    return np.concatenate(out,axis=0)

def scaled(x1,x2,fit,other):
    from sklearn.preprocessing import StandardScaler
    s1=StandardScaler().fit(x1[fit]);s2=StandardScaler().fit(x2[fit])
    def trans(ix):
        a=s1.transform(x1[ix]).astype(np.float32)/np.float32(np.sqrt(x1.shape[1]))
        b=s2.transform(x2[ix]).astype(np.float32)/np.float32(np.sqrt(x2.shape[1]))
        return a,b
    return trans(fit),trans(other)

def ridge_fit(x,y,alpha):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='cholesky'))
    return model.fit(x,y)

def ridge_oof(x,y,alpha,nfold,seed):
    from sklearn.model_selection import KFold
    out=np.full(y.shape,np.nan,dtype=np.float32)
    hits=np.zeros(len(y),dtype=np.int8)
    for train,val in KFold(n_splits=nfold,shuffle=True,random_state=seed).split(x):
        out[val]=ridge_fit(x[train],y[train],alpha).predict(x[val]).astype(np.float32)
        hits[val]+=1
    if not np.all(hits==1) or not np.isfinite(out).all():raise RuntimeError('OOF failure')
    return out

def fit_variant(variant,a,b,target,seed,width,dropout,wd,lr,epochs,batch,scale,penalty,
                val=None,patience=20,min_delta=1e-7):
    import torch
    seed_everything(seed)
    model=make_model(variant,a.shape[1],b.shape[1],target.shape[1],width,dropout).float()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    ta=torch.from_numpy(a);tb=torch.from_numpy(b);ty=torch.from_numpy(target)
    is_residual=variant!='gated_no_residual'
    best_epoch=0 if is_residual else None
    best=float(np.mean((val[2].astype(np.float64)-val[3].astype(np.float64))**2)) if val is not None and is_residual else float('inf')
    trace=[dict(epoch=0,validation_mse=best if is_residual and val is not None else None)]
    wait=0
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        perm=torch.randperm(len(ta),generator=torch.Generator().manual_seed(seed+epoch))
        for ids in perm.split(batch):
            opt.zero_grad(set_to_none=True)
            output=model(ta[ids],tb[ids])
            if is_residual:
                output=scale*output
                loss=(output-ty[ids]).square().mean()+penalty*output.square().mean()
            else:loss=(output-ty[ids]).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            losses.append(float(loss.detach()))
        row={'epoch':epoch,'training_loss':float(np.mean(losses))}
        if val is not None:
            av,bv,base_val,y_val=val
            p=inference(model,av,bv,batch)
            if is_residual:p=base_val+scale*p
            v=float(np.mean((p.astype(np.float64)-y_val.astype(np.float64))**2))
            row['validation_mse']=v
            if v < best-min_delta:
                best=v;best_epoch=epoch;wait=0
            else:wait+=1
        trace.append(row)
        if val is not None and wait>=patience:break
    return model,best_epoch,best,trace

def pcc_rows(y,p):
    a=y-y.mean(axis=1,keepdims=True);b=p-p.mean(axis=1,keepdims=True)
    d=np.sqrt(np.sum(a*a,axis=1)*np.sum(b*b,axis=1))
    return np.divide(np.sum(a*b,axis=1),d,out=np.full(len(y),np.nan),where=d>1e-12)

def score(y,p):
    y=y.astype(np.float64);p=p.astype(np.float64)
    a=y.ravel()-y.mean();b=p.ravel()-p.mean()
    den=np.sqrt((a*a).sum()*(b*b).sum())
    return dict(mse=float(np.mean((y-p)**2)),global_pcc=float((a*b).sum()/den) if den>1e-12 else np.nan,
                mean_perturbation_pcc=float(np.nanmean(pcc_rows(y,p))))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--outdir',default='results/step09B_ccrt_formal_ablation')
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    ap.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    ap.add_argument('--widths',nargs='+',type=int,default=[128])
    ap.add_argument('--weight-decays',nargs='+',type=float,default=[0.001,0.01])
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=20)
    ap.add_argument('--min-delta',type=float,default=1e-7)
    ap.add_argument('--dropout',type=float,default=0.1)
    ap.add_argument('--residual-penalty',type=float,default=0.1)
    ap.add_argument('--residual-scale',type=float,default=0.5)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--lr',type=float,default=0.0003)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--oof-folds',type=int,default=3)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if not args.seeds or len(set(args.seeds))!=len(args.seeds):raise ValueError('Duplicate/empty seeds')
    if not args.variants or len(set(args.variants))!=len(args.variants):raise ValueError('Duplicate/empty variants')
    if min(args.widths)<1 or min(args.weight_decays)<0 or args.epochs<1 or args.patience<1 or args.oof_folds<2:raise ValueError('Bad hyperparameters')
    import torch,joblib
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    root=Path(args.root).resolve();out=(root/args.outdir).resolve()
    rp=root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
    fdir=root/'results/step03_genept_uce_common/aligned_features'
    f1=fdir/'GenePT_3large.npz';f2=fdir/'UCE_input_tokens.npz'
    split_dir=root/'results/step05_tuned_genept_uce/inner_splits'
    perts,genes,y=load_npz(rp,['perturbations','genes','delta'])
    perts=perts.astype(str);genes=genes.astype(str);y=y.astype(np.float32)
    ids1,x1=load_npz(f1,['genes','embeddings']);ids2,x2=load_npz(f2,['genes','embeddings'])
    x1=x1.astype(np.float32);x2=x2.astype(np.float32)
    if not np.array_equal(ids1.astype(str),perts) or not np.array_equal(ids2.astype(str),perts):raise ValueError('Feature order mismatch')
    if y.shape!=(len(perts),len(genes)) or not np.isfinite(y).all() or not np.isfinite(x1).all() or not np.isfinite(x2).all():raise ValueError('Bad arrays')
    if len(set(perts))!=len(perts):raise ValueError('Duplicate perturbations')
    lookup={p:i for i,p in enumerate(perts)}
    splits={};alphas={}
    for seed in args.seeds:
        sp=split_dir/f'seed_{seed}.json';m=json.loads(sp.read_text());splits[seed]=m
        tr,te,it,iv=[m[k] for k in ('outer_train','outer_test','mlp_inner_train','mlp_inner_validation')]
        if set(tr)&set(te) or set(tr)|set(te)!=set(perts) or set(it)&set(iv) or set(it)|set(iv)!=set(tr):raise ValueError(f'Split invalid {seed}')
        if any(len(v)!=len(set(v)) for v in (tr,te,it,iv)):raise ValueError('Duplicate split labels')
        selection=root/f'results/step05_tuned_genept_uce/seed_{seed}/GenePT_3large/selection_summary.json'
        alphas[seed]=float(json.loads(selection.read_text())['ridge_alpha'])
        if alphas[seed]<=0:raise ValueError('Bad ridge alpha')
        original=root/f'results/step06B_ccrt_oof_cpu_v3/seed_{seed}/predictions.npz'
        if not original.exists():raise FileNotFoundError(original)
        print(f'[AUDIT] seed={seed} outer train={len(tr)} test={len(te)} inner train={len(it)} val={len(iv)} alpha={alphas[seed]}',flush=True)
    if args.dry_run:
        print(f'[DRY RUN PASS] {len(perts)} perturbations, {len(genes)} response genes, variants={args.variants}');return
    out.mkdir(parents=True,exist_ok=True)
    cfg={'args':vars(args),'inputs_sha256':{str(p):sha(p) for p in (rp,f1,f2)},
         'split_sha256':{str(s):sha(split_dir/f'seed_{s}.json') for s in args.seeds},
         'note':'Exploratory ablation; prior K562 outer-test has been inspected. No test-based tuning.'}
    cp=out/'experiment_config.json'
    if cp.exists() and json.loads(cp.read_text())!=cfg:raise RuntimeError('Config changed; use new --outdir')
    if not cp.exists():write_json(cp,cfg)
    import fcntl
    lock=(out/'run.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Output directory locked')
    with threadpool_limits(limits=args.threads):
        for seed in args.seeds:
            m=splits[seed];alpha=alphas[seed]
            tr,te,it,iv=[np.array([lookup[p] for p in m[k]]) for k in ('outer_train','outer_test','mlp_inner_train','mlp_inner_validation')]
            seed_dir=out/f'seed_{seed}';seed_dir.mkdir(exist_ok=True)
            original=root/f'results/step06B_ccrt_oof_cpu_v3/seed_{seed}/predictions.npz'
            op,og,oy,ofull,obase=load_npz(original,['test_perturbations','genes','y_true','pred_ccrt','pred_ridge_base'])
            if not np.array_equal(op.astype(str),perts[te]) or not np.array_equal(og.astype(str),genes) or not np.allclose(oy,y[te],atol=1e-6):raise ValueError('Original prediction alignment failed')
            # Reuse original full CCRT and base, never refit/tune them.
            original_models={'full_ccrt_frozen':ofull,'ridge_base':obase}
            for name,pred in original_models.items():
                fp=seed_dir/f'{name}_metrics.csv'
                pd.DataFrame([{'seed':seed,'model':name,**score(y[te],pred)}]).to_csv(fp,index=False)
            # Precompute inner OOF and outer OOF only when needed.
            need_resid=any(v!='gated_no_residual' for v in args.variants)
            if need_resid:
                print(f'[OOF] seed={seed} inner',flush=True)
                oof_i=ridge_oof(x1[it],y[it],alpha,args.oof_folds,seed+3000)
            inner_base=ridge_fit(x1[it],y[it],alpha)
            base_iv=inner_base.predict(x1[iv]).astype(np.float32)
            (ai,bi),(av,bv)=scaled(x1,x2,it,iv)
            inner_target=(y[it]-oof_i).astype(np.float32) if need_resid else None
            for variant in args.variants:
                vd=seed_dir/variant;vd.mkdir(exist_ok=True)
                complete=vd/'COMPLETE.json'
                if complete.exists():
                    for fname in ('metrics.csv','predictions.npz','selection.json','model.pt'):
                        if not (vd/fname).exists():raise RuntimeError(f'Missing completed {vd/fname}')
                    print(f'[SKIP] seed={seed} variant={variant}',flush=True);continue
                is_residual=variant!='gated_no_residual'
                target=inner_target if is_residual else y[it]
                candidates=[]
                for width in args.widths:
                    for wd in args.weight_decays:
                        model,ep,val,trace=fit_variant(variant,ai,bi,target,seed,width,args.dropout,wd,args.lr,
                            args.epochs,args.batch_size,args.residual_scale,args.residual_penalty,
                            val=(av,bv,base_iv,y[iv]),patience=args.patience,min_delta=args.min_delta)
                        candidates.append(dict(width=width,weight_decay=wd,selected_epoch=ep,validation_mse=val))
                        pd.DataFrame(trace).to_csv(vd/f'tuning_width{width}_wd{wd:g}.csv',index=False)
                        del model
                        print(f'[TUNE] seed={seed} {variant} width={width} wd={wd} epoch={ep} val={val:.8f}',flush=True)
                best=min(candidates,key=lambda z:(z['validation_mse'],z['width'],z['weight_decay']))
                pd.DataFrame(candidates).to_csv(vd/'tuning_summary.csv',index=False)
                # The inner validation is used for selecting epoch and WD, never for final model evaluation.
                if is_residual and best['selected_epoch']==0:
                    pred=obase.copy()
                    seed_everything(seed)
                    final=make_model(variant,x1.shape[1],x2.shape[1],len(genes),best['width'],args.dropout)
                    history=[dict(epoch=0,note='Selected no residual; original Ridge base')]
                else:
                    if is_residual:
                        print(f'[OOF] seed={seed} outer for {variant}',flush=True)
                        oof_t=ridge_oof(x1[tr],y[tr],alpha,args.oof_folds,seed+4000)
                        train_target=(y[tr]-oof_t).astype(np.float32)
                    else:train_target=y[tr]
                    (at,bt),(ae,be)=scaled(x1,x2,tr,te)
                    final,_,_,history=fit_variant(variant,at,bt,train_target,seed,best['width'],args.dropout,
                        best['weight_decay'],args.lr,int(best['selected_epoch']),args.batch_size,
                        args.residual_scale,args.residual_penalty)
                    raw=inference(final,ae,be,args.batch_size)
                    pred=(obase+args.residual_scale*raw) if is_residual else raw
                pred=np.asarray(pred,dtype=np.float32)
                if pred.shape!=y[te].shape or not np.isfinite(pred).all():raise RuntimeError('Bad prediction')
                torch.save({'variant':variant,'state_dict':final.state_dict(),'width':best['width'],
                            'dropout':args.dropout,'selected_epoch':best['selected_epoch'],
                            'residual_scale':args.residual_scale,'seed':seed},vd/'model.pt')
                np.savez_compressed(vd/'predictions.npz',test_perturbations=perts[te],genes=genes,y_true=y[te],prediction=pred)
                pd.DataFrame(history).to_csv(vd/'final_refit_history.csv',index=False)
                pd.DataFrame([{'seed':seed,'model':variant,**score(y[te],pred)}]).to_csv(vd/'metrics.csv',index=False)
                write_json(vd/'selection.json',{'seed':seed,'variant':variant,'best':best,'inner_selection_only':True,
                           'original_full_model_reused':False,'outer_test_used_for_selection':False})
                write_json(complete,{'seed':seed,'variant':variant,'status':'complete'})
                print(f'[COMPLETE] seed={seed} {variant} mse={score(y[te],pred)["mse"]:.8f}',flush=True)
    rows=[];pert_rows=[]
    for seed in args.seeds:
        sd=out/f'seed_{seed}';m=splits[seed]
        te=np.array([lookup[p] for p in m['outer_test']]);true=y[te]
        orig=root/f'results/step06B_ccrt_oof_cpu_v3/seed_{seed}/predictions.npz'
        _,_,_,full,base=load_npz(orig,['test_perturbations','genes','y_true','pred_ccrt','pred_ridge_base'])
        predictions={'ridge_base':base,'full_ccrt_frozen':full}
        for v in args.variants:
            z=load_npz(sd/v/'predictions.npz',['test_perturbations','genes','y_true','prediction'])
            if not np.array_equal(z[0].astype(str),perts[te]) or not np.array_equal(z[1].astype(str),genes):raise ValueError('Bad stored order')
            predictions[v]=z[3]
        for name,pred in predictions.items():
            rows.append({'seed':seed,'model':name,'n_perturbations':len(te),**score(true,pred)})
            mse=((true.astype(np.float64)-pred.astype(np.float64))**2).mean(axis=1)
            pp=pcc_rows(true.astype(np.float64),pred.astype(np.float64))
            for i,p in enumerate(perts[te]):pert_rows.append({'seed':seed,'perturbation':p,'model':name,'mse':mse[i],'pcc':pp[i]})
    df=pd.DataFrame(rows);df.to_csv(out/'metrics_per_seed.csv',index=False)
    summ=df.groupby('model',as_index=False).agg(n_seeds=('seed','nunique'),mse_mean=('mse','mean'),mse_sd=('mse','std'),
        global_pcc_mean=('global_pcc','mean'),global_pcc_sd=('global_pcc','std'),
        perturbation_pcc_mean=('mean_perturbation_pcc','mean'))
    summ.to_csv(out/'metrics_summary.csv',index=False)
    pd.DataFrame(pert_rows).to_csv(out/'per_perturbation_metrics.csv',index=False)
    print('\n=== STEP09B SUMMARY ===\n'+summ.to_string(index=False),flush=True)
    print('[COMPLETE]',out,flush=True)

if __name__=='__main__':main()
