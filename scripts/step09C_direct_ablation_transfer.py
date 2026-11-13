#!/usr/bin/env python3
"""Step09C: direct-response ablation and frozen K562->RPE1 transfer.

Requires step09B_ccrt_formal_ablation.py in the same scripts directory.
Exploratory: previously inspected K562 outer-test and RPE1 results.
"""
import argparse, importlib.util, json, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

NEW=('genept_only_direct','uce_only_direct','no_gate_direct')
ALL=NEW+('gated_no_residual',)

def load09b():
    p=Path(__file__).resolve().with_name('step09B_ccrt_formal_ablation.py')
    if not p.exists():raise FileNotFoundError('Place Step09B script alongside Step09C: '+str(p))
    spec=importlib.util.spec_from_file_location('step09b',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def make_direct(variant,d1,d2,n,width,dropout):
    from torch import nn
    class Direct(nn.Module):
        def __init__(self):
            super().__init__()
            if variant!='uce_only_direct':self.a=nn.Sequential(nn.Linear(d1,width),nn.LayerNorm(width),nn.GELU())
            if variant!='genept_only_direct':self.b=nn.Sequential(nn.Linear(d2,width),nn.LayerNorm(width),nn.GELU())
            self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(width,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,n))
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        def forward(self,x,y):
            if variant=='genept_only_direct':z=self.a(x)
            elif variant=='uce_only_direct':z=self.b(y)
            else:z=(self.a(x)+self.b(y))*0.5
            return self.head(z)
    return Direct()

def fit_direct(b,variant,a,c,target,seed,width,dropout,wd,lr,epochs,batch,val=None,patience=20,min_delta=1e-7):
    import torch
    b.seed_everything(seed)
    model=make_direct(variant,a.shape[1],c.shape[1],target.shape[1],width,dropout).float()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    ta,tb,ty=[torch.from_numpy(x) for x in (a,c,target)]
    best=float('inf');best_epoch=None;wait=0;trace=[]
    for ep in range(1,epochs+1):
        model.train();losses=[]
        order=torch.randperm(len(ta),generator=torch.Generator().manual_seed(seed+ep))
        for ids in order.split(batch):
            opt.zero_grad(set_to_none=True)
            loss=(model(ta[ids],tb[ids])-ty[ids]).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            losses.append(float(loss.detach()))
        row={'epoch':ep,'training_loss':float(np.mean(losses))}
        if val is not None:
            va,vb,vy=val
            pred=b.inference(model,va,vb,batch)
            v=float(np.mean((pred.astype(np.float64)-vy.astype(np.float64))**2))
            row['validation_mse']=v
            if v<best-min_delta:best=v;best_epoch=ep;wait=0
            else:wait+=1
        trace.append(row)
        if val is not None and wait>=patience:break
    return model,best_epoch,best,trace

def scaled_by_outer(x1,x2,train,other):
    from sklearn.preprocessing import StandardScaler
    a=StandardScaler().fit(x1[train]);b=StandardScaler().fit(x2[train])
    def transform(idx):
        return (a.transform(x1[idx]).astype(np.float32)/np.float32(np.sqrt(x1.shape[1])),
                b.transform(x2[idx]).astype(np.float32)/np.float32(np.sqrt(x2.shape[1])))
    return transform(other)

def save_json(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--outdir',default='results/step09C_direct_ablation_transfer')
    ap.add_argument('--step09b',default='results/step09B_ccrt_formal_ablation')
    ap.add_argument('--step08c',default='results/step08C_k562_to_rpe1_transfer')
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    ap.add_argument('--thresholds',nargs='+',type=int,default=[3,5,10])
    ap.add_argument('--widths',nargs='+',type=int,default=[128])
    ap.add_argument('--weight-decays',nargs='+',type=float,default=[.001,.01])
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=20)
    ap.add_argument('--dropout',type=float,default=.1)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--lr',type=float,default=.0003)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or not args.thresholds or len(set(args.thresholds))!=len(args.thresholds):raise ValueError('Invalid seed/threshold list')
    if min(args.widths)<1 or min(args.weight_decays)<0 or args.epochs<1 or args.patience<1 or args.batch_size<1:raise ValueError('Invalid hyperparameters')
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    b=load09b()
    root=Path(args.root).resolve();out=(root/args.outdir).resolve();bdir=root/args.step09b;cdir=root/args.step08c
    rp=root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
    fd=root/'results/step03_genept_uce_common/aligned_features'
    f1=fd/'GenePT_3large.npz';f2=fd/'UCE_input_tokens.npz'
    perts,genes,y=b.load_npz(rp,['perturbations','genes','delta'])
    p1,x1=b.load_npz(f1,['genes','embeddings']);p2,x2=b.load_npz(f2,['genes','embeddings'])
    perts=perts.astype(str);genes=genes.astype(str);y=y.astype(np.float32);x1=x1.astype(np.float32);x2=x2.astype(np.float32)
    if not np.array_equal(perts,p1.astype(str)) or not np.array_equal(perts,p2.astype(str)):raise ValueError('Embedding order mismatch')
    if len(set(perts))!=len(perts) or len(set(genes))!=len(genes):raise ValueError('Duplicate labels')
    if y.shape!=(len(perts),len(genes)) or any(not np.isfinite(z).all() for z in (y,x1,x2)):raise ValueError('Invalid matrices')
    ix={p:i for i,p in enumerate(perts)};gx={g:i for i,g in enumerate(genes)}
    split_dir=root/'results/step05_tuned_genept_uce/inner_splits'
    splits={}
    for seed in args.seeds:
        sp=split_dir/f'seed_{seed}.json';m=json.loads(sp.read_text());splits[seed]=m
        tr,te,it,iv=[m[k] for k in ('outer_train','outer_test','mlp_inner_train','mlp_inner_validation')]
        if set(tr)&set(te) or set(tr)|set(te)!=set(perts) or set(it)&set(iv) or set(it)|set(iv)!=set(tr):raise ValueError('Invalid split')
        modelpath=bdir/f'seed_{seed}/gated_no_residual/model.pt'
        if not modelpath.exists():raise FileNotFoundError(modelpath)
        for threshold in args.thresholds:
            pp=cdir/f'batch{threshold}/seed_{seed}/predictions.npz'
            if not pp.exists():raise FileNotFoundError(pp)
            names,g,truth=b.load_npz(pp,['perturbations','genes','y_true'])
            names=names.astype(str);g=g.astype(str)
            if len(set(names))!=len(names) or len(set(g))!=len(g) or not set(names).issubset(set(tr)) or not set(g).issubset(gx):raise ValueError(f'RPE1 alignment invalid seed={seed} batch={threshold}')
            if truth.shape!=(len(names),len(g)) or not np.isfinite(truth).all():raise ValueError('Invalid RPE1 truth')
        print(f'[AUDIT] seed={seed} train={len(tr)} test={len(te)} RPE1 thresholds={args.thresholds}',flush=True)
    if args.dry_run:
        print('[DRY RUN PASS] K562 arrays, splits, Step09B gated model and Step08C RPE1 files checked');return
    out.mkdir(parents=True,exist_ok=True)
    cfg={'args':vars(args),'source_sha256':{str(p):b.sha(p) for p in (rp,f1,f2)},'split_sha256':{str(s):b.sha(split_dir/f'seed_{s}.json') for s in args.seeds},'note':'Exploratory: K562 and RPE1 outcomes previously inspected; RPE1 never used for training/selection.'}
    cfgp=out/'experiment_config.json'
    if cfgp.exists() and json.loads(cfgp.read_text())!=cfg:raise RuntimeError('Changed configuration; choose new --outdir')
    if not cfgp.exists():save_json(cfgp,cfg)
    import fcntl
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=[];per_rows=[]
    with threadpool_limits(limits=args.threads):
        for seed in args.seeds:
            m=splits[seed];tr,te,it,iv=[np.array([ix[p] for p in m[k]]) for k in ('outer_train','outer_test','mlp_inner_train','mlp_inner_validation')]
            sd=out/f'seed_{seed}';sd.mkdir(exist_ok=True)
            models={}
            # Frozen direct gated model from Step09B; reconstruct scaler using exact outer train.
            gp=bdir/f'seed_{seed}/gated_no_residual/model.pt'
            ck=torch.load(gp,map_location='cpu',weights_only=True)
            if ck['variant']!='gated_no_residual' or ck['seed']!=seed:raise ValueError('Frozen model mismatch')
            gm=b.make_model('gated_no_residual',x1.shape[1],x2.shape[1],len(genes),ck['width'],ck['dropout'])
            gm.load_state_dict(ck['state_dict']);gm.eval();models['gated_no_residual']=gm
            (ai,bi),(av,bv)=b.scaled(x1,x2,it,iv)
            for variant in NEW:
                vd=sd/variant;vd.mkdir(exist_ok=True)
                done=vd/'COMPLETE.json'
                if done.exists():
                    for name in ('model.pt','predictions.npz','selection.json'):
                        if not (vd/name).exists():raise RuntimeError(f'Missing completed {vd/name}')
                    ck2=torch.load(vd/'model.pt',map_location='cpu',weights_only=True)
                    if ck2['variant']!=variant or ck2['seed']!=seed:raise ValueError('Cached model mismatch')
                    model=make_direct(variant,x1.shape[1],x2.shape[1],len(genes),ck2['width'],ck2['dropout'])
                    model.load_state_dict(ck2['state_dict']);model.eval();models[variant]=model
                    print(f'[SKIP] seed={seed} {variant}',flush=True);continue
                candidates=[]
                for width in args.widths:
                    for wd in args.weight_decays:
                        model,ep,val,trace=fit_direct(b,variant,ai,bi,y[it],seed,width,args.dropout,wd,args.lr,args.epochs,args.batch_size,val=(av,bv,y[iv]),patience=args.patience)
                        candidates.append({'width':width,'weight_decay':wd,'selected_epoch':ep,'validation_mse':val})
                        pd.DataFrame(trace).to_csv(vd/f'tuning_width{width}_wd{wd:g}.csv',index=False)
                        print(f'[TUNE] seed={seed} {variant} width={width} wd={wd} epoch={ep} val={val:.8f}',flush=True)
                best=min(candidates,key=lambda z:(z['validation_mse'],z['width'],z['weight_decay']))
                pd.DataFrame(candidates).to_csv(vd/'tuning_summary.csv',index=False)
                (at,bt),(ae,be)=b.scaled(x1,x2,tr,te)
                model,_,_,hist=fit_direct(b,variant,at,bt,y[tr],seed,best['width'],args.dropout,best['weight_decay'],args.lr,best['selected_epoch'],args.batch_size)
                pred=b.inference(model,ae,be,args.batch_size)
                np.savez_compressed(vd/'predictions.npz',test_perturbations=perts[te],genes=genes,y_true=y[te],prediction=pred)
                torch.save({'variant':variant,'state_dict':model.state_dict(),'width':best['width'],'dropout':args.dropout,'seed':seed,'selected_epoch':best['selected_epoch']},vd/'model.pt')
                pd.DataFrame(hist).to_csv(vd/'final_refit_history.csv',index=False)
                save_json(vd/'selection.json',{'seed':seed,'variant':variant,'best':best,'inner_selection_only':True,'outer_test_used_for_selection':False})
                save_json(done,{'status':'complete','seed':seed,'variant':variant})
                models[variant]=model
                print(f'[COMPLETE] seed={seed} {variant} K562 mse={b.score(y[te],pred)["mse"]:.8f}',flush=True)
            for name,model in models.items():
                if name=='gated_no_residual':
                    stored=b.load_npz(bdir/f'seed_{seed}/gated_no_residual/predictions.npz',['test_perturbations','genes','y_true','prediction'])
                    if not np.array_equal(stored[0].astype(str),perts[te]) or not np.array_equal(stored[1].astype(str),genes):raise ValueError('Step09B prediction mismatch')
                    pred=stored[3]
                    aa,bb=scaled_by_outer(x1,x2,tr,te)
                    rec=b.inference(model,aa,bb,args.batch_size)
                    if np.max(np.abs(rec-pred))>2e-5:raise RuntimeError(f'Frozen Step09B model reproduction failed seed={seed}')
                else:
                    pred=b.load_npz(sd/name/'predictions.npz',['prediction'])[0]
                metrics=b.score(y[te],pred)
                rows.append({'dataset':'K562_outer_test','threshold':0,'seed':seed,'model':name,'n_perturbations':len(te),**metrics})
                mser=((y[te].astype(np.float64)-pred.astype(np.float64))**2).mean(axis=1)
                for pert,err in zip(perts[te],mser):per_rows.append({'dataset':'K562_outer_test','threshold':0,'seed':seed,'model':name,'perturbation':pert,'mse':float(err)})
            for threshold in args.thresholds:
                rp2=cdir/f'batch{threshold}/seed_{seed}/predictions.npz'
                names,gg,truth=b.load_npz(rp2,['perturbations','genes','y_true']);names=names.astype(str);gg=gg.astype(str)
                ind=np.array([ix[p] for p in names]);gind=np.array([gx[g] for g in gg]);truth=truth.astype(np.float32)
                ra,rb=scaled_by_outer(x1,x2,tr,ind)
                for name,model in models.items():
                    pred=b.inference(model,ra,rb,args.batch_size)[:,gind]
                    if pred.shape!=truth.shape or not np.isfinite(pred).all():raise RuntimeError('RPE1 output invalid')
                    d=sd/name/f'RPE1_batch{threshold}';d.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(d/'predictions.npz',perturbations=names,genes=gg,y_true=truth,prediction=pred)
                    metrics=b.score(truth,pred)
                    rows.append({'dataset':'RPE1_seen_transfer','threshold':threshold,'seed':seed,'model':name,'n_perturbations':len(names),**metrics})
                    mser=((truth.astype(np.float64)-pred.astype(np.float64))**2).mean(axis=1)
                    for pert,err in zip(names,mser):per_rows.append({'dataset':'RPE1_seen_transfer','threshold':threshold,'seed':seed,'model':name,'perturbation':pert,'mse':float(err)})
                print(f'[TRANSFER] seed={seed} batch={threshold} n={len(names)}',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'metrics_per_seed.csv',index=False)
    summ=df.groupby(['dataset','threshold','model'],as_index=False).agg(n_seeds=('seed','nunique'),mse_mean=('mse','mean'),mse_sd=('mse','std'),global_pcc_mean=('global_pcc','mean'),global_pcc_sd=('global_pcc','std'),perturbation_pcc_mean=('mean_perturbation_pcc','mean'))
    summ.to_csv(out/'metrics_summary.csv',index=False)
    pd.DataFrame(per_rows).to_csv(out/'per_perturbation_mse.csv',index=False)
    print('\n=== STEP09C SUMMARY ===\n'+summ.to_string(index=False),flush=True)
    print('[COMPLETE]',out,flush=True)
if __name__=='__main__':main()
