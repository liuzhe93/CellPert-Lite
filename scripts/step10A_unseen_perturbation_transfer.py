#!/usr/bin/env python3
"""Step10A: held-out perturbation evaluation in K562 and K562->RPE1.

Train ONLY on K562 perturbations outside a locked held-out set. All model
selection uses an inner split of K562 training perturbations. Evaluate the
same held-out perturbations in K562 and RPE1 (when RPE1 truth is available).
This is unseen-perturbation cross-cell-line transfer, NOT unseen-cell-line
training, and remains exploratory because prior RPE1 results informed design.

Requires scripts/step09B_ccrt_formal_ablation.py and
scripts/step09C_direct_ablation_transfer.py alongside this script.
"""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

MODELS=('gated_direct','no_gate_direct','genept_only_direct','uce_only_direct')

def import_sibling(name,filename):
    path=Path(__file__).resolve().with_name(filename)
    if not path.exists():raise FileNotFoundError(f'Missing required sibling script: {path}')
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))

def per_pcc(y,p):
    a=y.astype(np.float64);b=p.astype(np.float64)
    a-=a.mean(axis=1,keepdims=True);b-=b.mean(axis=1,keepdims=True)
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return np.divide((a*b).sum(axis=1),den,out=np.full(len(a),np.nan),where=den>1e-12)

def evaluate(y,p):
    y=y.astype(np.float64);p=p.astype(np.float64)
    if y.shape!=p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Bad prediction/target')
    a=y.ravel()-y.mean();b=p.ravel()-p.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    pc=per_pcc(y,p)
    return {'mse':float(np.mean((y-p)**2)),
            'global_pcc':float(a@b/den) if den>1e-12 else None,
            'mean_perturbation_pcc':float(np.nanmean(pc)) if np.isfinite(pc).any() else None,
            'n_perturbations':len(y),'n_response_genes':y.shape[1]}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--outdir',default='results/step10A_unseen_perturbation_transfer')
    ap.add_argument('--rpe1-dir',default='results/step08_figure3_rpe1_v2')
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    ap.add_argument('--thresholds',type=int,nargs='+',default=[3,5,10])
    ap.add_argument('--holdout-fraction',type=float,default=.20)
    ap.add_argument('--inner-val-fraction',type=float,default=.20)
    ap.add_argument('--widths',type=int,nargs='+',default=[128])
    ap.add_argument('--weight-decays',type=float,nargs='+',default=[.001,.01])
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=20)
    ap.add_argument('--dropout',type=float,default=.1)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--lr',type=float,default=.0003)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if not 0<args.holdout_fraction<.5 or not 0<args.inner_val_fraction<.5:raise ValueError('Fractions must be in (0,0.5)')
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or not args.thresholds or len(set(args.thresholds))!=len(args.thresholds):raise ValueError('Duplicate or empty seeds/thresholds')
    if min(args.widths)<1 or min(args.weight_decays)<0 or args.epochs<1 or args.patience<1 or args.threads<1:raise ValueError('Invalid training args')
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    b=import_sibling('step09b_for10a','step09B_ccrt_formal_ablation.py')
    c=import_sibling('step09c_for10a','step09C_direct_ablation_transfer.py')
    root=Path(args.root).resolve();out=(root/args.outdir).resolve()
    rp=root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
    fd=root/'results/step03_genept_uce_common/aligned_features'
    f1=fd/'GenePT_3large.npz';f2=fd/'UCE_input_tokens.npz'
    perts,genes,y=b.load_npz(rp,['perturbations','genes','delta'])
    p1,x1=b.load_npz(f1,['genes','embeddings']);p2,x2=b.load_npz(f2,['genes','embeddings'])
    perts=perts.astype(str);genes=genes.astype(str);p1=p1.astype(str);p2=p2.astype(str)
    y=y.astype(np.float32);x1=x1.astype(np.float32);x2=x2.astype(np.float32)
    if not np.array_equal(perts,p1) or not np.array_equal(perts,p2):raise ValueError('Feature ID/order mismatch')
    if len(set(perts))!=len(perts) or len(set(genes))!=len(genes):raise ValueError('Duplicate K562 IDs')
    if y.shape!=(len(perts),len(genes)) or not all(np.isfinite(z).all() for z in (y,x1,x2)):raise ValueError('Bad K562 matrices')
    lookup={v:i for i,v in enumerate(perts)};gl={v:i for i,v in enumerate(genes)}
    rpe={};files=[rp,f1,f2]
    for th in args.thresholds:
        f=root/args.rpe1_dir/f'batch{th}/rpe1_response_common_genes.npz'
        ids,gg,delta=b.load_npz(f,['perturbations','genes','delta'])
        ids=ids.astype(str);gg=gg.astype(str);delta=delta.astype(np.float32)
        if len(set(ids))!=len(ids) or len(set(gg))!=len(gg):raise ValueError(f'Duplicate RPE1 labels batch{th}')
        if not set(ids).issubset(lookup) or not set(gg).issubset(gl):raise ValueError(f'RPE1 IDs not in K562 batch{th}')
        if delta.shape!=(len(ids),len(gg)) or not np.isfinite(delta).all():raise ValueError('Invalid RPE1 delta')
        rpe[th]=(ids,gg,delta,np.array([gl[g] for g in gg],dtype=int));files.append(f)
        print(f'[RPE1] batch{th} n={len(ids)} genes={len(gg)}',flush=True)
    # Use the least restrictive available threshold to define the test pool.
    # Stratify by threshold membership to preserve scarce batch10 perturbations.
    anchor=min(args.thresholds)
    pool=np.array(sorted(set(rpe[anchor][0])),dtype=str)
    if len(pool)<20:raise ValueError('Too few RPE1 perturbations for a held-out split')
    memberships={th:set(rpe[th][0]) for th in args.thresholds}
    groups=np.array([''.join('1' if p in memberships[th] else '0' for th in sorted(args.thresholds)) for p in pool])
    unique,counts=np.unique(groups,return_counts=True)
    strata=groups if min(counts)>=2 and len(unique)<len(pool)*args.holdout_fraction else None
    split_data={};holdout_files=[]
    for seed in args.seeds:
        try:
            test_names,train_pool=train_test_split(pool,test_size=1-args.holdout_fraction,random_state=seed,stratify=strata)
        except ValueError:
            # Rare membership strata may not fit the requested test fraction.
            test_names,train_pool=train_test_split(pool,test_size=1-args.holdout_fraction,random_state=seed)
        test_names=np.array(sorted(test_names));test_set=set(test_names)
        train_names=np.array(sorted(set(perts)-test_set))
        if len(test_set)!=len(test_names) or set(train_names)&test_set or set(train_names)|test_set!=set(perts):raise RuntimeError('Holdout leakage')
        inner_train,inner_val=train_test_split(train_names,test_size=args.inner_val_fraction,random_state=seed+10000)
        split_data[seed]={'train':np.array(sorted(train_names)),'test':test_names,'inner_train':np.array(sorted(inner_train)),'inner_val':np.array(sorted(inner_val))}
        print(f'[SPLIT] seed={seed} K562_train={len(train_names)} K562_test={len(test_names)} inner_val={len(inner_val)} '+
              ' '.join(f'RPE1_batch{th}={sum(p in memberships[th] for p in test_names)}' for th in args.thresholds),flush=True)
        if any(sum(p in memberships[th] for p in test_names)==0 for th in args.thresholds):
            print(f'[WARNING] seed={seed} at least one RPE1 threshold has zero held-out perturbations',flush=True)
    if args.dry_run:
        print('[DRY RUN PASS] All inputs and leakage checks passed. No outputs written.');return
    out.mkdir(parents=True,exist_ok=True)
    config={'args':vars(args),'input_sha256':{str(p.relative_to(root)):digest(p) for p in files},
            'note':'Exploratory design informed by previously inspected outcomes; no RPE1 truth used for training/tuning.'}
    cfg=out/'experiment_config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config:raise RuntimeError('Existing output has different config: choose a new --outdir')
    if not cfg.exists():dump(cfg,config)
    rows=[];per_rows=[]
    with threadpool_limits(limits=args.threads):
        for seed in args.seeds:
            split=split_data[seed];sd=out/f'seed_{seed}';sd.mkdir(exist_ok=True)
            sf=sd/'split.json'
            spl={k:v.tolist() for k,v in split.items()}
            if sf.exists() and json.loads(sf.read_text())!=spl:raise RuntimeError('Split changed')
            if not sf.exists():dump(sf,spl)
            tr,te,it,iv=[np.array([lookup[p] for p in split[k]],dtype=int) for k in ('train','test','inner_train','inner_val')]
            # Fit scalers exclusively on the relevant training subset.
            (ai,bi),(av,bv)=b.scaled(x1,x2,it,iv)
            (at,bt),(ae,be)=b.scaled(x1,x2,tr,te)
            s1=StandardScaler().fit(x1[tr]);s2=StandardScaler().fit(x2[tr])
            def trans(ids):
                return (s1.transform(x1[ids]).astype(np.float32)/np.float32(np.sqrt(x1.shape[1])),
                        s2.transform(x2[ids]).astype(np.float32)/np.float32(np.sqrt(x2.shape[1])))
            for model_name in MODELS:
                variant='gated_no_residual' if model_name=='gated_direct' else model_name
                md=sd/model_name;md.mkdir(exist_ok=True)
                checkpoint=md/'model.pt';sel=md/'selection.json'
                if checkpoint.exists() and sel.exists():
                    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    if ck['variant']!=variant or ck['seed']!=seed:raise RuntimeError('Cached checkpoint mismatch')
                    width=int(ck['width']);dropout=float(ck['dropout'])
                    model=b.make_model(variant,x1.shape[1],x2.shape[1],len(genes),width,dropout) if model_name=='gated_direct' else c.make_direct(variant,x1.shape[1],x2.shape[1],len(genes),width,dropout)
                    model.load_state_dict(ck['state_dict']);model.eval()
                    print(f'[SKIP TRAIN] seed={seed} {model_name}',flush=True)
                else:
                    if checkpoint.exists() or sel.exists():raise RuntimeError('Partial checkpoint/selection found; remove incomplete model directory')
                    candidates=[]
                    for width in args.widths:
                        for wd in args.weight_decays:
                            if model_name=='gated_direct':
                                _,ep,val,hist=b.fit_variant(variant,ai,bi,y[it],seed,width,args.dropout,wd,args.lr,args.epochs,args.batch_size,1.,0.,val=(av,bv,None,y[iv]),patience=args.patience)
                            else:
                                _,ep,val,hist=c.fit_direct(b,variant,ai,bi,y[it],seed,width,args.dropout,wd,args.lr,args.epochs,args.batch_size,val=(av,bv,y[iv]),patience=args.patience)
                            candidates.append({'width':width,'weight_decay':wd,'selected_epoch':ep,'validation_mse':val})
                            pd.DataFrame(hist).to_csv(md/f'tuning_w{width}_wd{wd:g}.csv',index=False)
                            print(f'[TUNE] seed={seed} {model_name} width={width} wd={wd:g} ep={ep} val={val:.8f}',flush=True)
                    best=min(candidates,key=lambda z:(z['validation_mse'],z['width'],z['weight_decay']))
                    if best['selected_epoch'] is None or best['selected_epoch']<1:raise RuntimeError('No valid selected epoch')
                    pd.DataFrame(candidates).to_csv(md/'tuning_summary.csv',index=False)
                    if model_name=='gated_direct':
                        model,_,_,hist=b.fit_variant(variant,at,bt,y[tr],seed,best['width'],args.dropout,best['weight_decay'],args.lr,best['selected_epoch'],args.batch_size,1.,0.)
                    else:
                        model,_,_,hist=c.fit_direct(b,variant,at,bt,y[tr],seed,best['width'],args.dropout,best['weight_decay'],args.lr,best['selected_epoch'],args.batch_size)
                    torch.save({'variant':variant,'seed':seed,'width':best['width'],'dropout':args.dropout,'state_dict':model.state_dict()},checkpoint)
                    dump(sel,{'best':best,'selection_scope':'K562 inner-train/validation only','seed':seed,'model':model_name})
                    pd.DataFrame(hist).to_csv(md/'final_refit_history.csv',index=False)
                pred=b.inference(model,ae,be,args.batch_size)
                np.savez_compressed(md/'K562_holdout_predictions.npz',perturbations=perts[te],genes=genes,y_true=y[te],prediction=pred)
                stats=evaluate(y[te],pred)
                k562_mse=stats['mse']
                rows.append({'dataset':'K562_unseen','threshold':0,'seed':seed,'model':model_name,**stats})
                pcs=per_pcc(y[te],pred);errs=np.mean((y[te].astype(np.float64)-pred.astype(np.float64))**2,axis=1)
                for p,e,pc in zip(perts[te],errs,pcs):per_rows.append({'dataset':'K562_unseen','threshold':0,'seed':seed,'model':model_name,'perturbation':p,'mse':float(e),'pcc':float(pc) if np.isfinite(pc) else np.nan})
                for th in args.thresholds:
                    ids,gg,truth,gidx=rpe[th];rowmap={p:i for i,p in enumerate(ids)}
                    selected=np.array([p for p in split['test'] if p in rowmap])
                    if not len(selected):continue
                    i1=np.array([lookup[p] for p in selected]);i2=np.array([rowmap[p] for p in selected])
                    rx,rz=trans(i1);pp=b.inference(model,rx,rz,args.batch_size)[:,gidx];yy=truth[i2]
                    if set(selected)&set(split['train']):raise RuntimeError('RPE1 held-out perturbation leaked into training')
                    d=md/f'RPE1_batch{th}';d.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(d/'predictions.npz',perturbations=selected,genes=gg,y_true=yy,prediction=pp)
                    stats=evaluate(yy,pp)
                    rows.append({'dataset':'RPE1_unseen_transfer','threshold':th,'seed':seed,'model':model_name,**stats})
                    pcs=per_pcc(yy,pp);errs=np.mean((yy.astype(np.float64)-pp.astype(np.float64))**2,axis=1)
                    for p,e,pc in zip(selected,errs,pcs):per_rows.append({'dataset':'RPE1_unseen_transfer','threshold':th,'seed':seed,'model':model_name,'perturbation':p,'mse':float(e),'pcc':float(pc) if np.isfinite(pc) else np.nan})
                print(f'[COMPLETE] seed={seed} {model_name} K562 MSE={k562_mse:.7f}',flush=True)
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics_per_seed.csv',index=False)
    pd.DataFrame(per_rows).to_csv(out/'per_perturbation_metrics.csv',index=False)
    summary=metrics.groupby(['dataset','threshold','model'],as_index=False).agg(
        n_seeds=('seed','nunique'),n_perturbations_mean=('n_perturbations','mean'),
        mse_mean=('mse','mean'),mse_sd=('mse','std'),
        global_pcc_mean=('global_pcc','mean'),global_pcc_sd=('global_pcc','std'),
        perturbation_pcc_mean=('mean_perturbation_pcc','mean'))
    summary.to_csv(out/'metrics_summary.csv',index=False)
    print('\n=== STEP10A SUMMARY ===\n'+summary.to_string(index=False),flush=True)
    print('[COMPLETE]',out,flush=True)

if __name__=='__main__':main()
