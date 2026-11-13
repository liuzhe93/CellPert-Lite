#!/usr/bin/env python3
"""Step11B independent zero-shot validation of FROZEN Step10A direct models.

Input truth NPZ: perturbations (P,), genes (G,), delta (P,G), all finite.
Input feature NPZs: genes (P,), embeddings (P,D), same GenePT/UCE dimensions
as K562 training features. Embeddings must be produced WITHOUT new truth labels.
Only previously trained K562 models are loaded; no retraining or selection.
The original K562 training response matrix is used ONLY to calculate the
per-seed train-mean baseline and training-fitted feature scalers.

New cell line targets can be included in K562 training; this is *zero-shot
cell-line transfer*, NOT unseen-target transfer unless target exclusion is
explicitly checked using --require-unseen-targets.

Use --dry-run to validate inputs without running models. No artificial data.
"""
import argparse, hashlib, json, importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd

def read(path,keys):
    if not path.is_file(): raise FileNotFoundError(path)
    with np.load(path,allow_pickle=False) as z:
        if not set(keys).issubset(z.files): raise ValueError(f'{path} needs {keys}; found {z.files}')
        return [z[k] for k in keys]

def ids_ok(ids,label):
    ids=np.asarray(ids).astype(str)
    if not len(ids) or len(set(ids))!=len(ids) or any(not x.strip() for x in ids):
        raise ValueError(f'{label}: empty or duplicated identifiers')
    return ids

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def pcc(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    a=a-a.mean();b=b-b.mean();d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/d) if d>1e-12 else np.nan

def metrics(y,p,mean):
    yy=np.asarray(y,dtype=np.float64);pp=np.asarray(p,dtype=np.float64)
    err=np.mean((yy-pp)**2);zerr=np.mean(yy**2);merr=np.mean((yy-mean)**2)
    return dict(mse=float(err),global_pcc=pcc(yy,pp),pooled_skill_vs_zero=float(1-err/zerr) if zerr>1e-15 else np.nan,
                pooled_skill_vs_train_mean=float(1-err/merr) if merr>1e-15 else np.nan,
                mean_perturbation_pcc=float(np.nanmean([pcc(a,b) for a,b in zip(yy,pp)])))

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--root',default='.')
    ap.add_argument('--truth',required=True,help='Independent response NPZ (perturbations,genes,delta)')
    ap.add_argument('--genept',required=True,help='External GenePT embeddings NPZ (genes,embeddings)')
    ap.add_argument('--uce',required=True,help='External UCE embeddings NPZ (genes,embeddings)')
    ap.add_argument('--step10a',default='results/step10A_unseen_perturbation_transfer')
    ap.add_argument('--outdir',default='results/step11B_independent_validation')
    ap.add_argument('--models',nargs='+',default=['gated_direct','no_gate_direct','genept_only_direct','uce_only_direct'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    ap.add_argument('--topk',nargs='+',type=int,default=[20,50,100])
    ap.add_argument('--require-unseen-targets',action='store_true',help='Fail if any test target appears in any K562 training split')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--threads',type=int,default=8)
    args=ap.parse_args()
    root=Path(args.root).resolve()
    resolve=lambda s:(root/s).resolve()
    out=resolve(args.outdir); aroot=resolve(args.step10a)
    truth,feat1,feat2=map(resolve,(args.truth,args.genept,args.uce))
    rp=root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
    fd=root/'results/step03_genept_uce_common/aligned_features'
    k1=fd/'GenePT_3large.npz';k2=fd/'UCE_input_tokens.npz'
    required=[truth,feat1,feat2,rp,k1,k2]
    for f in required:
        if not f.is_file():raise FileNotFoundError(f)
    if len(set(args.seeds))!=len(args.seeds) or not args.seeds:raise ValueError('Invalid seeds')
    allowed={'gated_direct','no_gate_direct','genept_only_direct','uce_only_direct'}
    if not set(args.models).issubset(allowed) or len(set(args.models))!=len(args.models):raise ValueError('Unsupported models')
    tid,tgenes,y=read(truth,['perturbations','genes','delta']);tid=ids_ok(tid,'external targets');tgenes=ids_ok(tgenes,'external response genes')
    y=np.asarray(y,dtype=np.float32)
    if y.shape!=(len(tid),len(tgenes)) or not np.isfinite(y).all():raise ValueError('Invalid external delta')
    fids1,x1=read(feat1,['genes','embeddings']);fids2,x2=read(feat2,['genes','embeddings'])
    fids1=ids_ok(fids1,'external GenePT IDs');fids2=ids_ok(fids2,'external UCE IDs')
    x1=np.asarray(x1,dtype=np.float32);x2=np.asarray(x2,dtype=np.float32)
    kid,kgenes,ky=read(rp,['perturbations','genes','delta']);kid=ids_ok(kid,'K562 targets');kgenes=ids_ok(kgenes,'K562 response genes')
    ky=np.asarray(ky,dtype=np.float32)
    kf1,kx1=read(k1,['genes','embeddings']);kf2,kx2=read(k2,['genes','embeddings'])
    kf1=ids_ok(kf1,'K562 GenePT IDs');kf2=ids_ok(kf2,'K562 UCE IDs')
    kx1=np.asarray(kx1,dtype=np.float32);kx2=np.asarray(kx2,dtype=np.float32)
    if not (np.array_equal(kid,kf1) and np.array_equal(kid,kf2)):raise ValueError('K562 embedding alignment mismatch')
    if ky.shape!=(len(kid),len(kgenes)):raise ValueError('K562 response dimensions mismatch')
    if kx1.shape!=(len(kid),x1.shape[1]) or kx2.shape!=(len(kid),x2.shape[1]):raise ValueError('Embedding dimension mismatch')
    if x1.shape!=(len(fids1),kx1.shape[1]) or x2.shape!=(len(fids2),kx2.shape[1]):raise ValueError('External embedding shape mismatch')
    if not all(np.isfinite(z).all() for z in [x1,x2,kx1,kx2,ky]):raise ValueError('Nonfinite values')
    e1={v:i for i,v in enumerate(fids1)};e2={v:i for i,v in enumerate(fids2)};gm={v:i for i,v in enumerate(kgenes)}
    missing_targets=[v for v in tid if v not in e1 or v not in e2]
    if missing_targets:raise ValueError(f'External targets missing embeddings: {missing_targets[:12]} (total {len(missing_targets)})')
    missing_genes=[v for v in tgenes if v not in gm]
    if missing_genes:raise ValueError(f'External response genes absent from K562 output space: {missing_genes[:12]} (total {len(missing_genes)})')
    ix1=[e1[v] for v in tid];ix2=[e2[v] for v in tid];gidx=np.array([gm[v] for v in tgenes]);km={v:i for i,v in enumerate(kid)}
    all_inputs=required[:]
    for seed in args.seeds:
        split=aroot/f'seed_{seed}/split.json';all_inputs.append(split)
        if not split.is_file():raise FileNotFoundError(split)
        s=json.loads(split.read_text());tr=set(s['train']);test=set(s['test'])
        if tr&test:raise ValueError(f'seed {seed}: K562 split leakage')
        if args.require_unseen_targets and tr&set(tid):raise ValueError(f'seed {seed}: {len(tr&set(tid))} targets present in K562 train')
        for m in args.models:
            ck=aroot/f'seed_{seed}'/m/'model.pt';all_inputs.append(ck)
            if not ck.is_file():raise FileNotFoundError(ck)
    print(f'[AUDIT PASS] external targets={len(tid)} response genes={len(tgenes)}; models={args.models}; seeds={args.seeds}')
    print(f'[TARGET OVERLAP] with K562 all targets: {len(set(tid)&set(kid))}/{len(tid)}')
    if args.dry_run:
        print('[DRY RUN PASS] no files written');return
    import torch
    from sklearn.preprocessing import StandardScaler
    torch.set_num_threads(args.threads)
    sibling=Path(__file__).resolve().with_name('step09B_ccrt_formal_ablation.py')
    sibling2=Path(__file__).resolve().with_name('step09C_direct_ablation_transfer.py')
    for f in [sibling,sibling2]:
        if not f.is_file():raise FileNotFoundError(f'Copy Step11B script into scripts/ alongside {f.name}')
    def imp(name,path):
        spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
        sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
    b=imp('step11b_b',sibling);c=imp('step11b_c',sibling2)
    out.mkdir(parents=True,exist_ok=True)
    cfg={'args':vars(args),'inputs_sha256':{str(p):digest(p) for p in all_inputs},'scope':'frozen K562 models, external zero-shot cell-line validation'}
    configfile=out/'frozen_evaluation_manifest.json'
    if configfile.exists() and json.loads(configfile.read_text())!=cfg:raise RuntimeError('Output exists with different configuration; use new --outdir')
    configfile.write_text(json.dumps(cfg,indent=2))
    rows=[];detail=[];audit=[]
    for seed in args.seeds:
        s=json.loads((aroot/f'seed_{seed}/split.json').read_text());tr=np.array([km[v] for v in s['train']],dtype=int)
        mu=ky[tr].mean(axis=0)[gidx].astype(np.float32)
        sc1=StandardScaler().fit(kx1[tr]);sc2=StandardScaler().fit(kx2[tr])
        aa=sc1.transform(x1[ix1]).astype(np.float32)/np.float32(np.sqrt(kx1.shape[1]))
        bb=sc2.transform(x2[ix2]).astype(np.float32)/np.float32(np.sqrt(kx2.shape[1]))
        preds={'zero_change':np.zeros_like(y),'k562_train_mean':np.broadcast_to(mu,y.shape)}
        for m in args.models:
            ck=torch.load(aroot/f'seed_{seed}'/m/'model.pt',map_location='cpu',weights_only=True)
            variant='gated_no_residual' if m=='gated_direct' else m
            if ck['variant']!=variant or ck['seed']!=seed:raise ValueError(f'checkpoint mismatch seed={seed} model={m}')
            width=int(ck['width']);drop=float(ck['dropout'])
            model=(b.make_model(variant,kx1.shape[1],kx2.shape[1],len(kgenes),width,drop) if m=='gated_direct' else c.make_direct(variant,kx1.shape[1],kx2.shape[1],len(kgenes),width,drop))
            model.load_state_dict(ck['state_dict']);model.eval()
            pp=b.inference(model,aa,bb,128)[:,gidx]
            if pp.shape!=y.shape or not np.isfinite(pp).all():raise ValueError('Invalid model prediction')
            preds[m]=pp
        for m,pp in preds.items():
            stat=metrics(y,pp,mu);rows.append(dict(seed=seed,model=m,n_perturbations=len(tid),n_genes=len(tgenes),**stat))
            for i,target in enumerate(tid):
                yy=y[i].astype(np.float64);pr=pp[i].astype(np.float64);err=float(np.mean((yy-pr)**2));ze=float(np.mean(yy**2))
                rec={'seed':seed,'model':m,'perturbation':target,'mse':err,'pcc':pcc(yy,pr),'true_rms':float(np.sqrt(ze)),
                     'skill_vs_zero':float(1-err/ze) if ze>1e-15 else np.nan}
                for k in args.topk:
                    if k>len(tgenes):continue
                    trueidx=np.argsort(-np.abs(yy),kind='stable')[:k]
                    predidx=np.argsort(-np.abs(pr),kind='stable')[:k]
                    rec[f'top{k}_overlap']=(float(len(set(trueidx)&set(predidx))/k) if np.count_nonzero(np.abs(pr)>1e-12)>=k else np.nan)
                    rec[f'top{k}_random_expectation']=k/len(tgenes)
                detail.append(rec)
        audit.append({'seed':seed,'external_targets':len(tid),'overlap_K562_train':len(set(tid)&set(s['train'])),
                      'overlap_K562_test':len(set(tid)&set(s['test']))})
        print(f'[DONE] seed={seed} external targets={len(tid)}',flush=True)
    pd.DataFrame(rows).to_csv(out/'global_metrics_per_seed.csv',index=False)
    pd.DataFrame(detail).to_csv(out/'per_perturbation_metrics.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'target_overlap_audit.csv',index=False)
    pd.DataFrame(rows).groupby('model')[['mse','global_pcc','mean_perturbation_pcc','pooled_skill_vs_zero','pooled_skill_vs_train_mean']].agg(['mean','std']).to_csv(out/'global_metrics_summary.csv')
    (out/'README_interpretation.md').write_text('''# Step11B independent validation\n\n- Check the provenance of the external dataset before calling it independent.\n- Model checkpoints and K562 training scalers are frozen. External labels are evaluation-only.\n- Target overlap with K562 train is reported: external cell-line transfer is NOT automatically unseen-target transfer.\n- Responses must be comparable deltas with the same normalization, control definition and gene ID namespace.\n- This evaluator does not infer a response matrix from raw single-cell data, compute GenePT/UCE embeddings, or retrain models.\n- Cross-cell-line comparison should prioritize relative skill within each dataset.\n- Five seeds are not five independent biological replicates.\n- Top-K uses absolute response magnitude, not FDR-defined DE genes.\n- Keep this independent dataset untouched for model selection.\n''')
    print('[COMPLETE]',out)

if __name__=='__main__':main()
