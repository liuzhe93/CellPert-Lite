#!/usr/bin/env python3
"""Step08C: frozen K562 models -> RPE1; verify K562 checkpoint reproduction first.

Models: Step06A Ridge-Fusion, MLP-Fusion; Step06B CCRT.
Only K562 outer-TRAIN perturbations are evaluated on RPE1 (seen transfer).
No training, tuning, or RPE1-based model selection.
"""
import argparse
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from threadpoolctl import threadpool_limits


def read_npz(path, keys):
    with np.load(path, allow_pickle=False) as z:
        for k in keys:
            if k not in z.files:
                raise KeyError(f'{path}: missing {k}; available={z.files}')
        return {k: z[k] for k in keys}


def ccrt_net(d1, d2, n, width, dropout):
    import torch
    from torch import nn
    class CCRT(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Sequential(nn.Linear(d1,width),nn.LayerNorm(width),nn.GELU())
            self.b = nn.Sequential(nn.Linear(d2,width),nn.LayerNorm(width),nn.GELU())
            self.gate = nn.Sequential(nn.Linear(width*2,width),nn.Sigmoid())
            self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(width,width),nn.GELU(),
                                      nn.Dropout(dropout),nn.Linear(width,n))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        def forward(self,x,y):
            a=self.a(x);b=self.b(y)
            g=self.gate(torch.cat([a,b],dim=1))
            return self.head(g*a+(1-g)*b),g
    return CCRT()


def ccrt_predict(seed_dir, x1, x2, batch_size=128):
    import torch
    ck = torch.load(seed_dir/'ccrt_model.pt', map_location='cpu', weights_only=True)
    pp = joblib.load(seed_dir/'feature_preprocessors.joblib')
    base = joblib.load(seed_dir/'ridge_base.joblib')
    d1,d2 = int(ck['genept_dim']),int(ck['uce_dim'])
    if x1.shape[1] != d1 or x2.shape[1] != d2:
        raise ValueError('CCRT embedding dimension mismatch')
    if list(pp['dims']) != [d1,d2] or pp['block_scaling'] != '1/sqrt(dim)':
        raise ValueError('CCRT feature preprocessing metadata mismatch')
    p_base = np.asarray(base.predict(x1),dtype=np.float32)
    if p_base.shape[1] != int(ck['response_dim']):
        raise ValueError('CCRT response dimension mismatch')
    if int(ck['selected_epoch']) == 0:
        return p_base
    a = np.asarray(pp['genept_scaler'].transform(x1),dtype=np.float32)/np.float32(np.sqrt(d1))
    b = np.asarray(pp['uce_scaler'].transform(x2),dtype=np.float32)/np.float32(np.sqrt(d2))
    net = ccrt_net(d1,d2,int(ck['response_dim']),int(ck['width']),float(ck['dropout']))
    net.load_state_dict(ck['state_dict'],strict=True)
    net.eval()
    outputs=[]
    with torch.no_grad():
        for i in range(0,len(a),batch_size):
            pa,_=net(torch.as_tensor(a[i:i+batch_size]),torch.as_tensor(b[i:i+batch_size]))
            outputs.append(pa.numpy())
    resid=np.concatenate(outputs,axis=0)
    return (p_base + np.float32(ck['residual_scale'])*resid).astype(np.float32)


def predict_all(seed, x1, x2, fusion_root, ccrt_root, batch_size):
    sd=fusion_root/f'seed_{seed}'
    ridge=joblib.load(sd/'ridge_model.joblib')
    mlp=joblib.load(sd/'mlp_model.joblib')
    if ridge.get('seed') != seed or mlp.get('seed') != seed:
        raise ValueError('Fusion seed mismatch')
    x=np.concatenate([x1,x2],axis=1)
    return {
        'ridge_fusion':np.asarray(ridge['model'].predict(x),dtype=np.float32),
        'mlp_fusion':np.asarray(mlp['model'].predict(mlp['preprocessor'].transform(x)),dtype=np.float32),
        'ccrt':ccrt_predict(ccrt_root/f'seed_{seed}',x1,x2,batch_size),
    },ridge,mlp


def check_reproduction(seed, pred, perts, genes, fusion_root, ccrt_root, tol):
    fi=read_npz(fusion_root/f'seed_{seed}'/'predictions.npz',
                 ['test_perturbations','genes','pred_ridge_fusion','pred_mlp_fusion'])
    ci=read_npz(ccrt_root/f'seed_{seed}'/'predictions.npz',
                 ['test_perturbations','genes','pred_ccrt'])
    if not np.array_equal(fi['genes'].astype(str),genes) or not np.array_equal(ci['genes'].astype(str),genes):
        raise ValueError(f'seed {seed}: saved response gene order differs')
    if not np.array_equal(fi['test_perturbations'].astype(str),ci['test_perturbations'].astype(str)):
        raise ValueError(f'seed {seed}: Step06A/B test IDs differ')
    lookup={p:i for i,p in enumerate(perts)}
    ids=fi['test_perturbations'].astype(str)
    idx=np.array([lookup[p] for p in ids])
    errors={}
    for model,key,reference in [('ridge_fusion','pred_ridge_fusion',fi),
                                ('mlp_fusion','pred_mlp_fusion',fi),
                                ('ccrt','pred_ccrt',ci)]:
        a=pred[model][idx]
        b=reference[key]
        if a.shape != b.shape or not np.isfinite(a).all():
            raise ValueError(f'seed {seed} {model}: bad K562 prediction shape/finite')
        err=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
        errors[model]=err
        if err > tol:
            raise RuntimeError(f'seed {seed} {model}: K562 checkpoint reproduction FAILED: max abs diff {err:.6g} > {tol}')
    return errors


def score(y,p,topk=50):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    mse=float(np.mean((y-p)**2))
    ym=y.ravel();pm=p.ravel()
    global_pcc=float(np.corrcoef(ym,pm)[0,1]) if np.std(ym)>0 and np.std(pm)>0 else np.nan
    a=y-y.mean(axis=1,keepdims=True);b=p-p.mean(axis=1,keepdims=True)
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    pc=np.divide((a*b).sum(axis=1),den,out=np.full(len(y),np.nan),where=den>1e-12)
    k=min(topk,y.shape[1]); overlaps=[]
    for i in range(len(y)):
        iy=np.argpartition(np.abs(y[i]),-k)[-k:]
        ip=np.argpartition(np.abs(p[i]),-k)[-k:]
        overlaps.append(len(set(iy)&set(ip))/k)
    return {'mse':mse,'global_pcc':global_pcc,
            'mean_perturbation_pcc':float(np.nanmean(pc)) if np.isfinite(pc).any() else np.nan,
            'median_perturbation_pcc':float(np.nanmedian(pc)) if np.isfinite(pc).any() else np.nan,
            'valid_pcc_fraction':float(np.isfinite(pc).mean()),
            'mean_top50_abs_overlap':float(np.mean(overlaps))},pc,np.asarray(overlaps)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.')
    p.add_argument('--rpe1-dir',default='results/step08_figure3_rpe1_v2')
    p.add_argument('--outdir',default='results/step08C_k562_to_rpe1_transfer')
    p.add_argument('--thresholds',nargs='+',type=int,default=[3,5,10])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--reproduction-tol',type=float,default=1e-4)
    args=p.parse_args()
    root=Path(args.root).resolve(); out=root/args.outdir
    out.mkdir(parents=True,exist_ok=True)
    fusion=root/'results/step06A_fusion_baselines'
    ccrt=root/'results/step06B_ccrt_oof_cpu_v3'
    split=root/'results/step05_tuned_genept_uce/inner_splits'
    response=read_npz(root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz',
                       ['perturbations','genes','delta'])
    perts=response['perturbations'].astype(str);genes=response['genes'].astype(str)
    if len(set(perts))!=len(perts) or len(set(genes))!=len(genes):raise ValueError('Duplicate K562 IDs')
    featdir=root/'results/step03_genept_uce_common/aligned_features'
    def features(name):
        f=read_npz(featdir/f'{name}.npz',['genes','embeddings'])
        if not np.array_equal(f['genes'].astype(str),perts):
            raise ValueError(f'{name} perturbation order differs from K562')
        x=f['embeddings'].astype(np.float32)
        if not np.isfinite(x).all():raise ValueError('Nonfinite embeddings')
        return x
    x1=features('GenePT_3large');x2=features('UCE_input_tokens')
    lookup={g:i for i,g in enumerate(perts)}
    summaries=[];details=[];repro=[]
    rpe1={}
    for th in args.thresholds:
        r=read_npz(root/args.rpe1_dir/f'batch{th}'/'rpe1_response_common_genes.npz',
                   ['perturbations','genes','delta'])
        rp=r['perturbations'].astype(str);rg=r['genes'].astype(str);ry=r['delta'].astype(np.float32)
        if len(set(rp))!=len(rp) or len(set(rg))!=len(rg):raise ValueError('Duplicate RPE1 IDs')
        if ry.shape!=(len(rp),len(rg)) or not np.isfinite(ry).all():raise ValueError('Invalid RPE1 matrix')
        if not set(rg).issubset(set(genes)):raise ValueError('RPE1 genes not subset of K562 genes')
        if not set(rp).issubset(set(perts)):raise ValueError('RPE1 perturbations not subset of K562')
        geneidx=np.array([np.where(genes==g)[0][0] for g in rg],dtype=int)
        rpe1[th]=(rp,rg,ry,geneidx)
        print(f'[RPE1] batch{th}: {len(rp)} perturbations, {len(rg)} genes',flush=True)
    with threadpool_limits(limits=args.threads):
        import torch
        torch.set_num_threads(args.threads)
        for seed in args.seeds:
            m=json.loads((split/f'seed_{seed}.json').read_text())
            train=set(m['outer_train']);test=set(m['outer_test'])
            if train&test or train|test!=set(perts):raise ValueError('Bad split')
            print(f'[SEED {seed}] loading frozen models',flush=True)
            predictions,ridge,mlp=predict_all(seed,x1,x2,fusion,ccrt,args.batch_size)
            if set(ridge['train_perturbations'])!=train or set(mlp['train_perturbations'])!=train:
                raise ValueError('Saved model training IDs do not match split')
            if not np.array_equal(np.asarray(ridge['response_genes']).astype(str),genes) or not np.array_equal(np.asarray(mlp['response_genes']).astype(str),genes):
                raise ValueError('Saved fusion output gene order mismatch')
            for name,v in predictions.items():
                if v.shape!=(len(perts),len(genes)) or not np.isfinite(v).all():
                    raise ValueError(f'{name} prediction shape or finite error')
            errs=check_reproduction(seed,predictions,perts,genes,fusion,ccrt,args.reproduction_tol)
            print(f'[K562 REPRO PASS] seed={seed} max abs errors={errs}',flush=True)
            repro.append({'seed':seed,**errs})
            for th,(rp,rg,ry,geneidx) in rpe1.items():
                ids=np.array([i for i,g in enumerate(rp) if g in train],dtype=int)
                selected=rp[ids]
                ix=np.array([lookup[g] for g in selected],dtype=int)
                y=ry[ids]
                if len(selected)==0:raise ValueError('Empty transfer set')
                sd=out/f'batch{th}'/f'seed_{seed}';sd.mkdir(parents=True,exist_ok=True)
                arrays={'perturbations':selected,'genes':rg,'y_true':y}
                for model,pred in predictions.items():
                    pr=pred[np.ix_(ix,geneidx)]
                    arrays[f'pred_{model}']=pr
                    stats,pc,ov=score(y,pr)
                    summaries.append({'threshold':th,'seed':seed,'model':model,'n_perturbations':len(selected),**stats})
                    for g,pp,oo in zip(selected,pc,ov):
                        details.append({'threshold':th,'seed':seed,'model':model,'perturbation':g,
                                        'perturbation_pcc':pp,'top50_abs_overlap':oo})
                zero=np.zeros_like(y)
                stats,_,_=score(y,zero)
                summaries.append({'threshold':th,'seed':seed,'model':'zero_response','n_perturbations':len(selected),**stats})
                np.savez_compressed(sd/'predictions.npz',**arrays)
                print(f'[TRANSFER] batch{th} seed={seed} n={len(selected)}',flush=True)
            del predictions,ridge,mlp
    pd.DataFrame(repro).to_csv(out/'k562_reproduction_audit.csv',index=False)
    df=pd.DataFrame(summaries);df.to_csv(out/'metrics_per_seed.csv',index=False)
    pd.DataFrame(details).to_csv(out/'per_perturbation_metrics.csv',index=False)
    df.groupby(['threshold','model'],as_index=False).agg(
        n_seeds=('seed','count'),n_perturbations_mean=('n_perturbations','mean'),
        mse_mean=('mse','mean'),mse_sd=('mse','std'),
        global_pcc_mean=('global_pcc','mean'),global_pcc_sd=('global_pcc','std'),
        perturbation_pcc_mean=('mean_perturbation_pcc','mean'),
        top50_overlap_mean=('mean_top50_abs_overlap','mean')
    ).to_csv(out/'metrics_summary.csv',index=False)
    print('[DONE]',out,flush=True)

if __name__=='__main__':main()
