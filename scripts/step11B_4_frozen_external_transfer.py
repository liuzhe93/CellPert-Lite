#!/usr/bin/env python3
"""Frozen Step10A K562 models -> HepG2/Jurkat; zero refitting, strict replay checks."""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f'Cannot import {path}')
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj

def read(path, keys):
    if not path.is_file(): raise FileNotFoundError(path)
    with np.load(path,allow_pickle=False) as z:
        missing=set(keys)-set(z.files)
        if missing:raise ValueError(f'{path}: missing {missing}; found {z.files}')
        return tuple(np.asarray(z[k]) for k in keys)

def unique(x,name):
    a=np.asarray(x).astype(str)
    if len(a)!=len(set(a)):raise ValueError(f'Duplicate {name}')
    return a

def pcc(y,p):
    a=y.astype(np.float64)-y.mean(axis=1,keepdims=True,dtype=np.float64)
    b=p.astype(np.float64)-p.mean(axis=1,keepdims=True,dtype=np.float64)
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return np.divide(np.einsum('ij,ij->i',a,b),den,out=np.full(len(y),np.nan),where=den>1e-12)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--outdir',default='results/step11B_4_frozen_external_transfer')
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--replay-tolerance',type=float,default=1e-4)
    ap.add_argument('--models',nargs='+',default=['gated_direct','no_gate_direct','genept_only_direct','uce_only_direct'])
    args=ap.parse_args();root=Path(args.root).resolve();out=root/args.outdir
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(args.threads)
    step10=root/'results/step10A_unseen_perturbation_transfer'
    b=module(root/'scripts/step09B_ccrt_formal_ablation.py','step09b_for11b4')
    c=module(root/'scripts/step09C_direct_ablation_transfer.py','step09c_for11b4')
    ids,genes,y=read(root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz',('perturbations','genes','delta'))
    p1,x1=read(root/'results/step03_genept_uce_common/aligned_features/GenePT_3large.npz',('genes','embeddings'))
    p2,x2=read(root/'results/step03_genept_uce_common/aligned_features/UCE_input_tokens.npz',('genes','embeddings'))
    ids=unique(ids,'K562 targets');genes=unique(genes,'K562 response genes')
    if not (np.array_equal(ids,p1.astype(str)) and np.array_equal(ids,p2.astype(str))):raise ValueError('Feature target order mismatch')
    x1=x1.astype(np.float32);x2=x2.astype(np.float32)
    if y.shape!=(len(ids),len(genes)) or not all(np.isfinite(a).all() for a in (x1,x2,y)):raise ValueError('Bad K562 inputs')
    idx={v:i for i,v in enumerate(ids)};gidx={v:i for i,v in enumerate(genes)}
    external={}
    for cell in ('HepG2','Jurkat'):
        f=root/'data/external_validation/step11B_2A_expanded_minbatch1'/cell/'response_matrix_k562_gene_intersection.npz'
        pp,gg,yy=read(f,('perturbations','genes','delta'))
        pp=unique(pp,cell+' targets');gg=unique(gg,cell+' response genes')
        if yy.shape!=(len(pp),len(gg)) or not np.isfinite(yy).all():raise ValueError(f'Bad {cell} response')
        if not set(gg).issubset(gidx):raise ValueError(f'{cell} response genes outside K562')
        if list(gg)!=[g for g in genes if g in set(gg)]:raise ValueError(f'{cell} gene order is not K562 subset')
        external[cell]=(pp,gg,yy.astype(np.float32),np.array([gidx[g] for g in gg],dtype=int))
        print(f'[{cell}] total={len(pp)} embedding-covered={sum(p in idx for p in pp)} response_genes={len(gg)}',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    metrics=[];perrows=[];audit=[]
    with threadpool_limits(limits=args.threads):
      for seed in args.seeds:
        sd=step10/f'seed_{seed}'
        split=json.loads((sd/'split.json').read_text())
        tr=unique(split['train'],f'seed{seed} train');te=unique(split['test'],f'seed{seed} test')
        if set(tr)&set(te) or set(tr)|set(te)!=set(ids):raise RuntimeError(f'seed{seed} invalid K562 split')
        ti=np.array([idx[p] for p in tr]);s1=StandardScaler().fit(x1[ti]);s2=StandardScaler().fit(x2[ti])
        def transform(q):
            q=np.array(q,dtype=int)
            return (s1.transform(x1[q]).astype(np.float32)/np.float32(np.sqrt(x1.shape[1])),
                    s2.transform(x2[q]).astype(np.float32)/np.float32(np.sqrt(x2.shape[1])))
        for model_name in args.models:
            variant='gated_no_residual' if model_name=='gated_direct' else model_name
            md=sd/model_name
            ck=torch.load(md/'model.pt',map_location='cpu',weights_only=True)
            if ck['variant']!=variant or int(ck['seed'])!=seed:raise RuntimeError(f'Checkpoint mismatch: {md}')
            width=int(ck['width']);dropout=float(ck['dropout'])
            net=(b.make_model(variant,x1.shape[1],x2.shape[1],len(genes),width,dropout)
                 if model_name=='gated_direct' else c.make_direct(variant,x1.shape[1],x2.shape[1],len(genes),width,dropout))
            net.load_state_dict(ck['state_dict'],strict=True);net.eval()
            # Check the exact same K562 held-out targets reproduce saved predictions.
            hold_ids,hold_genes,hold_true,hold_pred=read(md/'K562_holdout_predictions.npz',('perturbations','genes','y_true','prediction'))
            hold_ids=hold_ids.astype(str)
            if not np.array_equal(hold_genes.astype(str),genes) or not set(hold_ids).issubset(set(te)):
                raise RuntimeError(f'Holdout mismatch {md}')
            hx,hz=transform([idx[p] for p in hold_ids]);replay=b.inference(net,hx,hz,args.batch_size)
            if replay.shape!=hold_pred.shape or hold_true.shape!=replay.shape:raise RuntimeError('Replay shape mismatch')
            err=float(np.max(np.abs(replay.astype(np.float64)-hold_pred.astype(np.float64))))
            if err>args.replay_tolerance:raise RuntimeError(f'Replay FAILED {md}: max_abs={err:.6g}')
            print(f'[REPLAY PASS] seed={seed} {model_name} max_abs={err:.3g}',flush=True)
            for cell,(pp,gg,yy,gi) in external.items():
                chosen=[p for p in pp if p in idx]
                pi={p:i for i,p in enumerate(pp)}
                xx,zz=transform([idx[p] for p in chosen]);pred=b.inference(net,xx,zz,args.batch_size)[:,gi]
                truth=yy[[pi[p] for p in chosen]]
                if pred.shape!=truth.shape or not np.isfinite(pred).all():raise RuntimeError('Invalid external predictions')
                groups={'all_embedding_covered':chosen,'K562_train_seen':[p for p in chosen if p in set(tr)],
                        'K562_test_heldout':[p for p in chosen if p in set(te)]}
                dest=out/cell/f'seed_{seed}'/model_name;dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/'predictions.npz',perturbations=np.asarray(chosen),genes=gg,y_true=truth,prediction=pred,
                                    k562_train_seen=np.asarray([p in set(tr) for p in chosen],dtype=bool))
                for group,names in groups.items():
                    sub=np.array([i for i,p in enumerate(chosen) if p in set(names)],dtype=int)
                    if not len(sub):continue
                    a=truth[sub];v=pred[sub];pcs=pcc(a,v)
                    mse=float(np.mean((a.astype(np.float64)-v.astype(np.float64))**2))
                    baseline=float(np.mean(a.astype(np.float64)**2))
                    metrics.append({'dataset':cell,'seed':seed,'model':model_name,'group':group,
                                    'n_targets':len(sub),'n_response_genes':len(gg),'mse':mse,
                                    'zero_baseline_mse':baseline,'mse_improvement_vs_zero':baseline-mse,
                                    'mean_target_pcc':float(np.nanmean(pcs)) if np.isfinite(pcs).any() else np.nan,
                                    'median_target_pcc':float(np.nanmedian(pcs)) if np.isfinite(pcs).any() else np.nan,
                                    'valid_pcc_fraction':float(np.isfinite(pcs).mean())})
                pcs=pcc(truth,pred)
                for i,p in enumerate(chosen):
                    perrows.append({'dataset':cell,'seed':seed,'model':model_name,'perturbation':p,
                                    'K562_train_seen':p in set(tr),'K562_test_heldout':p in set(te),
                                    'pcc':pcs[i],'mse':float(np.mean((truth[i].astype(np.float64)-pred[i].astype(np.float64))**2))})
                audit.append({'dataset':cell,'seed':seed,'model':model_name,'checkpoint_replay_max_abs':err,
                              'n_all_external_targets':len(pp),'n_embedding_covered':len(chosen),
                              'n_K562_train_seen':len(groups['K562_train_seen']),
                              'n_K562_test_heldout':len(groups['K562_test_heldout'])})
                print(f'[{cell}] seed={seed} {model_name}: covered={len(chosen)} seen={len(groups["K562_train_seen"])} heldout={len(groups["K562_test_heldout"])}',flush=True)
    pd.DataFrame(metrics).to_csv(out/'metrics_per_seed.csv',index=False)
    pd.DataFrame(perrows).to_csv(out/'per_perturbation_metrics.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'replay_and_coverage_audit.csv',index=False)
    print('DONE',out,'(frozen models; no fitting or tuning)',flush=True)

if __name__=='__main__':main()
