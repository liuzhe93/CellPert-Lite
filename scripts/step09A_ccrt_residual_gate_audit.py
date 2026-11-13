#!/usr/bin/env python3
"""Step09A | Frozen CCRT residual/gate audit (NO retraining or tuning).

Primary: K562 outer-test (unseen perturbations per seed).
Secondary: RPE1 seen-transfer (Step08C); model-vs-base on the SAME genes.

This is diagnostic, not a formal gate ablation. Five seeds overlap: report
seed-level summaries and unique-perturbation descriptive statistics only.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODELS = ('ridge_base', 'ccrt')


def read_npz(path, required):
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        missing = set(required) - set(z.files)
        if missing:
            raise KeyError(f'{path}: missing {sorted(missing)}; available={z.files}')
        return {k: z[k] for k in required}


def row_pcc(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    a = y - y.mean(axis=1, keepdims=True)
    b = p - p.mean(axis=1, keepdims=True)
    den = np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
    return np.divide((a*b).sum(axis=1), den, out=np.full(len(y), np.nan), where=den > 1e-12)


def topk_overlap(y, p, k):
    k = min(k, y.shape[1])
    if k <= 0:
        raise ValueError('topk must be positive')
    yi = np.argpartition(np.abs(y), -k, axis=1)[:, -k:]
    pi = np.argpartition(np.abs(p), -k, axis=1)[:, -k:]
    return np.array([len(set(a.tolist()) & set(b.tolist()))/k for a,b in zip(yi,pi)])


def compute_rows(seed, dataset, threshold, perts, genes, y, base, residual, ccrt, gate=None, topk=50):
    perts = np.asarray(perts).astype(str)
    genes = np.asarray(genes).astype(str)
    arrays = [np.asarray(v, dtype=np.float64) for v in (y, base, residual, ccrt)]
    if len(set(perts)) != len(perts) or len(set(genes)) != len(genes):
        raise ValueError(f'{dataset} seed={seed}: duplicate perturbations/genes')
    if any(v.shape != (len(perts),len(genes)) for v in arrays):
        raise ValueError(f'{dataset} seed={seed}: shape mismatch')
    if any(not np.isfinite(v).all() for v in arrays):
        raise ValueError(f'{dataset} seed={seed}: nonfinite response/prediction')
    y,base,residual,ccrt = arrays
    discrepancy = float(np.max(np.abs((base+residual)-ccrt)))
    if discrepancy > 2e-5:
        raise ValueError(f'{dataset} seed={seed}: CCRT != base+residual, max diff={discrepancy}')
    bse = (base-y)**2
    cse = (ccrt-y)**2
    b_mse = bse.mean(axis=1)
    c_mse = cse.mean(axis=1)
    b_pcc = row_pcc(y,base)
    c_pcc = row_pcc(y,ccrt)
    b_top = topk_overlap(y,base,topk)
    c_top = topk_overlap(y,ccrt,topk)
    improve = b_mse-c_mse
    residual_rms = np.sqrt(np.mean(residual**2,axis=1))
    base_rms = np.sqrt(np.mean(base**2,axis=1))
    true_rms = np.sqrt(np.mean(y**2,axis=1))
    err = y-base
    alignment = np.sum(residual*err,axis=1)
    den = np.linalg.norm(residual,axis=1)*np.linalg.norm(err,axis=1)
    cosine = np.divide(alignment,den,out=np.full(len(y),np.nan),where=den>1e-12)
    # MSE(base)-MSE(base+resid) = 2 E[resid*(y-base)] - E[resid^2]
    linear_gain = 2*np.mean(residual*err,axis=1)
    penalty = np.mean(residual**2,axis=1)
    if not np.allclose(improve, linear_gain-penalty, atol=1e-9, rtol=1e-5):
        raise RuntimeError('Residual gain identity failed')
    if gate is None:
        gate = np.full(len(perts),np.nan)
    gate = np.asarray(gate,dtype=float).ravel()
    if len(gate) != len(perts):
        raise ValueError('gate length mismatch')
    out=[]
    for i,pert in enumerate(perts):
        out.append(dict(dataset=dataset,threshold=threshold,seed=seed,perturbation=pert,
                        n_genes=len(genes),base_mse=b_mse[i],ccrt_mse=c_mse[i],
                        mse_gain=improve[i],relative_mse_gain=improve[i]/b_mse[i] if b_mse[i]>1e-12 else np.nan,
                        base_pcc=b_pcc[i],ccrt_pcc=c_pcc[i],pcc_gain=c_pcc[i]-b_pcc[i],
                        base_topk_overlap=b_top[i],ccrt_topk_overlap=c_top[i],topk_gain=c_top[i]-b_top[i],
                        response_rms=true_rms[i],base_rms=base_rms[i],residual_rms=residual_rms[i],
                        residual_to_base_rms=residual_rms[i]/base_rms[i] if base_rms[i]>1e-12 else np.nan,
                        residual_error_cosine=cosine[i],linear_gain=linear_gain[i],residual_penalty=penalty[i],
                        gate_mean=gate[i],improved=bool(improve[i]>0)))
    gene_rows=[]
    gene_gain=(bse-cse).mean(axis=0)
    gene_resid_rms=np.sqrt(np.mean(residual**2,axis=0))
    for j,g in enumerate(genes):
        gene_rows.append(dict(dataset=dataset,threshold=threshold,seed=seed,gene=g,
                              base_mse=bse[:,j].mean(),ccrt_mse=cse[:,j].mean(),
                              mse_gain=gene_gain[j],residual_rms=gene_resid_rms[j],
                              n_perturbations=len(perts)))
    return out,gene_rows,discrepancy


def summarize(df):
    rows=[]
    for (dataset,th,seed), g in df.groupby(['dataset','threshold','seed'],dropna=False):
        rows.append(dict(dataset=dataset,threshold=th,seed=seed,n_perturbations=len(g),
                         base_mse=g.base_mse.mean(),ccrt_mse=g.ccrt_mse.mean(),
                         mse_gain=g.mse_gain.mean(),fraction_improved=g.improved.mean(),
                         mean_pcc_gain=g.pcc_gain.mean(),mean_topk_gain=g.topk_gain.mean(),
                         mean_residual_rms=g.residual_rms.mean(),
                         median_residual_to_base_rms=g.residual_to_base_rms.median(),
                         mean_gate=g.gate_mean.mean(),sd_gate=g.gate_mean.std(),
                         mean_response_rms=g.response_rms.mean()))
    return pd.DataFrame(rows)


def gate_detail(root, seed, perts, batch_size):
    """Infer per-hidden-unit gate on K562 outer-test, strictly from frozen checkpoint."""
    import joblib
    import torch
    from torch import nn
    sd=root/'results/step06B_ccrt_oof_cpu_v3'/f'seed_{seed}'
    try:
        ck=torch.load(sd/'ccrt_model.pt',map_location='cpu',weights_only=True)
    except TypeError:
        ck=torch.load(sd/'ccrt_model.pt',map_location='cpu')
    if int(ck['selected_epoch']) == 0:
        return None, {'seed':seed,'selected_epoch':0,'note':'Ridge-only selected; gate not trained'}
    feat=root/'results/step03_genept_uce_common/aligned_features'
    x1=read_npz(feat/'GenePT_3large.npz',['genes','embeddings'])
    x2=read_npz(feat/'UCE_input_tokens.npz',['genes','embeddings'])
    labels1=x1['genes'].astype(str);labels2=x2['genes'].astype(str)
    if not np.array_equal(labels1,labels2):
        raise ValueError('GenePT/UCE embedding perturbation order mismatch')
    ids={p:i for i,p in enumerate(labels1)}
    ix=np.array([ids[p] for p in perts])
    d1=int(ck['genept_dim']);d2=int(ck['uce_dim']);w=int(ck['width'])
    pp=joblib.load(sd/'feature_preprocessors.joblib')
    a=np.asarray(pp['genept_scaler'].transform(x1['embeddings'][ix]),dtype=np.float32)/np.float32(np.sqrt(d1))
    b=np.asarray(pp['uce_scaler'].transform(x2['embeddings'][ix]),dtype=np.float32)/np.float32(np.sqrt(d2))
    if a.shape[1]!=d1 or b.shape[1]!=d2:
        raise ValueError('Embedding dimension mismatch')
    class CCRT(nn.Module):
        def __init__(self):
            super().__init__()
            self.a=nn.Sequential(nn.Linear(d1,w),nn.LayerNorm(w),nn.GELU())
            self.b=nn.Sequential(nn.Linear(d2,w),nn.LayerNorm(w),nn.GELU())
            self.gate=nn.Sequential(nn.Linear(w*2,w),nn.Sigmoid())
            self.head=nn.Sequential(nn.Dropout(float(ck['dropout'])),nn.Linear(w,w),nn.GELU(),
                                    nn.Dropout(float(ck['dropout'])),nn.Linear(w,int(ck['response_dim'])))
        def forward(self,x,y):
            av=self.a(x);bv=self.b(y)
            return self.gate(torch.cat([av,bv],dim=1))
    net=CCRT().float();net.load_state_dict(ck['state_dict'],strict=True);net.eval()
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(ix),batch_size):
            chunks.append(net(torch.as_tensor(a[start:start+batch_size]),
                              torch.as_tensor(b[start:start+batch_size])).numpy())
    gates=np.concatenate(chunks,axis=0)
    if not np.isfinite(gates).all() or np.any((gates<0)|(gates>1)):
        raise ValueError('Invalid gate values')
    detail=pd.DataFrame({'seed':seed,'perturbation':perts,'gate_mean':gates.mean(axis=1),
                         'gate_std_across_units':gates.std(axis=1),
                         'gate_saturation_fraction':((gates<0.05)|(gates>0.95)).mean(axis=1),
                         'gate_fraction_above_half':(gates>0.5).mean(axis=1)})
    stat={'seed':seed,'selected_epoch':int(ck['selected_epoch']),
          'n_hidden_units':w,'gate_mean':float(gates.mean()),
          'gate_std_all':float(gates.std()),
          'gate_saturation_fraction':float(((gates<0.05)|(gates>0.95)).mean()),
          'between_perturbation_sd_gate_mean':float(gates.mean(axis=1).std()),
          'between_unit_sd_gate_mean':float(gates.mean(axis=0).std()),
          'note':'Gate values are NOT causal modality importance'}
    return detail,stat


def make_plots(df, summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    k=df[df.dataset=='K562_outer_test'].copy()
    if k.empty:return
    fig,ax=plt.subplots(figsize=(6.5,4.4))
    for seed,g in k.groupby('seed'):
        ax.scatter(g.response_rms,g.mse_gain,s=12,alpha=.45,label=f'Seed {seed}',rasterized=True)
    ax.axhline(0,color='black',lw=.9,ls='--')
    ax.set(xlabel='Observed response RMS (K562)',ylabel='MSE improvement (Ridge base - CCRT)',
           title='Residual correction vs response strength')
    ax.legend(frameon=False,fontsize=8,ncol=2)
    fig.tight_layout()
    for ext in ('pdf','svg'):fig.savefig(out/f'Figure4A_residual_gain_vs_signal.{ext}',bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4.3))
    vals=[g.mse_gain.values for _,g in k.groupby('seed')]
    labels=[str(seed) for seed,_ in k.groupby('seed')]
    ax.boxplot(vals,tick_labels=labels,showfliers=False)
    ax.axhline(0,color='black',lw=.9,ls='--')
    ax.set(xlabel='Seed',ylabel='Per-perturbation MSE improvement',title='CCRT vs its own Ridge base | K562 outer-test')
    fig.tight_layout()
    for ext in ('pdf','svg'):fig.savefig(out/f'Figure4A_residual_gain_by_seed.{ext}',bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4.2))
    s=summary[summary.dataset=='K562_outer_test'].sort_values('seed')
    ax.bar(s.seed.astype(str),s.fraction_improved)
    ax.axhline(.5,color='black',lw=.9,ls='--')
    ax.set(xlabel='Seed',ylabel='Fraction of perturbations improved',ylim=(0,1),
           title='CCRT correction success rate | K562 outer-test')
    fig.tight_layout()
    for ext in ('pdf','svg'):fig.savefig(out/f'Figure4A_fraction_improved.{ext}',bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.')
    p.add_argument('--outdir',default='results/step09A_ccrt_residual_gate_audit')
    p.add_argument('--step08c',default='results/step08C_k562_to_rpe1_transfer')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    p.add_argument('--thresholds',type=int,nargs='+',default=[3,5,10])
    p.add_argument('--topk',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--skip-gate-detail',action='store_true',help='Skip loading checkpoints/embeddings; use saved mean gate only')
    p.add_argument('--skip-rpe1',action='store_true')
    args=p.parse_args()
    os.environ['OMP_NUM_THREADS']=str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS']=str(args.threads)
    from threadpoolctl import threadpool_limits
    root=Path(args.root).resolve();out=root/args.outdir;out.mkdir(parents=True,exist_ok=True)
    base_dir=root/'results/step06B_ccrt_oof_cpu_v3'
    rows=[];gene_rows=[];audits=[];gate_rows=[];gate_stats=[]
    with threadpool_limits(limits=args.threads):
        for seed in args.seeds:
            sd=base_dir/f'seed_{seed}'
            z=read_npz(sd/'predictions.npz',
                ['test_perturbations','genes','y_true','pred_ccrt','pred_ridge_base',
                 'residual_ccrt','gate_mean_per_perturbation'])
            ids=z['test_perturbations'].astype(str)
            rr,gg,diff=compute_rows(seed,'K562_outer_test',0,ids,z['genes'],z['y_true'],
                 z['pred_ridge_base'],z['residual_ccrt'],z['pred_ccrt'],
                 z['gate_mean_per_perturbation'],args.topk)
            rows.extend(rr);gene_rows.extend(gg)
            audits.append({'seed':seed,'dataset':'K562_outer_test','max_reconstruction_difference':diff,
                           'n_perturbations':len(ids),'n_genes':len(z['genes'])})
            print(f'[K562] seed={seed} n={len(ids)} mse_gain={np.mean([v["mse_gain"] for v in rr]):.8g}',flush=True)
            if not args.skip_gate_detail:
                gd,gs=gate_detail(root,seed,ids,args.batch_size)
                gate_stats.append(gs)
                if gd is not None:
                    gate_rows.extend(gd.to_dict('records'))
                    # Validate gate against the original stored means.
                    err=float(np.nanmax(np.abs(gd.gate_mean.to_numpy()-z['gate_mean_per_perturbation'])))
                    if err>1e-5:
                        raise ValueError(f'seed={seed}: gate checkpoint reproduction failed ({err})')
                    print(f'[GATE REPRO] seed={seed} max abs={err:.3g}',flush=True)
            if args.skip_rpe1:continue
            # The Step08C RPE1 files do NOT save the CCRT-specific Ridge base.
            # Recreate its prediction with the frozen Step06B ridge_base.joblib.
            import joblib
            base=joblib.load(sd/'ridge_base.joblib')
            feat=root/'results/step03_genept_uce_common/aligned_features'/'GenePT_3large.npz'
            f=read_npz(feat,['genes','embeddings'])
            fmap={name:i for i,name in enumerate(f['genes'].astype(str))}
            for th in args.thresholds:
                path=root/args.step08c/f'batch{th}'/f'seed_{seed}'/'predictions.npz'
                t=read_npz(path,['perturbations','genes','y_true','pred_ccrt'])
                names=t['perturbations'].astype(str)
                indices=np.array([fmap[name] for name in names])
                full_base=np.asarray(base.predict(f['embeddings'][indices]),dtype=np.float32)
                kg=read_npz(root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz',['genes'])['genes'].astype(str)
                gene_map={g:i for i,g in enumerate(kg)}
                gi=np.array([gene_map[g] for g in t['genes'].astype(str)])
                b=full_base[:,gi]
                c=t['pred_ccrt']
                residual=np.asarray(c,dtype=np.float64)-np.asarray(b,dtype=np.float64)
                rr,gg,diff=compute_rows(seed,'RPE1_seen_transfer',th,names,t['genes'],t['y_true'],b,residual,c,
                                        None,args.topk)
                rows.extend(rr);gene_rows.extend(gg)
                audits.append({'seed':seed,'dataset':f'RPE1_batch{th}',
                               'max_reconstruction_difference':diff,'n_perturbations':len(names),
                               'n_genes':len(t['genes'])})
                print(f'[RPE1 batch{th}] seed={seed} n={len(names)} mse_gain={np.mean([v["mse_gain"] for v in rr]):.8g}',flush=True)
    df=pd.DataFrame(rows);genes=pd.DataFrame(gene_rows)
    df.to_csv(out/'per_perturbation_residual_audit.csv',index=False)
    genes.to_csv(out/'per_gene_residual_audit.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'reconstruction_audit.csv',index=False)
    if gate_rows:pd.DataFrame(gate_rows).to_csv(out/'per_perturbation_gate_detail.csv',index=False)
    if gate_stats:pd.DataFrame(gate_stats).to_csv(out/'gate_statistics_by_seed.csv',index=False)
    summary=summarize(df)
    summary.to_csv(out/'metrics_per_seed.csv',index=False)
    # Descriptive across seed summaries; seeds are NOT independent biological replicates.
    aggregate=summary.groupby(['dataset','threshold'],as_index=False).agg(
        n_seeds=('seed','size'),mean_base_mse=('base_mse','mean'),mean_ccrt_mse=('ccrt_mse','mean'),
        mean_mse_gain=('mse_gain','mean'),sd_seed_mse_gain=('mse_gain','std'),
        mean_fraction_improved=('fraction_improved','mean'),
        mean_pcc_gain=('mean_pcc_gain','mean'),mean_topk_gain=('mean_topk_gain','mean'),
        mean_residual_rms=('mean_residual_rms','mean'))
    aggregate.to_csv(out/'metrics_summary.csv',index=False)
    # Each perturbation may appear in multiple seeds; average first, do not pseudo-replicate.
    unique=df.groupby(['dataset','threshold','perturbation'],as_index=False).agg(
        n_seeds=('seed','size'),mean_mse_gain=('mse_gain','mean'),
        mean_pcc_gain=('pcc_gain','mean'),mean_response_rms=('response_rms','mean'),
        fraction_seeds_improved=('improved','mean'),mean_residual_rms=('residual_rms','mean'))
    unique.to_csv(out/'unique_perturbation_summary.csv',index=False)
    # Gene rankings: descriptive; same response genes across seeds, but shared perturbations.
    gene_summary=genes.groupby(['dataset','threshold','gene'],as_index=False).agg(
        mean_mse_gain=('mse_gain','mean'),sd_seed_mse_gain=('mse_gain','std'),
        mean_residual_rms=('residual_rms','mean'),n_seeds=('seed','size'))
    gene_summary.to_csv(out/'gene_level_summary.csv',index=False)
    make_plots(df,summary,out)
    config=vars(args).copy();config['root']=str(root)
    (out/'run_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
    print('\n=== PRIMARY: K562 outer-test ===',flush=True)
    print(aggregate[aggregate.dataset=='K562_outer_test'].to_string(index=False),flush=True)
    print('\n[COMPLETE]',out,flush=True)
    print('NOTE: Gate mean is NOT a causal modality contribution. This is NOT a formal ablation.',flush=True)

if __name__=='__main__':main()
