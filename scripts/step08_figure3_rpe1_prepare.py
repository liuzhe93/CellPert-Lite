#!/usr/bin/env python3
"""Figure 3 phase 1: audit RPE1 AnnData and optionally build a batch-matched response matrix.
CPU only. No model training or unverified cross-cell performance claims.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse


def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.')
    p.add_argument('--rpe1',default='data/raw/replogle_rpe1/ReplogleWeissman2022_rpe1.h5ad')
    p.add_argument('--k562',default='data/processed/benchmark_genept_uce_common/response_matrix_common.npz')
    p.add_argument('--outdir',default='results/step08_figure3_rpe1_audit')
    p.add_argument('--mode',choices=['audit','build'],default='audit')
    p.add_argument('--perturbation-col',default=None)
    p.add_argument('--batch-col',default=None)
    p.add_argument('--control-labels',nargs='+',default=None)
    p.add_argument('--layer',default='X',help='X or named layer; MUST contain raw nonnegative counts for build')
    p.add_argument('--min-cells',type=int,default=20)
    p.add_argument('--min-controls',type=int,default=20)
    p.add_argument('--target-sum',type=float,default=10000)
    p.add_argument('--chunk-size',type=int,default=2048)
    return p.parse_args()


def summary_obs(a,out):
    obs=a.obs
    rows=[]
    for col in obs.columns:
        s=obs[col]
        nunique=s.nunique(dropna=False)
        vals=s.value_counts(dropna=False).head(12)
        rows.append({'column':str(col),'dtype':str(s.dtype),'n_unique':int(nunique),
                     'top_values':json.dumps({str(k):int(v) for k,v in vals.items()},ensure_ascii=False)})
    pd.DataFrame(rows).to_csv(out/'rpe1_obs_columns.csv',index=False)
    print('\n[OBS COLUMNS]')
    for row in rows:
        if row['n_unique']<=5000:
            print(f"{row['column']}: unique={row['n_unique']} top={row['top_values'][:250]}")


def read_reference(path):
    with np.load(path,allow_pickle=False) as z:
        ids=z['perturbations'].astype(str); genes=z['genes'].astype(str)
        shape=z['delta'].shape
    return ids,genes,shape


def normalize_block(x,target_sum):
    if sparse.issparse(x):
        x=x.tocsr().astype(np.float64)
        if x.data.size and (not np.isfinite(x.data).all() or np.min(x.data)<0):
            raise ValueError('Negative/nonfinite expression; build requires raw nonnegative counts')
        if x.data.size and not np.allclose(x.data,np.rint(x.data),atol=1e-5):
            raise ValueError('Expression is not integer-like raw counts; STOP rather than normalize twice')
        totals=np.asarray(x.sum(axis=1)).ravel()
        factor=np.divide(target_sum,totals,out=np.zeros_like(totals),where=totals>0)
        x=x.multiply(factor[:,None]).tocsr();x.data=np.log1p(x.data)
        return np.asarray(x.sum(axis=0)).ravel(),len(totals)
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all() or (x<0).any() or not np.allclose(x,np.rint(x),atol=1e-5):
        raise ValueError('Expression is not integer-like raw counts; STOP')
    totals=x.sum(axis=1)
    x=x*np.divide(target_sum,totals,out=np.zeros_like(totals),where=totals>0)[:,None]
    return np.log1p(x).sum(axis=0),len(x)


def group_mean(a,indices,layer,chunk,target_sum):
    total=None;n=0
    for j in range(0,len(indices),chunk):
        ids=indices[j:j+chunk]
        x=a[ids,:].X if layer=='X' else a[ids,:].layers[layer]
        vec,nc=normalize_block(x,target_sum)
        total=vec if total is None else total+vec;n+=nc
    return total/n


def main():
    ar=args();root=Path(ar.root).resolve();out=(root/ar.outdir).resolve();out.mkdir(parents=True,exist_ok=True)
    rp=(root/ar.rpe1).resolve();kp=(root/ar.k562).resolve()
    if not rp.is_file() or not kp.is_file():raise FileNotFoundError(f'Missing input: {rp} or {kp}')
    try:import anndata as ad
    except ImportError:raise SystemExit('Install anndata: pip install anndata')
    a=ad.read_h5ad(rp,backed='r')
    kperts,kgenes,kshape=read_reference(kp)
    rgenes=np.asarray(a.var_names,dtype=str)
    if len(set(rgenes))!=len(rgenes):raise ValueError('Duplicate RPE1 gene IDs; resolve before proceeding')
    common_genes=sorted(set(rgenes)&set(kgenes))
    report={'rpe1_cells':int(a.n_obs),'rpe1_genes':int(a.n_vars),
            'k562_perturbations':int(len(kperts)),'k562_response_genes':int(len(kgenes)),
            'k562_delta_shape':list(kshape),'common_response_genes':len(common_genes),
            'common_response_gene_fraction_of_k562':len(common_genes)/len(kgenes),
            'expression_layer_requested':ar.layer,'mode':ar.mode,
            'warning':'K562 response normalization has NOT been verified against RPE1. Do not interpret cross-dataset MSE until harmonized.'}
    print('[INPUT]',rp,'RPE1',a.shape,'K562',kshape)
    print('[OVERLAP] response genes',len(common_genes))
    pd.Series(common_genes,name='gene').to_csv(out/'common_response_genes.csv',index=False)
    summary_obs(a,out)
    if ar.perturbation_col:
        if ar.perturbation_col not in a.obs:raise ValueError('Missing perturbation column')
        labels=a.obs[ar.perturbation_col].astype(str)
        rperts=set(labels.unique())
        report['rpe1_perturbation_labels']=len(rperts)
        report['overlap_perturbation_labels']=len(rperts&set(kperts))
        counts=labels.value_counts().rename_axis('perturbation').reset_index(name='n_cells')
        counts['in_k562_perturbations']=counts['perturbation'].isin(kperts)
        counts.to_csv(out/'rpe1_perturbation_counts.csv',index=False)
        print('[OVERLAP] perturbation labels',report['overlap_perturbation_labels'])
    if ar.mode=='build':
        if not ar.perturbation_col or not ar.batch_col or not ar.control_labels:
            raise ValueError('Build requires --perturbation-col, --batch-col and --control-labels')
        if ar.batch_col not in a.obs:raise ValueError('Missing batch column')
        if ar.layer!='X' and ar.layer not in a.layers:raise ValueError('Missing layer')
        if ar.min_cells<1 or ar.min_controls<1 or ar.chunk_size<1:raise ValueError('Invalid minimum/chunk')
        obs=a.obs[[ar.perturbation_col,ar.batch_col]].astype(str).copy()
        obs.columns=['perturbation','batch']
        obs['position']=np.arange(len(obs))
        is_control=obs['perturbation'].isin(ar.control_labels)
        if not is_control.any():raise ValueError('No control cells matched EXACT control labels')
        control_means={};control_counts={}
        for batch,sub in obs[is_control].groupby('batch',sort=True):
            if len(sub)<ar.min_controls:continue
            control_means[batch]=group_mean(a,sub.position.to_numpy(),ar.layer,ar.chunk_size,ar.target_sum)
            control_counts[batch]=len(sub)
        if not control_means:raise ValueError('No batch has sufficient controls')
        rows=[];groups=[]
        for (pert,batch),sub in obs[~is_control].groupby(['perturbation','batch'],sort=True):
            if len(sub)<ar.min_cells or batch not in control_means:continue
            avg=group_mean(a,sub.position.to_numpy(),ar.layer,ar.chunk_size,ar.target_sum)
            rows.append((avg-control_means[batch]).astype(np.float32))
            groups.append({'perturbation':pert,'batch':batch,'n_perturbed':len(sub),'n_controls':control_counts[batch]})
        if not rows:raise ValueError('No eligible perturbation-batch groups')
        group_table=pd.DataFrame(groups)
        group_table.to_csv(out/'eligible_perturbation_batches.csv',index=False)
        mat=np.stack(rows)
        # Pool batch-matched effects by number of perturbed cells; preserve a batch table for audit.
        perts=[];delta=[]
        for pert,sub in group_table.groupby('perturbation',sort=True):
            ix=sub.index.to_numpy();weights=sub.n_perturbed.to_numpy(dtype=float)
            perts.append(pert);delta.append(np.average(mat[ix],axis=0,weights=weights))
        delta=np.asarray(delta,dtype=np.float32)
        common_idx=np.array([np.where(rgenes==g)[0][0] for g in common_genes],dtype=int)
        np.savez_compressed(out/'rpe1_response_log1p_cp10k.npz',
            perturbations=np.asarray(perts,dtype=str),genes=rgenes,delta=delta)
        np.savez_compressed(out/'rpe1_response_common_genes.npz',
            perturbations=np.asarray(perts,dtype=str),genes=np.asarray(common_genes,dtype=str),delta=delta[:,common_idx])
        report['built_perturbations']=len(perts);report['eligible_perturbation_batch_groups']=len(groups)
        report['response_definition']='Weighted batch-matched mean log1p(CP10k) perturbed minus controls; weights=n_perturbed'
        print('[BUILD PASS]',delta.shape,'common',(len(perts),len(common_genes)))
    (out/'audit_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('[DONE]',out)
    print('[IMPORTANT] Do not compare RPE1 delta to K562 until K562 normalization/response definition is verified.')
    a.file.close()

if __name__=='__main__':main()
