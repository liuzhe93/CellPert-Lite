#!/usr/bin/env python3
"""Read-only metadata, expression-scale and control audit; no full-matrix loading."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

def vals(s, n=12):
    return {str(k):int(v) for k,v in s.value_counts(dropna=False).head(n).items()}

def audit(name, path, n=500):
    a=ad.read_h5ad(path,backed='r')
    try:
        obs=a.obs
        r={'dataset':name,'path':str(path),'shape':[int(a.n_obs),int(a.n_vars)],'layers':list(a.layers.keys()),'uns_keys':list(a.uns.keys()),'x_dtype':str(a.X.dtype),'obs_columns':list(obs.columns),'var_columns':list(a.var.columns),'var_names_preview':a.var_names[:8].astype(str).tolist()}
        for k in ('perturbation','gene_name','is_control','control','batch','batch_num','dataset_batch_num','perturbation_type','UMI_count','n_counts'):
            if k in obs.columns:
                if k in ('UMI_count','n_counts'):
                    q=pd.to_numeric(obs[k],errors='coerce'); r[k+'_summary']={'median':float(q.median()),'mean':float(q.mean())}
                else:r[k+'_value_counts']=vals(obs[k])
        if 'is_control' in obs.columns:
            flag=obs['is_control'].astype(str).str.lower().isin(['true','1','yes','t'])
            r['control_cells_from_is_control']=int(flag.sum())
            for b in ('batch_num','dataset_batch_num','batch'):
                if b in obs:
                    tab=pd.crosstab(obs[b].astype(str),flag)
                    r['control_by_'+b]=[{ 'batch':str(i),'control':int(tab.loc[i].get(True,0)),'noncontrol':int(tab.loc[i].get(False,0))} for i in tab.index]
                    break
        elif 'perturbation' in obs:
            flag=obs['perturbation'].astype(str).str.lower().eq('control')
            r['control_cells_from_perturbation']=int(flag.sum())
        if 'gene_name' in obs:
            s=obs.loc[~flag,'gene_name'].astype('string') if 'flag' in locals() else obs['gene_name'].astype('string')
            vc=s.value_counts()
            r['noncontrol_unique_targets']=int(len(vc))
            r['noncontrol_targets_ge_20_30_50_100']={str(t):int((vc>=t).sum()) for t in (20,30,50,100)}
        x=a.X[:min(n,a.n_obs)]
        if sparse.issparse(x):x=x.toarray()
        x=np.asarray(x,dtype=np.float64)
        nz=x[x!=0]
        sums=x.sum(axis=1)
        r['x_sample']={'n_cells':int(x.shape[0]),'min':float(x.min()),'max':float(x.max()),'mean':float(x.mean()),'zero_fraction':float(np.mean(x==0)),'nonzero_integer_fraction':float(np.mean(np.isclose(nz,np.rint(nz),atol=1e-5))) if len(nz) else None,'row_sum_quantiles':np.quantile(sums,[0,.25,.5,.75,1]).tolist(),'expm1_row_sum_quantiles':np.quantile(np.expm1(np.clip(x,0,20)).sum(axis=1),[0,.25,.5,.75,1]).tolist()}
        if 'UMI_count' in obs:
            umi=pd.to_numeric(obs['UMI_count'].iloc[:len(sums)],errors='coerce').to_numpy(dtype=float)
            good=np.isfinite(umi)
            if good.sum()>2:r['x_sample']['corr_X_rowsum_vs_UMI_count']=float(np.corrcoef(sums[good],umi[good])[0,1]) if np.std(sums[good])>0 and np.std(umi[good])>0 else None
        r['gene_name_duplicates']=int(a.var['gene_name'].astype(str).duplicated().sum()) if 'gene_name' in a.var else None
        return r
    finally:a.file.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.')
    p.add_argument('--rpe1-h5ad',default=None)
    p.add_argument('--outdir',default='results/step11B_0_four_cell_line_audit')
    p.add_argument('--sample-cells',type=int,default=500)
    args=p.parse_args()
    root=Path(args.root)
    out=root/args.outdir;out.mkdir(parents=True,exist_ok=True)
    paths={'K562':root/'data/raw/replogle22k562_processed_complete.h5ad','HepG2':root/'data/external_validation/HepG2/nadig_2024_hepg2.h5ad','Jurkat':root/'data/external_validation/Jurkat/nadig_2024_jurkat.h5ad'}
    if args.rpe1_h5ad:paths['RPE1']=Path(args.rpe1_h5ad)
    result={}
    for name,path in paths.items():
        print('\n###',name,path,flush=True)
        if not path.exists():print('MISSING',flush=True);result[name]={'error':'file missing','path':str(path)};continue
        try:
            d=audit(name,path,args.sample_cells);result[name]=d
            print('shape',d['shape'],'x',d['x_sample'],flush=True)
            print('controls',d.get('control_cells_from_is_control',d.get('control_cells_from_perturbation','unknown')),flush=True)
        except Exception as e:
            result[name]={'error':repr(e),'path':str(path)};print('ERROR',repr(e),flush=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    rows=[]
    for name,d in result.items():
        x=d.get('x_sample',{})
        rows.append({'dataset':name,'path':d.get('path'),'cells':d.get('shape',[None,None])[0],'genes':d.get('shape',[None,None])[1],'controls':d.get('control_cells_from_is_control',d.get('control_cells_from_perturbation')),'noncontrol_targets':d.get('noncontrol_unique_targets'),'max_X':x.get('max'),'zero_fraction':x.get('zero_fraction'),'nonzero_integer_fraction':x.get('nonzero_integer_fraction'),'median_rowsum_X':x.get('row_sum_quantiles',[None]*5)[2],'median_rowsum_expm1_X':x.get('expm1_row_sum_quantiles',[None]*5)[2],'error':d.get('error')})
    pd.DataFrame(rows).to_csv(out/'audit_summary.csv',index=False)
    print('\nSaved',out/'audit.json','and',out/'audit_summary.csv')
    print('IMPORTANT: This audit suggests expression scale but cannot prove normalization provenance.')
if __name__=='__main__':main()
