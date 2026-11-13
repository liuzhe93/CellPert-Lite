#!/usr/bin/env python3
"""Step11B-7: matched min_batch=1 vs min_batch=5 external response sensitivity.

Frozen Step11B-4 predictions are re-evaluated against BOTH truth definitions on
identical perturbations and genes. No model fitting, prediction regeneration, or tuning.
The thresholds change truth construction; the prediction vector remains fixed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

MODELS=('gated_direct','no_gate_direct','genept_only_direct','uce_only_direct')
GROUPS=('all_embedding_covered','K562_train_seen','K562_test_heldout')

def load(path,keys):
    if not path.is_file(): raise FileNotFoundError(f'Missing: {path}')
    with np.load(path,allow_pickle=False) as z:
        if not set(keys).issubset(z.files): raise ValueError(f'{path}: expected {keys}; found {z.files}')
        return tuple(np.asarray(z[k]) for k in keys)

def unique(x,label):
    a=np.asarray(x).astype(str)
    if len(a)!=len(set(a)): raise ValueError(f'Duplicate {label}')
    return a

def pcc(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    yc=y-y.mean(axis=1,keepdims=True);pc=p-p.mean(axis=1,keepdims=True)
    den=np.linalg.norm(yc,axis=1)*np.linalg.norm(pc,axis=1)
    return np.divide(np.einsum('ij,ij->i',yc,pc),den,out=np.full(len(y),np.nan),where=den>1e-12)

def summary(d):
    if not len(d): return {}
    return {'n_targets':len(d),'n_valid_pcc':int(np.isfinite(d.pcc).sum()),
            'mean_target_pcc':float(d.pcc.mean()),'median_target_pcc':float(d.pcc.median()),
            'mean_target_mse':float(d.mse.mean()),
            'mean_zero_baseline_mse':float(d.zero_baseline_mse.mean()),
            'mse_improvement_vs_zero':float(d.mse_improvement_vs_zero.mean()),
            'relative_skill_vs_zero':float(d.mse_improvement_vs_zero.mean()/d.zero_baseline_mse.mean()) if d.zero_baseline_mse.mean()>0 else np.nan}

def boot_delta(a,b,n,rng):
    a=np.asarray(a,float);b=np.asarray(b,float)
    good=np.isfinite(a)&np.isfinite(b);a=a[good];b=b[good]
    if not len(a):return {'n_paired':0,'mean_delta':np.nan,'ci_low':np.nan,'ci_high':np.nan}
    diff=a-b
    vals=np.empty(n)
    for j in range(n): vals[j]=np.mean(diff[rng.integers(len(diff),size=len(diff))])
    return {'n_paired':len(diff),'mean_delta':float(diff.mean()),'ci_low':float(np.quantile(vals,.025)),
            'ci_high':float(np.quantile(vals,.975))}

def savefig(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.')
    ap.add_argument('--prediction-dir',default='results/step11B_4_frozen_external_transfer')
    ap.add_argument('--strict-dir',default='data/external_validation/step11B_2_responses')
    ap.add_argument('--expanded-dir',default='data/external_validation/step11B_2A_expanded_minbatch1')
    ap.add_argument('--outdir',default='results/step11B_7_threshold_sensitivity')
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    ap.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--bootstrap-seed',type=int,default=2026)
    args=ap.parse_args()
    if args.bootstrap<100:ap.error('--bootstrap must be >=100')
    root=Path(args.root).resolve();out=root/args.outdir;out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.bootstrap_seed)
    all_rows=[];coverage=[];consistency=[]
    for cell in ('HepG2','Jurkat'):
        f='response_matrix_k562_gene_intersection.npz'
        p1,g1,y1=load(root/args.expanded_dir/cell/f,('perturbations','genes','delta'))
        p5,g5,y5=load(root/args.strict_dir/cell/f,('perturbations','genes','delta'))
        p1=unique(p1,f'{cell} min1 targets');p5=unique(p5,f'{cell} min5 targets')
        g1=unique(g1,f'{cell} min1 genes');g5=unique(g5,f'{cell} min5 genes')
        if y1.shape!=(len(p1),len(g1)) or y5.shape!=(len(p5),len(g5)):
            raise ValueError(f'{cell}: invalid response shapes')
        if not np.isfinite(y1).all() or not np.isfinite(y5).all():raise ValueError(f'{cell}: nonfinite truth')
        common_targets=[p for p in p1 if p in set(p5)]
        common_genes=[g for g in g1 if g in set(g5)]
        if not common_targets or not common_genes:raise ValueError(f'{cell}: empty matched intersection')
        i1={p:i for i,p in enumerate(p1)};i5={p:i for i,p in enumerate(p5)}
        j1={g:i for i,g in enumerate(g1)};j5={g:i for i,g in enumerate(g5)}
        a=y1[np.ix_([i1[p] for p in common_targets],[j1[g] for g in common_genes])].astype(np.float64)
        b=y5[np.ix_([i5[p] for p in common_targets],[j5[g] for g in common_genes])].astype(np.float64)
        coverage.append({'dataset':cell,'n_min1_targets':len(p1),'n_min5_targets':len(p5),
                         'n_common_targets_before_embedding_filter':len(common_targets),
                         'n_min1_genes':len(g1),'n_min5_genes':len(g5),'n_matched_genes':len(common_genes)})
        tc=pcc(a,b)
        for k,p in enumerate(common_targets):
            consistency.append({'dataset':cell,'perturbation':p,'truth_min1_min5_pcc':tc[k],
                                'truth_min1_min5_mse':float(np.mean((a[k]-b[k])**2)),
                                'truth_min1_rms':float(np.sqrt(np.mean(a[k]**2))),
                                'truth_min5_rms':float(np.sqrt(np.mean(b[k]**2)))})
        print(f'[{cell}] min1={len(p1)}, min5={len(p5)}, common={len(common_targets)}, genes={len(common_genes)}',flush=True)
        truth1={p:a[i] for i,p in enumerate(common_targets)}
        truth5={p:b[i] for i,p in enumerate(common_targets)}
        for seed in args.seeds:
            for model in args.models:
                path=root/args.prediction_dir/cell/f'seed_{seed}'/model/'predictions.npz'
                pp,gg,yt,pred,seen=load(path,('perturbations','genes','y_true','prediction','k562_train_seen'))
                pp=unique(pp,f'{cell}/{seed}/{model} prediction targets');gg=unique(gg,'prediction genes')
                if len(seen)!=len(pp) or pred.shape!=(len(pp),len(gg)) or yt.shape!=pred.shape:raise ValueError(f'Invalid predictions {path}')
                pj={g:i for i,g in enumerate(gg)}
                if not set(common_genes).issubset(pj):raise ValueError(f'{path}: matched genes missing')
                if not np.array_equal(gg,g1):raise ValueError(f'{path}: prediction genes differ from min1 response; verify Step11B-4 inputs')
                if not np.allclose(yt,y1[[i1[p] for p in pp]],rtol=1e-5,atol=1e-6):
                    raise ValueError(f'{path}: saved min1 y_true differs from expanded response matrix')
                cols=[pj[g] for g in common_genes]
                picked=[i for i,p in enumerate(pp) if p in truth5]
                if not picked:raise ValueError(f'No common embedding-covered targets for {path}')
                names=[pp[i] for i in picked];v=pred[np.ix_(picked,cols)].astype(np.float64)
                for threshold,lookup in [('min_batch1',truth1),('min_batch5',truth5)]:
                    yy=np.stack([lookup[p] for p in names]);pcs=pcc(yy,v)
                    mse=np.mean((yy-v)**2,axis=1);base=np.mean(yy**2,axis=1)
                    for k,p in enumerate(names):
                        all_rows.append({'dataset':cell,'seed':seed,'model':model,'perturbation':p,
                                         'group':'K562_train_seen' if bool(seen[picked[k]]) else 'K562_test_heldout',
                                         'threshold':threshold,'pcc':pcs[k],'mse':mse[k],
                                         'zero_baseline_mse':base[k],'mse_improvement_vs_zero':base[k]-mse[k]})
                print(f'  seed={seed} {model}: matched covered targets={len(names)}',flush=True)
    df=pd.DataFrame(all_rows)
    df.to_csv(out/'per_target_matched_metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'coverage_audit.csv',index=False)
    pd.DataFrame(consistency).to_csv(out/'truth_threshold_consistency.csv',index=False)
    sums=[]
    for (cell,seed,model,threshold),d in df.groupby(['dataset','seed','model','threshold'],sort=True):
        for group in GROUPS:
            x=d if group=='all_embedding_covered' else d[d.group==group]
            if len(x):sums.append({'dataset':cell,'seed':seed,'model':model,'threshold':threshold,'group':group,**summary(x)})
    pd.DataFrame(sums).to_csv(out/'metrics_per_seed_matched.csv',index=False)
    deltas=[]
    for (cell,seed,model),d in df.groupby(['dataset','seed','model'],sort=True):
        for group in GROUPS:
            x=d if group=='all_embedding_covered' else d[d.group==group]
            for metric in ('pcc','mse','mse_improvement_vs_zero'):
                wide=x.pivot(index='perturbation',columns='threshold',values=metric)
                if not {'min_batch1','min_batch5'}.issubset(wide.columns):continue
                st=boot_delta(wide.min_batch5,wide.min_batch1,args.bootstrap,rng)
                deltas.append({'dataset':cell,'seed':seed,'model':model,'group':group,'metric':metric,
                               'comparison':'min_batch5_minus_min_batch1',**st})
    pd.DataFrame(deltas).to_csv(out/'paired_threshold_differences.csv',index=False)
    # Compare models within each threshold using identical perturbations, independently by seed.
    models=[]
    for (cell,seed,threshold),d in df.groupby(['dataset','seed','threshold'],sort=True):
        for group in GROUPS:
            x=d if group=='all_embedding_covered' else d[d.group==group]
            for competitor in args.models:
                if competitor=='gated_direct' or 'gated_direct' not in args.models:continue
                for metric in ('pcc','mse_improvement_vs_zero'):
                    wide=x.pivot(index='perturbation',columns='model',values=metric)
                    if not {'gated_direct',competitor}.issubset(wide.columns):continue
                    st=boot_delta(wide.gated_direct,wide[competitor],args.bootstrap,rng)
                    models.append({'dataset':cell,'seed':seed,'threshold':threshold,'group':group,
                                   'metric':metric,'comparison':f'gated_direct_minus_{competitor}',**st})
    pd.DataFrame(models).to_csv(out/'paired_model_differences_by_threshold.csv',index=False)
    # Seed-averaged descriptive summaries, not independent biological replication.
    summ=pd.DataFrame(sums)
    avg=(summ.groupby(['dataset','model','threshold','group'],as_index=False)
         [['n_targets','mean_target_pcc','mean_target_mse','mse_improvement_vs_zero','relative_skill_vs_zero']].mean())
    avg.to_csv(out/'model_threshold_summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for cell in ('HepG2','Jurkat'):
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
        for ax,group in zip(axes,('K562_train_seen','K562_test_heldout')):
            sub=avg[(avg.dataset==cell)&(avg.group==group)]
            xx=np.arange(len(args.models));w=.36
            for t,offset in [('min_batch1',-w/2),('min_batch5',w/2)]:
                vals=[sub[(sub.model==m)&(sub.threshold==t)].mean_target_pcc.mean() for m in args.models]
                ax.bar(xx+offset,vals,width=w,label=t)
            ax.set_xticks(xx,[m.replace('_direct','').replace('_','\n') for m in args.models]);ax.set_title(group)
            ax.set_ylabel('Mean per-target PCC');ax.axhline(0,color='gray',linewidth=.8)
            ax.legend(fontsize=8)
        fig.suptitle(f'{cell}: matched-target threshold sensitivity (seed mean)')
        fig.tight_layout();savefig(fig,out/f'Figure7_{cell}_matched_PCC')
    (out/'analysis_manifest.json').write_text(json.dumps(vars(args),indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nSUMMARY (five-seed means; identical targets/genes within each cell line):',flush=True)
    print(avg[(avg.model=='gated_direct')][['dataset','threshold','group','n_targets','mean_target_pcc','mse_improvement_vs_zero']].to_string(index=False),flush=True)
    print('\nDONE:',out,flush=True)

if __name__=='__main__':main()
