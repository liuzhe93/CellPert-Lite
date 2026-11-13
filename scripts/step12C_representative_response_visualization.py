#!/usr/bin/env python3
"""Step12C: frozen CellPert-Lite prediction visualizations (no model training).

Inputs: Step12A selected_candidates_unique.csv and Step11B-4 predictions.npz.
Uses K562 observed response matrix for biological comparison, NEVER presents K562
observations as model predictions. All comparisons align response genes by name.
Generates scatter, observed/predicted response profiles, and cross-cell-line heatmaps.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_npz(path, keys):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}')
    with np.load(path, allow_pickle=False) as z:
        missing=set(keys)-set(z.files)
        if missing: raise ValueError(f'{path}: missing {sorted(missing)}; available {z.files}')
        return [np.asarray(z[k]) for k in keys]


def unique_names(arr, label):
    v=np.asarray(arr).astype(str)
    if len(v)!=len(set(v)): raise ValueError(f'Duplicate {label}')
    return v


def pcc(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:return np.nan
    return float(np.corrcoef(a,b)[0,1])


def safe_name(s):
    return ''.join(c if c.isalnum() or c in '_-' else '_' for c in str(s))


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--root',default='.')
    ap.add_argument('--candidates',default='results/step12A_representative_perturbations/selected_candidates_unique.csv')
    ap.add_argument('--predictions-dir',default='results/step11B_4_frozen_external_transfer')
    ap.add_argument('--k562',default='data/processed/benchmark_genept_uce_common/response_matrix_common.npz')
    ap.add_argument('--outdir',default='results/step12C_representative_response_visualization')
    ap.add_argument('--model',default='gated_direct')
    ap.add_argument('--genes',nargs='*',default=None,help='Explicit target gene symbols; otherwise rank Step12A candidates')
    ap.add_argument('--top-n',type=int,default=8,help='Maximum targets per external cell line, unless --genes given')
    ap.add_argument('--top-response-genes',type=int,default=30)
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    ap.add_argument('--max-plot-points',type=int,default=3500)
    args=ap.parse_args()
    if args.top_n<1 or args.top_response_genes<1:raise ValueError('Counts must be positive')
    root=Path(args.root).resolve();out=root/args.outdir;out.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    candidates=pd.read_csv(root/args.candidates)
    needed={'dataset','perturbation','group','mean_pcc','external_rms'}
    if not needed.issubset(candidates.columns):raise ValueError(f'Candidate CSV missing {needed-set(candidates.columns)}')
    candidates=candidates[candidates.dataset.isin(['HepG2','Jurkat'])].copy()
    candidates['perturbation']=candidates.perturbation.astype(str)
    candidates['group']=candidates.group.astype(str)
    k_ids,k_genes,k_y=load_npz(root/args.k562,('perturbations','genes','delta'))
    k_ids=unique_names(k_ids,'K562 targets');k_genes=unique_names(k_genes,'K562 genes')
    if k_y.shape!=(len(k_ids),len(k_genes)):raise ValueError('K562 response matrix shape mismatch')
    k_target={v:i for i,v in enumerate(k_ids)};k_gene={v:i for i,v in enumerate(k_genes)}
    # Cache each frozen prediction NPZ only once. Verify that its stored true response
    # and target/gene labels agree with the other seeds for the same cell line.
    cache={};truth_reference={};rows=[];profiles=[];audit=[]
    for cell in ('HepG2','Jurkat'):
        for seed in args.seeds:
            path=root/args.predictions_dir/cell/f'seed_{seed}'/args.model/'predictions.npz'
            if not path.is_file():
                print(f'[WARN] Missing {path}',flush=True);continue
            ids,genes,truth,pred,seen=load_npz(path,('perturbations','genes','y_true','prediction','k562_train_seen'))
            ids=unique_names(ids,f'{cell} seed{seed} targets');genes=unique_names(genes,f'{cell} seed{seed} genes')
            if truth.shape!=pred.shape or truth.shape!=(len(ids),len(genes)):raise ValueError(f'Bad matrix shapes {path}')
            if len(seen)!=len(ids):raise ValueError(f'Bad seen vector {path}')
            if not set(genes).issubset(k_gene):raise ValueError(f'{path}: gene outside K562')
            if not np.isfinite(truth).all() or not np.isfinite(pred).all():raise ValueError(f'Nonfinite predictions {path}')
            if cell not in truth_reference:
                truth_reference[cell]=(ids,genes,truth)
            else:
                refids,refgenes,reftruth=truth_reference[cell]
                if not np.array_equal(refids,ids) or not np.array_equal(refgenes,genes) or not np.allclose(reftruth,truth,atol=1e-7,rtol=1e-6):
                    raise ValueError(f'External truth differs between seeds in {cell}; investigate preprocessing')
            cache[(cell,seed)]=(ids,genes,truth,pred,seen,{v:i for i,v in enumerate(ids)})
    if not cache:raise FileNotFoundError('No Step11B-4 prediction NPZ found')
    # Priority: candidate ranks from Step12A, then strong cross-cell-line contrasts.
    for cell in ('HepG2','Jurkat'):
        c=candidates[candidates.dataset==cell].copy()
        if args.genes:
            targets=list(dict.fromkeys(args.genes))
        else:
            if 'selection_priority' in c.columns:
                c=c.sort_values(['selection_priority','perturbation'],na_position='last')
            else:c=c.sort_values(['mean_pcc','perturbation'],ascending=[False,True])
            targets=list(dict.fromkeys(c.perturbation))[:args.top_n]
        for target in targets:
            matched=c[c.perturbation==target]
            requested_groups=set(matched.group) if not matched.empty else {'seen','heldout'}
            for group in ('seen','heldout'):
                if group not in requested_groups and not args.genes:continue
                entries=[]
                for seed in args.seeds:
                    if (cell,seed) not in cache:continue
                    ids,genes,truth,pred,seen,index=cache[(cell,seed)]
                    if target not in index:continue
                    i=index[target]
                    actual_group='seen' if bool(seen[i]) else 'heldout'
                    if actual_group!=group:continue
                    entries.append((seed,genes,truth[i].astype(float),pred[i].astype(float)))
                if not entries:
                    audit.append({'dataset':cell,'perturbation':target,'group':group,'status':'no_matching_seed_predictions'})
                    continue
                genes=entries[0][1]
                if any(not np.array_equal(x[1],genes) for x in entries):raise ValueError('Gene order differs between seed predictions')
                obs=entries[0][2]
                if any(not np.allclose(x[2],obs,atol=1e-7) for x in entries):raise ValueError('Truth differs between seeds')
                preds=np.stack([x[3] for x in entries]);mean_pred=preds.mean(axis=0)
                seed_pcc=[pcc(obs,x[3]) for x in entries]
                gene_idx=np.array([k_gene[g] for g in genes],int)
                k_obs=k_y[k_target[target],gene_idx].astype(float) if target in k_target else None
                order=np.argsort(-np.abs(obs),kind='stable')[:min(args.top_response_genes,len(genes))]
                # Compare across lines on the SAME response gene set, with no normalization of observed amplitudes.
                for j in order:
                    profiles.append({'dataset':cell,'perturbation':target,'group':group,'gene':genes[j],
                                     'external_observed':obs[j],'external_prediction_mean':mean_pred[j],
                                     'k562_observed':k_obs[j] if k_obs is not None else np.nan,
                                     'n_seeds':len(entries)})
                rows.append({'dataset':cell,'perturbation':target,'group':group,'n_seeds':len(entries),
                             'seeds':';'.join(str(x[0]) for x in entries),
                             'mean_seed_pcc':np.nanmean(seed_pcc),'pcc_of_mean_prediction':pcc(obs,mean_pred),
                             'pcc_k562_external_observed':pcc(k_obs,obs) if k_obs is not None else np.nan,
                             'observed_rms':float(np.sqrt(np.mean(obs**2))),
                             'predicted_rms':float(np.sqrt(np.mean(mean_pred**2))),
                             'mse_mean_prediction':float(np.mean((obs-mean_pred)**2)),
                             'skill_vs_zero_mean_prediction':float(1-np.mean((obs-mean_pred)**2)/np.mean(obs**2)) if np.mean(obs**2)>0 else np.nan})
                audit.append({'dataset':cell,'perturbation':target,'group':group,'status':'plotted','n_seeds':len(entries)})
                fig,axs=plt.subplots(1,3,figsize=(14.5,4.2),gridspec_kw={'width_ratios':[1,1.25,1.25]})
                rng=np.random.default_rng(42)
                sample=np.arange(len(genes))
                if len(sample)>args.max_plot_points:sample=np.sort(rng.choice(sample,args.max_plot_points,replace=False))
                axs[0].scatter(obs[sample],mean_pred[sample],s=7,alpha=.32,rasterized=True)
                low=min(obs.min(),mean_pred.min());high=max(obs.max(),mean_pred.max())
                axs[0].plot([low,high],[low,high],ls='--',color='gray',lw=.9)
                axs[0].set(xlabel='Observed external response',ylabel='Frozen model prediction',title=f'PCC={pcc(obs,mean_pred):.3f} | {len(entries)} seeds')
                x=np.arange(len(order)); labels=genes[order]
                axs[1].plot(x,obs[order],marker='o',ms=3,lw=1.1,label=f'{cell} observed')
                axs[1].plot(x,mean_pred[order],marker='s',ms=3,lw=1.1,label='Frozen prediction')
                axs[1].axhline(0,color='gray',lw=.7)
                axs[1].set_xticks(x);axs[1].set_xticklabels(labels,rotation=90,fontsize=6)
                axs[1].set(ylabel='Response (delta expression)',title='Top genes by |external observed|')
                axs[1].legend(frameon=False,fontsize=7)
                if k_obs is not None:
                    axs[2].plot(x,k_obs[order],marker='o',ms=3,lw=1.1,label='K562 observed')
                axs[2].plot(x,obs[order],marker='o',ms=3,lw=1.1,label=f'{cell} observed')
                axs[2].axhline(0,color='gray',lw=.7)
                axs[2].set_xticks(x);axs[2].set_xticklabels(labels,rotation=90,fontsize=6)
                axs[2].set(ylabel='Observed response (delta expression)',title=f'Cross-line observed PCC={pcc(k_obs,obs):.3f}' if k_obs is not None else 'K562 response unavailable')
                axs[2].legend(frameon=False,fontsize=7)
                fig.suptitle(f'{target} | {cell} | K562 {group} | {args.model}',fontsize=12)
                fig.tight_layout()
                stem=f'Figure8_{safe_name(cell)}_{safe_name(group)}_{safe_name(target)}'
                for ext in ('png','pdf','svg'):fig.savefig(out/f'{stem}.{ext}',dpi=220,bbox_inches='tight')
                plt.close(fig)
                print(f'[PLOT] {cell} {target} {group}: n_seeds={len(entries)}, mean_PCC={np.nanmean(seed_pcc):.3f}',flush=True)
    pd.DataFrame(rows).to_csv(out/'representative_prediction_metrics.csv',index=False)
    pd.DataFrame(profiles).to_csv(out/'representative_top_response_genes.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'plot_audit.csv',index=False)
    # Cross-cell-line comparison of the SAME target: only observational truths are used.
    # Common genes are explicitly aligned; target-wise top genes are selected using the
    # average of absolute observed responses in both lines, not model predictions.
    if all(cell in truth_reference for cell in ('HepG2','Jurkat')):
        a_ids,a_genes,a_truth=truth_reference['HepG2'];b_ids,b_genes,b_truth=truth_reference['Jurkat']
        aidx={v:i for i,v in enumerate(a_ids)};bidx={v:i for i,v in enumerate(b_ids)}
        ag={v:i for i,v in enumerate(a_genes)};bg={v:i for i,v in enumerate(b_genes)}
        common=[g for g in a_genes if g in bg]
        ia=np.array([ag[g] for g in common]);ib=np.array([bg[g] for g in common])
        chosen=sorted(set(pd.DataFrame(rows).perturbation)) if rows else []
        cross=[]
        for target in chosen:
            if target not in aidx or target not in bidx:continue
            av=a_truth[aidx[target],ia].astype(float);bv=b_truth[bidx[target],ib].astype(float)
            cross.append({'perturbation':target,'n_common_genes':len(common),'HepG2_Jurkat_observed_PCC':pcc(av,bv),
                          'HepG2_RMS':float(np.sqrt(np.mean(av**2))),'Jurkat_RMS':float(np.sqrt(np.mean(bv**2)))})
            n=min(args.top_response_genes,len(common));ix=np.argsort(-(np.abs(av)+np.abs(bv)),kind='stable')[:n]
            mat=np.stack([av[ix],bv[ix]])
            bound=max(float(np.max(np.abs(mat))),1e-8)
            fig,ax=plt.subplots(figsize=(max(7,n*.3),2.8))
            im=ax.imshow(mat,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
            ax.set_yticks([0,1]);ax.set_yticklabels(['HepG2 observed','Jurkat observed'])
            ax.set_xticks(np.arange(n));ax.set_xticklabels(np.asarray(common)[ix],rotation=90,fontsize=7)
            ax.set_title(f'{target}: observed cross-line responses (common genes)')
            fig.colorbar(im,ax=ax,label='Observed delta expression',shrink=.7)
            fig.tight_layout()
            for ext in ('png','pdf','svg'):fig.savefig(out/f'Figure8_crossline_{safe_name(target)}.{ext}',dpi=220,bbox_inches='tight')
            plt.close(fig)
        pd.DataFrame(cross,columns=['perturbation','n_common_genes','HepG2_Jurkat_observed_PCC','HepG2_RMS','Jurkat_RMS']).to_csv(out/'cross_line_observed_comparison.csv',index=False)
    print(f'DONE {out} | {len(rows)} plotted target/group comparisons',flush=True)

if __name__=='__main__':main()
