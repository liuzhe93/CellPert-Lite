#!/usr/bin/env python3
"""Publication-oriented plots from verified Step03 outputs. No retraining.

Five seeds are repeated random holdouts, not five-fold CV or independent cohorts.
Per-perturbation panels use the preselected seed 42, not pooled repeats.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPS = ['GenePT_ada', 'GenePT_3large', 'UCE_input_tokens']
LABELS = ['GenePT-ada', 'GenePT-3-large', 'UCE input tokens']
COLORS = ['#477FB3', '#CC7355', '#559C88']
MODELS = ['no_change', 'train_mean', 'ridge_gene', 'mlp_gene']


def row_pcc(y, p):
    a = y.astype(np.float64) - y.mean(axis=1, keepdims=True, dtype=np.float64)
    b = p.astype(np.float64) - p.mean(axis=1, keepdims=True, dtype=np.float64)
    den = np.sqrt((a*a).sum(1) * (b*b).sum(1))
    # A constant prediction has undefined PCC, not PCC=0.
    return np.divide((a*b).sum(1), den, out=np.full(len(y), np.nan), where=den > 1e-12)


def save(fig, out, name):
    for extension in ['svg', 'pdf', 'png']:
        fig.savefig(out / f'{name}.{extension}', dpi=300)
    plt.close(fig)
    print('[FIGURE]', name, flush=True)


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', color='#E6E6E6', linewidth=.5)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='.')
    ap.add_argument('--results', default='results/step03_genept_uce_common')
    ap.add_argument('--outdir', default='results/step04_Figure2')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42,43,44,45,46])
    ap.add_argument('--distribution-seed', type=int, default=42)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    base, out = root / args.results, root / args.outdir
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Use at least two distinct seeds for mean +/- SD')
    if args.distribution_seed not in args.seeds:
        raise ValueError('Distribution seed must be included in --seeds')
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'Output directory is nonempty; choose a fresh --outdir: {out}')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                         'axes.labelsize':9, 'axes.titlesize':10,
                         'svg.fonttype':'none', 'pdf.fonttype':42,
                         'ps.fonttype':42, 'savefig.bbox':'tight'})
    response_path = root / 'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
    with np.load(response_path, allow_pickle=False) as z:
        response_names = z['perturbations'].astype(str)
        response_genes = z['genes'].astype(str)
        response = z['delta']
    if len(set(response_names)) != len(response_names):
        raise ValueError('Duplicate response perturbations')
    response_lookup = {g:i for i,g in enumerate(response_names)}
    all_metrics, per_rows = [], []
    for seed in args.seeds:
        reference_names = reference_truth = reference_baseline = None
        for rep in REPS:
            run = base / f'seed_{seed}' / rep
            verified = json.loads((run/'STEP03_VERIFIED.json').read_text())
            metrics = pd.read_csv(run/'metrics.csv')
            if len(metrics) != 4 or set(metrics['model']) != set(MODELS):
                raise ValueError(f'Unexpected model rows: {run}')
            required = ['mse','global_pcc_delta','mean_perturbation_pcc',
                        'deg_recovery_top20','deg_recovery_top50','deg_recovery_top100']
            for key in required:
                if key not in metrics:
                    raise KeyError(f'Missing {key} in {run}')
            metrics.insert(0,'representation',rep)
            metrics.insert(1,'seed',seed)
            all_metrics.append(metrics)
            with np.load(run/'predictions.npz', allow_pickle=False) as z:
                test = z['test_perturbations'].astype(str)
                y = z['y_true']
                if len(set(test)) != len(test) or y.shape != (len(test),len(response_genes)):
                    raise ValueError(f'Invalid prediction dimensions/labels: {run}')
                if not np.array_equal(z['genes'].astype(str),response_genes):
                    raise ValueError(f'Response gene order mismatch: {run}')
                if set(test) != set(verified['split']['test']):
                    raise ValueError(f'Prediction labels disagree with verified split: {run}')
                expected_y = response[[response_lookup[g] for g in test]]
                if not np.array_equal(y, expected_y):
                    raise ValueError(f'y_true differs from source response: {run}')
                if reference_names is None:
                    reference_names = sorted(test.tolist())
                if set(test) != set(reference_names):
                    raise ValueError(f'Test perturbations differ across representations: seed {seed}')
                index = {g:i for i,g in enumerate(test)}
                order = np.array([index[g] for g in reference_names])
                y = y[order]
                if reference_truth is None:
                    reference_truth = y.copy()
                elif not np.array_equal(y, reference_truth):
                    raise ValueError('Cross-representation ground-truth mismatch')
                baseline = z['pred_train_mean'][order]
                if reference_baseline is None:
                    reference_baseline = baseline.copy()
                elif not np.array_equal(baseline, reference_baseline):
                    raise ValueError('Cross-representation train-mean baseline mismatch')
                for model in MODELS:
                    p = z['pred_'+model][order]
                    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
                        raise ValueError(f'Invalid prediction values: {run}/{model}')
                    pcc = row_pcc(y,p)
                    record = metrics.loc[metrics.model.eq(model)].iloc[0]
                    mse = np.mean((y.astype(float)-p)**2)
                    if not np.isclose(mse,record['mse'],rtol=5e-4,atol=1e-8):
                        raise ValueError(f'MSE audit failed: {run}/{model}')
                    mean_pcc = np.nanmean(pcc) if np.isfinite(pcc).any() else np.nan
                    if not np.isclose(mean_pcc,record['mean_perturbation_pcc'],
                                      atol=5e-5,rtol=5e-4,equal_nan=True):
                        raise ValueError(f'PCC audit failed: {run}/{model}; review metric implementation')
                    per_rows.append(pd.DataFrame({'seed':seed,'representation':rep,
                                                   'model':model,'perturbation':reference_names,
                                                   'pcc':pcc}))
            print('[AUDIT PASS]', seed, rep, flush=True)

    metrics = pd.concat(all_metrics,ignore_index=True)
    per = pd.concat(per_rows,ignore_index=True)
    # Baselines are representation-independent. Verify before deduplicating.
    for seed in args.seeds:
        for model in ['no_change','train_mean']:
            rows = metrics.loc[metrics.seed.eq(seed)&metrics.model.eq(model)]
            a = rows[required].to_numpy(float)
            if not np.allclose(a,a[0],equal_nan=True,rtol=1e-6,atol=1e-9):
                raise ValueError('Baseline metrics differ across representations')
    out.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(out/'metrics_all_runs_audited.csv',index=False)
    per.to_csv(out/'per_perturbation_pcc_all_runs.csv',index=False)
    metric_columns = [c for c in metrics.select_dtypes(include=np.number) if c != 'seed']
    metrics.groupby(['representation','model'])[metric_columns].agg(['mean','std','count']).to_csv(
        out/'metrics_mean_sd.csv')
    nseeds = len(args.seeds)
    # B: seed-level values, paired within representation across Ridge/MLP.
    panels = [('mse','MSE (lower is better)'),
              ('global_pcc_delta','Global PCC'),
              ('mean_perturbation_pcc','Mean per-perturbation PCC'),
              ('deg_recovery_top100','Top-100 recovery')]
    fig,axes=plt.subplots(2,2,figsize=(7.4,6.0),layout='constrained')
    for panel,(ax,(key,label)) in enumerate(zip(axes.flat,panels)):
        for i,(rep,color) in enumerate(zip(REPS,COLORS)):
            for model,offset,marker in [('ridge_gene',-.16,'s'),('mlp_gene',.16,'o')]:
                values=metrics.loc[metrics.representation.eq(rep)&metrics.model.eq(model)].set_index('seed').loc[args.seeds,key].to_numpy()
                x=i+offset
                ax.errorbar(x,np.mean(values),yerr=np.std(values,ddof=1),fmt=marker,
                            color=color,capsize=3,markersize=6,zorder=4)
                ax.scatter(x+np.linspace(-.055,.055,nseeds),values,s=12,
                           color=color,alpha=.65,zorder=3)
        for model,color,ls in [('train_mean','#555555','--'),('no_change','#999999',':')]:
            values=metrics.loc[metrics.representation.eq(REPS[0])&metrics.model.eq(model),key].to_numpy()
            if np.isfinite(values).all():
                ax.axhline(values.mean(),color=color,ls=ls,lw=1)
        ax.set_xticks(range(3),['GenePT-ada','GenePT-3-large','UCE input\ntokens'])
        ax.set_ylabel(label); ax.set_title(chr(97+panel),loc='left',fontweight='bold')
        if key=='mse':ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        style(ax)
    handles=[Line2D([],[],color='black',marker='s',ls='',label='Ridge'),
             Line2D([],[],color='black',marker='o',ls='',label='MLP'),
             Line2D([],[],color='#555555',ls='--',label='Train-mean baseline'),
             Line2D([],[],color='#999999',ls=':',label='No-change baseline (MSE only)')]
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False,fontsize=8)
    fig.suptitle(f'{nseeds} repeated random holdouts; points = seeds; mean ± SD',fontsize=10)
    save(fig,out,'Figure2B_Performance')

    # C: one preselected split; repeated test appearances are not independent observations.
    sub=per.loc[per.seed.eq(args.distribution_seed)]
    baseline=sub.loc[sub.representation.eq(REPS[0])&sub.model.eq('train_mean')].set_index('perturbation')['pcc']
    groups=[baseline.to_numpy()]
    differences=[]
    for rep in REPS:
        vals=sub.loc[sub.representation.eq(rep)&sub.model.eq('mlp_gene')].set_index('perturbation')['pcc'].reindex(baseline.index)
        groups.append(vals.to_numpy())
        differences.append((vals-baseline).to_numpy())
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.5),layout='constrained')
    for ax,arrays,colors,labels,ylabel in [
        (axes[0],groups,['#999999']+COLORS,['Train mean','GenePT-ada','GenePT-3-large','UCE input\ntokens'],'Per-perturbation PCC'),
        (axes[1],differences,COLORS,['GenePT-ada','GenePT-3-large','UCE input\ntokens'],'MLP PCC − train-mean PCC')]:
        counts=[]
        for i,(arr,color) in enumerate(zip(arrays,colors)):
            arr=arr[np.isfinite(arr)];counts.append(len(arr))
            if not len(arr):continue
            if len(np.unique(arr))>1:
                violin=ax.violinplot([arr],positions=[i],widths=.7,showextrema=False)
                for body in violin['bodies']:body.set_facecolor(color);body.set_alpha(.4)
            ax.boxplot([arr],positions=[i],widths=.18,showfliers=False,
                       medianprops={'color':'black'},manage_ticks=False)
        ax.axhline(0,color='#777777',ls='--',lw=.8)
        ax.set_xticks(range(len(labels)),[f'{s}\n(n={n})' for s,n in zip(labels,counts)],rotation=20,ha='right')
        ax.set_ylabel(ylabel);style(ax)
    axes[0].set_title('a',loc='left',fontweight='bold');axes[1].set_title('b',loc='left',fontweight='bold')
    fig.suptitle(f'Preselected seed {args.distribution_seed}; MLP predictions',fontsize=10)
    save(fig,out,'Figure2C_PerturbationPCC')
    # Descriptive paired wins per seed, without treating overlapping holdouts as independent.
    pivot=per.loc[per.model.eq('mlp_gene')].pivot(index=['seed','perturbation'],columns='representation',values='pcc')
    baseline_all=per.loc[per.representation.eq(REPS[0])&per.model.eq('train_mean')].set_index(['seed','perturbation'])['pcc']
    wins=[]
    for seed in args.seeds:
        for rep in REPS:
            diff=(pivot[rep]-baseline_all).xs(seed).dropna()
            wins.append({'seed':seed,'representation':rep,'n_valid_pairs':len(diff),
                         'fraction_pcc_improved':(diff>0).mean(),
                         'median_pcc_improvement':diff.median()})
    pd.DataFrame(wins).to_csv(out/'paired_improvement_by_seed.csv',index=False)

    fig,axes=plt.subplots(1,3,figsize=(7.4,3.2),layout='constrained')
    for i,(ax,k) in enumerate(zip(axes,[20,50,100])):
        key=f'deg_recovery_top{k}'
        for j,(rep,color) in enumerate(zip(REPS,COLORS)):
            vals=metrics.loc[metrics.representation.eq(rep)&metrics.model.eq('mlp_gene')].set_index('seed').loc[args.seeds,key].to_numpy()
            ax.errorbar(j,vals.mean(),yerr=vals.std(ddof=1),fmt='o',color=color,capsize=3)
            ax.scatter(j+np.linspace(-.06,.06,nseeds),vals,s=12,color=color,alpha=.7)
        baseline_values=metrics.loc[metrics.representation.eq(REPS[0])&metrics.model.eq('train_mean'),key]
        ax.axhline(baseline_values.mean(),ls='--',color='#555555',lw=1)
        ax.set_xticks(range(3),['GenePT-ada','GenePT-3-large','UCE input\ntokens'],rotation=30,ha='right')
        ax.set_title(f'{chr(97+i)}  Top-{k}',loc='left')
        ax.set_ylabel('Recovery fraction');ax.set_ylim(bottom=0);style(ax)
    fig.suptitle(f'MLP; {nseeds} holdouts; mean ± SD; dashed line = train mean',fontsize=10)
    save(fig,out,'Figure2D_TopKRecovery')
    notes=f'''Figure 2B. Fixed-configuration comparison across {nseeds} repeated random 80/20 holdouts
(seeds: {args.seeds}). Small points represent individual seeds; larger markers and error bars
show mean and sample SD, not confidence intervals. Baseline lines show means across seeds.
No-change PCC is undefined and is not plotted. Ridge and MLP are included without selecting
the better model on the test set. These holdouts are not five-fold cross-validation.

Figure 2C. Per-perturbation PCC distributions and paired differences from the train-mean
baseline on preselected seed {args.distribution_seed}. Boxes show median and interquartile range;
whiskers extend to 1.5 IQR. Counts report finite values. Repeated test appearances across seeds
are not pooled as independent observations. No hypothesis-test stars are shown.

Figure 2D. Top-K recovery for MLP using the existing metrics.csv implementation. These are
ranked response-gene recovery metrics; statistical DEG significance is not established by
this plot. Points and SD summarize repeated random holdouts, not independent cohorts.

UCE features are static UCE input gene tokens, not contextual UCE cell embeddings. All runs
use the same 1950 perturbations and 8563 response genes. Input/prediction labels, y_true,
baseline consistency, MSE and mean per-perturbation PCC were audited before plotting.
Feature scaling, tuning and early-stopping leakage require a separate training-code review;
these plots alone do not establish their absence. Hyperparameters should be selected using
training-only validation before a final confirmatory evaluation.
'''
    (out/'Figure_legends_and_notes.txt').write_text(notes)
    print('[COMPLETE]',out)


if __name__=='__main__':
    main()
