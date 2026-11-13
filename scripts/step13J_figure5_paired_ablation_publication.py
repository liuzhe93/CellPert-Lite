#!/usr/bin/env python3
"""Figure 5: publication-style paired ablation across HepG2 and Jurkat.
Uses existing Step11B-5 statistics. No model training or invented values.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

MODELS=['no_gate_direct','genept_only_direct','uce_only_direct']
LABELS={'no_gate_direct':'No-gate fusion','genept_only_direct':'GenePT-only','uce_only_direct':'UCE-only'}
COLORS={'no_gate_direct':'#407B85','genept_only_direct':'#C28D3D','uce_only_direct':'#8C70A9'}
DATASETS=['HepG2','Jurkat']; GROUPS=['seen','heldout']

def args_parse():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.')
    p.add_argument('--source',default='results/step11B_5_external_transfer_analysis')
    p.add_argument('--outdir',default='results/step13J_figure5_paired_ablation_publication')
    p.add_argument('--dpi',type=int,default=300)
    return p.parse_args()

def require(df,cols,name):
    miss=set(cols)-set(df.columns)
    if miss:raise ValueError(f'{name} missing columns: {sorted(miss)}')

def main():
    a=args_parse();root=Path(a.root).resolve(); src=root/a.source;out=root/a.outdir;out.mkdir(parents=True,exist_ok=True)
    pair=pd.read_csv(src/'paired_model_differences.csv')
    target=pd.read_csv(src/'paired_pcc_per_perturbation.csv')
    require(pair,['dataset','group','metric','model_a','model_b','mean_paired_difference','ci95_low','ci95_high','n_unique_perturbations','n_seed_perturbation_pairs'],'paired_model_differences.csv')
    require(target,['dataset','group','seed','perturbation','competitor','difference'],'paired_pcc_per_perturbation.csv')
    pair=pair[(pair.model_a=='gated_direct') & pair.dataset.isin(DATASETS)&pair.group.isin(GROUPS)&pair.model_b.isin(MODELS)&pair.metric.isin(['pcc','mse'])].copy()
    target=target[target.dataset.isin(DATASETS)&target.group.eq('heldout')&target.competitor.isin(MODELS)].copy()
    if pair.duplicated(['dataset','group','metric','model_b']).any():raise ValueError('Duplicate paired comparison rows')
    expected={(ds,g,m,mod) for ds in DATASETS for g in GROUPS for m in ['pcc','mse'] for mod in MODELS}
    got=set(map(tuple,pair[['dataset','group','metric','model_b']].itertuples(index=False,name=None)))
    if expected!=got:raise ValueError(f'Missing comparisons: {sorted(expected-got)}; unexpected: {sorted(got-expected)}')
    if target.duplicated(['dataset','group','seed','perturbation','competitor']).any():raise ValueError('Duplicate target paired records')
    if not np.isfinite(pair[['mean_paired_difference','ci95_low','ci95_high']].to_numpy(dtype=float)).all():raise ValueError('Nonfinite bootstrap values')
    if (pair.ci95_low>pair.ci95_high).any():raise ValueError('Invalid confidence interval')
    if not np.isfinite(target.difference.to_numpy(dtype=float)).all():raise ValueError('Nonfinite target differences')
    # Source uses gated minus competitor for both metrics; negative MSE is better.
    pair['plot_effect']=np.where(pair.metric.eq('mse'),-pair.mean_paired_difference,pair.mean_paired_difference)
    pair['plot_low']=np.where(pair.metric.eq('mse'),-pair.ci95_high,pair.ci95_low)
    pair['plot_high']=np.where(pair.metric.eq('mse'),-pair.ci95_low,pair.ci95_high)
    pair['ci_excludes_zero']=(pair.plot_low>0)|(pair.plot_high<0)
    pair.to_csv(out/'Figure5_effects_and_CI.csv',index=False)
    target.to_csv(out/'Figure5_heldout_target_differences.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.titleweight':'bold','axes.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(15.8,10.2),facecolor='white')
    gs=fig.add_gridspec(2,2,left=.085,right=.965,top=.90,bottom=.105,wspace=.32,hspace=.42)
    axA=fig.add_subplot(gs[0,0]);axB=fig.add_subplot(gs[0,1]);axC=fig.add_subplot(gs[1,0]);axD=fig.add_subplot(gs[1,1])
    # A and B: 12 comparisons each, grouped by cell line and target familiarity.
    rows=[(ds,g) for ds in DATASETS for g in GROUPS]
    centers=[10.5,7.5,4.5,1.5]
    def forest(ax,metric,title):
        sub=pair[pair.metric==metric]
        for k,(ds,g) in enumerate(rows):
            cy=centers[k]
            if k%2==0:ax.axhspan(cy-1.4,cy+1.4,color='#F4F6F8',zorder=0)
            for j,mod in enumerate(MODELS):
                r=sub[(sub.dataset==ds)&(sub.group==g)&(sub.model_b==mod)].iloc[0]
                y=cy+(1-j)*.67
                v,lo,hi=[float(r[x]) for x in ['plot_effect','plot_low','plot_high']]
                ax.plot([lo,hi],[y,y],lw=1.8,color=COLORS[mod],solid_capstyle='round',zorder=2)
                ax.plot([lo,lo],[y-.10,y+.10],lw=1.2,color=COLORS[mod]);ax.plot([hi,hi],[y-.10,y+.10],lw=1.2,color=COLORS[mod])
                ax.scatter(v,y,s=35,color=COLORS[mod],edgecolors='white',linewidths=.65,zorder=3)
        ax.axvline(0,color='#66717B',ls='--',lw=1)
        ax.set_ylim(.0,12.1);ax.set_yticks(centers,[f'{ds}  |  {"Seen" if g=="seen" else "Held-out"}' for ds,g in rows]);ax.invert_yaxis()
        ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True)
        ax.set_title(title,loc='left',pad=12)
        ax.set_xlabel('Paired improvement (positive favors gated fusion)')
        if metric=='mse':
            from matplotlib.ticker import ScalarFormatter
            ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,-3))
    forest(axA,'pcc','A   Paired change in per-target PCC')
    forest(axB,'mse','B   Paired reduction in mean squared error')
    # C: heatmap, exact effect size, not significance masking.
    hm=np.array([[float(pair[(pair.dataset==ds)&(pair.group==g)&(pair.metric=='pcc')&(pair.model_b==mod)].plot_effect.iloc[0]) for mod in MODELS] for ds,g in rows])
    lim=max(float(np.max(np.abs(hm))),1e-6)
    im=axC.imshow(hm,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
    axC.set_xticks(range(3),[LABELS[m] for m in MODELS]);axC.set_yticks(range(4),[f'{ds} · {"Seen" if g=="seen" else "Held-out"}' for ds,g in rows]);axC.tick_params(length=0)
    for i,(ds,g) in enumerate(rows):
        for j,mod in enumerate(MODELS):
            r=pair[(pair.dataset==ds)&(pair.group==g)&(pair.metric=='pcc')&(pair.model_b==mod)].iloc[0]
            v=hm[i,j]; sig='*' if r.ci_excludes_zero else ''
            axC.text(j,i,f'{v:+.4f}{sig}',ha='center',va='center',fontsize=9,color='white' if abs(v)>.58*lim else '#202A35',fontweight='bold')
    axC.set_title('C   Context dependence of paired PCC effects',loc='left',pad=12)
    cb=fig.colorbar(im,ax=axC,fraction=.046,pad=.035);cb.set_label('ΔPCC (gated − comparator)',fontsize=8)
    # D: heldout target-averaged differences, each perturbation once per competitor.
    pos=np.arange(6)
    rng=np.random.default_rng(42)
    summary=[]
    for k,ds in enumerate(DATASETS):
        for j,mod in enumerate(MODELS):
            x=target[(target.dataset==ds)&(target.competitor==mod)]
            values=x.groupby('perturbation',observed=True).difference.mean().dropna().to_numpy()
            if len(values)==0:raise ValueError(f'No heldout paired target differences: {ds}/{mod}')
            p=k*3+j
            # Boxplot avoids treating seed records as independent targets.
            bp=axD.boxplot(values,positions=[p],widths=.48,showfliers=False,patch_artist=True,medianprops={'color':'#253447','linewidth':1.5},boxprops={'linewidth':1.0,'edgecolor':COLORS[mod]},whiskerprops={'color':COLORS[mod]},capprops={'color':COLORS[mod]})
            bp['boxes'][0].set_facecolor(COLORS[mod]);bp['boxes'][0].set_alpha(.20)
            if len(values)<=600:
                xx=p+rng.uniform(-.20,.20,len(values));axD.scatter(xx,values,s=5,color=COLORS[mod],alpha=.13,rasterized=True)
            summary.append({'dataset':ds,'competitor':mod,'n_unique_targets':len(values),'mean_target_difference':float(values.mean()),'median_target_difference':float(np.median(values)),'fraction_positive':float(np.mean(values>0))})
    axD.axhline(0,color='#66717B',ls='--',lw=1)
    axD.set_xticks(pos,[LABELS[m] for _ in DATASETS for m in MODELS],rotation=35,ha='right')
    axD.axvline(2.5,color='#C7CDD3',lw=.8)
    axD.text(1,1.025,'HepG2',transform=axD.get_xaxis_transform(),ha='center',fontsize=9,fontweight='bold')
    axD.text(4,1.025,'Jurkat',transform=axD.get_xaxis_transform(),ha='center',fontsize=9,fontweight='bold')
    axD.set_ylabel('Target-averaged paired ΔPCC')
    axD.set_title('D   Held-out perturbation-level effect distributions',loc='left',pad=20)
    axD.grid(axis='y',alpha=.13);axD.set_axisbelow(True)
    pd.DataFrame(summary).to_csv(out/'Figure5_heldout_target_distribution_summary.csv',index=False)
    legend=[Line2D([0],[0],marker='o',lw=2,color=COLORS[m],label=LABELS[m],markersize=5) for m in MODELS]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.948),ncol=3,frameon=False)
    fig.suptitle('Paired ablation across external cell lines',x=.085,y=.993,ha='left',fontsize=16,fontweight='bold')
    fig.text(.085,.042,'Seen/Held-out denotes target presence in K562 training; all external predictions use frozen K562 weights.\n'
             'Error bars: target-cluster bootstrap 95% CI from Step11B-5. * CI excludes zero (descriptive, unadjusted).\n'
             'Panel D averages repeated splits per unique target before plotting; these are not independent biological replicates.',fontsize=8,color='#59636F',va='bottom')
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'Figure5_paired_ablation_publication.{ext}',dpi=a.dpi,facecolor='white')
    plt.close(fig)
    (out/'Figure5_legend.md').write_text('''**Figure 5 | Paired ablation across external cell lines.** (A) Per-target Pearson correlation differences between gated fusion and each ablation model for seen and held-out perturbations in HepG2 and Jurkat. (B) Matched mean squared error reduction, defined as MSE(comparator) − MSE(gated). Positive values favor gated fusion in both panels. Points show the paired mean effect and error bars show 95% target-cluster bootstrap confidence intervals from Step11B-5. (C) Heatmap of paired PCC effects; asterisks denote bootstrap intervals excluding zero without multiple-testing correction. (D) Distribution of per-target paired PCC differences in held-out perturbations, after averaging each unique target over available random splits. Seen/held-out refers to whether a target was included in K562 model training, not to training on HepG2 or Jurkat. The external predictions use frozen K562-trained model weights. Seed repetitions are not independent biological replicates; confidence intervals are descriptive.\n''',encoding='utf-8')
    print('DONE:',out)
    for f in sorted(out.iterdir()):print(f.name)
if __name__=='__main__':main()
