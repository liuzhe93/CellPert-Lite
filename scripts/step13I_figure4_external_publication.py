#!/usr/bin/env python3
"""Publication Figure 4: frozen K562 -> HepG2/Jurkat transfer.
Uses Step11B-5 CSV outputs; never fabricates observations or retrains models.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODELS=['gated_direct','no_gate_direct','genept_only_direct','uce_only_direct']
LABEL={'gated_direct':'Gated fusion','no_gate_direct':'No-gate fusion','genept_only_direct':'GenePT only','uce_only_direct':'UCE only'}
COLOR={'gated_direct':'#247C91','no_gate_direct':'#507E6B','genept_only_direct':'#D19B39','uce_only_direct':'#8466AD'}
GROUPS=['seen','heldout']; CELLS=['HepG2','Jurkat']

def require_cols(df, cols, filename):
    miss=set(cols)-set(df.columns)
    if miss: raise ValueError(f'{filename}: missing columns {sorted(miss)}; available={list(df.columns)}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--source',default='results/step11B_5_external_transfer_analysis')
    ap.add_argument('--outdir',default='results/step13I_figure4_external_publication')
    ap.add_argument('--dpi',type=int,default=300)
    args=ap.parse_args()
    root=Path(args.root).resolve(); src=root/args.source; out=root/args.outdir;out.mkdir(parents=True,exist_ok=True)
    s=pd.read_csv(src/'metrics_per_seed_recomputed.csv')
    p=pd.read_csv(src/'paired_model_differences.csv')
    g=pd.read_csv(src/'seen_heldout_gap_per_seed.csv')
    require_cols(s,['dataset','group','model','seed','n','mean_pcc'], 'metrics_per_seed_recomputed.csv')
    require_cols(p,['dataset','group','metric','model_a','model_b','mean_paired_difference','ci95_low','ci95_high','n_unique_perturbations'], 'paired_model_differences.csv')
    require_cols(g,['dataset','model','seed','seen_mean_pcc','heldout_mean_pcc','heldout_minus_seen_pcc'], 'seen_heldout_gap_per_seed.csv')
    for ds in CELLS:
        for grp in GROUPS:
            for model in MODELS:
                z=s[(s.dataset==ds)&(s.group==grp)&(s.model==model)]
                if len(z)<2: raise ValueError(f'Missing >=2 seed records for {ds}/{grp}/{model}: {len(z)}')
                if z.seed.duplicated().any(): raise ValueError(f'Duplicate seeds for {ds}/{grp}/{model}')
    s=s[s.dataset.isin(CELLS)&s.model.isin(MODELS)&s.group.isin(GROUPS)].copy()
    p=p[p.dataset.isin(CELLS)&p.group.eq('heldout')&p.metric.eq('pcc')&p.model_a.eq('gated_direct')&p.model_b.isin(MODELS[1:])].copy()
    if len(p)!=6: raise ValueError(f'Expected six heldout paired comparisons, found {len(p)}')
    g=g[g.dataset.isin(CELLS)&g.model.isin(MODELS)].copy()
    s.to_csv(out/'figure4_source_seed_metrics.csv',index=False)
    p.to_csv(out/'figure4_heldout_paired_pcc.csv',index=False)
    g.to_csv(out/'figure4_seen_heldout_gap.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
        'savefig.facecolor':'white','axes.linewidth':0.8})
    fig=plt.figure(figsize=(13.2,9.2),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.05],wspace=.18,hspace=.23)
    axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    fig.suptitle('Cross-cell-line generalization depends on perturbation target familiarity',fontweight='bold',fontsize=15)
    rng=np.random.default_rng(2026)
    for j,ds in enumerate(CELLS):
        ax=axs[j]
        for k,grp in enumerate(GROUPS):
            for m,model in enumerate(MODELS):
                z=s[(s.dataset==ds)&(s.group==grp)&(s.model==model)].sort_values('seed')
                x=k+(m-1.5)*.16
                vals=z.mean_pcc.to_numpy(float)
                jitter=np.linspace(-.035,.035,len(vals))
                ax.scatter(np.full(len(vals),x)+jitter,vals,s=20,color=COLOR[model],alpha=.53,linewidths=0,zorder=3)
                mu=np.mean(vals);sd=np.std(vals,ddof=1)
                ax.errorbar([x],[mu],yerr=[sd],fmt='o',markersize=6.5,elinewidth=1.5,capsize=3,
                    color=COLOR[model],markeredgecolor='white',markeredgewidth=.6,zorder=5)
        ax.set_xticks([0,1],['Seen in K562 training','Held out from K562'])
        ax.set_xlim(-.5,1.5)
        ax.set_title(f'{chr(65+j)}   {ds} | Frozen-weight transfer',loc='left',fontweight='bold')
        ax.set_ylabel('Mean per-target Pearson r' if j==0 else '')
        ax.grid(axis='y',alpha=.17)
        ax.axhline(0,color='#888',lw=.7,ls='--')
        for k,grp in enumerate(GROUPS):
            zz=s[(s.dataset==ds)&(s.group==grp)&(s.model=='gated_direct')]
            counts=zz.n.to_numpy(float)
            label=f'n/seed: {int(counts[0])}' if np.all(counts==counts[0]) else f'n/seed: {int(counts.min())}–{int(counts.max())}'
            ax.text(k,.02,label,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=7.5,color='#666')
    handles=[Line2D([0],[0],marker='o',linestyle='none',color=COLOR[m],label=LABEL[m],markersize=7) for m in MODELS]
    axs[0].legend(handles=handles,frameon=False,ncol=2,loc='upper right',fontsize=8)
    # C: descriptive seen/heldout gap; different target populations, not paired targets.
    ax=axs[2]
    for i,ds in enumerate(CELLS):
        for j,model in enumerate(MODELS):
            z=g[(g.dataset==ds)&(g.model==model)].sort_values('seed')
            vals=z.heldout_minus_seen_pcc.to_numpy(float)
            if not len(vals):continue
            x=i+(j-1.5)*.16
            ax.scatter(np.full(len(vals),x)+np.linspace(-.035,.035,len(vals)),vals,color=COLOR[model],s=17,alpha=.48)
            ax.errorbar(x,vals.mean(),yerr=vals.std(ddof=1) if len(vals)>1 else 0,fmt='o',color=COLOR[model],capsize=3,markersize=6)
    ax.set_xticks([0,1],CELLS);ax.set_xlim(-.5,1.5)
    ax.set_ylabel('Held-out minus seen | mean target PCC')
    ax.set_title('C   Familiarity-associated performance gap',loc='left',fontweight='bold')
    ax.axhline(0,color='#777',ls='--',lw=1);ax.grid(axis='y',alpha=.17)
    ax.text(.01,.03,'Descriptive: seen and held-out are distinct target sets',transform=ax.transAxes,fontsize=7.6,color='#666')
    # D: paired comparisons on identical seed and heldout target, with existing target-cluster bootstrap CI.
    ax=axs[3]
    yy=[];labels=[]
    for i,ds in enumerate(CELLS):
        for j,model in enumerate(MODELS[1:]):
            z=p[(p.dataset==ds)&(p.model_b==model)]
            if len(z)!=1:raise ValueError(f'Missing unique paired result {ds}/{model}')
            r=z.iloc[0];y=i*4+j
            est=float(r.mean_paired_difference);lo=float(r.ci95_low);hi=float(r.ci95_high)
            ax.plot([lo,hi],[y,y],lw=2,color=COLOR[model],solid_capstyle='round')
            ax.plot(est,y,'o',ms=6,color=COLOR[model],markeredgecolor='white',markeredgewidth=.5)
            yy.append(y);labels.append(f'{ds}  ·  {LABEL[model]}')
    ax.set_yticks(yy,labels,fontsize=8);ax.invert_yaxis()
    ax.axvline(0,color='#777',ls='--',lw=1)
    ax.set_xlabel('Δ mean target PCC (gated − comparator)')
    ax.set_title('D   Held-out targets | Paired model differences',loc='left',fontweight='bold')
    ax.grid(axis='x',alpha=.17)
    ax.text(.99,.03,'95% target-cluster bootstrap CI',transform=ax.transAxes,ha='right',fontsize=8,color='#666')
    fig.text(.015,.006,'Dots: individual random splits; error bars in A–C: mean ± SD across seeds. D: matched seed–target comparisons.\n'
        'Frozen K562 weights; repeated splits are not biological replicates. Seen/held-out groups contain different perturbation targets.',fontsize=8,color='#555')
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'Figure4_external_publication.{ext}',dpi=args.dpi,bbox_inches='tight',pad_inches=.18)
    plt.close(fig)
    with open(out/'Figure4_legend.md','w') as f:
        f.write('**Figure 4 | Cross-cell-line generalization depends on perturbation target familiarity.** '
          'K562-trained models were evaluated with frozen weights in HepG2 and Jurkat. '
          '(A,B) Mean per-target Pearson correlations for perturbation targets seen in K562 training or held out from training. '
          'Small points show individual random seeds; larger points and whiskers show mean ± SD across seeds. '
          '(C) Descriptive held-out minus seen difference per seed; these groups contain distinct targets and are not paired observations. '
          '(D) Paired mean per-target Pearson correlation differences (gated fusion minus comparator) for the same seed and held-out target, '
          'with 95% confidence intervals obtained by target-cluster bootstrap from Step11B-5. '
          'Random splits are not independent biological replicates.\n')
    print('Created:',out/'Figure4_external_publication.pdf')
    print('Created:',out/'Figure4_external_publication.svg')

if __name__=='__main__': main()
