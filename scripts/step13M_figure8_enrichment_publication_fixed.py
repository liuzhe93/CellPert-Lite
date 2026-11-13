#!/usr/bin/env python3
"""Figure 8: exploratory enrichment, transparent set sizes and selection.
Usage: python scripts/step13M_figure8_enrichment_publication.py --source results/<enrichment-dir> --outdir results/step13M_figure8_publication
Or pass --all-csv and --sig-csv explicitly.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

REQUIRED=['dataset','group','category','database','term','universe_size','query_size','term_size_in_universe','overlap','fold_enrichment','overlap_genes','fdr_bh']
COLORS={'HepG2':'#26718C','Jurkat':'#C68A31','cross_cell_line':'#79669B'}

def read(path):
    d=pd.read_csv(path)
    missing=[c for c in REQUIRED if c not in d.columns]
    if missing: raise ValueError(f'{path}: missing {missing}; available {list(d.columns)}')
    for c in ['universe_size','query_size','term_size_in_universe','overlap','fold_enrichment','fdr_bh']:
        d[c]=pd.to_numeric(d[c],errors='raise')
    if ((d.fdr_bh<0)|(d.fdr_bh>1)).any():raise ValueError('FDR must be in [0,1]')
    if ((d['overlap']>d['query_size'])|(d['overlap']>d['term_size_in_universe'])).any():raise ValueError('Invalid overlap counts')
    return d

def short_term(s,maxlen=43):
    import re
    s=re.sub(r'\s*\((GO:\d+|R-HSA-\d+)\)\s*$','',str(s))
    return s if len(s)<=maxlen else s[:maxlen-1]+'…'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='results/step10G_ccrt_rescue_biological_evaluation')
    p.add_argument('--all-csv',default=None)
    p.add_argument('--sig-csv',default=None)
    p.add_argument('--outdir',default='results/step13M_figure8_publication')
    p.add_argument('--top',type=int,default=5)
    p.add_argument('--dpi',type=int,default=300)
    a=p.parse_args()
    source=Path(a.source)
    if (a.all_csv is None) != (a.sig_csv is None):
        p.error('Please supply both --all-csv and --sig-csv, or neither.')
    if a.all_csv:
        allp, sigp = Path(a.all_csv), Path(a.sig_csv)
    else:
        # Prefer the declared source, then discover an exact pair under results/.
        allp, sigp = source/'enrichment_all_terms.csv', source/'enrichment_significant.csv'
        if not (allp.is_file() and sigp.is_file()):
            candidates=[]
            for base in (Path('results'),Path('data/processed')):
                if base.is_dir():
                    for f in base.rglob('enrichment_all_terms.csv'):
                        other=f.with_name('enrichment_significant.csv')
                        if other.is_file(): candidates.append((f,other))
            # deduplicate any paths found through multiple roots
            candidates=list({str(x[0].resolve()):x for x in candidates}.values())
            if len(candidates)==1:
                allp,sigp=candidates[0]
                print(f'[INFO] Auto-discovered enrichment inputs in {allp.parent}',flush=True)
            elif len(candidates)>1:
                choices='\n'.join(f'  {x.parent}' for x,_ in candidates)
                p.error('Multiple enrichment datasets found. Specify --source explicitly:\n'+choices)
            else:
                p.error('No enrichment_all_terms.csv + enrichment_significant.csv pair found. '
                        'Run: find results/ -type f -name "enrichment_all_terms.csv"; '
                        'or supply both exact --all-csv and --sig-csv paths.')
    for f in (allp,sigp):
        if not f.is_file(): p.error(f'Input file not found: {f}')
    print(f'[INFO] All terms: {allp}\n[INFO] Significant terms: {sigp}',flush=True)
    full=read(allp); sig=read(sigp)
    if a.top<1:raise ValueError('--top must be >=1')
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    sig=sig.loc[sig.fdr_bh<0.05].copy()
    if sig.empty:raise ValueError('No FDR < 0.05 terms. Do not plot significance that is absent.')
    keys=['dataset','group','category','database']
    sig=sig.sort_values(['fdr_bh','fold_enrichment','overlap'],ascending=[True,False,False])
    chosen=sig.groupby(keys,sort=False,dropna=False).head(a.top).copy()
    chosen['panel_label']=chosen.apply(lambda r:f"{r.dataset} / {r.group}",axis=1)
    chosen['term_label']=chosen.term.map(short_term)
    chosen['neglog10fdr']=-np.log10(chosen.fdr_bh.clip(lower=1e-300))
    chosen.to_csv(out/'Figure8_plotted_terms.csv',index=False)
    # audit counts: one tested term row per row of all_terms
    counts=full.groupby(['dataset','group','category'],dropna=False).agg(n_tested=('term','size'),query_min=('query_size','min'),query_max=('query_size','max'),universe_min=('universe_size','min'),universe_max=('universe_size','max')).reset_index()
    nc=sig.groupby(['dataset','group','category'],dropna=False).size().rename('n_fdr_lt_0_05').reset_index()
    counts=counts.merge(nc,how='left',on=['dataset','group','category']);counts['n_fdr_lt_0_05']=counts.n_fdr_lt_0_05.fillna(0).astype(int)
    counts.to_csv(out/'Figure8_enrichment_audit.csv',index=False)
    # select max 4 strata (actual observed)
    strata=list(dict.fromkeys(chosen.panel_label))
    if len(strata)>6:raise ValueError('More than six dataset/group strata: use a supplementary plot or filter source before drawing.')
    fig=plt.figure(figsize=(15.8,9.3),facecolor='white')
    gs=GridSpec(2,2,figure=fig,width_ratios=[1.28,1],height_ratios=[1.55,1],left=.07,right=.95,top=.88,bottom=.14,hspace=.38,wspace=.30)
    ax=fig.add_subplot(gs[0,0]);ax2=fig.add_subplot(gs[0,1]);ax3=fig.add_subplot(gs[1,:])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    # A: top term dotplot; colored by adjusted p, sized by overlap
    plotted=chosen.sort_values(['panel_label','neglog10fdr'],ascending=[True,False]).reset_index(drop=True)
    labels=[f"{short_term(r.term,39)}" for r in plotted.itertuples()]
    ys=np.arange(len(plotted))[::-1]
    vmax=max(2.,float(np.quantile(plotted.neglog10fdr,.98)))
    norm=Normalize(vmin=0,vmax=vmax)
    sizes=30+plotted.overlap.to_numpy()*25
    sc=ax.scatter(plotted.fold_enrichment,ys,s=sizes,c=[COLORS.get(r.dataset,'#666') for r in plotted.itertuples()],edgecolors='white',linewidth=.7,zorder=3)
    ax.set_yticks(ys,labels,fontsize=8)
    ax.set_xlabel('Fold enrichment (observed / expected)')
    ax.set_title('A   Enriched functional terms',loc='left',weight='bold',fontsize=12,pad=13)
    ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
    ax.spines[['top','right','left']].set_visible(False)
    ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=COLORS[k],markeredgecolor='white',markersize=7,label=k.replace('cross_cell_line','Cross-cell')) for k in COLORS],loc='lower right',frameon=False,fontsize=8)
    ax.set_xlim(left=0)
    # B: term-gene overlap matrix, top 12 unique genes by incidence; explicit membership
    gene_sets=[]
    for row in plotted.itertuples():
        gene_sets.append(set(str(row.overlap_genes).replace(',',';').split(';'))-{'','nan'})
    freq={g:sum(g in ss for ss in gene_sets) for g in set().union(*gene_sets)}
    genes=sorted(freq,key=lambda g:(-freq[g],g))[:14]
    if genes:
        mat=np.array([[int(g in ss) for g in genes] for ss in gene_sets],dtype=float)
        ax2.imshow(mat,aspect='auto',cmap=matplotlib.colors.ListedColormap(['#F0F2F4','#26718C']),interpolation='nearest',vmin=0,vmax=1)
        ax2.set_xticks(range(len(genes)),genes,rotation=60,ha='right',fontsize=7)
        ax2.set_yticks(range(len(plotted)),[f'{i+1:02d} · {r.dataset[:3]}-{r.group[:1]}' for i,r in enumerate(plotted.itertuples())],fontsize=8)
        ax2.set_ylabel('Selected pathways')
        ax2.tick_params(length=0)
        ax2.spines[:].set_visible(False)
    else:
        ax2.text(.5,.5,'No overlap gene IDs provided',ha='center',va='center',transform=ax2.transAxes)
    ax2.set_title('B   Genes contributing to selected terms',loc='left',weight='bold',fontsize=12,pad=13)
    # C: context of selection: all tested vs significant counts
    count2=counts.loc[counts.n_fdr_lt_0_05>0].copy();count2['label']=count2.dataset+' / '+count2.group+' / '+count2.category
    count2=count2.sort_values(['dataset','group','category']).reset_index(drop=True)
    yy=np.arange(len(count2))
    ax3.barh(yy,count2.n_tested,color='#E5E9EC',height=.58,label='Terms tested')
    ax3.barh(yy,count2.n_fdr_lt_0_05,color='#26718C',height=.58,label='BH FDR < 0.05')
    ax3.set_yticks(yy,count2.label,fontsize=8)
    ax3.invert_yaxis();ax3.set_xlabel('Number of enriched terms')
    ax3.set_title('C   Enrichment search space and FDR-qualified terms',loc='left',weight='bold',fontsize=12,pad=10)
    ax3.legend(loc='upper right',frameon=False,ncol=2,fontsize=8)
    ax3.grid(axis='x',alpha=.13);ax3.set_axisbelow(True);ax3.spines[['top','right','left']].set_visible(False)
    fig.suptitle('Figure 8  |  Exploratory functional enrichment across transfer contexts',x=.07,y=.975,ha='left',fontsize=16,weight='bold')
    fig.text(.07,.925,'Top terms selected within dataset × familiarity × analysis category; gene memberships reflect overlap, not causal mechanisms.',fontsize=9,color='#555')
    fig.text(.07,.07,'BH FDR is taken from the input table; query and universe sizes are group-specific. Terms may overlap substantially in gene membership.',fontsize=8,color='#444')
    fig.text(.07,.046,'Descriptive enrichment only. Cross-group bar heights and FDR values are not direct tests of between-group pathway differences.',fontsize=8,color='#444')
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'Figure8_enrichment_publication.{ext}',dpi=a.dpi,facecolor='white',bbox_inches='tight')
    plt.close(fig)
    (out/'Figure8_legend.md').write_text('Figure 8 | Exploratory functional enrichment across transfer contexts. (A) Top FDR-qualified terms within each dataset, familiarity group, and analysis category, showing fold enrichment and gene overlap; only terms with BH-adjusted FDR < 0.05 are included. (B) Membership of overlapping query genes in selected terms; terms and genes are not independent. (C) Number of tested and FDR-qualified terms per stratum. Enrichment uses group-specific query and background universes. All panels are descriptive and cannot establish mechanistic causality or between-group differences.\n',encoding='utf-8')
    print('Saved to',out,'| plotted terms:',len(chosen),'| tested:',len(full),'| FDR-qualified:',len(sig))

if __name__=='__main__':main()
