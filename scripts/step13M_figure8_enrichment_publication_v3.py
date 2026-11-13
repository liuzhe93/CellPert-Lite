#!/usr/bin/env python3
"""Figure 8 v3: publication-oriented enrichment figure from audited selected terms.
No re-selection of terms. Usage:
 python scripts/step13M_figure8_enrichment_publication_v3.py \
   --terms results/step13M_figure8_publication/Figure8_plotted_terms.csv \
   --audit results/step13M_figure8_publication/Figure8_enrichment_audit.csv \
   --outdir results/step13M_figure8_publication_v3
"""
import argparse, re, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

PALETTE={'HepG2':'#1D708A','Jurkat':'#B78335','cross_cell_line':'#80659B'}

def clean_term(term):
    term=re.sub(r'\s*\(GO:\d+\)\s*$','',str(term))
    term=re.sub(r'\s+R-HSA-\d+\s*$','',term)
    term=term.replace('Nuclear-Transcribed','Nuclear-transcribed').replace('Of ','of ')
    return term

def gene_set(s):
    if pd.isna(s):return set()
    return {x.strip() for x in re.split(r'[;,]',str(s)) if x.strip()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--terms',default='results/step13M_figure8_publication/Figure8_plotted_terms.csv')
    p.add_argument('--audit',default='results/step13M_figure8_publication/Figure8_enrichment_audit.csv')
    p.add_argument('--outdir',default='results/step13M_figure8_publication_v3')
    p.add_argument('--dpi',type=int,default=350)
    p.add_argument('--max-genes',type=int,default=18)
    a=p.parse_args()
    df=pd.read_csv(a.terms)
    required={'dataset','group','category','term','overlap','fold_enrichment','fdr_bh','overlap_genes'}
    if required-set(df.columns):raise ValueError(f'Missing columns: {required-set(df.columns)}')
    if df.empty:raise ValueError('Empty selected term file')
    if (df.fdr_bh<=0).any() or (df.fdr_bh>=.05).any():raise ValueError('Expected strictly 0 < FDR < 0.05 for selected terms')
    audit=pd.read_csv(a.audit) if Path(a.audit).is_file() else None
    df=df.copy()
    df['label']=df.term.map(clean_term)
    df['set']=df.overlap_genes.map(gene_set)
    df['context']=df.apply(lambda r: f"{('Cross-cell' if r.dataset=='cross_cell_line' else r.dataset)} · {r.group.title()}",axis=1)
    # Preserve all selected terms; order groups and FDR, without changing selection.
    order={'HepG2':0,'Jurkat':1,'cross_cell_line':2}
    df['ord']=df.dataset.map(order).fillna(9)
    df=df.sort_values(['ord','group','fdr_bh','label'],kind='stable').reset_index(drop=True)
    # Gene columns ranked by incidence across selected terms. Omitted genes remain in CSV audit.
    universe=sorted(set().union(*df['set'].tolist()))
    ranked=sorted(universe,key=lambda g:(-sum(g in x for x in df['set']),g))
    genes=ranked[:a.max_genes]
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    df.drop(columns=['set','ord']).to_csv(out/'Figure8_v3_displayed_terms.csv',index=False)
    pd.DataFrame([{'gene':g,'term_memberships':sum(g in x for x in df['set']),'displayed':g in genes} for g in ranked]).to_csv(out/'Figure8_v3_gene_coverage.csv',index=False)
    n=len(df); y=np.arange(n)[::-1]
    # Wider figure and dedicated text column; full names are wrapped rather than truncated.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
    fig=plt.figure(figsize=(17.0,max(8.3,0.43*n+3.2)),facecolor='white')
    gs=fig.add_gridspec(1,3,width_ratios=[3.9,2.1,2.65],left=.045,right=.97,top=.82,bottom=.19,wspace=.045)
    axname=fig.add_subplot(gs[0,0]);axdot=fig.add_subplot(gs[0,1],sharey=axname);axmat=fig.add_subplot(gs[0,2],sharey=axname)
    axname.set_xlim(0,1);axname.set_ylim(-.8,n-.2);axname.axis('off')
    groups=df['context'].tolist()
    for i,row in df.iterrows():
        yy=y[i]
        if i%2==0:
            for ax in (axname,axdot,axmat):ax.axhspan(yy-.48,yy+.48,color='#F4F6F7',zorder=0)
        name=row['label']
        # Two-line wrapping for long labels, never ellipsize.
        wrapped='\n'.join(textwrap.wrap(name,width=47,break_long_words=False,break_on_hyphens=False))
        axname.plot([.006,.006],[yy-.30,yy+.30],color=PALETTE.get(row.dataset,'#555'),lw=3,solid_capstyle='round')
        axname.text(.024,yy,wrapped,va='center',ha='left',fontsize=8.2,color='#23323C',linespacing=1.1)
        # Group identities are preserved in Figure8_v3_displayed_terms.csv; colored stripe denotes dataset.
    maxx=float(df.fold_enrichment.max())*1.14
    axdot.set_xlim(0,maxx);axdot.set_ylim(-.8,n-.2)
    sizes=34+df.overlap.to_numpy()*27
    neglog=-np.log10(df.fdr_bh.to_numpy())
    norm=Normalize(vmin=0,vmax=max(2.5,float(np.ceil(neglog.max()))))
    cmap=plt.get_cmap('viridis')
    axdot.scatter(df.fold_enrichment,y,s=sizes,c=neglog,cmap=cmap,norm=norm,edgecolors='white',linewidths=.65,zorder=3)
    axdot.set_xlabel('Fold enrichment',labelpad=9,fontsize=9)
    axdot.grid(axis='x',alpha=.13,zorder=0);axdot.set_axisbelow(True)
    axdot.tick_params(axis='y',left=False,labelleft=False)
    axdot.spines[['left','top','right']].set_visible(False)
    axdot.spines['bottom'].set_color('#B8C3CA')
    axmat.set_xlim(-.5,len(genes)-.5);axmat.set_ylim(-.8,n-.2)
    for i,s in enumerate(df['set']):
        for j,g in enumerate(genes):
            if g in s:axmat.scatter(j,y[i],s=39,marker='s',color='#26788F',linewidths=0,zorder=3)
    axmat.set_xticks(range(len(genes)),genes,rotation=65,ha='right',fontsize=7.7)
    axmat.set_yticks([]);axmat.tick_params(length=0)
    axmat.spines[:].set_visible(False)
    axmat.set_xlabel('Overlapping query genes',labelpad=10,fontsize=9)
    for x in range(len(genes)):
        axmat.axvline(x,color='#E8EDEF',lw=.6,zorder=0)
    fig.text(.045,.95,'Figure 8  |  Functional enrichment across transfer contexts',fontsize=17,weight='bold',color='#1B2B36',ha='left')
    fig.text(.045,.914,'Selected FDR-qualified terms · colored stripes denote HepG2 (blue), Jurkat (gold), cross-cell (purple); familiarity groups are in the accompanying table',fontsize=10,color='#526471',ha='left')
    for x,t in [(.045,'PATHWAY / FUNCTION'),(.495,'A   ENRICHMENT'),(.695,'B   GENE MEMBERSHIP')]:
        fig.text(x,.858,t,fontsize=9.2,weight='bold',color='#2D4554')
    # Legends below main panels with no overlap with x tick labels
    cbax=fig.add_axes([.43,.075,.18,.014])
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cbax,orientation='horizontal')
    cb.set_label('−log10(BH-adjusted FDR)',fontsize=8,labelpad=4)
    cb.ax.tick_params(labelsize=7)
    lax=fig.add_axes([.66,.065,.29,.07]);lax.axis('off')
    lax.text(.0,.78,'Gene overlap',fontsize=8,color='#425463')
    for k,(num,sz) in enumerate([(2,34+2*27),(4,34+4*27),(6,34+6*27)]):
        x=.12+.28*k
        lax.scatter([x],[.35],s=sz,c='#DCE8ED',edgecolors='#26788F',linewidths=.7,transform=lax.transAxes)
        lax.text(x+.065,.32,str(num),transform=lax.transAxes,fontsize=8,va='center')
    note=f'{len(df)} selected terms; {len(universe)} unique overlapping genes, {len(genes)} displayed.'
    if audit is not None:
        note+=f' Full audit: {int(audit.n_tested.sum())} tested records; {int(audit.n_fdr_lt_0_05.sum())} FDR-qualified records.'
    fig.text(.045,.125,note,fontsize=8,color='#526471')
    fig.text(.045,.107,'Terms were selected upstream; this figure does not re-rank or add enrichment results. Gene membership denotes overlap, not causal evidence.',fontsize=8,color='#526471')
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'Figure8_enrichment_publication_v3.{ext}',dpi=a.dpi,facecolor='white')
    plt.close(fig)
    (out/'Figure8_v3_legend.md').write_text('Figure 8 | Functional enrichment across transfer contexts. (A) Fold enrichment of selected BH-FDR-qualified functional terms; dot size represents the number of overlapping query genes, and color represents −log10(BH-adjusted FDR). (B) Membership of overlapping query genes across selected terms; the most recurrent genes are displayed and the full gene list is retained in the exported coverage table. Terms were selected in the upstream analysis; this visualization does not independently test between-context enrichment differences. All associations are exploratory.\n')
    print(f'[OK] {out} | terms={len(df)} genes shown={len(genes)}/{len(universe)}')
if __name__=='__main__':main()
