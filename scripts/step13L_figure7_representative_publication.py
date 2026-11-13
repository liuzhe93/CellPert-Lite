#!/usr/bin/env python3
"""Figure 7: representative external perturbation responses, source-based.
No model fitting. Requires Step11B_4 predictions.npz, seeds 42-46.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

COLORS={'HepG2':'#276C91','Jurkat':'#C58B30'}
MODELS=['gated_direct']

def pcc(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12: return np.nan
    return float(np.corrcoef(a,b)[0,1])

def read_prediction(root,cell,seed):
    f=root/'results/step11B_4_frozen_external_transfer'/cell/f'seed_{seed}'/'gated_direct'/'predictions.npz'
    if not f.exists(): return None
    with np.load(f,allow_pickle=False) as z:
        needed={'perturbations','genes','y_true','prediction','k562_train_seen'}
        if not needed.issubset(z.files): raise ValueError(f'{f}: missing {needed-set(z.files)}')
        ids=z['perturbations'].astype(str); genes=z['genes'].astype(str)
        yt=z['y_true']; yp=z['prediction']; seen=z['k562_train_seen']
        if yt.shape!=yp.shape or yt.shape!=(len(ids),len(genes)) or len(seen)!=len(ids):
            raise ValueError(f'{f}: inconsistent dimensions')
        if len(set(ids))!=len(ids) or len(set(genes))!=len(genes): raise ValueError(f'{f}: duplicate ids/genes')
        return {'ids':ids,'genes':genes,'true':yt,'pred':yp,'seen':seen,'file':str(f)}

def load_all(root,seeds):
    data={}; records=[]; audit=[]
    for cell in COLORS:
        for seed in seeds:
            z=read_prediction(root,cell,seed)
            if z is None: continue
            data[(cell,seed)]=z
            audit.append({'dataset':cell,'seed':seed,'file':z['file'],'n_targets':len(z['ids']),'n_genes':len(z['genes'])})
            for i,t in enumerate(z['ids']):
                if z['seen'][i]: continue
                y=z['true'][i].astype(float); pred=z['pred'][i].astype(float)
                if not (np.isfinite(y).all() and np.isfinite(pred).all()): continue
                yr=np.sqrt(np.mean(y*y)); pr=np.sqrt(np.mean(pred*pred))
                records.append({'dataset':cell,'seed':seed,'perturbation':t,'pcc':pcc(y,pred),
                                'observed_rms':yr,'predicted_rms':pr,'rms_ratio':pr/yr if yr>1e-12 else np.nan,
                                'n_response_genes':len(y)})
    if not data: raise RuntimeError('No prediction files found; check --root and seeds')
    df=pd.DataFrame(records)
    if df.empty: raise RuntimeError('No held-out targets found')
    return data,df,pd.DataFrame(audit)

def choose_targets(df,n=4,min_seeds=2):
    g=df.groupby(['dataset','perturbation'],as_index=False).agg(mean_pcc=('pcc','mean'),mean_rms=('observed_rms','mean'),n_seeds=('seed','nunique'))
    wide=g.pivot(index='perturbation',columns='dataset',values=['mean_pcc','mean_rms','n_seeds'])
    candidates=[]
    for t in wide.index:
        try:
            a=float(wide.loc[t,('mean_pcc','HepG2')]); b=float(wide.loc[t,('mean_pcc','Jurkat')])
            na=float(wide.loc[t,('n_seeds','HepG2')]); nb=float(wide.loc[t,('n_seeds','Jurkat')])
            ra=float(wide.loc[t,('mean_rms','HepG2')]); rb=float(wide.loc[t,('mean_rms','Jurkat')])
        except (KeyError,TypeError,ValueError): continue
        if not all(np.isfinite(v) for v in (a,b,na,nb,ra,rb)): continue
        if min(na,nb)<min_seeds or min(ra,rb)<0.025: continue
        candidates.append({'perturbation':t,'HepG2_pcc':a,'Jurkat_pcc':b,'HepG2_n_seeds':int(na),
                           'Jurkat_n_seeds':int(nb),'pcc_gap':a-b,'mean_pcc':(a+b)/2,'min_rms':min(ra,rb)})
    c=pd.DataFrame(candidates)
    if c.empty: raise RuntimeError('No common held-out targets meet min-seed and RMS filters. Try --min-seeds 1.')
    # Predeclared criteria, not post-hoc selection of only successes.
    selectors=[('HepG2-favored',c.sort_values('pcc_gap',ascending=False)),
               ('Jurkat-favored',c.sort_values('pcc_gap')),
               ('Both stronger',c.sort_values('mean_pcc',ascending=False)),
               ('Both weaker',c.sort_values('mean_pcc'))]
    selected=[]; used=set()
    for label,frame in selectors[:n]:
        for _,r in frame.iterrows():
            if r['perturbation'] not in used:
                d=r.to_dict();d['selection_category']=label;selected.append(d);used.add(r['perturbation']);break
    if len(selected)<n:
        for _,r in c.sort_values('perturbation').iterrows():
            if r['perturbation'] not in used:
                d=r.to_dict();d['selection_category']='Additional';selected.append(d);used.add(r['perturbation'])
                if len(selected)>=n:break
    return pd.DataFrame(selected),c

def averaged_vectors(data,cell,target):
    entries=[]
    for (c,seed),z in data.items():
        if c!=cell:continue
        inds=np.where(z['ids']==target)[0]
        if len(inds)!=1:continue
        i=int(inds[0]);
        if z['seen'][i]:continue
        entries.append((seed,z['genes'],z['true'][i].astype(float),z['pred'][i].astype(float)))
    if not entries:return None
    # Average across available seeds after explicit gene alignment; validate observed truth consistency.
    common=set(entries[0][1]);
    for _,genes,_,_ in entries[1:]:common &= set(genes)
    genes=np.array(sorted(common))
    if len(genes)<100:raise ValueError(f'Too few shared response genes for {cell}/{target}')
    ys=[];ps=[]
    for _,g,y,p in entries:
        lookup={x:i for i,x in enumerate(g)}; ix=np.array([lookup[x] for x in genes]);ys.append(y[ix]);ps.append(p[ix])
    Y=np.vstack(ys);P=np.vstack(ps)
    truth_disagreement=float(np.max(np.abs(Y-Y[0])))
    # Truth can differ across seeds if evaluation response preparation differs. Do not silently merge.
    if truth_disagreement>1e-4:
        raise ValueError(f'{cell}/{target}: y_true differs across seeds (max absolute diff {truth_disagreement:.5g}); inspect data provenance')
    return genes,np.mean(Y,axis=0),np.mean(P,axis=0),len(entries)

def plot(data,df,chosen,out,dpi):
    targets=chosen['perturbation'].tolist(); colors=COLORS
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig=plt.figure(figsize=(15.0,10.8),facecolor='white')
    gs=GridSpec(2,2,figure=fig,width_ratios=[1.18,1],height_ratios=[1.04,1],left=.075,right=.965,top=.925,bottom=.12,wspace=.25,hspace=.37)
    axA=fig.add_subplot(gs[0,0]);axB=fig.add_subplot(gs[0,1]);axC=fig.add_subplot(gs[1,0]);axD=fig.add_subplot(gs[1,1])
    # Panel A: rows are cell-target pairs; observed/predicted stacked rows, columns top response genes per pair.
    matrix=[];labels=[]; genes_used=[]; details=[]
    for t in targets:
        for cell in COLORS:
            res=averaged_vectors(data,cell,t)
            if res is None:raise RuntimeError(f'Missing {cell}/{t}')
            genes,y,p,nseed=res
            ix=np.argsort(-np.abs(y))[:24]
            matrix.extend([y[ix],p[ix]]);labels.extend([f'{t} · {cell} | Obs',f'{t} · {cell} | Pred'])
            genes_used.append({'perturbation':t,'dataset':cell,'genes_top24':';'.join(genes[ix]),'n_seeds':nseed})
            details.append((t,cell,genes,y,p,nseed))
    mat=np.vstack(matrix); cap=max(.04,float(np.quantile(np.abs(mat),.98)));cap=min(cap,.6)
    im=axA.imshow(np.clip(mat,-cap,cap),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-cap,vcenter=0,vmax=cap),aspect='auto',interpolation='nearest')
    axA.set_yticks(np.arange(len(labels)));axA.set_yticklabels(labels,fontsize=6.8)
    axA.set_xticks([0,5,11,17,23]);axA.set_xticklabels(['1','6','12','18','24'])
    axA.set_xlabel('Top 24 observed-response genes per target × cell line (ranked by |observed Δ|)')
    for yline in np.arange(4,len(labels),4):axA.axhline(yline-.5,color='white',lw=2.0)
    cb=fig.colorbar(im,ax=axA,fraction=.025,pad=.012);cb.set_label('Expression response Δ (shared scale)',fontsize=7)
    axA.set_title('A  Observed versus predicted response patterns',loc='left',fontweight='bold',fontsize=10,pad=10)
    # Panel B: observed/predicted response of one preselected high-contrast example, show all genes as faint points
    example=chosen.iloc[0]['perturbation']; maxabs=0
    for cell in COLORS:
        _,y,p,_=averaged_vectors(data,cell,example)
        rng=np.random.default_rng(2026)
        ix=rng.choice(len(y),size=min(1800,len(y)),replace=False)
        axB.scatter(y[ix],p[ix],s=6,alpha=.18,color=colors[cell],rasterized=True,label=cell)
        maxabs=max(maxabs,float(np.quantile(np.abs(np.r_[y,p]),.995)))
    lim=max(.06,min(.8,maxabs*1.15));axB.plot([-lim,lim],[-lim,lim],ls='--',lw=1,color='#777777',label='Identity')
    axB.axhline(0,color='#AAAAAA',lw=.5);axB.axvline(0,color='#AAAAAA',lw=.5)
    axB.set_xlim(-lim,lim);axB.set_ylim(-lim,lim);axB.set_aspect('equal',adjustable='box')
    axB.set_xlabel('Observed gene expression response Δ');axB.set_ylabel('Predicted gene expression response Δ')
    axB.legend(frameon=False,fontsize=7,loc='upper left')
    axB.set_title(f'B  Gene-level recovery: {example}',loc='left',fontweight='bold',fontsize=10,pad=10)
    # Panel C: matched target-level pcc, average repeated seeds and show seed range (descriptive)
    for i,t in enumerate(targets):
        pair=[]
        for cell in COLORS:
            sub=df[(df.dataset==cell)&(df.perturbation==t)]
            mean=float(sub.pcc.mean());sd=float(sub.pcc.std(ddof=1)) if len(sub)>1 else 0
            pair.append(mean)
            axC.errorbar(i+(-.12 if cell=='HepG2' else .12),mean,yerr=sd,fmt='o',color=colors[cell],capsize=2.5,ms=5,lw=1.2,label=cell if i==0 else None)
        axC.plot([i-.12,i+.12],pair,color='#999999',lw=1,zorder=0)
    axC.axhline(0,color='#888888',ls='--',lw=.8)
    axC.set_xticks(range(len(targets)));axC.set_xticklabels(targets,rotation=25,ha='right')
    axC.set_ylabel('Per-target Pearson r (mean ± SD across seeds)')
    axC.legend(frameon=False,ncol=2,loc='upper right',fontsize=7)
    axC.set_title('C  Cell-context-dependent prediction accuracy',loc='left',fontweight='bold',fontsize=10,pad=10)
    # Panel D: amplitude ratio for same examples; reference line 1
    for i,t in enumerate(targets):
        for cell in COLORS:
            sub=df[(df.dataset==cell)&(df.perturbation==t)]
            vals=sub.rms_ratio.replace([np.inf,-np.inf],np.nan).dropna()
            if vals.empty:continue
            offset=-.12 if cell=='HepG2' else .12
            axD.errorbar(i+offset,vals.mean(),yerr=vals.std(ddof=1) if len(vals)>1 else 0,
                         fmt='o',color=colors[cell],capsize=2.5,ms=5,lw=1.2)
    axD.axhline(1,color='#777777',ls='--',lw=1,label='Perfect amplitude')
    axD.set_xticks(range(len(targets)));axD.set_xticklabels(targets,rotation=25,ha='right')
    axD.set_ylabel('Predicted / observed response RMS')
    axD.set_title('D  Response amplitude recovery',loc='left',fontweight='bold',fontsize=10,pad=10)
    fig.suptitle('Figure 7 | Representative perturbations reveal cell-context-dependent response recovery',x=.075,y=.982,ha='left',fontsize=14,fontweight='bold')
    fig.text(.075,.064,'Frozen K562-trained gated model · External held-out perturbations only · Seeds 42–46 · Descriptive case selection; not a population-level performance estimate.',fontsize=8)
    fig.text(.075,.043,'Heatmap gene rankings are selected using observed response magnitude (post hoc); observed and predicted values use the same color scale.',fontsize=7.5,color='#555555')
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'Figure7_representative_publication.{ext}',dpi=dpi)
    plt.close(fig)
    return pd.DataFrame(genes_used),pd.DataFrame([{'target':t,'dataset':c,'n_seeds':n,'pcc_mean_prediction':pcc(y,p),'observed_rms':float(np.sqrt(np.mean(y*y))),
                                                  'predicted_rms':float(np.sqrt(np.mean(p*p)))} for t,c,g,y,p,n in details])

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--outdir',default='results/step13L_figure7_representative_publication');p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46]);p.add_argument('--min-seeds',type=int,default=2);p.add_argument('--dpi',type=int,default=300)
    a=p.parse_args();root=Path(a.root).resolve();out=(root/a.outdir).resolve();out.mkdir(parents=True,exist_ok=True)
    data,df,audit=load_all(root,a.seeds)
    chosen,candidates=choose_targets(df,min_seeds=a.min_seeds)
    geneaudit,detail=plot(data,df,chosen,out,a.dpi)
    df.to_csv(out/'per_target_seed_metrics.csv',index=False);chosen.to_csv(out/'selected_representative_targets.csv',index=False)
    candidates.to_csv(out/'eligible_target_candidates.csv',index=False);audit.to_csv(out/'input_audit.csv',index=False)
    geneaudit.to_csv(out/'heatmap_gene_selection.csv',index=False);detail.to_csv(out/'selected_target_response_summary.csv',index=False)
    (out/'Figure7_legend.md').write_text('Figure 7. Representative external perturbation responses. (A) For four rule-selected targets, the top 24 genes ranked by absolute observed response are displayed for each cell line. Observed and predicted responses share a diverging color scale; gene columns are ranked independently for each target–cell-line pair and should not be interpreted as globally aligned genes. (B) Gene-level predicted versus observed responses for the HepG2-favored example, using a reproducible subset of up to 1,800 genes per cell line. (C) Per-target Pearson correlation and (D) predicted/observed response RMS ratio for HepG2 and Jurkat; points represent means and error bars SD across available seeds. All cases are K562-training-held-out perturbations and frozen model predictions. Targets are selected descriptively based on minimum seed coverage and distinct performance profiles; no inference about general performance is made from these selected examples.\n',encoding='utf-8')
    print('Selected targets:');print(chosen.to_string(index=False));print('Saved to',out)
if __name__=='__main__':main()
