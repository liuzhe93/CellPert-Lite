#!/usr/bin/env python3
"""Figure 6: frozen cross-cell-line held-out response diagnostics.
Uses actual per-seed predictions.npz; no model fitting or simulated values.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODELS={'gated_direct':'Gated fusion','no_gate_direct':'No-gate fusion',
        'genept_only_direct':'GenePT-only','uce_only_direct':'UCE-only'}
COL={'gated_direct':'#286B87','no_gate_direct':'#25816C',
     'genept_only_direct':'#C68C30','uce_only_direct':'#8264A7'}
CELL={'HepG2':'o','Jurkat':'s'}
EPS=1e-12

def corr_rows(y,p):
    a=y-y.mean(axis=1,keepdims=True);b=p-p.mean(axis=1,keepdims=True)
    den=np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
    return np.divide((a*b).sum(axis=1),den,out=np.full(len(y),np.nan),where=den>EPS)

def read_npz(path):
    with np.load(path,allow_pickle=False) as z:
        need={'perturbations','genes','y_true','prediction','k562_train_seen'}
        if not need.issubset(z.files):raise ValueError(f'{path}: missing {need-set(z.files)}; keys={z.files}')
        ids=z['perturbations'].astype(str); genes=z['genes'].astype(str)
        y=z['y_true'].astype('float64');p=z['prediction'].astype('float64')
        seen=z['k562_train_seen'].astype(bool)
    if y.shape!=p.shape or y.shape!=(len(ids),len(genes)) or seen.shape!=(len(ids),):
        raise ValueError(f'Incompatible shapes: {path}')
    if len(np.unique(ids))!=len(ids):raise ValueError(f'Duplicate perturbation names: {path}')
    if not (np.isfinite(y).all() and np.isfinite(p).all()):raise ValueError(f'Nonfinite response: {path}')
    return ids,genes,y,p,seen

def load(root,seeds):
    rows=[]; audit=[]
    base=root/'results/step11B_4_frozen_external_transfer'
    for cell in CELL:
      for seed in seeds:
        reference=None
        for model in MODELS:
          f=base/cell/f'seed_{seed}'/model/'predictions.npz'
          if not f.is_file():raise FileNotFoundError(f)
          ids,genes,y,p,seen=read_npz(f)
          if reference is None:reference=(ids,genes,y,seen)
          else:
            ri,rg,ry,rs=reference
            if not (np.array_equal(ids,ri) and np.array_equal(genes,rg) and
                    np.array_equal(seen,rs) and np.allclose(y,ry,atol=1e-7,rtol=1e-6)):
                raise ValueError(f'Model truth/target alignment differs: {f}')
          keep=~seen
          yy=y[keep];pp=p[keep]; ii=ids[keep]
          pcc=corr_rows(yy,pp)
          ss=np.mean(yy*yy,axis=1)
          mm=np.mean((pp-yy)**2,axis=1)
          rms_true=np.sqrt(ss)
          rms_pred=np.sqrt(np.mean(pp*pp,axis=1))
          skill=np.divide(ss-mm,ss,out=np.full(len(ss),np.nan),where=ss>EPS)
          ratio=np.divide(rms_pred,rms_true,out=np.full(len(ss),np.nan),where=rms_true>EPS)
          for j,t in enumerate(ii):
            rows.append((cell,seed,model,t,pcc[j],ss[j],mm[j],rms_true[j],rms_pred[j],ratio[j],skill[j]))
          audit.append(dict(cell_line=cell,seed=seed,model=model,n_total=len(ids),n_heldout=len(ii),
                            n_valid_pcc=np.isfinite(pcc).sum(),n_valid_skill=np.isfinite(skill).sum()))
    cols=['cell_line','seed','model','perturbation','pcc','truth_mse_vs_zero','model_mse',
          'truth_rms','predicted_rms','rms_ratio','target_skill']
    return pd.DataFrame(rows,columns=cols),pd.DataFrame(audit)

def summaries(d):
    rows=[]
    for (cell,model,seed),g in d.groupby(['cell_line','model','seed'],sort=False):
      base=g.truth_mse_vs_zero.sum();loss=g.model_mse.sum()
      rows.append(dict(cell_line=cell,model=model,seed=seed,n_targets=len(g),
                       micro_skill=(base-loss)/base if base>EPS else np.nan,
                       macro_skill=g.target_skill.mean(),mean_pcc=g.pcc.mean(),
                       median_rms_ratio=g.rms_ratio.median()))
    return pd.DataFrame(rows)

def plot(d,s,out,dpi):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.8,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(13.3,9.3),constrained_layout=False)
    gs=fig.add_gridspec(2,2,left=.075,right=.97,bottom=.145,top=.895,wspace=.32,hspace=.40)
    axes=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    ax=axes[0]
    # Per-seed values, micro and macro, separated by position and cell line
    for k,model in enumerate(MODELS):
      for cell,offset in [('HepG2',-.12),('Jurkat',.12)]:
        q=s[(s.model==model)&(s.cell_line==cell)]
        for t,(metric,shift) in enumerate([('micro_skill',-.20),('macro_skill',.20)]):
          vals=q[metric].to_numpy();x=k+offset+shift
          ax.scatter(np.full(len(vals),x),vals,s=15,color=COL[model],marker=CELL[cell],alpha=.45,zorder=3)
          if len(vals):ax.plot([x-.075,x+.075],[np.nanmean(vals)]*2,color=COL[model],lw=2.1)
    ax.axhline(0,color='#777',ls='--',lw=.9)
    ax.set_xticks(range(4),list(MODELS.values()),rotation=16,ha='right')
    ax.set_ylabel('Relative skill vs zero-response baseline')
    ax.set_title('A   Micro and macro relative skill',loc='left',weight='bold',pad=10)
    ax.text(.01,.97,'Left cluster: micro   |   Right cluster: macro',transform=ax.transAxes,va='top',fontsize=7.4)
    ax.grid(axis='y',alpha=.15)
    # For B and C: target-level, average across seeds for unique target to avoid pseudo replication
    q=d[d.model=='gated_direct'].groupby(['cell_line','perturbation'],as_index=False).agg(
        truth_rms=('truth_rms','mean'),pcc=('pcc','mean'),rms_ratio=('rms_ratio','mean'),
        target_skill=('target_skill','mean'))
    for cell in CELL:
      x=q[q.cell_line==cell].dropna(subset=['truth_rms','pcc'])
      axes[1].scatter(x.truth_rms,x.pcc,s=8,alpha=.14,color='#286B87' if cell=='HepG2' else '#C68C30',rasterized=True)
      # Equal-count bins for robust descriptive trend
      x=x.copy();x['bin']=pd.qcut(x.truth_rms.rank(method='first'),q=min(8,len(x)),labels=False)
      b=x.groupby('bin').agg(x=('truth_rms','median'),y=('pcc','median'))
      axes[1].plot(b.x,b.y,lw=2,marker=CELL[cell],ms=4,label=cell,color='#286B87' if cell=='HepG2' else '#C68C30')
      z=q[q.cell_line==cell].dropna(subset=['truth_rms','rms_ratio']).copy()
      z=z[z.truth_rms>1e-9]
      z['bin']=pd.qcut(z.truth_rms.rank(method='first'),q=min(8,len(z)),labels=False)
      bb=z.groupby('bin').agg(x=('truth_rms','median'),y=('rms_ratio','median'))
      axes[2].plot(bb.x,bb.y,lw=2,marker=CELL[cell],ms=4,label=cell,color='#286B87' if cell=='HepG2' else '#C68C30')
    axes[1].set(title='B   Response strength and prediction correlation',xlabel='Observed response RMS (target level)',ylabel='Per-target Pearson r')
    axes[1].legend(frameon=False,loc='best');axes[1].grid(alpha=.14)
    axes[2].axhline(1,ls='--',lw=.9,color='#777')
    axes[2].set(title='C   Predicted response amplitude',xlabel='Observed response RMS (target level)',ylabel='Median predicted / observed RMS')
    axes[2].legend(frameon=False);axes[2].grid(alpha=.14)
    # D: target-level skill quartiles (descriptive, truth-defined)
    ax=axes[3]
    for cell,off in [('HepG2',-.15),('Jurkat',.15)]:
      z=q[q.cell_line==cell].dropna(subset=['truth_rms','target_skill']).copy()
      z['quartile']=pd.qcut(z.truth_rms.rank(method='first'),4,labels=False)
      for k in range(4):
        v=z.loc[z.quartile==k,'target_skill'].to_numpy()
        # Avoid hiding extremes: show median and IQR, not mean affected by near-zero denominator
        if len(v):
          xx=k+1+off; lo,med,hi=np.percentile(v,[25,50,75]);
          ax.plot([xx,xx],[lo,hi],color='#286B87' if cell=='HepG2' else '#C68C30',lw=5,alpha=.45)
          ax.scatter([xx],[med],marker=CELL[cell],s=34,color='#286B87' if cell=='HepG2' else '#C68C30',zorder=3)
    ax.axhline(0,ls='--',lw=.9,color='#777')
    ax.set_xticks([1,2,3,4],['Q1 (weak)','Q2','Q3','Q4 (strong)'])
    ax.set(title='D   Relative skill by observed-response quartile',ylabel='Target relative skill (median and IQR)')
    ax.grid(axis='y',alpha=.14)
    fig.suptitle('Figure 6 | Response strength and calibration limit external transfer',x=.075,ha='left',y=.975,fontsize=15,weight='bold')
    legend=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#555',label='HepG2',markersize=6),
            Line2D([0],[0],marker='s',color='none',markerfacecolor='#555',label='Jurkat',markersize=6)]
    fig.legend(handles=legend,loc='upper right',bbox_to_anchor=(.97,.956),frameon=False,ncol=2)
    fig.text(.075,.074,'Held-out perturbations only | Frozen K562-trained weights | Seeds 42–46; target-level analyses average repeated seeds.',fontsize=8)
    fig.text(.075,.050,'Micro skill = 1 − ΣMSE / Σzero-baseline MSE; macro skill = mean target-level skill. Q1–Q4 use observed truth (post hoc).',fontsize=7.8)
    fig.text(.075,.028,'Panels B–D: Gated fusion; descriptive analyses, not deployable truth-based selection. All metrics recomputed from predictions.npz.',fontsize=7.8)
    for ext in ['pdf','svg','png']:
      fig.savefig(out/f'Figure6_response_diagnostics_publication.{ext}',dpi=dpi)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--outdir',default='results/step13K_figure6_response_diagnostics')
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46]);ap.add_argument('--dpi',type=int,default=300)
    a=ap.parse_args();root=Path(a.root).resolve();out=root/a.outdir;out.mkdir(parents=True,exist_ok=True)
    d,audit=load(root,a.seeds);s=summaries(d)
    d.to_csv(out/'per_target_response_diagnostics.csv',index=False)
    s.to_csv(out/'micro_macro_skill_per_seed.csv',index=False)
    audit.to_csv(out/'input_coverage_audit.csv',index=False)
    plot(d,s,out,a.dpi)
    print('Saved Figure6_response_diagnostics_publication.pdf/.svg/.png')
    print(s.groupby(['cell_line','model'])[['micro_skill','macro_skill','mean_pcc']].mean().to_string())
if __name__=='__main__':main()
