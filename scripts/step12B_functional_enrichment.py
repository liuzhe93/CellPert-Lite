#!/usr/bin/env python3
"""Step12B: exploratory GO BP / Reactome over-representation with matched assay universes.

Dependencies: pandas numpy scipy matplotlib; optional gseapy (only for --fetch-libraries).
Gene symbols are human HGNC symbols; GMTs must also use human gene symbols.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LIBS={'GO_BP':'GO_Biological_Process_2025','Reactome':'Reactome_2022'}

def get_args():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 p.add_argument('--root',default='.')
 p.add_argument('--candidates',default='results/step12A_representative_perturbations/selected_candidates_by_category.csv')
 p.add_argument('--background',default='results/step12A_representative_perturbations/enrichment_background_by_dataset_group.csv')
 p.add_argument('--outdir',default='results/step12B_functional_enrichment')
 p.add_argument('--gmt-dir',default='data/gene_sets')
 p.add_argument('--fetch-libraries',action='store_true',help='Download Enrichr gene sets using gseapy; internet required on server')
 p.add_argument('--min-term-size',type=int,default=5)
 p.add_argument('--max-term-size',type=int,default=500)
 p.add_argument('--min-overlap',type=int,default=2)
 p.add_argument('--fdr',type=float,default=.05)
 p.add_argument('--top-plot',type=int,default=15)
 p.add_argument('--min-query',type=int,default=5)
 p.add_argument('--seed',type=int,default=42)
 return p.parse_args()

def resolve(root,s):
 x=Path(s);return x if x.is_absolute() else root/x

def parse_gmt(path):
 d={}
 with open(path,encoding='utf-8') as f:
  for line in f:
   a=line.rstrip('\n').split('\t')
   if len(a)>=3:
    genes={x.strip().upper() for x in a[2:] if x.strip()}
    if genes:d[a[0]]=genes
 return d

def fetch_gmt(outdir):
 try:import gseapy as gp
 except ImportError:raise SystemExit('Please install gseapy (pip install gseapy) or provide local GMT files.')
 outdir.mkdir(parents=True,exist_ok=True)
 for name,library in LIBS.items():
  path=outdir/f'{name}.gmt'
  if path.exists():continue
  print(f'[DOWNLOAD] {library}',flush=True)
  try: terms=gp.get_library(name=library,organism='Human')
  except Exception as e:raise SystemExit(f'Failed to fetch {library}: {e}')
  with path.open('w') as f:
   for term,genes in terms.items():f.write('\t'.join([term,library]+list(genes))+'\n')

def bh(p):
 p=np.asarray(p,dtype=float); n=len(p)
 if not n:return p
 idx=np.argsort(p);out=np.empty(n);out[idx]=np.minimum.accumulate((p[idx]*n/np.arange(1,n+1))[::-1])[::-1].clip(0,1)
 return out

def enrich(query,universe,terms,db,min_size,max_size,min_overlap):
 query=query&universe
 N=len(universe);n=len(query); rows=[]
 if not n or not N:return pd.DataFrame()
 for term,raw in terms.items():
  genes=raw&universe;K=len(genes)
  if K<min_size or K>max_size:continue
  hits=query&genes;k=len(hits)
  # Include zero-hit and single-hit terms in the FDR correction family.
  pv=float(hypergeom.sf(k-1,N,K,n)) if k else 1.0
  rows.append({'database':db,'term':term,'universe_size':N,'query_size':n,
               'term_size_in_universe':K,'overlap':k,'p_value':pv,
               'fold_enrichment':(k/n)/(K/N) if k else 0.0,
               'overlap_genes':';'.join(sorted(hits))})
 if not rows:return pd.DataFrame()
 x=pd.DataFrame(rows);x['fdr_bh']=bh(x.p_value.values)
 return x[x.overlap>=min_overlap].sort_values(['fdr_bh','p_value','term']).reset_index(drop=True)

def draw(x,outbase,n,fdr):
 x=x[(x.fdr_bh<fdr)&(x.overlap>=2)].head(n).iloc[::-1]
 if x.empty:return False
 labels=[s[:68] for s in x.term]
 fig,ax=plt.subplots(figsize=(10,max(3.8,.39*len(x)+1.5)))
 y=np.arange(len(x));ax.scatter(-np.log10(np.maximum(x.fdr_bh.to_numpy(),1e-300)),y,
                               s=25+18*x.overlap.to_numpy(),alpha=.85)
 ax.set_yticks(y,labels);ax.set_xlabel('-log10(BH-adjusted P)')
 ax.set_title(outbase.stem.replace('_',' '));ax.grid(axis='x',alpha=.2)
 fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(outbase.with_suffix('.'+ext),dpi=250,bbox_inches='tight')
 plt.close(fig);return True

def main():
 a=get_args();root=Path(a.root).resolve();out=resolve(root,a.outdir);out.mkdir(parents=True,exist_ok=True)
 cand_path=resolve(root,a.candidates);bg_path=resolve(root,a.background)
 if not cand_path.exists() or not bg_path.exists():
  raise SystemExit(f'Missing inputs: {cand_path} or {bg_path}')
 cand=pd.read_csv(cand_path);bg=pd.read_csv(bg_path)
 for label,df,cols in [('candidates',cand,['dataset','perturbation','group','category']),('background',bg,['dataset','perturbation','group'])]:
  missing=set(cols)-set(df.columns)
  if missing:raise SystemExit(f'{label} missing columns: {sorted(missing)}')
 for df in [cand,bg]:
  for c in ['dataset','group','perturbation']:df[c]=df[c].astype(str).str.strip()
  df['perturbation']=df['perturbation'].str.upper()
 cand['category']=cand.category.astype(str).str.strip()
 gdir=resolve(root,a.gmt_dir)
 if a.fetch_libraries:fetch_gmt(gdir)
 libraries={}
 for name in LIBS:
  p=gdir/f'{name}.gmt'
  if p.exists():libraries[name]=parse_gmt(p);print(f'[GMT] {name}: {len(libraries[name])} terms from {p}')
  else:print(f'[WARN] Missing {p}',file=sys.stderr)
 if not libraries:raise SystemExit('No GMTs found. Run with --fetch-libraries (internet + gseapy) or place GO_BP.gmt and Reactome.gmt in --gmt-dir.')
 all_results=[];audits=[];coverage=[]
 keys=sorted(set(zip(cand.dataset,cand.group)))
 for dataset,group in keys:
  U=set(bg.loc[(bg.dataset==dataset)&(bg.group==group),'perturbation'])
  if not U:raise SystemExit(f'Empty universe for {dataset}/{group}; check background file.')
  sub=cand[(cand.dataset==dataset)&(cand.group==group)]
  for category,s in sub.groupby('category'):
   raw=set(s.perturbation);Q=raw&U
   audit={'dataset':dataset,'group':group,'category':category,'n_background':len(U),
          'n_candidates_raw':len(raw),'n_query_in_background':len(Q),
          'n_outside_background':len(raw-U),'outside_background_genes':';'.join(sorted(raw-U)),
          'low_power_query_lt_min':len(Q)<a.min_query}
   audits.append(audit)
   if len(Q)<a.min_query:
    print(f'[WARN] Small query ({len(Q)}): {dataset}/{group}/{category}')
   for db,terms in libraries.items():
    eligible={t:genes for t,genes in terms.items() if a.min_term_size<=len(genes&U)<=a.max_term_size}
    annotated=set().union(*[g&U for g in eligible.values()]) if eligible else set()
    coverage.append({**{k:audit[k] for k in ['dataset','group','category']},'database':db,
                    'query_n':len(Q),'query_annotated_n':len(Q&annotated),
                    'background_n':len(U),'background_annotated_n':len(U&annotated),
                    'tested_terms':len(eligible)})
    result=enrich(Q,U,terms,db,a.min_term_size,a.max_term_size,a.min_overlap)
    if result.empty:continue
    result.insert(0,'category',category);result.insert(0,'group',group);result.insert(0,'dataset',dataset)
    all_results.append(result)
    base=out/'figures'/f'{dataset}_{group}_{category}_{db}'
    base.parent.mkdir(exist_ok=True,parents=True)
    draw(result,base,a.top_plot,a.fdr)
 auditdf=pd.DataFrame(audits);auditdf.to_csv(out/'selection_and_background_audit.csv',index=False)
 pd.DataFrame(coverage).to_csv(out/'gene_set_coverage.csv',index=False)
 if all_results:
  results=pd.concat(all_results,ignore_index=True)
  results.to_csv(out/'enrichment_all_terms.csv',index=False)
  sig=results[results.fdr_bh<a.fdr].copy();sig.to_csv(out/'enrichment_significant.csv',index=False)
  print(f'[DONE] {len(results)} displayed tested terms with >= {a.min_overlap} overlaps; {len(sig)} FDR<{a.fdr}')
 else:
  pd.DataFrame().to_csv(out/'enrichment_all_terms.csv',index=False)
  pd.DataFrame().to_csv(out/'enrichment_significant.csv',index=False)
  print('[DONE] No terms passed overlap cutoff.')
 print(f'Output: {out}')
 print('CAUTION: exploratory ORA; query selected by model performance, possible signal/QC confounding.')
 print('BH correction is within each dataset/group/category/library, across all eligible terms including zero-hit terms.')

if __name__=='__main__':main()
