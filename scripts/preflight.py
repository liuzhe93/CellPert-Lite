#!/usr/bin/env python3
"""Check local inputs for the published K562 research workflow."""
import argparse
from pathlib import Path
import numpy as np

REQUIRED={
 'response':('data/processed/benchmark_genept_uce_common/response_matrix_common.npz',('perturbations','genes','delta')),
 'GenePT':('results/step03_genept_uce_common/aligned_features/GenePT_3large.npz',('genes','embeddings')),
 'UCE':('results/step03_genept_uce_common/aligned_features/UCE_input_tokens.npz',('genes','embeddings')),
}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args()
 root=Path(a.root); ids={};ok=True
 for name,(relative,keys) in REQUIRED.items():
  path=root/relative
  if not path.is_file():print(f'MISSING {name}: {path}');ok=False;continue
  with np.load(path,allow_pickle=False) as z:
   missing=set(keys)-set(z.files)
   if missing:print(f'INVALID {name}: missing {sorted(missing)}');ok=False;continue
   for key in keys:print(f'{name}.{key}: {z[key].shape}')
   ids[name]=z['perturbations'].astype(str) if name=='response' else z['genes'].astype(str)
 if len(ids)==3 and not (np.array_equal(ids['response'],ids['GenePT']) and np.array_equal(ids['response'],ids['UCE'])):
  print('INVALID: target IDs/order differ across response and embeddings');ok=False
 print('PREFLIGHT PASS' if ok else 'PREFLIGHT FAIL')
 raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
