#!/usr/bin/env python3
"""Step11B-3: read-only, no-training compatibility audit for CellPert-Lite.
Outputs CSV/JSON reports. Does NOT load untrusted pickle/checkpoint weights.
"""
import argparse, json, csv, zipfile, re
from pathlib import Path
from collections import Counter
import numpy as np

def resolve(root, value):
    p=Path(value)
    return p if p.is_absolute() else root/p

def npz_info(path, required):
    if not path.is_file():
        return None, {"path":str(path),"status":"MISSING"}
    with np.load(path,allow_pickle=False) as z:
        keys=list(z.files)
        if not set(required).issubset(keys):
            return None,{"path":str(path),"status":"INVALID_KEYS","keys":keys,"required":required}
        data={k:np.asarray(z[k]) for k in required}
        extra={k:{"shape":list(z[k].shape),"dtype":str(z[k].dtype)} for k in keys if k not in required}
    return data,{"path":str(path),"status":"OK","keys":keys,
                 "shapes":{k:list(v.shape) for k,v in data.items()},
                 "extra_arrays":extra}

def symbols(a, name):
    a=np.asarray(a)
    if a.ndim!=1: raise ValueError(f"{name} must be 1D")
    s=[str(x).strip() for x in a]
    if not s or any(not x or x.lower() in ("nan","none") for x in s):
        raise ValueError(f"{name}: empty/invalid identifiers")
    if len(set(s))!=len(s):
        raise ValueError(f"{name}: duplicate identifiers, examples {list(k for k,v in Counter(s).items() if v>1)[:8]}")
    return s

def feature(path):
    data,info=npz_info(path,["genes","embeddings"])
    if data is None:return None,info
    ids=symbols(data["genes"],str(path))
    x=data["embeddings"]
    if x.ndim!=2 or x.shape[0]!=len(ids):
        raise ValueError(f"{path}: invalid embeddings shape")
    info["embedding_dim"]=int(x.shape[1])
    info["n_genes"]=len(ids)
    info["finite"]=bool(np.isfinite(x).all())
    return {"genes":ids,"dim":x.shape[1]},info

def response(path):
    data,info=npz_info(path,["perturbations","genes","delta"])
    if data is None:return None,info
    targets=symbols(data["perturbations"],str(path)+" targets")
    genes=symbols(data["genes"],str(path)+" genes")
    y=data["delta"]
    if y.ndim!=2 or y.shape!=(len(targets),len(genes)):
        raise ValueError(f"{path}: delta shape mismatch")
    info["n_targets"]=len(targets)
    info["n_response_genes"]=len(genes)
    info["finite"]=bool(np.isfinite(y).all())
    return {"targets":targets,"genes":genes},info

def discover(root, pattern, max_count=30):
    return sorted(str(x.relative_to(root)) for x in root.glob(pattern) if x.is_file())[:max_count]

def checkpoint_metadata(path):
    item={"path":str(path),"bytes":path.stat().st_size,"suffix":path.suffix}
    # Inspect archive metadata only. Never torch.load/pickle.load arbitrary files.
    if path.suffix.lower() in (".pt",".pth",".ckpt") and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names=z.namelist()
            item["archive_entries"]=len(names)
            item["contains_pickle"]=any(x.endswith(".pkl") for x in names)
            item["contains_tensor_storage"]=any("/data/" in x for x in names)
    return item

def csvout(path, rows, fields):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",default=".")
    ap.add_argument("--outdir",default="results/step11B_3_model_compatibility")
    ap.add_argument("--k562-response",default="data/processed/benchmark_genept_uce_common/response_matrix_common.npz")
    ap.add_argument("--k562-genept",default="results/step03_genept_uce_common/aligned_features/GenePT_3large.npz")
    ap.add_argument("--k562-uce",default="results/step03_genept_uce_common/aligned_features/UCE_input_tokens.npz")
    ap.add_argument("--hepg2",default="data/external_validation/step11B_2A_expanded_minbatch1/HepG2/response_matrix_k562_gene_intersection.npz")
    ap.add_argument("--jurkat",default="data/external_validation/step11B_2A_expanded_minbatch1/Jurkat/response_matrix_k562_gene_intersection.npz")
    ap.add_argument("--step10a",default="results/step10A_unseen_perturbation_transfer")
    ap.add_argument("--hepg2-genept",default=None,help="Optional HepG2 GenePT features NPZ")
    ap.add_argument("--hepg2-uce",default=None,help="Optional HepG2 UCE features NPZ")
    ap.add_argument("--jurkat-genept",default=None)
    ap.add_argument("--jurkat-uce",default=None)
    args=ap.parse_args()
    root=Path(args.root).resolve();out=resolve(root,args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    paths={"K562_response":args.k562_response,"K562_GenePT":args.k562_genept,
           "K562_UCE":args.k562_uce,"HepG2_response":args.hepg2,"Jurkat_response":args.jurkat}
    for cell in ("hepg2","jurkat"):
        for emb in ("genept","uce"):
            v=getattr(args,f"{cell}_{emb}")
            if v:paths[f"{cell.title()}_{emb}"]=v
    paths={k:resolve(root,v) for k,v in paths.items()}
    report={"settings":vars(args),"inputs":{},"status":"AUDIT_ONLY"}
    res={}
    for key in ("K562_response","HepG2_response","Jurkat_response"):
        res[key],report["inputs"][key]=response(paths[key])
    feat={}
    for key in paths:
        if "GenePT" in key or "UCE" in key or key.endswith(("_genept","_uce")):
            feat[key],report["inputs"][key]=feature(paths[key])
    kr=res["K562_response"]
    if kr is None:raise FileNotFoundError("K562 reference response is required")
    kt=set(kr["targets"]);kg=set(kr["genes"])
    rows=[];genes_rows=[]
    for cell in ("HepG2","Jurkat"):
        r=res[cell+"_response"]
        if r is None:
            print(f"[WARN] {cell}: response missing")
            continue
        t=set(r["targets"]);g=set(r["genes"])
        g1=feat.get(cell+"_genept");g2=feat.get(cell+"_uce")
        k1=feat.get("K562_GenePT");k2=feat.get("K562_UCE")
        e1=set(g1["genes"]) if g1 else None
        e2=set(g2["genes"]) if g2 else None
        both=e1&e2 if e1 is not None and e2 is not None else None
        row={"dataset":cell,"n_external_targets":len(t),
             "n_targets_in_K562_response":len(t&kt),
             "n_targets_not_in_K562_response":len(t-kt),
             "n_response_genes":len(g),"n_response_genes_in_K562":len(g&kg),
             "n_response_genes_missing_from_K562":len(g-kg),
             "response_gene_order_is_K562_subset":r["genes"]==[x for x in kr["genes"] if x in g],
             "n_genept_covered":len(t&e1) if e1 is not None else "",
             "n_uce_covered":len(t&e2) if e2 is not None else "",
             "n_both_covered":len(t&both) if both is not None else "",
             "genept_dimension_match":(g1["dim"]==k1["dim"]) if g1 and k1 else "",
             "uce_dimension_match":(g2["dim"]==k2["dim"]) if g2 and k2 else ""}
        rows.append(row)
        for target in r["targets"]:
            genes_rows.append({"dataset":cell,"target":target,
                "in_K562_response":int(target in kt),
                "has_GenePT":int(target in e1) if e1 is not None else "",
                "has_UCE":int(target in e2) if e2 is not None else "",
                "has_both":int(target in both) if both is not None else ""})
        print(f"[{cell}] {len(t)} targets; K562 overlap={len(t&kt)}; "
              f"response genes K562 overlap={len(g&kg)}/{len(g)}; "
              f"GenePT+UCE coverage={row['n_both_covered'] or 'NOT PROVIDED'}")
    if rows:csvout(out/"dataset_compatibility.csv",rows,list(rows[0]))
    if genes_rows:csvout(out/"target_coverage.csv",genes_rows,list(genes_rows[0]))
    aroot=resolve(root,args.step10a)
    report["step10a_exists"]=aroot.is_dir()
    report["step10a_candidate_files"]=[]
    report["checkpoint_metadata"]=[]
    if aroot.is_dir():
        for p in sorted(aroot.rglob("*")):
            if not p.is_file():continue
            if len(report["step10a_candidate_files"])<150:
                report["step10a_candidate_files"].append(str(p.relative_to(root)))
            if p.suffix.lower() in (".pt",".pth",".ckpt",".joblib",".pkl",".safetensors") and len(report["checkpoint_metadata"])<80:
                report["checkpoint_metadata"].append(checkpoint_metadata(p))
    report["split_manifest_candidates"]=discover(root,"results/**/*split*.json")+discover(root,"results/**/*manifest*.json")
    report["train_split_status"]="NOT_VERIFIED: K562 response membership is NOT K562 training membership"
    report["model_input_output_status"]="NOT_VERIFIED: checkpoint tensor shapes and architecture require trusted model loader/manifest"
    report["datasets"]=rows
    report["warnings"]=[]
    if not aroot.is_dir():report["warnings"].append("Step10A directory missing")
    if not report["checkpoint_metadata"]:report["warnings"].append("No checkpoint files found in Step10A tree")
    if not all(k in feat and feat[k] is not None for k in ("K562_GenePT","K562_UCE")):
        report["warnings"].append("K562 embedding features missing")
    if not all(k in feat and feat[k] is not None for k in ("HepG2_genept","HepG2_uce","Jurkat_genept","Jurkat_uce")):
        report["warnings"].append("External embedding feature NPZs not provided; target feature coverage remains unknown")
    with (out/"audit_report.json").open("w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,default=str)
    print("\nSaved:",out)
    print("NOTE: in_K562_response != seen_in_K562_training. Do not label unseen-target until splits are verified.")
    print("NOTE: checkpoint metadata is discovery only; NO frozen model was executed.")
    for w in report["warnings"]:print("[WARN]",w)

if __name__=="__main__":
    main()
