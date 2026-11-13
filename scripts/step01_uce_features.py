#!/usr/bin/env python3
"""UCE input-token extraction, using official UCE gene-to-token mapping.
Run from project root. This is ESM2-based UCE INPUT tokens, NOT contextual UCE cell embeddings.
"""
import argparse, json, os, sys, tarfile
from pathlib import Path
import numpy as np


def safe_extract(tar_path, dest):
    dest = Path(dest).resolve()
    with tarfile.open(tar_path, 'r:gz') as tf:
        members = tf.getmembers()
        for m in members:
            resolved = (dest / m.name).resolve()
            if not resolved.is_relative_to(dest) or m.issym() or m.islnk() or m.isdev():
                raise ValueError(f'Unsafe archive entry: {m.name}')
        tf.extractall(dest, members=members, filter='data')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--extract', action='store_true', help='Extract validated tar.gz in model_files')
    args=ap.parse_args()
    root=Path(args.root).resolve(); uce=root/'external/UCE'; mf=uce/'model_files'
    for f in ['4layer_model.torch','all_tokens.torch','species_chrom.csv','species_offsets.pkl','protein_embeddings.tar.gz']:
        p=mf/f
        if not p.is_file() or p.stat().st_size==0: raise FileNotFoundError(f'Missing/empty: {p}')
        print(f'[OK] {f}: {p.stat().st_size/1e9:.3f} GB')
    with tarfile.open(mf/'protein_embeddings.tar.gz','r:gz') as tf:
        members=tf.getmembers(); print('[TAR] entries:',len(members),'examples:',[x.name for x in members[:6]])
    if args.extract:
        print('[EXTRACT] Checking and extracting archive...')
        safe_extract(mf/'protein_embeddings.tar.gz',mf)
    import torch, pandas as pd, anndata as ad
    sys.path.insert(0,str(uce))
    os.chdir(uce)  # UCE internally reads ./model_files/...
    from data_proc.data_utils import get_species_to_pe, get_spec_chrom_csv, adata_path_to_prot_chrom_starts
    import pickle
    with open(mf/'species_offsets.pkl','rb') as f: offsets=pickle.load(f)
    print('[OFFSETS]',offsets)
    # Official UCE expects model_files/protein_embeddings/; fail rather than guessing mapping.
    pe_dir=mf/'protein_embeddings'
    if not pe_dir.is_dir():
        raise RuntimeError(f'{pe_dir} missing. Re-run with --extract, inspect archive layout.')
    species_to_pe=get_species_to_pe(str(pe_dir)+'/')
    if 'human' not in species_to_pe: raise ValueError(f'human not in species_to_pe: {list(species_to_pe)}')
    spec_pe_genes=list(species_to_pe['human'].keys())
    chrom=get_spec_chrom_csv(str(mf/'species_chrom.csv'))
    with np.load(root/'data/processed/k562/response_matrix_genept_common.npz',allow_pickle=False) as z:
        if 'perturbations' not in z: raise KeyError(f'expected perturbations, got {z.files}')
        genes=np.asarray(z['perturbations']).astype(str)
    # Keep original perturbation names in the saved features.
    # Resolve explicit aliases only for UCE lookup.
    # HGNC previous approved symbols only; no fuzzy/alias matching.
    hgnc_path = root / "data/reference/hgnc/hgnc_complete_set.txt"
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str).fillna("")
    hgnc = hgnc.loc[hgnc["status"] == "Approved"]
    approved = set(hgnc["symbol"])

    previous_to_current = {}
    for current, previous in zip(hgnc["symbol"], hgnc["prev_symbol"]):
        for old_symbol in previous.split("|"):
            old_symbol = old_symbol.strip()
            if old_symbol:
                previous_to_current.setdefault(old_symbol, set()).add(current)

    aliases = {
        old_symbol: next(iter(candidates))
        for old_symbol, candidates in previous_to_current.items()
        if len(candidates) == 1 and old_symbol not in approved
    }
    # Normalize HGNC names exactly as the input lookup names.
    # Accept only unique mappings; do not guess ambiguous aliases.
    approved_upper = {
        str(s).strip().upper() for s in hgnc["symbol"]
    }
    previous_lookup = {}
    alias_lookup = {}

    for _, row in hgnc.iterrows():
        current = str(row["symbol"]).strip().upper()
        for column, lookup in [
            ("prev_symbol", previous_lookup),
            ("alias_symbol", alias_lookup),
        ]:
            for name in str(row[column]).split("|"):
                name = name.strip().upper()
                if name:
                    lookup.setdefault(name, set()).add(current)

    aliases = {}
    for name in set(previous_lookup) | set(alias_lookup):
        # An approved current symbol must not be redirected.
        if name in approved_upper:
            continue
        candidates = (
            previous_lookup.get(name, set()) |
            alias_lookup.get(name, set())
        )
        if len(candidates) == 1:
            aliases[name] = next(iter(candidates))

    print("[HGNC] unique normalized mappings:", len(aliases))
    for name in ["C14ORF178", "QARS"]:
        print("[HGNC LOOKUP]", name, "->",
              aliases.get(name, "NO UNIQUE MAPPING"))


    # Specific QARS resolution, validated against the local HGNC snapshot.
    qars_rows = hgnc.loc[
        hgnc["symbol"].astype(str).str.upper().eq("QARS1")
    ]
    qars_supported = False
    for _, row in qars_rows.iterrows():
        for column in ["prev_symbol", "alias_symbol"]:
            names = {
                x.strip().upper()
                for x in str(row[column]).split("|") if x.strip()
            }
            if "QARS" in names:
                qars_supported = True

    if not qars_supported:
        raise RuntimeError(
            "Local HGNC table does not support QARS -> QARS1; stop for review."
        )
    aliases["QARS"] = "QARS1"
    print("[CURATED MAPPING] QARS -> QARS1; supported by local HGNC")
    pe_names = set(spec_pe_genes)
    human_chrom = chrom.loc[chrom["species"] == "human"]
    chrom_names = set(human_chrom["gene_symbol"].astype(str))

    mapped_names = []
    audit = []
    for original in genes:
        query = str(original).strip().upper()
        mapped = query
        if query not in pe_names and query in aliases:
            mapped = aliases[query]

        in_pe = mapped in pe_names
        in_chrom = mapped in chrom_names
        audit.append({
            "original_gene": str(original),
            "uce_gene": mapped,
            "alias_applied": mapped != query,
            "in_protein_embeddings": in_pe,
            "in_chromosome_table": in_chrom,
        })
        mapped_names.append(mapped)

    out = root / "data/processed/embeddings/uce"
    out.mkdir(parents=True, exist_ok=True)
    audit_df = pd.DataFrame(audit)
    audit_path = out / "uce_gene_mapping_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    changed = audit_df.loc[audit_df["alias_applied"]]
    if len(changed):
        print("[ALIASES]")
        print(changed.to_string(index=False))

    missing = audit_df.loc[
        ~(audit_df["in_protein_embeddings"] &
          audit_df["in_chromosome_table"])
    ]
    print(f"[MAPPING] matched {len(genes)-len(missing)}/{len(genes)}")
    print("[AUDIT]", audit_path)
    if len(missing):
        print("[UNMATCHED]")
        print(missing.to_string(index=False))

        # Only this explicitly reviewed exclusion is allowed.
        allowed_exclusions = {"C14ORF178"}
        unexpected = set(missing["original_gene"]) - allowed_exclusions
        if unexpected:
            raise RuntimeError(
                "Unexpected unmapped genes remain: " +
                ", ".join(sorted(unexpected))
            )

        exclusion_path = out / "uce_excluded_perturbations.csv"
        missing.to_csv(exclusion_path, index=False)

        keep = (
            audit_df["in_protein_embeddings"] &
            audit_df["in_chromosome_table"]
        ).to_numpy(dtype=bool)

        genes = genes[keep]
        mapped_names = [
            name for name, retained in zip(mapped_names, keep) if retained
        ]
        audit_df = audit_df.loc[keep].reset_index(drop=True)

        print("[EXCLUSIONS]", exclusion_path)
        print("[RETAINED]", len(genes), "perturbations")

    duplicates = pd.Index(mapped_names).duplicated(keep=False)
    if duplicates.any():
        print(audit_df.loc[duplicates].to_string(index=False))
        raise RuntimeError("Multiple perturbations map to one UCE gene; review first.")

    # Use the official mapper with verified names and original row order.
    dummy=ad.AnnData(X=np.zeros((1,len(genes)),dtype=np.float32))
    dummy.var_names=pd.Index(mapped_names)
    pe_idx, chroms, positions=adata_path_to_prot_chrom_starts(
        dummy,'human',spec_pe_genes,chrom,offsets['human'])
    pe_idx=np.asarray(pe_idx).reshape(-1)
    if len(pe_idx)!=len(genes):
        raise RuntimeError(f'Official mapper returned {len(pe_idx)} indices for {len(genes)} genes. '
                           'It likely filters unmapped genes; inspect data_proc/data_utils.py and use its retained gene order. '
                           'Do NOT align indices by position until mapping is verified.')
    tokens=torch.load(mf/'all_tokens.torch',map_location='cpu',weights_only=True)
    if not isinstance(tokens,torch.Tensor): raise TypeError('all_tokens.torch is not a tensor')
    if tokens.ndim!=2: raise ValueError(f'Unexpected token shape: {tokens.shape}')
    if not np.issubdtype(pe_idx.dtype,np.integer): raise TypeError('token indices are not integers')
    valid=(pe_idx>=0)&(pe_idx<len(tokens))
    if not valid.all():
        raise RuntimeError('Mapper contains missing/out-of-bounds indices; inspect official mapping, do not silently drop.')
    if len(set(pe_idx.tolist()))!=len(pe_idx):
        raise RuntimeError('Multiple genes mapped to the same token; audit mapping before proceeding.')
    # Do not claim model's contextual gene features: these are static UCE input tokens.
    arr=tokens[torch.as_tensor(pe_idx,dtype=torch.long)].cpu().numpy().astype(np.float32)
    out=root/'data/processed/embeddings/uce';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'uce_gene_tokens.npz',genes=genes,embeddings=arr)
    (out/'uce_gene_mapping_summary.json').write_text(json.dumps({
        'feature_definition':'UCE input gene tokens (ESM2-derived; not contextual UCE cell embeddings)',
        'n_genes':len(genes),'embedding_dim':arr.shape[1], 'uce_git_revision':'9c416007be15ad6753dc84af4468c1dc10421ab9'
    },indent=2))
    print('[SAVED]',out/'uce_gene_tokens.npz',arr.shape)

if __name__=='__main__':main()
