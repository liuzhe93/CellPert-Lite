import argparse
import json
from pathlib import Path
from . import __version__
from .data import inspect_h5ad, prepare_h5ad
from .benchmark import run_demo, run_response_benchmark
from .predict import predict_one

def build_parser():
    p=argparse.ArgumentParser(prog="cellpert", description="CellPert-Lite reproducible perturbation-response baselines")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub=p.add_subparsers(dest="command", required=True)
    q=sub.add_parser("demo", help="Run complete synthetic smoke test")
    q.add_argument("--outdir", default="results/demo"); q.add_argument("--seed", type=int, default=42)
    q.add_argument("--n-perturbations", type=int, default=120); q.add_argument("--n-genes", type=int, default=250)
    q.add_argument("--embedding-dim", type=int, default=48); q.add_argument("--gene-embeddings", default=None)
    q=sub.add_parser("inspect-data", help="Inspect .h5ad metadata")
    q.add_argument("--input", required=True); q.add_argument("--perturbation-key", default=None)
    q=sub.add_parser("prepare-h5ad", help="Export group-mean delta matrix from .h5ad")
    q.add_argument("--input", required=True); q.add_argument("--outdir", required=True)
    q.add_argument("--perturbation-key", required=True); q.add_argument("--control-label", required=True)
    q.add_argument("--min-cells", type=int, default=20); q.add_argument("--layer", default=None)
    q=sub.add_parser("train", help="Benchmark models on prepared response matrix")
    q.add_argument("--response", required=True); q.add_argument("--gene-embeddings", required=True)
    q.add_argument("--outdir", required=True); q.add_argument("--seed", type=int, default=42)
    q.add_argument("--test-size", type=float, default=0.2)
    q=sub.add_parser("predict", help="Predict one perturbation using saved Ridge model")
    q.add_argument("--model", required=True); q.add_argument("--gene-embeddings", required=True)
    q.add_argument("--perturbation", required=True); q.add_argument("--out", required=True)
    return p

def main(argv=None):
    a=build_parser().parse_args(argv)
    if a.command=="demo":
        df=run_demo(a.outdir,a.seed,a.n_perturbations,a.n_genes,a.embedding_dim,a.gene_embeddings)
        print(df.to_string(index=False)); print(f"\nResults: {Path(a.outdir).resolve()}")
    elif a.command=="inspect-data":
        print(json.dumps(inspect_h5ad(a.input,a.perturbation_key),indent=2,default=str))
    elif a.command=="prepare-h5ad":
        print(json.dumps(prepare_h5ad(a.input,a.outdir,a.perturbation_key,a.control_label,a.min_cells,a.layer),indent=2))
    elif a.command=="train":
        df=run_response_benchmark(a.response,a.gene_embeddings,a.outdir,a.seed,a.test_size)
        print(df.to_string(index=False)); print(f"\nResults: {Path(a.outdir).resolve()}")
    elif a.command=="predict":
        df=predict_one(a.model,a.gene_embeddings,a.perturbation,a.out)
        print(df.head(10).to_string(index=False)); print(f"\nPrediction: {Path(a.out).resolve()}")

if __name__=="__main__": main()
