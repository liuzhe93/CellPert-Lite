
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def print_title(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def audit_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[READ ERROR] {path}: {e}")
        return

    print(f"\n[CSV] {path}")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    relevant = [
        c for c in df.columns
        if any(k in c.lower() for k in [
            "seed", "model", "method", "feature",
            "split", "train", "val", "test",
            "pcc", "top", "overlap", "recovery",
            "mse", "mae", "alpha"
        ])
    ]

    if relevant:
        print(df[relevant].head(8).to_string(index=False))

    for c in df.columns:
        if any(k in c.lower() for k in [
            "n_test", "test_count", "test_size",
            "n_train", "n_val", "n_valid"
        ]):
            print(f"  {c}: {df[c].dropna().unique()[:20]}")

    pcc_cols = [
        c for c in df.columns
        if "pcc" in c.lower()
    ]
    if pcc_cols:
        print("PCC non-null counts:")
        print(df[pcc_cols].notna().sum().to_string())


def inspect_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"[READ ERROR] {path}: {e}")
        return

    print(f"\n[JSON] {path}")

    if not isinstance(obj, dict):
        print("Top-level type:", type(obj).__name__)
        return

    for k, v in obj.items():
        if isinstance(v, list):
            print(f"  {k}: list length={len(v)}")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys={list(v)[:15]}")
        else:
            print(f"  {k}: {str(v)[:180]}")

    # Common split-key formats
    aliases = {
        "train": ["train", "train_genes", "train_ids",
                  "train_perturbations"],
        "val": ["val", "valid", "validation",
                "val_genes", "val_ids"],
        "test": ["test", "test_genes", "test_ids",
                 "test_perturbations"],
    }

    sets = {}
    for part, keys in aliases.items():
        for key in keys:
            if key in obj and isinstance(obj[key], list):
                sets[part] = set(map(str, obj[key]))
                break

    if sets:
        print("Detected split sizes:",
              {k: len(v) for k, v in sets.items()})

        names = list(sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = sets[a] & sets[b]
                print(f"  {a}/{b} overlap: {len(overlap)}")
                if overlap:
                    print("  Examples:", sorted(overlap)[:10])

        if len(sets) == 3:
            total = sum(map(len, sets.values()))
            print("Total unique split membership:", total)
            print("Split fractions:", {
                k: round(len(v) / total, 4)
                for k, v in sets.items()
            })


def inspect_npz(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            print(f"\n[NPZ] {path}")
            for k in data.files:
                arr = data[k]
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    except Exception as e:
        print(f"[NPZ INFO ERROR] {path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--results",
        default="results/step05_tuned_genept_uce"
    )
    parser.add_argument(
        "--figdir",
        default="results/step05_tuned_Figure2"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paths = [
        root / args.results,
        root / args.figdir,
    ]

    print_title("CellPert-Lite Figure 2 Audit")
    print("Project root:", root)

    for directory in paths:
        print_title(f"Scanning: {directory}")

        if not directory.exists():
            print("[NOT FOUND]")
            continue

        csvs = sorted(directory.rglob("*.csv"))
        jsons = sorted(directory.rglob("*.json"))
        npzs = sorted(directory.rglob("*.npz"))

        print("CSV:", len(csvs))
        print("JSON:", len(jsons))
        print("NPZ:", len(npzs))

        for path in csvs:
            if any(k in path.name.lower() for k in [
                "metric", "summary", "audit"
            ]):
                audit_csv(path)

        for path in jsons:
            if any(k in path.name.lower() for k in [
                "split", "manifest", "config", "audit"
            ]):
                inspect_json(path)

        for path in npzs[:5]:
            inspect_npz(path)

    print_title("Audit finished")
    print("NOTE: This is a metadata audit.")
    print("Check training and metric source code separately.")
    print("Matching test counts do NOT prove matching test IDs.")


if __name__ == "__main__":
    main()

