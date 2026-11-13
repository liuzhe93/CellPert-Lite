
import pickle
import sys

path = sys.argv[1]

targets = {
    "ALG1L", "C12ORF45", "C12ORF60", "C14ORF178",
    "C16ORF86", "C17ORF58", "C19ORF25", "C19ORF53",
    "C1ORF109", "C6ORF15", "C7ORF26", "C7ORF50",
    "C9ORF16", "C9ORF78", "CCDC130", "CCDC144NL",
    "DDN", "FRG2", "INTS2", "MBTPS1", "PHB",
    "POLR2B", "SPATA5", "SPATA5L1", "TDGF1",
    "WDR61", "WDR92", "ZNF720",
}

with open(path, "rb") as f:
    obj = pickle.load(f)

if not isinstance(obj, dict):
    raise TypeError(f"Expected dict, got {type(obj)}")

found = {gene: [] for gene in targets}

for key in obj:
    if isinstance(key, str):
        normalized = key.strip().upper()
        if normalized in targets:
            found[normalized].append(key)

for gene in sorted(targets):
    keys = found[gene]
    print(f"{gene}: {keys if keys else 'NOT FOUND'}")

