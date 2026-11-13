#!/usr/bin/env python3
"""Audit a CellPert-Lite checkout before preparing a reproducible release."""
import argparse
import ast
import csv
import json
from pathlib import Path

CORE = ('cellpert/models.py', 'cellpert/data.py', 'cellpert/metrics.py',
        'cellpert/cli.py', 'cellpert/train.py', 'cellpert/predict.py')
PATTERNS = ('step09B_', 'step09C_', 'step10A_', 'step11B_4_',
            'step13', 'step12B_')

def audit(root: Path):
    rows = []
    for p in sorted((root / 'scripts').glob('*.py')):
        text = p.read_text(encoding='utf-8')
        try:
            ast.parse(text, filename=str(p))
            status = 'ok'
        except SyntaxError as e:
            status = f'syntax_error:{e.lineno}'
        rows.append({'path': p.relative_to(root).as_posix(),
                     'bytes': p.stat().st_size,
                     'syntax': status,
                     'paper_candidate': any(p.name.startswith(x) for x in PATTERNS)})
    core = {name: (root / name).is_file() for name in CORE}
    return {'root': str(root.resolve()), 'core': core,
            'scripts': rows,
            'n_scripts': len(rows),
            'n_syntax_errors': sum(r['syntax'] != 'ok' for r in rows)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--outdir', type=Path, default=Path('release_audit'))
    args = ap.parse_args()
    root = args.root.resolve()
    if not (root / 'cellpert').is_dir():
        ap.error(f'Not a CellPert-Lite project root: {root}')
    result = audit(root)
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'inventory.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out / 'script_inventory.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'bytes', 'syntax', 'paper_candidate'])
        writer.writeheader()
        writer.writerows(result['scripts'])
    print(f"Scripts: {result['n_scripts']} | Syntax errors: {result['n_syntax_errors']}")
    print('Core files:', result['core'])
    print('Audit:', out)

if __name__ == '__main__':
    main()
