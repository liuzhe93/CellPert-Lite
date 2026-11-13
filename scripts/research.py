#!/usr/bin/env python3
"""Research workflow dispatcher; forwards arguments to audited legacy scripts."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "k562-unseen": "step10A_unseen_perturbation_transfer.py",
    "external-prepare": "step11B_2_prepare_hepg2_jurkat.py",
    "external-transfer": "step11B_4_frozen_external_transfer.py",
    "external-analysis": "step11B_5_external_transfer_analysis.py",
    "response-diagnostics": "step11B_6_response_strength_and_biological_shift.py",
    "enrichment": "step12B_functional_enrichment_fixed.py",
    "figure2": "step13G_figure2_k562_publication.py",
    "figure3": "step13H_figure3_rpe1_publication.py",
    "figure4": "step13I_figure4_external_publication.py",
    "figure5": "step13J_figure5_paired_ablation_publication.py",
    "figure6": "step13K_figure6_response_diagnostics.py",
    "figure7": "step13L_figure7_representative_publication.py",
    "figure8": "step13M_figure8_enrichment_publication_v3.py",
}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=sorted(COMMANDS))
    p.add_argument('--dry-run',action='store_true',help='Print underlying command without executing')
    p.add_argument('--script-help',action='store_true',help='Show the underlying script help')
    args,extra=p.parse_known_args(argv)
    script=ROOT/'scripts'/COMMANDS[args.command]
    if not script.is_file():p.error(f'Missing script: {script}')
    command=[sys.executable,str(script),*(['--help'] if args.script_help else extra)]
    if args.dry_run:
        print(' '.join(command));return 0
    return subprocess.call(command,cwd=ROOT)

if __name__=='__main__':raise SystemExit(main())
