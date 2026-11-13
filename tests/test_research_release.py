import subprocess
import sys
from pathlib import Path
from cellpert import __version__

ROOT = Path(__file__).resolve().parents[1]

def test_version():
    assert __version__ == '1.1.0'

def test_dispatcher_dry_run():
    r = subprocess.run([sys.executable, str(ROOT/'scripts/research.py'), 'k562-unseen', '--dry-run'],
                       capture_output=True, text=True, check=True)
    assert 'step10A_unseen_perturbation_transfer.py' in r.stdout

def test_figure_dispatcher_dry_run():
    r = subprocess.run([sys.executable, str(ROOT/'scripts/research.py'), 'figure8', '--dry-run'],
                       capture_output=True, text=True, check=True)
    assert 'step13M_figure8_enrichment_publication_v3.py' in r.stdout
