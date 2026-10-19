#!/usr/bin/env bash
set -euo pipefail
pip install -e ".[dev]"
pytest -q
cellpert demo --outdir results/demo --seed 42
