# v1.1 Research Release (candidate)

- Bumps Python package version to 1.1.0 while preserving the v1.0 baseline CLI.
- Adds optional `research` dependencies for PyTorch-based experiments.
- Packages the supplied Step-series research source scripts, excluding backups.
- Adds `scripts/research.py` workflow dispatcher and `scripts/preflight.py` input audit.
- Adds explicit study input contracts, workflow ordering, figure map and scientific limitations.
- Adds release smoke tests.

**Release status:** candidate for server-side validation; not automatically pushed to GitHub, not a claim of full experiment reproduction. Before tagging, run tests and Step10A/Step11B with the actual inputs, review licenses and data provenance, and verify figure outputs.
