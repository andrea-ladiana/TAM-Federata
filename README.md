# UNSUP — Tripartite Associative Memory (TAM) Experiments

This repository contains the core modules and experiment scripts for TAM-based unsupervised/federated experiments.

The codebase has been decluttered and reorganized to a `src/` layout with clean imports and separated experiments.

## Quick Start

- Python 3.9+
- Create a virtualenv and install deps:

```
python -m venv .venv
. .venv/Scripts/Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

## Project Layout

- `src/unsup/`: core library
  - `networks.py`: Hopfield/TAM network primitives (NumPy backend)
  - `dynamics.py`: high-level routines and utilities used by experiments
  - `dynamics_single_mode.py`: single-round variant utilities
  - `functions.py`: dataset/pattern generators and linear algebra helpers
- `experiments/`: runnable experiment scripts (formerly `stress_tests/`)
- `notebooks/`: Jupyter notebooks (moved here from root)
- `results/`: archived legacy results (moved from `risultati/`)
- `docs/`: additional technical notes

## Running an Experiment

All experiment scripts add `src/` to `sys.path` and import the library as `unsup.*`.
From the repository root:

```
python experiments/exp_01_partial_archetypes.py
```

Other examples:

```
python experiments/exp_05_byzantine_robust_agg.py
python experiments/exp_06_mixing_drift_k_3.py
```

## Notes on TensorFlow

The core modules run with a lightweight NumPy backend. If TensorFlow is present, some shims may use it, but TF is optional.

## Housekeeping

- Generated outputs under `experiments/**` are ignored by git via `.gitignore`.
- Legacy result assets were moved to `results/archive_risultati/`.
- Caches (`__pycache__`, `*.pyc`) and OS junk files are excluded.

## Development

- Code style: PEP 8, snake_case for files and functions; PascalCase for classes.
- Use a virtualenv and pin versions if needed for reproducibility.

## License

Not specified. Add a LICENSE if distribution is intended.
