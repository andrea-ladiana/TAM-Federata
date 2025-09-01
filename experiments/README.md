# Experiments

This folder contains runnable experiment scripts (formerly under `stress_tests/`).

- Scripts import the core library as `unsup.*` and ensure `src/` is on `sys.path` at runtime.
- Generated outputs (logs, tables, figures) are ignored by git; see `.gitignore`.

Run examples from the repository root:

```
python experiments/exp_01_partial_archetypes.py
python experiments/exp_05_byzantine_robust_agg.py
```

Note: some experiments rely on `unsup.dynamics`. If you see import errors, ensure the module has valid ASCII identifiers (fix any corrupted characters in `src/unsup/dynamics.py`).
