# Architecture Overview

- Core library under `src/unsup` exposes:
  - `networks.py`: Hopfield_Network, TAM_Network and field/dynamics primitives.
  - `functions.py`: pattern/dataset generators and J/K computations (Hebb, JK_real, etc.).
  - `dynamics.py`: experiment routines (retrieval, disentangling, K_eff estimators, utilities).
  - `dynamics_single_mode.py`: single-round metrics and grid search helper.
- Experiments under `experiments/` import `unsup.*` and write their own result folders.
- Notebooks live in `notebooks/` and can import `unsup` by ensuring `src/` is on `sys.path`.

Import paths were standardized to `unsup.<module>` and scripts add `src/` to `sys.path` at runtime.
