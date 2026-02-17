# Panel 8 – Heavy-tail stress test

This folder contains the scripts required to reproduce the heavy-tail stress test described in the UNSUP panel protocol. The workflow mirrors the existing `panel2/` structure:

1. **Data generation** – `heavy_tail_demo.py` samples spiked elliptical datasets with Student-*t* radial components, runs both the sample covariance (SCM) and the Tyler shape estimator, and stores the per-replica metrics (K_eff, false outliers, Frobenius error, bulk edge) in `output/panel8_heavy_tail_data.npz`. A JSON log with aggregate statistics is written alongside the NPZ.
2. **Plotting** – `plot_panel8.py` loads the NPZ and renders the boxplots plus the diagnostic histogram required for the panel.

## Quick start

```bash
python graphs/panel8/heavy_tail_demo.py --preset quick --replicates 100
python graphs/panel8/plot_panel8.py --input graphs/panel8/output/panel8_heavy_tail_data.npz
```

Use `--preset full` to sweep the entire grid described in the protocol (`N ∈ {256,512}`, `q ∈ {0.3,0.6}`, `ν ∈ {3,5,8,∞}`, `K ∈ {2,4}`, `κ ∈ {1.2,1.5}`). All knobs (lists of `N`, `q`, `nu`, `K`, `kappa`) can also be overridden explicitly.

## Implementation notes

- **Robust scatter** – The Tyler estimator lives in `src/unsup/spectral/robust.py` (`tyler_shape_matrix`, `normalize_diagonal`). The panel script simply imports it, so other experiments can reuse the same implementation.
- **Thresholding** – The MP edge is computed via `marchenko_pastur_edge(q)` and futher adjusted with a small cushion (default 2%) to mimic the finite-`N` guard band mentioned in the protocol. All spectra are diagonal-normalised before thresholding to keep the variance at one per marginal.
- **Metrics** – For each replica we store:
  - `keff`: count of eigenvalues above the cushioned MP edge;
  - `false_outliers`: `max(0, keff - K_true)`;
  - `fro`: relative Frobenius error between the estimator and the ground-truth scatter (after diagonal normalisation);
  - `bulk_edge`: proxy for `λ_max(bulk)` given by the `(K_true + 1)`-th eigenvalue;
  - `lambda_max`: top eigenvalue (useful for debugging).
- **Radial sampling** – Heavy tails come from Student-*t* radii (`r_i = sqrt(ν / χ^2_ν)`); `ν = ∞` falls back to Gaussian noise.

The plotting script groups statistics by `ν` (boxplots) and aggregates all runs for the histogram of `λ_max(bulk)` vs the MP edge band. Styling (Okabe-Ito palette, grid lines, etc.) matches `panel2` for visual consistency.
