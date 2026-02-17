# Panel 9 – Correlated noise / deformed-MP

Reproduces the separable-noise stress test that compares naive Marchenko–Pastur thresholding against a deformed edge (and whitening) when the noise covariance is not identity.

## Workflow

1. **Data generation** (`separable_noise_demo.py`)
   - Builds structured noise covariances (`AR(1)` with configurable ρ or block-diagonal variances), injects `K` spikes with strength `κ`, and draws `n` samples with aspect ratio `q = N/n`.
   - Records SCM eigenvalues under three decision rules: classical MP, deformed-MP (estimated via a Monte-Carlo proxy on the known covariance), and whitening with the true scatter.
   - Stores per-replica metrics and thresholds inside `output/panel9_separable_data.npz` and writes a JSON summary.

2. **Plotting** (`plot_panel9.py`)
   - Aggregates conditions by noise setting (`kind`, `ρ`, `q`, `K`) and renders the four-panel layout: bulk spectrum vs thresholds, false-outlier bars, `K_eff` (mean ± IQR) with reference lines, and the whitening ablation.

## Usage

```bash
python graphs/panel9/separable_noise_demo.py --preset quick --replicates 100
python graphs/panel9/plot_panel9.py --input graphs/panel9/output/panel9_separable_data.npz
```

Pass `--preset full` to sweep `N ∈ {256,512}`, `q ∈ {0.3,0.6}`, both AR(1) levels (`ρ ∈ {0.3,0.6}`) plus the block-variance model, and spike grids `K ∈ {2,4}`, `κ ∈ {1.2,1.5}`. As with Panel 8, every list can be overridden manually.

## Notes

- **Noise helpers** – Structured covariances are assembled here, while the threshold logic reuses `src/unsup/spectral.deformed_mp.approximate_deformed_edge` (Monte-Carlo proxy) and `marchenko_pastur_edge`.
- **Whitening** – Since the experiments are synthetic, we whiten with the *true* noise covariance. This mirrors the “ideal” scenario described in the protocol and makes the optional Panel 9D easy to interpret.
- **Metrics tracked per replica**
  - `false_outliers`: `max(0, K_eff - K_true)` for each rule;
  - `bias`: `(λ_max(bulk) - τ)` where `τ` is the corresponding threshold;
  - `bulk_edge`: `(K_true + 1)`-th eigenvalue for reference;
  - `lambda_max`: top eigenvalue post-thresholding;
  - `keff`: total number of eigenvalues above the threshold.
- **Cushion** – Same `(1+η)` guard band (default `η=0.02`) is applied to all thresholds; tweak `--cushion` to align with your finite-`N` calibration.

The plotting script keeps the Okabe-Ito palette and outputs both PDF and PNG assets under `graphs/panel9/output/` ready to be dropped into the final manuscript.
