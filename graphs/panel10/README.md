# Panel 10 – DP-PCA / Wishart Trade-off

This directory implements the privacy–utility study described in the protocol for Panel 10. The workflow mirrors the earlier panels:

1. **Data generation** – `dp_pca_tradeoff.py` simulates a federated covariance aggregation pipeline with per-client operator-norm clipping and Gaussian mechanism noise calibrated via the zCDP rule-of-thumb. For each privacy budget `ε_total` it records novelty AUC, `ΔK_eff` (vs the non-DP baseline), eigenvector overlaps with true archetypes, and per-round spectra.
2. **Plotting** – `plot_panel10.py` loads the NPZ file, builds the three curves (AUC, `ΔK_eff`, overlap) against `ε_total`, and renders the spectral inset comparing `ε ∈ {1,4,∞}`.

## Usage

```bash
# Generate Monte-Carlo data (full grid: ε = 0.5…∞, 10 replicates)
python graphs/panel10/dp_pca_tradeoff.py --replicates 10 --output graphs/panel10/output/panel10_dp_pca_data.npz

# Produce the publication figure
python graphs/panel10/plot_panel10.py --input graphs/panel10/output/panel10_dp_pca_data.npz --output graphs/panel10/output/panel10_dp_pca.pdf
```

Flags allow overriding the ε grid (`--epsilons`), novelty probability, or the number of Monte-Carlo replicates. The JSON log (`panel10_dp_pca_log.json`) records the DP noise scales (`σ`) used for each ε, together with the aggregate means of the three metrics.

## Implementation highlights

- **Data model**: `N = 256`, `q = 0.5` (`n = 512`), `L = 50` clients with per-client sample size `m ≈ 10`. Two base spikes (`κ = 1.5`) are always active; a third spike (`κ = 1.2`) appears with probability `p_novel = 0.35` at each round, providing ground-truth labels for the novelty AUC.
- **Clipping & DP**: Each client covariance is projected onto the spectral ball `‖M‖_op ≤ C` (`C = 1`). The server adds a symmetric Gaussian matrix with variance `σ(ε) = 2Δ_F sqrt(T log(1/δ)) / ε`, with `Δ_F = 2C/L`, `T = 50`, `δ = 1e-5`. Setting `ε = ∞` skips the perturbation (baseline).
- **Metrics**:
  - `AUC`: ROC AUC of `K_eff^{DP}` (used as score) against the true novelty labels.
  - `ΔK_eff`: mean absolute deviation between DP and clean effective ranks per round.
  - `Overlap`: average of the best squared overlap between DP outlier eigenvectors and the active archetypes; the curve in the panel reports this mean directly (higher is better).

The spectral inset reuses the stored eigenvalues for the first novel round, illustrating how DP noise at `ε = 1` squeezes the bulk above `(1+η)λ_+` compared to the `ε = ∞` baseline.
