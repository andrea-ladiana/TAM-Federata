# Panel 11 – Fake Spikes / Byzantine Defenses

This folder implements the multi-round Byzantine stress test with persistence and cross-shard filtering.

## Workflow

1. **Data generation** – `byzantine_persistence_demo.py` simulates `L = 50` clients split into `s = 5` shards, each contributing a covariance estimate at every round. Adversarial clients inject rank-1 spikes with configurable amplitude `α ∈ {0.1, 0.2, 0.3, 0.5}` and population (`c ∈ {0, 1, 3, 5}`) using either stationary or wandering directions. The script logs:
   - ROC curves (baseline vs persistence vs persistence+cross-shard);
   - FPR at TPR = 0.9 per `(c, α)` scenario;
   - `ΔK_eff` distributions for each defense;
   - cross-shard vote traces `S_t` for a representative seed.
2. **Plotting** – `plot_panel11.py` renders the four-panel layout (ROC, operating-point bars, `ΔK_eff` boxplots, and a persistence heatmap).

## Usage

```bash
# Generate Monte-Carlo data (default: 8 replicates, T = 80 rounds each)
python graphs/panel11/byzantine_persistence_demo.py --replicates 8 --output graphs/panel11/output/panel11_byzantine_data.npz

# Produce the figure
python graphs/panel11/plot_panel11.py --input graphs/panel11/output/panel11_byzantine_data.npz --output graphs/panel11/output/panel11_byzantine.pdf
```

The accompanying JSON log (`panel11_byzantine_log.json`) summarises mean `ΔK_eff` per defense and the ROC AUCs.

## Implementation notes

- **Ground truth**: Two genuine spikes (`κ = 1.5`) remain present at all times; fake injections add a third rank-1 component but should be rejected by the defenses.
- **Persistence filter**: a new outlier is accepted only after `m = 3` consecutive rounds with `λ > (1+η)λ_+` and directional alignment ≥ `η = 0.9`. The cross-shard module counts shard-level eigenpairs that align with the global candidate (`η_s = 0.8`) and requires a majority `S_t ≥ θ = 0.6`.
- **Metrics**:
  - `score_baseline = max(0, λ_max - τ)`;
  - `score_persistence = score_baseline` if the persistence rule holds, else 0;
  - `score_combo = score_baseline * S_t` when both persistence and cross-shard votes pass, else 0.
  These continuous scores feed the ROC (Panel 11A) and the FPR@TPR=0.9 bars (Panel 11B). `ΔK_eff` compares the retained spikes against the true rank (2) to quantify false positives and missed detections (Panel 11C). Panel 11D visualises the vote trace `S_t`, with dashed lines over rounds where an attack is present.
