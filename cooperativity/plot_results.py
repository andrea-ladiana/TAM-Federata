
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# --- Style from client_aware_w/visualization.py ---

def _apply_rc():
    mpl.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 1.5,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "patch.linewidth": 1.2,
    })

# Colour palette (Okabe-Ito, colorblind-friendly)
COLOR_ATTACKER = "#D55E00"   # Vermillion
COLOR_GOOD_MEAN = "#0072B2"  # Blue
COLOR_GOOD_FILL = "#56B4E9"  # Sky blue (envelope)
COLOR_RETRIEVAL = "#009E73"  # Bluish green
# We use COLOR_RETRIEVAL for the main overlap curve

def plot_cooperativity(
    data_path: Path,
    out_path: Path
):
    _apply_rc()
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    data = np.load(data_path)
    # data keys: 'High', 'Hetero', 'Low', 't_steps'
    # Shape of results: (n_seeds, T)
    
    t_steps = list(range(12)) # Hardcoded or infer from data
    if 't_steps' in data:
         # In the run_experiment, run_single_seed returns a list of length hp.n_batch.
         # But in main(), we save as np.array(regime_retrievals).
         # However, t_steps was saved as np.arange(T), so it might be an array inside the npz.
         pass
         
    regimes = [
        ("High (Homogeneous)", data["High"]),
        ("Heterogeneous", data["Hetero"]),
        ("Low (Homogeneous)", data["Low"]),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    
    # Calculate global min/max for y-axis sharing consistency?
    # Actually, let's keep y-axis [0, 1] for overlap.
    
    for ax, (name, matrix) in zip(axes, regimes):
        # matrix: (n_seeds, T)
        n_seeds, T = matrix.shape
        steps = np.arange(T)
        
        mean_curve = matrix.mean(axis=0)
        std_curve = matrix.std(axis=0, ddof=1)
        se_curve = std_curve / np.sqrt(n_seeds)
        
        # Plot
        ax.fill_between(
            steps,
            mean_curve - se_curve,
            mean_curve + se_curve,
            color=COLOR_RETRIEVAL,
            alpha=0.3
        )
        ax.plot(steps, mean_curve, color=COLOR_RETRIEVAL, lw=2.5, marker='o', markersize=4, label="Max Overlap")
        
        # Dress axis
        ax.set_title(name, fontsize=16, fontweight="bold")
        ax.set_xlabel("Round $t$")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(steps)
        # Show fewer x-ticks if crowded? 12 is fine.
        ax.grid(True, ls="--", alpha=0.3)
        
        # Remove top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Add seed count annotation
        ax.text(
            0.05, 0.05,
            f"$S={n_seeds}$ seeds",
            transform=ax.transAxes,
            fontsize=10,
            color="#555555"
        )

    axes[0].set_ylabel("Max Overlap (Magnetization)")
    
    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_cooperativity(
        Path("cooperativity/data/results.npz"),
        Path("cooperativity/cooperativity_plot.png")
    )
