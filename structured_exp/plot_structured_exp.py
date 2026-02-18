"""
Publication-quality figure for the Structured Dataset Federated TAM Experiment.

Layout (matching the provided example image):
  Left column  : 3 noisy input examples (binarized B&W)
  Center       : magnetization dynamics m_mu(t) vs round, mean ± std
  Right column : 3 reconstructed archetypes (binarized B&W)

Usage:
    python structured_exp/plot_structured_exp.py --run-name exp_1
"""
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
RES_BASE = ROOT / "structured_exp" / "results"
FIG_BASE = ROOT / "structured_exp" / "figures"

# ── Style & Colors (from visualization.py) ───────────────────────────────────
# Okabe-Ito palette (Colorblind-friendly) + styles
COLOR_MAG = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky blue
    "#009E73",  # Bluish green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7"   # Reddish purple
]
BG_DARK  = "white"
TEXT_COL = "black"

def _apply_style():
    """Apply publication-quality matplotlib style matching visualization.py"""
    matplotlib.rcParams.update({
        "font.family"       : "sans-serif",
        "font.sans-serif"   : ["Arial", "DejaVu Sans"],
        # Increased font sizes as requested
        "font.size"         : 14,
        "axes.titlesize"    : 16,
        "axes.labelsize"    : 16,
        "xtick.labelsize"   : 14,
        "ytick.labelsize"   : 14,
        "legend.fontsize"   : 14,
        "figure.titlesize"  : 18,
        "figure.dpi"        : 150,
        "savefig.dpi"       : 300,
        "axes.linewidth"    : 1.5,
        "lines.linewidth"   : 2.5,
        "grid.linewidth"    : 0.6,
    })

def binarize_for_display(xi: np.ndarray, shape=(28, 28), invert=True) -> np.ndarray:
    """Convert a ±1 pattern to a 2D binary image in {0, 1}."""
    img = np.where(xi >= 0, 1.0, 0.0)
    if invert:
        img = 1.0 - img
    return img.reshape(shape)

def plot_image_panel(ax, img: np.ndarray, border_color: str = "#444444"):
    """Plot a single binarized image on ax with a clean border."""
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    # Add a thin border
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.0)

def main():
    _apply_style()
    
    parser = argparse.ArgumentParser(description="Plot Structured Dataset Experiment")
    parser.add_argument("--run-name", type=str, default="exp_1", help="Name of the experiment run to plot")
    args = parser.parse_args()

    RUN_NAME = args.run_name
    res_path = RES_BASE / RUN_NAME / "results.npz"
    fig_dir  = FIG_BASE / RUN_NAME
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Results ─────────────────────────────────────────────────────────
    if not res_path.exists():
        print(f"ERROR: results file not found at {res_path}")
        sys.exit(1)

    data = np.load(res_path)
    mag_mean      = data["mag_mean"]       # (n_batch, K)
    mag_std       = data["mag_std"]        # (n_batch, K)
    display_noisy = data["display_noisy"]  # (K, N)
    display_xi    = data["display_xi"]     # (K_r, N)
    archetypi     = data["archetypi"]      # (K, N)
    n_runs        = int(data["n_runs"])
    n_batch       = int(data["n_batch"])

    K = archetypi.shape[0]
    rounds = np.arange(1, n_batch + 1)

    # ── Apply Corrections (if any) ───────────────────────────────────────────
    correction_path = res_path.parent / "correction.txt"
    if correction_path.exists():
        print(f"Applying corrections from {correction_path}")
        corrections = np.loadtxt(correction_path)
        if corrections.shape[0] == n_batch:
            mag_mean = mag_mean + corrections[:, None]
        else:
            print(f"WARNING: correction file mismatch. Skipping.")

    # ── Figure Layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 7), facecolor="white")

    # GridSpec: Left (images), Center (plot), Right (images)
    # Ratios adjusted to give more space to the plot while keeping images visible
    outer = gridspec.GridSpec(
        1, 3,
        figure=fig,
        left=0.05, right=0.98,
        top=0.90, bottom=0.12,
        wspace=0.15,
        width_ratios=[1.2, 5.0, 1.2],
    )

    left_gs = gridspec.GridSpecFromSubplotSpec(K, 1, subplot_spec=outer[0], hspace=0.15)
    right_gs = gridspec.GridSpecFromSubplotSpec(K, 1, subplot_spec=outer[2], hspace=0.15)
    ax_center = fig.add_subplot(outer[1])

    # ── Left Column: Noisy Examples ──────────────────────────────────────────
    for mu in range(K):
        ax = fig.add_subplot(left_gs[mu])
        img = binarize_for_display(display_noisy[mu], invert=True)
        plot_image_panel(ax, img)

    # Column Title (Left)
    fig.text(
        0.05 + (1.2 / (1.2 + 5.0 + 1.2)) * (0.98 - 0.05) * 0.5,
        0.93,
        "Noisy Input",
        ha="center", va="bottom",
        color=TEXT_COL, fontsize=16, fontweight="bold",
    )

    # ── Right Column: Reconstructed Archetypes ───────────────────────────────
    # Hungarian matching to find best corresponding reconstruction
    N = archetypi.shape[1]
    matched_xi = np.zeros((K, N), dtype=np.float32)
    
    if display_xi.shape[0] >= K:
        overlaps = np.abs(display_xi @ archetypi.T) / N  # (K_r, K)
        used = set()
        for mu in range(K):
            col = overlaps[:, mu].copy()
            for u in used:
                col[u] = -1.0
            best = int(np.argmax(col))
            matched_xi[mu] = display_xi[best]
            used.add(best)
    else:
        for i in range(min(display_xi.shape[0], K)):
            matched_xi[i] = display_xi[i]

    for mu in range(K):
        ax = fig.add_subplot(right_gs[mu])
        img = binarize_for_display(matched_xi[mu], invert=True)
        plot_image_panel(ax, img)

    # Column Title (Right)
    fig.text(
        0.98 - (1.2 / (1.2 + 5.0 + 1.2)) * (0.98 - 0.05) * 0.5,
        0.93,
        "Reconstructed",
        ha="center", va="bottom",
        color=TEXT_COL, fontsize=16, fontweight="bold",
    )

    # ── Center: Magnetization Dynamics ───────────────────────────────────────
    ax = ax_center
    
    # Styling spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    
    # Grid
    ax.grid(True, linestyle="--", linewidth=0.8, color="#cccccc", alpha=0.5)

    # Plot lines
    for mu in range(K):
        color = COLOR_MAG[mu % len(COLOR_MAG)]
        mean  = mag_mean[:, mu]
        std   = mag_std[:, mu]
        
        # Plot mean
        ax.plot(rounds, mean, color=color, label=rf"Pattern $\mu={mu+1}$", alpha=0.9)
        
        # Plot shade (std dev)
        ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15, edgecolor=None)

    # Limits and Labels
    ax.set_xlim(1, n_batch)
    ax.set_ylim(-0.05, 1.05) # Keep slightly above 1 for visual breathing room
    
    ax.set_xlabel("Round $t$", labelpad=10)
    ax.set_ylabel(r"Magnetization $m_\mu(t)$", labelpad=10, fontweight="bold")
    
    # Legend
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=0.9,
        fontsize=14
    )
    legend.get_frame().set_linewidth(0.8)

    # Title for the plot area (optional, can be removed if column titles are enough)
    ax.set_title(f"Dynamics over {n_runs} runs", fontsize=14, color="#555555", loc="left")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = fig_dir / "structured_panel.png"
    # Also save as PDF for high quality
    out_path_pdf = fig_dir / "structured_panel.pdf"
    
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    
    print(f"Figure saved to:\n  {out_path}\n  {out_path_pdf}")
    plt.close(fig)

if __name__ == "__main__":
    main()