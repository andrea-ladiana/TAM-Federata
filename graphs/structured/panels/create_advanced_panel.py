"""
Advanced visualization panel for structured federated TAM experiment.

Layout (adapted from MNIST panel):
- (A) Magnetization heatmap: |m|_{k,t} (K×T) with exposure contours
- (B) Evolution strips: 3 archetypes showing init→mid1→mid2→final→true
      with #flips and energy changes
- (C) Confusion matrices: beginning vs end (same scale)
- (D) Basin of attraction (UMAP): initial states (pale) → final (full), convex hulls
- (E) Metrics evolution: retrieval, FRO, K_eff over rounds (multi-panel)

This provides comprehensive visual analysis of the federated TAM pipeline.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from pathlib import Path
import sys

# Add paths
root_path = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from graphs.structured.verification.hopfield_retrieval_test import hopfield_dynamics
from unsup.functions import Hebb_J


def plot_magnetization_heatmap(ax, archetypi, xi_retrieved_all, exposure_percentile=80):
    """
    Plot magnetization heatmap |m|_{k,t} over rounds.
    
    Parameters
    ----------
    ax : matplotlib axis
    archetypi : np.ndarray
        True archetypes (K, N)
    xi_retrieved_all : np.ndarray
        Retrieved patterns for all rounds (object array of varying shapes)
    exposure_percentile : float
        Percentile for highlighting high-exposure cells
    """
    K, N = archetypi.shape
    n_rounds = len(xi_retrieved_all)
    
    # Compute magnetizations per round
    magnetizations = np.zeros((K, n_rounds))
    
    for t in range(n_rounds):
        xi_t = xi_retrieved_all[t]
        if xi_t.shape[0] == 0:
            continue
        # Overlap with true archetypes
        overlaps = (xi_t @ archetypi.T) / float(N)  # (K_ret, K)
        # Max absolute overlap for each true archetype
        mag = np.abs(overlaps).max(axis=0) if overlaps.size > 0 else np.zeros(K)
        magnetizations[:, t] = mag
    
    # Plot heatmap
    im = ax.imshow(magnetizations, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    
    # Add contours for high-magnetization cells
    threshold = np.percentile(magnetizations, exposure_percentile)
    for k in range(K):
        for t in range(n_rounds):
            if magnetizations[k, t] >= threshold:
                rect = Rectangle((t - 0.5, k - 0.5), 1, 1,
                                fill=False, edgecolor='white', linewidth=2)
                ax.add_patch(rect)
    
    ax.set_xlabel('Federated round', fontsize=9)
    ax.set_ylabel('Archetype', fontsize=9)
    ax.set_title(f'Magnetization Evolution (top {100-exposure_percentile}% contoured)', fontsize=10)
    ax.set_xticks(range(0, n_rounds, max(1, n_rounds // 6)))
    ax.set_yticks(range(K))
    ax.set_yticklabels([f'{k+1}' for k in range(K)])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('|m|', fontsize=8)


def plot_evolution_strips(ax, trajectories, archetypi, side=28):
    """
    Plot evolution strips: init → mid1 → mid2 → final → true.
    
    Parameters
    ----------
    ax : matplotlib axis
    trajectories : dict
        Sample trajectories from Hopfield test (k → trajectory)
    archetypi : np.ndarray
        True archetypes (K, N)
    side : int
        Image side length
    """
    K = archetypi.shape[0]
    
    if not trajectories or len(trajectories) == 0:
        ax.text(0.5, 0.5, 'No trajectories available',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return
    
    ax.axis('off')
    ax.set_title('Hopfield Evolution Trajectories', fontsize=10, pad=10)
    
    # Show up to K archetypes
    archetype_keys = sorted(list(trajectories.keys()))[:K]
    n_show = len(archetype_keys)
    
    # 5 snapshots per archetype: init, mid1, mid2, final, true
    n_snapshots = 5
    
    for i, k in enumerate(archetype_keys):
        traj_data = trajectories[k]
        traj = traj_data['trajectory']
        true_k = traj_data['true_k']
        
        # Select snapshots
        n_steps = len(traj)
        idx_init = 0
        idx_mid1 = n_steps // 3
        idx_mid2 = 2 * n_steps // 3
        idx_final = -1
        
        snapshots = [
            traj[idx_init],
            traj[idx_mid1],
            traj[idx_mid2],
            traj[idx_final],
            archetypi[true_k]
        ]
        
        labels = ['Init', 'Mid1', 'Mid2', 'Final', 'True']
        
        # Compute flips and energy for each step
        def count_flips(s1, s2):
            return np.sum(s1 != s2)
        
        # Energy: E = -0.5 * s^T J s (but we don't have J here, use overlap as proxy)
        overlaps = [(s @ archetypi[true_k]) / archetypi.shape[1] for s in snapshots]
        
        for j, (snap, label, ovl) in enumerate(zip(snapshots, labels, overlaps)):
            left = j / n_snapshots + 0.01
            bottom = 1.0 - (i + 1) / n_show
            width = 0.18
            height = (1.0 / n_show) - 0.05
            
            sub_ax = ax.inset_axes([left, bottom, width, height])
            img = snap.reshape(side, side)
            sub_ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
            sub_ax.axis('off')
            
            # Add title with stats
            if j < 4:  # Not the true archetype
                next_snap = snapshots[j + 1] if j < 3 else snapshots[j]
                flips = count_flips(snap, next_snap)
                title_text = f'{label}\nm={ovl:.2f}'
                if j < 3:
                    title_text += f'\nΔ={flips}'
            else:
                title_text = f'{label}\n(k={true_k+1})'
            
            sub_ax.set_title(title_text, fontsize=6)


def plot_confusion_matrices(ax, confusion_init, confusion_final, K):
    """
    Plot initial and final confusion matrices side-by-side.
    
    Parameters
    ----------
    ax : matplotlib axis
    confusion_init : np.ndarray or None
        Initial confusion matrix (K, K)
    confusion_final : np.ndarray
        Final confusion matrix (K, K)
    K : int
        Number of archetypes
    """
    ax.axis('off')
    
    # If no initial confusion, use identity-like (perfect retrieval assumption)
    if confusion_init is None:
        confusion_init = np.eye(K) * 3  # Assuming 3 examples per archetype
    
    # Normalize to [0, 1]
    vmax = max(confusion_init.max(), confusion_final.max())
    
    # Left: Initial
    ax_left = ax.inset_axes([0.05, 0.15, 0.4, 0.7])
    im_left = ax_left.imshow(confusion_init, cmap='Blues', vmin=0, vmax=vmax, aspect='auto')
    ax_left.set_xlabel('Target', fontsize=8)
    ax_left.set_ylabel('True', fontsize=8)
    ax_left.set_title('Initial (t=0)', fontsize=9)
    ax_left.set_xticks(range(K))
    ax_left.set_yticks(range(K))
    ax_left.set_xticklabels([f'{k+1}' for k in range(K)], fontsize=7)
    ax_left.set_yticklabels([f'{k+1}' for k in range(K)], fontsize=7)
    
    # Right: Final
    ax_right = ax.inset_axes([0.55, 0.15, 0.4, 0.7])
    im_right = ax_right.imshow(confusion_final, cmap='Blues', vmin=0, vmax=vmax, aspect='auto')
    ax_right.set_xlabel('Target', fontsize=8)
    ax_right.set_ylabel('True', fontsize=8)
    ax_right.set_title('Final (t=T)', fontsize=9)
    ax_right.set_xticks(range(K))
    ax_right.set_yticks(range(K))
    ax_right.set_xticklabels([f'{k+1}' for k in range(K)], fontsize=7)
    ax_right.set_yticklabels([f'{k+1}' for k in range(K)], fontsize=7)
    
    # Add values
    for i in range(K):
        for j in range(K):
            ax_left.text(j, i, f'{int(confusion_init[i, j])}',
                        ha='center', va='center', fontsize=7,
                        color='white' if confusion_init[i, j] > vmax/2 else 'black')
            ax_right.text(j, i, f'{int(confusion_final[i, j])}',
                         ha='center', va='center', fontsize=7,
                         color='white' if confusion_final[i, j] > vmax/2 else 'black')
    
    # Shared colorbar
    cbar_ax = ax.inset_axes([0.45, 0.15, 0.02, 0.7])
    plt.colorbar(im_right, cax=cbar_ax)
    
    # Main title
    ax.text(0.5, 0.95, 'Confusion Matrices: Hopfield Convergence',
            ha='center', va='top', transform=ax.transAxes, fontsize=10)


def plot_basin_umap(ax, examples, labels, xi_retrieved, archetypi, J_hopfield):
    """
    Plot basin of attraction using UMAP dimensionality reduction.
    
    Parameters
    ----------
    ax : matplotlib axis
    examples : np.ndarray
        Noisy examples (L, M_c, N)
    labels : np.ndarray
        Labels (L, M_c)
    xi_retrieved : np.ndarray
        Retrieved archetypes (K, N)
    archetypi : np.ndarray
        True archetypes (K, N)
    J_hopfield : np.ndarray
        Hopfield matrix (N, N)
    """
    try:
        from umap import UMAP
    except ImportError:
        ax.text(0.5, 0.5, 'UMAP not available\n(pip install umap-learn)',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return
    
    K, N = archetypi.shape
    
    # Flatten examples
    examples_flat = examples.reshape(-1, N)
    labels_flat = labels.flatten()
    
    # Run Hopfield dynamics
    finals = []
    for i in range(len(examples_flat)):
        s_final, _ = hopfield_dynamics(examples_flat[i], J_hopfield, n_steps=50)
        finals.append(s_final)
    finals = np.array(finals)
    
    # UMAP on combined (initial + final + archetypes)
    all_states = np.vstack([examples_flat, finals, archetypi])
    
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
    embedding = reducer.fit_transform(all_states)
    
    n_ex = len(examples_flat)
    emb_init = embedding[:n_ex]
    emb_final = embedding[n_ex:2*n_ex]
    emb_arch = embedding[2*n_ex:]
    
    # Colors for each archetype
    colors = plt.cm.tab10(np.arange(K))
    
    # Plot initial states (pale)
    for k in range(K):
        mask = labels_flat == k
        if mask.sum() > 0:
            ax.scatter(emb_init[mask, 0], emb_init[mask, 1],
                      c=[colors[k]], alpha=0.3, s=30, marker='o',
                      label=f'Init k={k+1}' if k == 0 else '')
    
    # Plot final states (full)
    for k in range(K):
        mask = labels_flat == k
        if mask.sum() > 0:
            ax.scatter(emb_final[mask, 0], emb_final[mask, 1],
                      c=[colors[k]], alpha=0.8, s=50, marker='s',
                      edgecolors='black', linewidth=0.5)
    
    # Plot archetypes (stars)
    ax.scatter(emb_arch[:, 0], emb_arch[:, 1],
              c=colors[:K], s=200, marker='*',
              edgecolors='black', linewidth=1.5, label='Archetypes')
    
    # Convex hulls (optional, requires scipy)
    try:
        from scipy.spatial import ConvexHull
        for k in range(K):
            mask = labels_flat == k
            if mask.sum() >= 3:  # Need at least 3 points
                points = emb_final[mask]
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1],
                           color=colors[k], alpha=0.4, linewidth=1)
    except:
        pass
    
    ax.set_xlabel('UMAP 1', fontsize=9)
    ax.set_ylabel('UMAP 2', fontsize=9)
    ax.set_title('Basin of Attraction (UMAP)', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)


def plot_metrics_evolution(ax, retrieval, fro, keff, K):
    """
    Plot evolution of retrieval, FRO, and K_eff over rounds.
    
    Parameters
    ----------
    ax : matplotlib axis
    retrieval : np.ndarray
        Retrieval per round
    fro : np.ndarray
        FRO per round
    keff : np.ndarray
        K_eff per round
    K : int
        Target number of archetypes
    """
    ax.axis('off')
    
    n_rounds = len(retrieval)
    rounds = np.arange(1, n_rounds + 1)
    
    # Three sub-panels
    ax1 = ax.inset_axes([0.05, 0.55, 0.9, 0.35])
    ax2 = ax.inset_axes([0.05, 0.15, 0.43, 0.35])
    ax3 = ax.inset_axes([0.52, 0.15, 0.43, 0.35])
    
    # Retrieval
    ax1.plot(rounds, retrieval, marker='o', linewidth=2, markersize=4,
            color='darkgreen', label='Retrieval')
    ax1.axhline(1.0, linestyle='--', color='gray', alpha=0.5)
    ax1.set_ylabel('Retrieval', fontsize=9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(range(1, n_rounds + 1, max(1, n_rounds // 6)))
    
    # FRO
    ax2.plot(rounds, fro, marker='s', linewidth=2, markersize=4,
            color='darkred', label='FRO')
    ax2.axhline(0.0, linestyle='--', color='gray', alpha=0.5)
    ax2.set_xlabel('Round', fontsize=9)
    ax2.set_ylabel('FRO distance', fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(range(1, n_rounds + 1, max(1, n_rounds // 6)))
    
    # K_eff
    ax3.plot(rounds, keff, marker='^', linewidth=2, markersize=4,
            color='darkorange', label='K_eff')
    ax3.axhline(K, linestyle='--', color='gray', alpha=0.5, label=f'Target (K={K})')
    ax3.set_xlabel('Round', fontsize=9)
    ax3.set_ylabel('K_eff', fontsize=9)
    ax3.set_ylim(0, K + 1)
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=7)
    ax3.set_xticks(range(1, n_rounds + 1, max(1, n_rounds // 6)))
    
    # Main title
    ax.text(0.5, 0.95, 'Metrics Evolution Over Rounds',
            ha='center', va='top', transform=ax.transAxes, fontsize=10, fontweight='bold')


def create_advanced_panel(
    archetypi: np.ndarray,
    results_npz_path: str,
    hopfield_npz_path: str,
    out_path: str = "structured_panel_ADVANCED.pdf"
):
    """
    Create advanced 2×3 panel with comprehensive analysis.
    
    Layout:
    Row 1: (A) Magnetization heatmap | (B) Evolution strips
    Row 2: (C) Confusion matrices    | (D) Basin UMAP
    Row 3: (E) Metrics evolution (spans 2 columns)
    """
    # Load data
    results = np.load(results_npz_path, allow_pickle=True)
    hopfield = np.load(hopfield_npz_path, allow_pickle=True)
    
    K, N = archetypi.shape
    side = int(np.sqrt(N))
    
    # Extract xi_retrieved_all
    xi_retrieved_all = results.get('xi_retrieved_all', None)
    
    # Create figure
    fig = plt.figure(figsize=(16, 18))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8],
                  hspace=0.25, wspace=0.25)
    
    # (A) Magnetization heatmap
    ax_mag = fig.add_subplot(gs[0, 0])
    if xi_retrieved_all is not None:
        plot_magnetization_heatmap(ax_mag, archetypi, xi_retrieved_all, exposure_percentile=80)
    else:
        ax_mag.text(0.5, 0.5, 'Magnetization data not available',
                   ha='center', va='center', transform=ax_mag.transAxes)
        ax_mag.axis('off')
    
    # (B) Evolution strips
    ax_evol = fig.add_subplot(gs[0, 1])
    sample_trajectories = hopfield['sample_trajectories'].item()
    plot_evolution_strips(ax_evol, sample_trajectories, archetypi, side=side)
    
    # (C) Confusion matrices
    ax_conf = fig.add_subplot(gs[1, 0])
    confusion_final = hopfield['confusion_matrix']
    plot_confusion_matrices(ax_conf, None, confusion_final, K)
    
    # (D) Basin UMAP
    ax_basin = fig.add_subplot(gs[1, 1])
    examples_final = results['examples_final']
    labels_final = results['labels_final']
    xi_retrieved = results['xi_retrieved_final']
    J_hopfield = hopfield['J_hopfield']
    plot_basin_umap(ax_basin, examples_final, labels_final,
                    xi_retrieved, archetypi, J_hopfield)
    
    # (E) Metrics evolution (spans both columns)
    ax_metrics = fig.add_subplot(gs[2, :])
    plot_metrics_evolution(ax_metrics, results['retrieval'],
                          results['fro'], results['keff'], K)
    
    # Super title
    fig.suptitle('Structured Federated TAM: Advanced Analysis Panel (K=3)',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nAdvanced panel saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    # Load data
    results_path = Path("out_structured/federated_run/results.npz")
    hopfield_path = Path("out_structured/federated_run/hopfield_verification.npz")
    
    if not results_path.exists() or not hopfield_path.exists():
        print("ERROR: Missing results files")
        print("Run exp_structured_federated.py and hopfield_retrieval_test.py first")
        sys.exit(1)
    
    data = np.load(results_path)
    archetypi = data['archetypi']
    
    print(f"Creating advanced panel for K={archetypi.shape[0]} archetypes...")
    
    create_advanced_panel(
        archetypi=archetypi,
        results_npz_path=str(results_path),
        hopfield_npz_path=str(hopfield_path),
        out_path="out_structured/federated_run/structured_panel_ADVANCED.pdf"
    )
