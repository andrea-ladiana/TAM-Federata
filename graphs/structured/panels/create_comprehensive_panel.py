"""
Comprehensive visualization panel for structured federated TAM experiment.

Panel structure (4 rows):
- Row 1: Original archetypes (3×3) | Retrieved patterns (3×3) from final round
- Row 2: Basin visualization | Archetype alignment curve (over rounds)
- Row 3: Subspace distance | Permutation stability  
- Row 4: Overlap matrix (final round) | Hopfield retrieval demonstration (noisy→clean)

This visualization demonstrates whether the federated TAM pipeline successfully
recovers structured archetypes and whether spurious attractors are present.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys

# Add paths
root_path = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from graphs.structured.prep.load_structured_dataset import load_structured_archetypes


def plot_patterns_grid(ax, patterns, titles=None, cmap='gray', side=28):
    """
    Plot K patterns in 3×3 grid on given axis.
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    patterns : np.ndarray
        Patterns (K, N) where N = side*side
    titles : list, optional
        Titles for each pattern
    cmap : str
        Colormap
    side : int
        Side length (assume square)
    """
    K = patterns.shape[0]
    
    # Create subgrid
    n_rows = int(np.ceil(np.sqrt(K)))
    n_cols = int(np.ceil(K / n_rows))
    
    ax.axis('off')
    
    for k in range(K):
        # Compute subplot position
        row = k // n_cols
        col = k % n_cols
        
        # Create sub-axis
        left = col / n_cols
        bottom = 1.0 - (row + 1) / n_rows
        width = 1.0 / n_cols
        height = 1.0 / n_rows
        
        sub_ax = ax.inset_axes([left, bottom, width, height])
        
        # Plot pattern
        pattern = patterns[k].reshape(side, side)
        sub_ax.imshow(pattern, cmap=cmap, vmin=-1, vmax=1)
        sub_ax.axis('off')
        
        if titles is not None and k < len(titles):
            sub_ax.set_title(titles[k], fontsize=6)


def plot_basin_visualization(ax, J, xi_true, n_init=100, n_steps=20, side=28):
    """
    Visualize basins of attraction by testing random initial states.
    
    Parameters
    ----------
    ax : matplotlib axis
    J : np.ndarray
        Hopfield matrix (N, N)
    xi_true : np.ndarray
        True archetypes (K, N)
    n_init : int
        Number of random initial states to test
    n_steps : int
        Number of Hopfield steps
    side : int
        Pattern side length
    """
    from graphs.structured.verification.hopfield_retrieval_test import hopfield_dynamics
    
    K, N = xi_true.shape
    
    # Generate random initial states
    s_init_all = np.sign(np.random.randn(n_init, N))
    
    # Track which archetype each converges to
    convergence_count = np.zeros(K, dtype=int)
    
    for i in range(n_init):
        s_init = s_init_all[i]
        s_final, _ = hopfield_dynamics(s_init, J, n_steps=n_steps, beta=1.0)
        
        # Find closest archetype
        overlaps = (1.0 / N) * (s_final @ xi_true.T)
        target_k = np.argmax(overlaps)
        convergence_count[target_k] += 1
    
    # Plot as bar chart
    ax.bar(range(K), convergence_count, color='steelblue', alpha=0.7)
    ax.set_xlabel('Archetype index', fontsize=9)
    ax.set_ylabel(f'Basin size (out of {n_init})', fontsize=9)
    ax.set_title('Basin of Attraction Analysis', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'{k+1}' for k in range(K)])


def plot_alignment_curve(ax, retrieval_history):
    """
    Plot retrieval (alignment) over federated rounds.
    
    Parameters
    ----------
    ax : matplotlib axis
    retrieval_history : np.ndarray
        Retrieval values per round (n_batch,)
    """
    n_batch = len(retrieval_history)
    rounds = np.arange(1, n_batch + 1)
    
    ax.plot(rounds, retrieval_history, marker='o', linewidth=2, markersize=4,
            color='darkgreen', label='Retrieval')
    ax.axhline(1.0, linestyle='--', color='gray', alpha=0.5, label='Perfect')
    ax.set_xlabel('Federated round', fontsize=9)
    ax.set_ylabel('Retrieval (Hungarian)', fontsize=9)
    ax.set_title('Archetype Alignment Over Rounds', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)


def plot_subspace_distance(ax, fro_history):
    """
    Plot Frobenius distance over rounds.
    
    Parameters
    ----------
    ax : matplotlib axis
    fro_history : np.ndarray
        Frobenius relative distance per round (n_batch,)
    """
    n_batch = len(fro_history)
    rounds = np.arange(1, n_batch + 1)
    
    ax.plot(rounds, fro_history, marker='s', linewidth=2, markersize=4,
            color='darkred', label='FRO distance')
    ax.axhline(0.0, linestyle='--', color='gray', alpha=0.5, label='Perfect')
    ax.set_xlabel('Federated round', fontsize=9)
    ax.set_ylabel('Frobenius distance (relative)', fontsize=9)
    ax.set_title('Subspace Distance Over Rounds', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_permutation_stability(ax, keff_history, K):
    """
    Plot K_eff over rounds.
    
    Parameters
    ----------
    ax : matplotlib axis
    keff_history : np.ndarray
        K_eff values per round (n_batch,)
    K : int
        Total number of archetypes
    """
    n_batch = len(keff_history)
    rounds = np.arange(1, n_batch + 1)
    
    ax.plot(rounds, keff_history, marker='^', linewidth=2, markersize=4,
            color='darkorange', label='K_eff')
    ax.axhline(K, linestyle='--', color='gray', alpha=0.5, label=f'Target (K={K})')
    ax.set_xlabel('Federated round', fontsize=9)
    ax.set_ylabel('Effective rank (K_eff)', fontsize=9)
    ax.set_title('Permutation Stability (K_eff)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0, K + 1)


def plot_overlap_matrix(ax, xi_retrieved, xi_true):
    """
    Plot overlap matrix between retrieved and true archetypes.
    
    Parameters
    ----------
    ax : matplotlib axis
    xi_retrieved : np.ndarray
        Retrieved archetypes (K_ret, N)
    xi_true : np.ndarray
        True archetypes (K, N)
    """
    K, N = xi_true.shape
    K_ret = xi_retrieved.shape[0]
    
    if K_ret == 0:
        ax.text(0.5, 0.5, 'No archetypes retrieved\n(K_eff=0)',
                ha='center', va='center', fontsize=10, color='red')
        ax.set_title('Overlap Matrix (Final Round)', fontsize=10)
        ax.axis('off')
        return
    
    # Compute overlap matrix
    overlap = (1.0 / N) * (xi_retrieved @ xi_true.T)  # (K_ret, K)
    
    # Plot as heatmap
    im = ax.imshow(overlap, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel('True archetype', fontsize=9)
    ax.set_ylabel('Retrieved archetype', fontsize=9)
    ax.set_title('Overlap Matrix (Final Round)', fontsize=10)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K_ret))
    ax.set_xticklabels([f'{k+1}' for k in range(K)], fontsize=7)
    ax.set_yticklabels([f'{k+1}' for k in range(K_ret)], fontsize=7)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Overlap', fontsize=8)
    
    # Add values
    for i in range(K_ret):
        for j in range(K):
            text = ax.text(j, i, f'{overlap[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=6)


def plot_hopfield_retrieval_demo(ax, sample_trajectories, xi_true, side=28):
    """
    Plot Hopfield retrieval demonstration: noisy → clean for sample trajectories.
    
    Parameters
    ----------
    ax : matplotlib axis
    sample_trajectories : dict
        Sample trajectories from Hopfield test
    xi_true : np.ndarray
        True archetypes (K, N)
    side : int
        Pattern side length
    """
    if not sample_trajectories:
        ax.text(0.5, 0.5, 'No trajectories available\n(retrieval test failed)',
                ha='center', va='center', fontsize=10, color='red')
        ax.set_title('Hopfield Retrieval Verification', fontsize=10)
        ax.axis('off')
        return
    
    # Show first 3 archetypes
    n_show = min(3, len(sample_trajectories))
    archetype_keys = sorted(list(sample_trajectories.keys()))[:n_show]
    
    ax.axis('off')
    ax.set_title('Hopfield Retrieval: Noisy → Converged', fontsize=10)
    
    for i, k in enumerate(archetype_keys):
        traj = sample_trajectories[k]
        s_init = traj['init']
        s_final = traj['final']
        xi_true_k = xi_true[k]
        
        # Plot: noisy | converged | true
        patterns = [s_init, s_final, xi_true_k]
        titles = [f'Noisy\n(k={k+1})', f'Converged', f'True']
        
        for j, (pattern, title) in enumerate(zip(patterns, titles)):
            left = (j / 3.0) + 0.01
            bottom = 1.0 - (i + 1) / n_show
            width = 0.30
            height = 1.0 / n_show - 0.05
            
            sub_ax = ax.inset_axes([left, bottom, width, height])
            img = pattern.reshape(side, side)
            sub_ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
            sub_ax.axis('off')
            sub_ax.set_title(title, fontsize=7)


def create_structured_panel(
    archetypi: np.ndarray,
    xi_retrieved: np.ndarray,
    J_hopfield: np.ndarray,
    retrieval_history: np.ndarray,
    fro_history: np.ndarray,
    keff_history: np.ndarray,
    sample_trajectories: dict,
    out_path: str = "structured_panel.pdf"
):
    """
    Create comprehensive 4-row panel.
    
    Parameters
    ----------
    archetypi : np.ndarray
        True archetypes (K, N)
    xi_retrieved : np.ndarray
        Retrieved archetypes (K_ret, N)
    J_hopfield : np.ndarray
        Hopfield matrix (N, N)
    retrieval_history : np.ndarray
        Retrieval per round
    fro_history : np.ndarray
        FRO per round
    keff_history : np.ndarray
        K_eff per round
    sample_trajectories : dict
        Sample Hopfield trajectories
    out_path : str
        Output PDF path
    """
    K, N = archetypi.shape
    side = int(np.sqrt(N))
    
    # Create figure
    fig = plt.figure(figsize=(14, 16))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1],
                  hspace=0.35, wspace=0.3)
    
    # Row 1: Original vs Retrieved
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_retr = fig.add_subplot(gs[0, 1])
    
    plot_patterns_grid(ax_orig, archetypi, titles=[f'Arch {i+1}' for i in range(K)],
                      cmap='gray', side=side)
    ax_orig.set_title('Original Archetypes', fontsize=12, fontweight='bold')
    
    if xi_retrieved.shape[0] > 0:
        plot_patterns_grid(ax_retr, xi_retrieved,
                          titles=[f'Ret {i+1}' for i in range(xi_retrieved.shape[0])],
                          cmap='gray', side=side)
    else:
        ax_retr.text(0.5, 0.5, 'No archetypes retrieved\n(K_eff=0)',
                    ha='center', va='center', fontsize=12, color='red',
                    transform=ax_retr.transAxes)
        ax_retr.axis('off')
    ax_retr.set_title('Retrieved Patterns (Final Round)', fontsize=12, fontweight='bold')
    
    # Row 2: Basin + Alignment
    ax_basin = fig.add_subplot(gs[1, 0])
    ax_align = fig.add_subplot(gs[1, 1])
    
    plot_basin_visualization(ax_basin, J_hopfield, archetypi,
                            n_init=100, n_steps=20, side=side)
    plot_alignment_curve(ax_align, retrieval_history)
    
    # Row 3: Subspace + Permutation
    ax_subspace = fig.add_subplot(gs[2, 0])
    ax_perm = fig.add_subplot(gs[2, 1])
    
    plot_subspace_distance(ax_subspace, fro_history)
    plot_permutation_stability(ax_perm, keff_history, K)
    
    # Row 4: Overlap matrix + Hopfield demo
    ax_overlap = fig.add_subplot(gs[3, 0])
    ax_hopfield = fig.add_subplot(gs[3, 1])
    
    plot_overlap_matrix(ax_overlap, xi_retrieved, archetypi)
    plot_hopfield_retrieval_demo(ax_hopfield, sample_trajectories, archetypi, side=side)
    
    # Super title
    fig.suptitle('Structured Federated TAM: Comprehensive Analysis Panel',
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPanel saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    # Load data
    results_path = Path("out_structured/federated_run/results.npz")
    hopfield_path = Path("out_structured/federated_run/hopfield_verification.npz")
    
    if not results_path.exists():
        print(f"ERROR: Results not found at {results_path}")
        print("Run exp_structured_federated.py first")
        sys.exit(1)
    
    if not hopfield_path.exists():
        print(f"ERROR: Hopfield verification not found at {hopfield_path}")
        print("Run hopfield_retrieval_test.py first")
        sys.exit(1)
    
    # Load results
    data = np.load(results_path)
    hopfield_data = np.load(hopfield_path, allow_pickle=True)
    
    archetypi = data['archetypi']
    retrieval = data['retrieval']
    fro = data['fro']
    keff = data['keff']
    xi_retrieved = data['xi_retrieved_final']
    
    J_hopfield = hopfield_data['J_hopfield']
    sample_trajectories = hopfield_data['sample_trajectories'].item()
    
    print("Creating comprehensive panel...")
    print(f"  K={archetypi.shape[0]} archetypes")
    print(f"  K_ret={xi_retrieved.shape[0]} retrieved")
    print(f"  n_rounds={len(retrieval)}")
    print(f"  Final retrieval={retrieval[-1]:.3f}")
    print(f"  Final K_eff={keff[-1]}")
    
    # Create panel
    create_structured_panel(
        archetypi=archetypi,
        xi_retrieved=xi_retrieved,
        J_hopfield=J_hopfield,
        retrieval_history=retrieval,
        fro_history=fro,
        keff_history=keff,
        sample_trajectories=sample_trajectories,
        out_path="out_structured/federated_run/structured_panel_FULL.pdf"
    )
