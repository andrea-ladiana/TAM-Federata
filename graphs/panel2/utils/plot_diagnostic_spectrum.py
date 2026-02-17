"""
Diagnostic plot showing the full eigenspectrum to verify weak archetypes behavior.

This creates a simple plot showing:
- All top eigenvalues (not just top 3)
- MP threshold λ₊
- Theoretical positions for both strong and weak archetypes
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_full_spectrum():
    """Create diagnostic plot of full eigenspectrum."""
    
    script_dir = Path(__file__).parent.parent  # Go up to panel2/
    data_path = script_dir / "output" / "bbp_demo_data.npz"
    
    if not data_path.exists():
        print("❌ Data file not found. Run bbp_theorem_demo.py first.")
        return
    
    data = np.load(data_path, allow_pickle=True)
    results = data['results'].item()
    K_strong = int(data['K'])
    K_weak = int(data['K_weak'])
    K_total = K_strong + K_weak
    
    # Pick middle M value for visualization
    M_values = data['M_values']
    M = int(M_values[0])  # Use first M (q=1, clearest BBP threshold)
    
    res = results[M]
    theory = res['theory']
    theory_all = res['theory_all']
    emp = res['empirical']
    
    # Get full spectrum
    all_eigs = emp['all_eigs']
    sorted_eigs = np.sort(all_eigs)[::-1]
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n_plot = 30  # Show top 30 eigenvalues
    indices = np.arange(1, n_plot + 1)
    top_eigs = sorted_eigs[:n_plot]
    
    lambda_plus = theory['lambda_plus']
    
    # Separate detected and undetected
    above = top_eigs > lambda_plus
    
    # Plot eigenvalues
    ax.scatter(indices[above], top_eigs[above], 
              s=100, color='red', alpha=0.7, 
              label='Detected (above λ₊)', 
              edgecolor='black', linewidth=1, zorder=3)
    
    ax.scatter(indices[~above], top_eigs[~above], 
              s=60, color='lightgray', alpha=0.7,
              label='Bulk (below λ₊)',
              edgecolor='gray', linewidth=1, zorder=2)
    
    # MP threshold
    ax.axhline(lambda_plus, color='blue', linestyle='--', linewidth=2, 
              label=f'MP edge λ₊ = {lambda_plus:.3f}', alpha=0.7)
    
    # Show theoretical positions for STRONG archetypes
    lambda_out_strong = theory['lambda_out']
    for i, lam in enumerate(lambda_out_strong):
        if not np.isnan(lam):
            ax.axhline(lam, color='green', linestyle=':', linewidth=1.5, 
                      alpha=0.5)
            ax.text(n_plot + 0.5, lam, f'ξ{i+1} (strong)', 
                   fontsize=9, va='center', color='green')
    
    # Show theoretical positions for WEAK archetypes
    lambda_out_weak = theory_all['lambda_out'][K_strong:]
    for i, lam in enumerate(lambda_out_weak):
        if not np.isnan(lam):
            ax.axhline(lam, color='orange', linestyle=':', linewidth=1.0, 
                      alpha=0.3)
            ax.text(n_plot + 0.5, lam, f'ξ{K_strong+i+1} (weak)', 
                   fontsize=8, va='center', color='orange', alpha=0.7)
        else:
            # Below threshold - show where it would be if detectable
            pass
    
    # Reference line at y=0
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel('Eigenvalue rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Eigenvalue λ', fontsize=12, fontweight='bold')
    ax.set_title(f'Full Eigenspectrum Diagnostic (M={M}, N=400, q={theory["q"]:.1f})\n'
                f'K_strong={K_strong} (red), K_weak={K_weak} (orange theory lines)',
                fontsize=13, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Set limits to show interesting region
    ax.set_xlim([0, n_plot + 1])
    ax.set_ylim([min(-1, top_eigs[~above].min() * 1.1), 
                 top_eigs[0] * 1.05])
    
    # Save
    out_path = script_dir / "output" / "diagnostic_full_spectrum.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"✅ Diagnostic plot saved: {out_path}")
    
    plt.close()
    
    # Print summary
    print()
    print("="*70)
    print("SPECTRUM ANALYSIS")
    print("="*70)
    print(f"Configuration: M={M}, N=400, q={theory['q']:.2f}")
    print(f"MP threshold: λ₊ = {lambda_plus:.4f}")
    print()
    print(f"Eigenvalues above threshold: {above.sum()}/{n_plot}")
    print(f"  → Should match K_strong = {K_strong}")
    print()
    print("STRONG archetypes (expected to be detected):")
    for i in range(K_strong):
        lam_theo = theory['lambda_out'][i]
        lam_emp = sorted_eigs[i] if i < len(sorted_eigs) else np.nan
        gap = lam_emp - lambda_plus
        print(f"  ξ{i+1}: λ_theory={lam_theo:.2f}, λ_emp={lam_emp:.2f}, gap={gap:.2f}")
    
    print()
    print("WEAK archetypes (expected to remain in bulk):")
    kappa_all = theory_all['kappa']
    sqrt_q = theory['sqrt_q']
    for i in range(K_strong, K_total):
        kappa = kappa_all[i]
        detectable = "YES" if kappa > sqrt_q else "NO"
        lam_theo = theory_all['lambda_out'][i]
        if np.isnan(lam_theo):
            print(f"  ξ{i+1}: κ={kappa:.3f} < √q={sqrt_q:.3f} → NOT detectable ✓")
        else:
            print(f"  ξ{i+1}: κ={kappa:.3f}, λ_theory={lam_theo:.3f} (barely above threshold)")
    
    print("="*70)


if __name__ == '__main__':
    plot_full_spectrum()
