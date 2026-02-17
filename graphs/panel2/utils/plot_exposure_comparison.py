"""
Create a simple visualization showing the relationship between exposure and detectability.

This plot shows:
- X-axis: Archetype index (1-6)
- Y-axis: Exposure α (log scale)
- Markers: Color-coded by detectability (green=detected, red=not detected)
- Horizontal line: BBP threshold exposure level
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def plot_exposure_vs_detectability():
    """Create exposure vs detectability visualization."""
    
    script_dir = Path(__file__).parent.parent  # Go up to panel2/
    json_path = script_dir / "output" / "bbp_demo_log.json"
    data_path = script_dir / "output" / "bbp_demo_data.npz"
    
    if not json_path.exists() or not data_path.exists():
        print("❌ Data files not found. Run bbp_theorem_demo.py first.")
        return
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    npz_data = np.load(data_path, allow_pickle=True)
    results = npz_data['results'].item()
    
    metadata = json_data['metadata']
    
    # Get exposure values
    alpha_strong = np.array(metadata['exposure_alpha_strong'])
    alpha_weak = np.array(metadata['exposure_alpha_weak'])
    alpha_all = np.concatenate([alpha_strong, alpha_weak])
    
    K_strong = metadata['K_strong']
    K_total = metadata['K_total']
    
    # Use first M for visualization
    M = metadata['M_values'][0]
    M_str = str(M)
    
    res = json_data['results_by_M'][M_str]
    theory = res['theory']
    sqrt_q = theory['sqrt_q']
    
    # Get kappa values for all archetypes
    res_M = results[M]
    theory_all = res_M['theory_all']
    kappa_all = theory_all['kappa']
    
    # Determine detectability
    detected = kappa_all > sqrt_q
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    archetype_indices = np.arange(1, K_total + 1)
    
    # ========== Panel 1: Exposure α ==========
    ax1.semilogy(archetype_indices[:K_strong], alpha_all[:K_strong], 
                 'o', markersize=12, color='green', alpha=0.7,
                 label='Strong (detected)', markeredgecolor='black', linewidth=1.5)
    
    ax1.semilogy(archetype_indices[K_strong:], alpha_all[K_strong:], 
                 'o', markersize=12, color='red', alpha=0.7,
                 label='Weak (not detected)', markeredgecolor='black', linewidth=1.5)
    
    # Draw vertical separator
    ax1.axvline(K_strong + 0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    
    ax1.set_ylabel('Exposure α (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Exposure Distribution (M={M}, N=400, q={theory["q"]:.1f})',
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    
    # ========== Panel 2: Signal Strength κ ==========
    ax2.semilogy(archetype_indices[:K_strong], kappa_all[:K_strong], 
                 's', markersize=12, color='green', alpha=0.7,
                 label='Strong archetypes', markeredgecolor='black', linewidth=1.5)
    
    ax2.semilogy(archetype_indices[K_strong:], kappa_all[K_strong:], 
                 's', markersize=12, color='red', alpha=0.7,
                 label='Weak archetypes', markeredgecolor='black', linewidth=1.5)
    
    # BBP threshold line
    ax2.axhline(sqrt_q, color='blue', linestyle='-', linewidth=2.5, alpha=0.7,
               label=f'BBP threshold √q = {sqrt_q:.2f}')
    
    # Draw vertical separator
    ax2.axvline(K_strong + 0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    
    ax2.set_xlabel('Archetype index ξᵢ', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Signal strength κ (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('BBP Detectability: Signal Strength vs Threshold',
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax2.legend(fontsize=11, loc='upper right', framealpha=0.9)
    
    # Set x-ticks
    ax2.set_xticks(archetype_indices)
    ax2.set_xticklabels([f'ξ{i}' for i in archetype_indices])
    
    # Add text annotations for κ values
    for i, (idx, k) in enumerate(zip(archetype_indices, kappa_all)):
        if i < K_strong:
            # Strong archetypes - annotate above
            ax2.text(idx, k * 1.3, f'{k:.0f}', 
                    ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')
        else:
            # Weak archetypes - annotate below
            ax2.text(idx, k * 0.7, f'{k:.2f}', 
                    ha='center', va='top', fontsize=9, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    out_path = script_dir / "output" / "exposure_detectability_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"✅ Comparison plot saved: {out_path}")
    
    plt.close()
    
    # Print summary
    print()
    print("="*70)
    print("EXPOSURE vs DETECTABILITY SUMMARY")
    print("="*70)
    print()
    print(f"Configuration: M={M}, N=400, q={theory['q']:.2f}, √q={sqrt_q:.2f}")
    print()
    print("Archetype  |  Type    |  Exposure α   |  Signal κ  |  Detection")
    print("-" * 70)
    
    for i in range(K_total):
        if i < K_strong:
            arch_type = "STRONG"
            color = "✓"
        else:
            arch_type = "WEAK  "
            color = "✗"
        
        a = alpha_all[i]
        k = kappa_all[i]
        det = "YES" if detected[i] else "NO "
        
        print(f"    ξ{i+1}     |  {arch_type}  |  {a:.6f}  |  {k:8.2f}  |  {color} {det}")
    
    print()
    print("KEY:")
    print("  • Exposure α determines signal strength κ via κ = (r²/σ²) × α × N")
    print("  • BBP threshold: κ > √q required for spectral detection")
    print("  • Strong archetypes: κ >> √q → always detected")
    print("  • Weak archetypes: κ < √q → remain hidden in bulk")
    print()
    print("="*70)


if __name__ == '__main__':
    plot_exposure_vs_detectability()
