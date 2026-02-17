"""
Test alternativo Panel C: Overlap vs q

Mostra dipendenza dell'overlap da q invece che da γ(κ,q).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data = np.load('graphs/panel2/output/bbp_demo_data.npz', allow_pickle=True)
results = data['results'].item()
M_values = data['M_values']
K = int(data['K'])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
colors_M = plt.cm.viridis(np.linspace(0.2, 0.9, len(M_values)))
markers = ['o', 's', '^', 'D', 'v', 'p']

for arch_idx in range(K):
    ax = axes[arch_idx]
    
    q_vals = []
    overlap_emp = []
    gamma_theo = []
    
    for i, M in enumerate(M_values):
        res = results[M]
        theory = res['theory']
        emp = res['empirical']
        
        if not np.isnan(theory['gamma'][arch_idx]):
            q_vals.append(theory['q'])
            overlap_emp.append(emp['overlap'][arch_idx])
            gamma_theo.append(theory['gamma'][arch_idx])
    
    # Sort by q for cleaner line plot
    sorted_idx = np.argsort(q_vals)
    q_sorted = np.array(q_vals)[sorted_idx]
    overlap_sorted = np.array(overlap_emp)[sorted_idx]
    gamma_sorted = np.array(gamma_theo)[sorted_idx]
    
    # Plot empirical
    for i, M in enumerate(M_values):
        res = results[M]
        theory = res['theory']
        emp = res['empirical']
        
        if not np.isnan(theory['gamma'][arch_idx]):
            ax.scatter(theory['q'], emp['overlap'][arch_idx], 
                      marker=markers[i], s=100, color=colors_M[i],
                      alpha=0.8, edgecolor='black', linewidth=1.5,
                      label=f'M={M}' if arch_idx == 0 else None)
    
    # Plot theoretical line
    ax.plot(q_sorted, gamma_sorted, 'k--', linewidth=2, 
            alpha=0.7, label='Theory γ(κ,q)' if arch_idx == 0 else None)
    
    # Labels
    if arch_idx == 0:
        ax.set_ylabel('|⟨v,u⟩|² / γ(κ,q)', fontsize=11)
    ax.set_xlabel('q = N/M', fontsize=11)
    
    alpha_val = results[M_values[0]]['info']['exposure_theory'][arch_idx]
    ax.set_title(f'Archetype {arch_idx+1} (α={alpha_val:.2f})', 
                fontsize=11, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.5, 9])
    ax.set_ylim([0.96, 1.0])
    
    if arch_idx == 0:
        ax.legend(fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig('graphs/panel2/output/panel_C_alternative_vs_q.png', dpi=150)
print("✅ Grafico alternativo salvato: panel_C_alternative_vs_q.png")
print("\nQuesta visualizzazione mostra:")
print("  - Overlap empirico vs q (N/M)")
print("  - Curva teorica γ(κ,q) per confronto")
print("  - Tendenza: overlap → 1 quando q → 0 (M grande)")
