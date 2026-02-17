#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analisi autovalori per capire la soglia tau appropriata."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.unsup.config import HyperParams, PropagationParams
from src.unsup.data import make_client_subsets, gen_dataset_partial_archetypes, new_round_single
from src.unsup.estimators import build_unsup_J_single, blend_with_memory
from src.unsup.functions import gen_patterns, JK_real, propagate_J
from src.unsup.metrics import frobenius_relative

# Test con K=3
hp = HyperParams(
    L=3, K=3, N=400, n_batch=20, M_total=2400, K_per_client=3,
    w=0.5, use_tqdm=False,
    prop=PropagationParams(iters=50),
)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, r_ex in enumerate([0.2, 0.5, 0.7, 0.95]):
    ax = axes[idx // 2, idx % 2]
    
    rng = np.random.default_rng(42 + idx)
    xi_true = np.asarray(gen_patterns(hp.N, hp.K), dtype=np.int32)
    J_star = np.asarray(JK_real(xi_true), dtype=np.float32)
    
    subsets = make_client_subsets(K=hp.K, L=hp.L, K_per_client=hp.K_per_client, rng=rng)
    hp_local = hp.copy_with(r_ex=r_ex)
    ETA, _labels = gen_dataset_partial_archetypes(
        xi_true=xi_true, M_total=hp_local.M_total, r_ex=hp_local.r_ex,
        n_batch=hp_local.n_batch, L=hp_local.L, client_subsets=subsets,
        rng=rng, use_tqdm=False,
    )
    
    # Analizza autovalori per diversi round
    xi_ref = None
    evals_all = []
    
    for t in range(hp.n_batch):
        ETA_t = new_round_single(ETA, t)
        J_unsup, M_eff = build_unsup_J_single(ETA_t, K=hp.K)
        J_rec = blend_with_memory(J_unsup, xi_prev=xi_ref, w=hp.w)
        J_KS = np.asarray(propagate_J(J_rec, J_real=-1, verbose=False, iters=hp.prop.iters), dtype=np.float32)
        
        # Calcola autovalori
        evals = np.linalg.eigvalsh(J_KS)
        evals_sorted = np.sort(evals)[::-1]
        evals_all.append(evals_sorted[:10])  # top 10
    
    evals_all = np.array(evals_all)
    
    # Plot top 10 autovalori per round
    for k in range(min(10, evals_all.shape[1])):
        ax.plot(range(hp.n_batch), evals_all[:, k], alpha=0.6, linewidth=1)
    
    # Linea tau=0.5
    ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='tau=0.5')
    ax.axhline(0.0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_title(f'r_ex = {r_ex}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Eigenvalue', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    # Stampa statistiche
    print(f"\nr_ex={r_ex}: Top 5 eigenvalues (final round): {evals_all[-1, :5]}")
    print(f"  Max eigenvalue: {evals_all[-1, 0]:.4f}")
    print(f"  # eigenvalues > 0.5: {np.sum(evals_all[-1] > 0.5)}")
    print(f"  # eigenvalues > 0.1: {np.sum(evals_all[-1] > 0.1)}")

plt.tight_layout()
plt.savefig('graphs/panel12/eigenvalue_analysis_K3.png', dpi=200)
print("\n✓ Saved eigenvalue analysis plot")
plt.close()
