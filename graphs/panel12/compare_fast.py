#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Panel 12 - FAST VERSION per test rapido.
Riduce: K=3, n_batch=5, n_experiments=10, r_values=5 punti
"""
from __future__ import annotations

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.unsup.config import HyperParams, PropagationParams, SpectralParams
from src.unsup.data import make_client_subsets, gen_dataset_partial_archetypes, new_round_single
from src.unsup.estimators import build_unsup_J_single, blend_with_memory
from src.unsup.functions import gen_patterns, JK_real, propagate_J, Hebb_J
from src.unsup.spectrum import estimate_keff
from src.unsup.dynamics import dis_check
from src.unsup.metrics import frobenius_relative


def run_standard_pipeline(xi_true: np.ndarray, ETA: np.ndarray, hp: HyperParams) -> np.ndarray:
    xi_ref = None
    for t in range(hp.n_batch):
        ETA_t = new_round_single(ETA, t)
        J_unsup, M_eff = build_unsup_J_single(ETA_t, K=hp.K)
        J_rec = blend_with_memory(J_unsup, xi_prev=xi_ref, w=hp.w)
        J_KS = np.asarray(propagate_J(J_rec, J_real=-1, verbose=False, iters=hp.prop.iters), dtype=np.float32)
        
        # Eigen cut adattivo
        K_eff_est, keep_mask, info = estimate_keff(J_KS, method="shuffle", M_eff=M_eff)
        evals, evecs = np.linalg.eigh(J_KS)
        order = np.argsort(evals)[::-1]
        evecs_sorted = evecs[:, order]
        n_keep = max(hp.K, K_eff_est)
        V = evecs_sorted[:, :n_keep].T
        
        # TAM
        xi_r, _m_vec = dis_check(V=V, K=hp.K, L=hp.L, J_rec=J_rec, JKS_iter=J_KS,
                                  xi_true=xi_true, tam=hp.tam, spec=hp.spec, show_progress=False)
        
        if xi_r.shape[0] >= hp.K:
            xi_ref = xi_r[:hp.K].astype(int)
        else:
            xi_ref = xi_r.astype(int)
    
    J_final = Hebb_J(xi_ref)
    return J_final


def run_baseline_pipeline(ETA: np.ndarray, hp: HyperParams) -> np.ndarray:
    J_mem = None
    for t in range(hp.n_batch):
        ETA_t = new_round_single(ETA, t)
        J_unsup, _M_eff = build_unsup_J_single(ETA_t, K=hp.K)
        if J_mem is None:
            J_blended = J_unsup
        else:
            J_blended = float(hp.w) * J_unsup + float(1.0 - hp.w) * J_mem
        J_mem = J_blended
    return J_mem


def run_single_experiment(r_ex: float, hp: HyperParams, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    xi_true = np.asarray(gen_patterns(hp.N, hp.K), dtype=np.int32)
    J_star = np.asarray(JK_real(xi_true), dtype=np.float32)
    subsets = make_client_subsets(K=hp.K, L=hp.L, K_per_client=hp.K_per_client, rng=rng)
    hp_local = hp.copy_with(r_ex=r_ex)
    ETA, _labels = gen_dataset_partial_archetypes(
        xi_true=xi_true, M_total=hp_local.M_total, r_ex=hp_local.r_ex,
        n_batch=hp_local.n_batch, L=hp_local.L, client_subsets=subsets,
        rng=rng, use_tqdm=False,
    )
    
    J_standard = run_standard_pipeline(xi_true, ETA, hp_local)
    fro_standard = frobenius_relative(J_standard, J_star)
    
    J_baseline = run_baseline_pipeline(ETA, hp_local)
    fro_baseline = frobenius_relative(J_baseline, J_star)
    
    return float(fro_standard), float(fro_baseline)


def main():
    sns.set_theme(style="whitegrid", context="paper", palette="tab10")
    
    # FAST CONFIG
    hp = HyperParams(
        L=3, K=3, N=200, n_batch=5, M_total=600, K_per_client=3,  # RIDOTTO!
        w=0.5, n_seeds=1, use_tqdm=False,
        prop=PropagationParams(iters=30),  # RIDOTTO!
        spec=SpectralParams(tau=0.5, rho=0.0, qthr=0.95),
    )
    
    r_values = np.linspace(0.2, 0.95, 5)  # 5 punti
    n_experiments = 10  # 10 repliche
    
    fro_standard_all = np.zeros((len(r_values), n_experiments))
    fro_baseline_all = np.zeros((len(r_values), n_experiments))
    
    print("=" * 70)
    print("Panel 12 FAST - Test Comparison")
    print("=" * 70)
    print(f"K={hp.K}, N={hp.N}, L={hp.L}, n_batch={hp.n_batch}")
    print(f"r_ex values: {r_values}")
    print(f"Experiments per point: {n_experiments}")
    print("=" * 70)
    
    max_workers = 6
    
    for i, r_ex in enumerate(r_values):
        print(f"\n[{i+1}/{len(r_values)}] r_ex = {r_ex:.3f}")
        seeds = [2000 + i * n_experiments + exp_idx for exp_idx in range(n_experiments)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_experiment, r_ex, hp, seed): idx
                for idx, seed in enumerate(seeds)
            }
            
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    fro_std, fro_base = future.result()
                    fro_standard_all[i, idx] = fro_std
                    fro_baseline_all[i, idx] = fro_base
                    completed += 1
                    if completed % 5 == 0:
                        print(f"  [{completed}/{n_experiments}] completed")
                except Exception as e:
                    print(f"  ERROR in experiment {idx}: {e}")
        
        print(f"  Standard: {fro_standard_all[i].mean():.4f} ± {fro_standard_all[i].std():.4f}")
        print(f"  Baseline: {fro_baseline_all[i].mean():.4f} ± {fro_baseline_all[i].std():.4f}")
    
    # Statistiche
    fro_std_mean = fro_standard_all.mean(axis=1)
    fro_std_std = fro_standard_all.std(axis=1)
    fro_base_mean = fro_baseline_all.mean(axis=1)
    fro_base_std = fro_baseline_all.std(axis=1)
    
    # Salva
    out_dir = Path(__file__).parent
    np.savez(out_dir / "comparison_fast.npz",
             r_values=r_values, fro_standard_mean=fro_std_mean,
             fro_standard_std=fro_std_std, fro_baseline_mean=fro_base_mean,
             fro_baseline_std=fro_base_std)
    print(f"\n✓ Saved data")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r_values, fro_std_mean, 'o-', linewidth=2, markersize=6,
            label='Standard (TAM+Sharpening+Pruning)', color='tab:blue')
    ax.fill_between(r_values, fro_std_mean - fro_std_std, fro_std_mean + fro_std_std,
                     alpha=0.25, color='tab:blue')
    ax.plot(r_values, fro_base_mean, 's-', linewidth=2, markersize=6,
            label='Baseline (solo aggregazione)', color='tab:orange')
    ax.fill_between(r_values, fro_base_mean - fro_base_std, fro_base_mean + fro_base_std,
                     alpha=0.25, color='tab:orange')
    ax.set_xlabel(r'$r$ (data quality)', fontsize=13)
    ax.set_ylabel(r'Frobenius distance to $J^*$', fontsize=13)
    ax.set_title('Standard vs Baseline [FAST TEST]', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_dir / "panel12_fast.png", dpi=200)
    plt.savefig(out_dir / "panel12_fast.pdf")
    print(f"✓ Saved plots")
    plt.close()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, r in enumerate(r_values):
        print(f"r={r:.2f}: STD={fro_std_mean[i]:.4f}±{fro_std_std[i]:.4f}  BASE={fro_base_mean[i]:.4f}±{fro_base_std[i]:.4f}")


if __name__ == "__main__":
    main()
