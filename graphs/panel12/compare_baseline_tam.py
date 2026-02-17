#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Panel 12: Confronto Pipeline Standard (TAM+Sharpening+Pruning) vs Baseline (solo aggregazione)

Confronta due approcci alla federazione:
1. STANDARD: aggregazione → propagazione → eigen_cut → TAM → pruning → J_Hebb da xi_r
2. BASELINE: aggregazione → blending diretto (NO propagazione, NO TAM, NO pruning)

Metrica: Norma di Frobenius rispetto a J_star (matrice ideale) alla fine dei 20 round.
Variabile indipendente: r_ex (qualità dei dati) in (0, 1].
Aggregazione: 50 esperimenti indipendenti per punto, con media ± std (fill_between).
"""
from __future__ import annotations

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup path per imports locali
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.unsup.config import HyperParams, PropagationParams, SpectralParams
from src.unsup.data import make_client_subsets, gen_dataset_partial_archetypes, new_round_single
from src.unsup.estimators import build_unsup_J_single, blend_with_memory
from src.unsup.functions import gen_patterns, JK_real, propagate_J, Hebb_J
from src.unsup.spectrum import eigen_cut as spectral_cut, estimate_keff
from src.unsup.dynamics import dis_check
from src.unsup.metrics import frobenius_relative


def run_standard_pipeline(
    xi_true: np.ndarray,
    ETA: np.ndarray,
    hp: HyperParams,
    debug: bool = False,
) -> np.ndarray:
    """
    Pipeline STANDARD: aggregazione → propagate → eigen_cut → TAM → pruning → J_Hebb.
    
    Returns
    -------
    J_final : (N, N)
        J_Hebb costruita dagli archetipi estratti xi_r al termine dei 20 round.
    """
    xi_ref = None  # memoria Hebbian (pattern)
    
    for t in range(hp.n_batch):
        # Estrai dati round t
        ETA_t = new_round_single(ETA, t)  # (L, M_c, N)
        
        # 1) Stima J_unsup e blending
        J_unsup, M_eff = build_unsup_J_single(ETA_t, K=hp.K)
        J_rec = blend_with_memory(J_unsup, xi_prev=xi_ref, w=hp.w)
        
        # 2) Propagazione pseudo-inversa
        J_KS = np.asarray(propagate_J(J_rec, J_real=-1, verbose=False, iters=hp.prop.iters), dtype=np.float32)
        
        # 3) Eigen cut ADATTIVO (usa K_eff per determinare soglia invece di tau fisso)
        # Questo risolve il problema: tau fisso è troppo alto per dati rumorosi
        K_eff_est, keep_mask, info = estimate_keff(J_KS, method="shuffle", M_eff=M_eff)
        
        # Estrai autovettori significativi basandoti su keep_mask
        evals, evecs = np.linalg.eigh(J_KS)
        order = np.argsort(evals)[::-1]
        evals_sorted = evals[order]
        evecs_sorted = evecs[:, order]
        
        # Prendi top K_eff_est autovettori (o almeno K se K_eff_est < K)
        n_keep = max(hp.K, K_eff_est)  # almeno K per non rimanere senza candidati
        V = evecs_sorted[:, :n_keep].T  # (n_keep, N)
        
        # 4) Disentangling (TAM + pruning)
        xi_r, _m_vec = dis_check(
            V=V,
            K=hp.K,
            L=hp.L,
            J_rec=J_rec,
            JKS_iter=J_KS,
            xi_true=xi_true,
            tam=hp.tam,
            spec=hp.spec,
            show_progress=False,
        )
        
        # 5) Aggiorna memoria per round successivo
        if xi_r.shape[0] >= hp.K:
            xi_ref = xi_r[:hp.K].astype(int)
        else:
            xi_ref = xi_r.astype(int)
        
        # Debug: controlla overlap con archetipi veri
        if debug and t == hp.n_batch - 1:
            from src.unsup.metrics import overlap_matrix
            M_overlap = overlap_matrix(xi_ref, xi_true)
            max_overlaps = M_overlap.max(axis=1)
            print(f"    [DEBUG] Final round: {xi_ref.shape[0]} candidates, overlaps: {max_overlaps[:5]}")
    
    # Al termine: costruisci J_Hebb dagli archetipi estratti
    J_final = Hebb_J(xi_ref)  # (N, N)
    return J_final


def run_baseline_pipeline(
    ETA: np.ndarray,
    hp: HyperParams,
) -> np.ndarray:
    """
    Pipeline BASELINE: aggregazione → blending diretto (NO propagate, NO TAM, NO pruning).
    
    La "memoria" è semplicemente la J_unsup del round precedente.
    
    Returns
    -------
    J_final : (N, N)
        J_unsup aggregata al termine dei 20 round (usata come J_Hebb surrogata).
    """
    J_mem = None  # memoria (matrice, non pattern)
    
    for t in range(hp.n_batch):
        # Estrai dati round t
        ETA_t = new_round_single(ETA, t)  # (L, M_c, N)
        
        # 1) Stima J_unsup (aggregazione sui client)
        J_unsup, _M_eff = build_unsup_J_single(ETA_t, K=hp.K)
        
        # 2) Blending con memoria precedente (se esiste)
        if J_mem is None:
            J_blended = J_unsup
        else:
            # blend: w * J_unsup + (1-w) * J_mem
            J_blended = float(hp.w) * J_unsup + float(1.0 - hp.w) * J_mem
        
        # 3) Aggiorna memoria per round successivo (NO propagate, NO TAM)
        J_mem = J_blended
    
    # Al termine: restituisci l'ultima J aggregata
    J_final = J_mem
    return J_final


def run_single_experiment(
    r_ex: float,
    hp: HyperParams,
    seed: int,
) -> tuple[float, float]:
    """
    Esegue un esperimento completo (federazione 20 round) per un dato r_ex e seed.
    
    Returns
    -------
    fro_standard : float
        Frobenius relativo per metodo standard.
    fro_baseline : float
        Frobenius relativo per metodo baseline.
    """
    rng = np.random.default_rng(seed)
    
    # Genera archetipi veri
    xi_true = np.asarray(gen_patterns(hp.N, hp.K), dtype=np.int32)
    
    # Matrice ideale J_star
    J_star = np.asarray(JK_real(xi_true), dtype=np.float32)
    
    # Subset per client
    subsets = make_client_subsets(K=hp.K, L=hp.L, K_per_client=hp.K_per_client, rng=rng)
    
    # Genera dataset federato con qualità r_ex
    hp_local = hp.copy_with(r_ex=r_ex)
    ETA, _labels = gen_dataset_partial_archetypes(
        xi_true=xi_true,
        M_total=hp_local.M_total,
        r_ex=hp_local.r_ex,
        n_batch=hp_local.n_batch,
        L=hp_local.L,
        client_subsets=subsets,
        rng=rng,
        use_tqdm=False,
    )
    
    # Pipeline STANDARD
    J_standard = run_standard_pipeline(xi_true, ETA, hp_local, debug=False)
    fro_standard = frobenius_relative(J_standard, J_star)
    
    # Pipeline BASELINE
    J_baseline = run_baseline_pipeline(ETA, hp_local)
    fro_baseline = frobenius_relative(J_baseline, J_star)
    
    return float(fro_standard), float(fro_baseline)


def main():
    """Esegue confronto su range di r_ex con 50 repliche per punto."""
    
    # Setup seaborn
    sns.set_theme(style="whitegrid", context="paper", palette="tab10")
    
    # Parametri federazione (con propagazione ridotta per velocità)
    hp = HyperParams(
        L=3,
        K=10,
        N=400,
        n_batch=20,  # 20 round
        M_total=2400,
        K_per_client=10,
        w=0.5,  # peso blending fisso
        n_seeds=1,  # non usato direttamente (loop manuale)
        use_tqdm=False,
        prop=PropagationParams(iters=50),  # ridotto da 200 per velocità
        spec=SpectralParams(tau=0.5, rho=0.6, qthr=0.4),  # default pruning
    )
    
    # Range qualità dati: r_ex in (0, 1]
    r_values = np.linspace(0.1, 1.0, 10)  # 10 punti da 0.1 a 1.0
    n_experiments = 50  # repliche per punto
    
    # Storage risultati
    fro_standard_all = np.zeros((len(r_values), n_experiments))
    fro_baseline_all = np.zeros((len(r_values), n_experiments))
    
    print("=" * 70)
    print("Panel 12: Standard vs Baseline Pipeline Comparison")
    print("=" * 70)
    print(f"K={hp.K}, N={hp.N}, L={hp.L}, n_batch={hp.n_batch}, M_total={hp.M_total}")
    print(f"w={hp.w}, K_per_client={hp.K_per_client}")
    print(f"r_ex values: {r_values}")
    print(f"Experiments per point: {n_experiments}")
    print("=" * 70)
    
    # Loop su r_ex con parallelizzazione per esperimenti
    max_workers = 6  # parallelizza esperimenti
    
    for i, r_ex in enumerate(r_values):
        print(f"\n[{i+1}/{len(r_values)}] r_ex = {r_ex:.3f}")
        
        # Parallelizza esperimenti per questo valore di r_ex
        seeds = [1000 + i * n_experiments + exp_idx for exp_idx in range(n_experiments)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tutti gli esperimenti
            futures = {
                executor.submit(run_single_experiment, r_ex, hp, seed): idx
                for idx, seed in enumerate(seeds)
            }
            
            # Raccogli risultati man mano che completano
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    fro_std, fro_base = future.result()
                    fro_standard_all[i, idx] = fro_std
                    fro_baseline_all[i, idx] = fro_base
                    completed += 1
                    
                    # Progress ogni 10 esperimenti
                    if completed % 10 == 0:
                        print(f"  [{completed}/{n_experiments}] completed")
                except Exception as e:
                    print(f"  ERROR in experiment {idx}: {e}")
        
        # Statistiche punto corrente
        print(f"  Standard: {fro_standard_all[i].mean():.4f} ± {fro_standard_all[i].std():.4f}")
        print(f"  Baseline: {fro_baseline_all[i].mean():.4f} ± {fro_baseline_all[i].std():.4f}")
        
        # Salva progressi intermedi
        out_dir = Path(__file__).parent
        np.savez(out_dir / "comparison_partial.npz",
                 r_values=r_values, fro_standard_all=fro_standard_all,
                 fro_baseline_all=fro_baseline_all, n_completed=i+1)
    
    # Calcola medie e std
    fro_std_mean = fro_standard_all.mean(axis=1)
    fro_std_std = fro_standard_all.std(axis=1)
    fro_base_mean = fro_baseline_all.mean(axis=1)
    fro_base_std = fro_baseline_all.std(axis=1)
    
    # Salva dati numerici
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        out_dir / "comparison_data.npz",
        r_values=r_values,
        fro_standard_mean=fro_std_mean,
        fro_standard_std=fro_std_std,
        fro_baseline_mean=fro_base_mean,
        fro_baseline_std=fro_base_std,
        fro_standard_all=fro_standard_all,
        fro_baseline_all=fro_baseline_all,
    )
    print(f"\n✓ Saved data to {out_dir / 'comparison_data.npz'}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Standard pipeline (blu)
    ax.plot(r_values, fro_std_mean, 'o-', linewidth=2, markersize=6, 
            label='Standard (TAM+Sharpening+Pruning)', color='tab:blue')
    ax.fill_between(r_values, 
                     fro_std_mean - fro_std_std, 
                     fro_std_mean + fro_std_std,
                     alpha=0.25, color='tab:blue')
    
    # Baseline pipeline (arancione)
    ax.plot(r_values, fro_base_mean, 's-', linewidth=2, markersize=6,
            label='Baseline (solo aggregazione)', color='tab:orange')
    ax.fill_between(r_values,
                     fro_base_mean - fro_base_std,
                     fro_base_mean + fro_base_std,
                     alpha=0.25, color='tab:orange')
    
    # Formatting
    ax.set_xlabel(r'$r$ (data quality)', fontsize=13)
    ax.set_ylabel(r'Frobenius distance to $J^*$', fontsize=13)
    ax.set_title('Standard vs Baseline Pipeline: Final Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Tight layout e salvataggio
    plt.tight_layout()
    out_path = out_dir / "panel12_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {out_path}")
    
    # PDF per pubblicazione
    out_path_pdf = out_dir / "panel12_comparison.pdf"
    plt.savefig(out_path_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF to {out_path_pdf}")
    
    plt.close()
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'r_ex':<8} {'Standard Mean':<15} {'Standard Std':<15} {'Baseline Mean':<15} {'Baseline Std':<15}")
    print("-" * 70)
    for i, r in enumerate(r_values):
        print(f"{r:<8.3f} {fro_std_mean[i]:<15.5f} {fro_std_std[i]:<15.5f} {fro_base_mean[i]:<15.5f} {fro_base_std[i]:<15.5f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
