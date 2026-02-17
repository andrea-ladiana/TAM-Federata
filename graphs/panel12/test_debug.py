#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test debug per capire il problema della pipeline standard."""
import sys
from pathlib import Path
import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.unsup.config import HyperParams, PropagationParams, SpectralParams
from src.unsup.data import make_client_subsets, gen_dataset_partial_archetypes, new_round_single
from src.unsup.estimators import build_unsup_J_single, blend_with_memory
from src.unsup.functions import gen_patterns, JK_real, propagate_J, Hebb_J
from src.unsup.spectrum import eigen_cut as spectral_cut, estimate_keff
from src.unsup.dynamics import dis_check
from src.unsup.metrics import frobenius_relative, overlap_matrix

# Test con r_ex basso (0.2) e alto (0.9) - ORA CON K=3
hp = HyperParams(
    L=3, K=3, N=400, n_batch=20, M_total=2400, K_per_client=3,
    w=0.5, use_tqdm=False,
    prop=PropagationParams(iters=50),
    spec=SpectralParams(tau=0.5, rho=0.0, qthr=0.95),  # disabilita pruning
)

for r_ex in [0.2, 0.9]:
    print(f"\n{'='*70}")
    print(f"Testing r_ex = {r_ex}")
    print('='*70)
    
    rng = np.random.default_rng(42)
    xi_true = np.asarray(gen_patterns(hp.N, hp.K), dtype=np.int32)
    J_star = np.asarray(JK_real(xi_true), dtype=np.float32)
    
    subsets = make_client_subsets(K=hp.K, L=hp.L, K_per_client=hp.K_per_client, rng=rng)
    hp_local = hp.copy_with(r_ex=r_ex)
    ETA, _labels = gen_dataset_partial_archetypes(
        xi_true=xi_true, M_total=hp_local.M_total, r_ex=hp_local.r_ex,
        n_batch=hp_local.n_batch, L=hp_local.L, client_subsets=subsets,
        rng=rng, use_tqdm=False,
    )
    
    # Simula pipeline standard
    xi_ref = None
    for t in [0, 10, 19]:  # primo, metà, ultimo round
        ETA_t = new_round_single(ETA, t)
        J_unsup, M_eff = build_unsup_J_single(ETA_t, K=hp.K)
        J_rec = blend_with_memory(J_unsup, xi_prev=xi_ref, w=hp.w)
        J_KS = np.asarray(propagate_J(J_rec, J_real=-1, verbose=False, iters=hp.prop.iters), dtype=np.float32)
        
        # Eigen cut ADATTIVO (usa K_eff invece di tau fisso)
        K_eff_est, keep_mask, info = estimate_keff(J_KS, method="shuffle", M_eff=M_eff)
        
        # Estrai autovettori
        evals, evecs = np.linalg.eigh(J_KS)
        order = np.argsort(evals)[::-1]
        evals_sorted = evals[order]
        evecs_sorted = evecs[:, order]
        
        n_keep = max(hp.K, K_eff_est)
        V = evecs_sorted[:, :n_keep].T
        
        print(f"\nRound {t}: K_eff_est={K_eff_est}, n_keep={n_keep}, top evals={evals_sorted[:5]}")
        
        # TAM
        xi_r, m_vec = dis_check(
            V=V, K=hp.K, L=hp.L, J_rec=J_rec, JKS_iter=J_KS,
            xi_true=xi_true, tam=hp.tam, spec=hp.spec, show_progress=False,
        )
        
        # Overlap con archetipi veri
        M_overlap = overlap_matrix(xi_r, xi_true)
        max_overlaps = M_overlap.max(axis=1)
        print(f"  Candidates: {xi_r.shape[0]}")
        print(f"  Max overlaps (top 5): {max_overlaps[:5]}")
        print(f"  Mean max overlap: {max_overlaps.mean():.4f}")
        
        # Aggiorna memoria
        if xi_r.shape[0] >= hp.K:
            xi_ref = xi_r[:hp.K].astype(int)
        else:
            xi_ref = xi_r.astype(int)
    
    # Finale
    J_final = Hebb_J(xi_ref)
    fro = frobenius_relative(J_final, J_star)
    print(f"\n  Final Frobenius: {fro:.6f}")
    print(f"  Final xi_ref shape: {xi_ref.shape}")
