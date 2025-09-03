# -*- coding: utf-8 -*-
"""
Orchestratore di UN round in modalità SINGLE.

Fasi per round t:
  1) build_unsup_J_single(ETA_t) -> J_unsup, M_eff
  2) blend_with_memory(J_unsup, xi_prev, w) -> J_rec
  3) propagate_J(J_rec) -> J_KS
  4) eigen_cut(J_KS, tau) -> V
  5) dis_check(V, K, L, J_rec, J_KS, xi_true, tam, spec) -> xi_r, m
  6) metriche: retrieval (Hungarian), FRO, K_eff (shuffle|mp), coverage(labels_t)
  7) aggiorna memoria xi_ref per t+1

Compatibilità:
- Nessun riferimento a modalità 'extend'.
- Usa le stesse primitive già fornite nei moduli precedenti.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.unsup.config import HyperParams
from src.unsup.estimators import build_unsup_J_single, blend_with_memory
from src.unsup.spectrum import eigen_cut as spectral_cut, estimate_keff
from src.unsup.dynamics import dis_check
from src.unsup.metrics import frobenius_relative, retrieval_mean_hungarian
from src.unsup.data import compute_round_coverage
from src.unsup.functions import propagate_J


__all__ = ["RoundLog", "single_round_step"]


@dataclass
class RoundLog:
    """Metriche aggregate del round."""
    retrieval: float
    fro: float
    keff: int
    coverage: float


def single_round_step(
    ETA_t: np.ndarray,        # (L, M_c, N)
    labels_t: np.ndarray,     # (L, M_c)
    xi_true: np.ndarray,      # (K, N)
    J_star: np.ndarray,       # (N, N) - riferimento ideale per FRO
    xi_prev: Optional[np.ndarray],  # None al round 0, poi (S, N)
    hp: HyperParams,
) -> Tuple[np.ndarray, np.ndarray, RoundLog]:
    """
    Esegue un round completo in SINGLE-mode e restituisce:
      - xi_ref_new   : memoria aggiornata per round successivo
      - J_KS         : matrice server (post-propagation) del round
      - RoundLog     : metriche

    Note:
      - mp: passa M_eff del *round corrente* a estimate_keff.
      - fallback memoria: se xi_r ha meno di K candidati, conserva tutti.
    """
    if ETA_t.ndim != 3:
        raise ValueError("ETA_t atteso (L, M_c, N).")
    if labels_t.ndim != 2:
        raise ValueError("labels_t atteso (L, M_c).")

    # 1) stima unsup per round t
    J_unsup, M_eff = build_unsup_J_single(ETA_t, K=hp.K)

    # 2) blending con memoria ebraica precedente (se presente)
    J_rec = blend_with_memory(J_unsup, xi_prev=xi_prev, w=hp.w)

    # 3) propagazione pseudo-inversa
    J_KS = np.asarray(propagate_J(J_rec, J_real=-1, verbose=False, iters=hp.prop.iters), dtype=np.float32)

    # 4) cut spettrale
    V, *_ = spectral_cut(J_KS, tau=hp.spec.tau)

    # 5) disentangling + magnetizzazioni (robusto con fallback interno)
    xi_r, _m_vec = dis_check(
        V=V,
        K=hp.K,
        L=hp.L,
        J_rec=J_rec,
        JKS_iter=J_KS,
        xi_true=xi_true,
        tam=hp.tam,
        spec=hp.spec,
        show_progress=hp.use_tqdm,
    )

    # 6) metriche
    #    6a) retrieval (matching ungherese)
    retr = retrieval_mean_hungarian(xi_r.astype(int), xi_true.astype(int))
    #    6b) coverage su questo round
    cov = compute_round_coverage(labels_t, K=hp.K)
    #    6c) FRO vs J*
    fro = frobenius_relative(J_KS, J_star)
    #    6d) K_eff
    if hp.estimate_keff_method == "mp":
        K_eff, _, _ = estimate_keff(J_KS, method="mp", M_eff=M_eff)
    else:
        K_eff, _, _ = estimate_keff(J_KS, method="shuffle")

    # 7) aggiorna memoria xi_ref per round successivo
    if xi_r.shape[0] >= hp.K:
        xi_ref_new = xi_r[: hp.K].astype(int)
    else:
        xi_ref_new = xi_r.astype(int)

    return xi_ref_new, J_KS, RoundLog(retrieval=float(retr), fro=float(fro), keff=int(K_eff), coverage=float(cov))
