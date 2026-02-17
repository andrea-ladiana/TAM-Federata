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

    # 2) blending con memoria  precedente (se presente)
    J_rec = blend_with_memory(J_unsup, xi_prev=xi_prev, w=hp.w)

    # 3) propagazione pseudo-inversa
    J_KS = np.asarray(propagate_J(J_rec, J_real=-1, verbose=False, iters=hp.prop.iters), dtype=np.float32)

    # 4) cut spettrale
    _spec_out = spectral_cut(J_KS, tau=hp.spec.tau, return_info=True)
    if len(_spec_out) == 3:
        V, k_eff_cut, info_spec = _spec_out
    else:  # fallback (shouldn't happen with return_info=True but guard defensively)
        V, k_eff_cut = _spec_out
        info_spec = {"evals": None}

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
    # NOTE: BUGFIX (2025-09-04): in precedenza si usava xi_r.astype(int) che, per valori float in (-1,1),
    #       li troncava a 0 abbattendo gli overlap (~0.1 medio). Ora binarizziamo con segno in {+1,-1}.
    xi_r_bin = np.where(xi_r >= 0, 1, -1).astype(np.int8)
    retr = retrieval_mean_hungarian(xi_r_bin, xi_true.astype(int))
    #    6b) coverage su questo round
    cov = compute_round_coverage(labels_t, K=hp.K)
    #    6c) FRO vs J*
    fro = frobenius_relative(J_KS, J_star)
    #    6d) K_eff
    if hp.estimate_keff_method == "mp":
        K_eff, _, _ = estimate_keff(J_KS, method="mp", M_eff=M_eff)
    else:
        K_eff, _, _ = estimate_keff(J_KS, method="shuffle")

    # --- DIAGNOSTICA FACOLTATIVA ---
    # Abilita impostando una variabile di ambiente UNSUP_DEBUG=1 per evitare stampe rumorose di default.
    import os
    if os.environ.get("UNSUP_DEBUG", "0") == "1":
        evals = info_spec.get("evals")
        # prime 5 autovalori
        top_evals = np.array2string(evals[:5], precision=3) if evals is not None else "[]"
        print(f"[DEBUG single_round] k_spec={k_eff_cut} K_eff={K_eff} retr~{float(retr):.3f} fro={float(fro):.3f} top_eigs={top_evals}")

    # 7) aggiorna memoria xi_ref per round successivo
    if xi_r_bin.shape[0] >= hp.K:
        xi_ref_new = xi_r_bin[: hp.K]
    else:
        xi_ref_new = xi_r_bin

    return xi_ref_new, J_KS, RoundLog(retrieval=float(retr), fro=float(fro), keff=int(K_eff), coverage=float(cov))
