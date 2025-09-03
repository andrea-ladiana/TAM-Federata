#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 02 — Feature Masking per Client
=========================================
Scenario (b): ogni client osserva solo una frazione m dei neuroni (feature masking). L'unione dei client
copre (quasi) tutte le feature, ma ogni client invia statistiche parziali. I round servono a mediare il rumore
e a ricomporre l'informazione strutturale.

Output per ogni configurazione:
- hyperparams.json
- log.jsonl (una riga per seed con serie round-wise)
- fig_metrics.png (grafico Seaborn con bande SE)

Dipendenze: numpy, matplotlib, seaborn, scipy, tqdm (opzionale), + i tuoi file Functions.py, Networks.py, Dynamics.py.
"""

from __future__ import annotations
import os
import sys
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# Silenzia log TensorFlow PRIMA di importare moduli locali che potrebbero
# importare TensorFlow. Questo evita i messaggi INFO (oneDNN, cpu_feature_guard).
# ---------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 3=ERROR only
# Disabilita ottimizzazioni oneDNN per eliminare l'avviso opzionale
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
try:  # import lazy: se TF non è installato non fallisce
    import tensorflow as tf  # type: ignore
    try:
        tf.get_logger().setLevel('ERROR')
    except Exception:
        pass
    try:
        tf.autograph.set_verbosity(0)  # type: ignore
    except Exception:
        pass
    try:  # TF1 compat (se disponibile)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # type: ignore[attr-defined]
    except Exception:
        pass
except Exception:
    pass

# ---------------------------------------------------------------------
# tqdm (opzionale) per barre di avanzamento
# ---------------------------------------------------------------------
try:  # uso auto per compatibilità notebook/console
    from tqdm.auto import trange, tqdm  # type: ignore
    _TQDM_AVAILABLE = True
except Exception:  # pragma: no cover - fallback se tqdm non presente
    _TQDM_AVAILABLE = False
    def trange(*args, **kwargs):  # type: ignore
        return range(*args)
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return iterable if iterable is not None else range(0)

# ---------------------------------------------------------------------
# Import moduli locali del progetto
# ---------------------------------------------------------------------
# Root del progetto (cartella UNSUP) = parent della directory 'stress_tests'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Functions import (
    gen_patterns,
    JK_real,
    unsupervised_J,
    propagate_J,
    estimate_K_eff_from_J,
)
from Networks import TAM_Network
from Dynamics import dis_check

# ---------------------------------------------------------------------
# Dataclass iperparametri e contenitori
# ---------------------------------------------------------------------
@dataclass
class HyperParams:
    # Modello / dataset
    L: int = 3
    K: int = 6
    N: int = 500
    n_batch: int = 10
    M_total: int = 3000  # totale su tutti i client e round
    r_ex: float = 0.6
    m_min: float = 0.3   # frazione minima di feature osservate per client
    m_max: float = 0.6   # frazione massima di feature osservate per client (inclusivo)

    # Dinamica TAM
    updates: int = 60
    beta_T: float = 2.5
    lam: float = 0.2
    h_in: float = 0.1

    # Blending federato (unsup vs Hebb su archetipi ricostruiti)
    w: float = 0.9

    # Esperimento
    n_seeds: int = 6
    seed_base: int = 123

    # Plot
    palette: str = "deep"
    style: str = "whitegrid"
    # Utility
    # tqdm flags
    use_tqdm_seeds: bool = True    # barra esterna sui seed
    use_tqdm_rounds: bool = True   # barra interna sui round
    use_tqdm_updates: bool = False # barra iterazioni interne (dis_check)
    # retro-compat: vecchio flag (non più usato direttamente)
    use_tqdm: bool = True

@dataclass
class RoundSeries:
    rounds: List[int]
    m_extend_mean: List[float]
    fro_extend: List[float]
    keff_extend: List[int]
    snr_extend: List[float]

@dataclass
class SeedResult:
    seed: int
    m_first: float
    m_final: float
    G_ext: float
    fro_final: float
    deltaK: int
    pair_coverage: float
    series: RoundSeries

# ---------------------------------------------------------------------
# Generazione maschere e dataset
# ---------------------------------------------------------------------

def make_feature_masks(L: int, N: int, m_min: float, m_max: float, rng: np.random.Generator) -> np.ndarray:
    """Restituisce un array (L, N) booleano: mask[l, i] = True se il client l osserva il neurone i."""
    masks = np.zeros((L, N), dtype=bool)
    for l in range(L):
        m = rng.uniform(m_min, m_max)
        n_obs = max(1, int(round(m * N)))
        idx = rng.choice(N, size=n_obs, replace=False)
        masks[l, idx] = True
    return masks


def gen_dataset_feature_masking(
    xi_true: np.ndarray,
    M_total: int,
    r_ex: float,
    n_batch: int,
    L: int,
    masks: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera ETA: ogni client l vede solo le feature consentite da masks[l].

    Ritorna
    -------
    ETA : array (L, n_batch, M_c, N)
    labels : array (L, n_batch, M_c) con l'indice μ generativo per ogni esempio
    """
    K, N = xi_true.shape
    M_c = math.ceil(M_total / (L * n_batch))
    ETA = np.zeros((L, n_batch, M_c, N), dtype=np.float32)
    labels = np.zeros((L, n_batch, M_c), dtype=np.int32)
    p_keep = 0.5 * (1.0 + r_ex)

    for l in range(L):
        mask_l = masks[l]
        for b in range(n_batch):
            mus = rng.integers(0, K, size=M_c)
            labels[l, b] = mus
            probs = rng.uniform(size=(M_c, N))
            chi = np.where(probs <= p_keep, 1.0, -1.0)
            xi_sel = xi_true[mus]  # (M_c, N)
            E = (chi * xi_sel).astype(np.float32)
            E[:, ~mask_l] = 0.0  # maschera feature non osservate
            ETA[l, b] = E
    return ETA, labels

# ---------------------------------------------------------------------
# Aggregazione pesata (pairwise) di J con maschere feature
# ---------------------------------------------------------------------

def _client_J_with_mask(E_l: np.ndarray, mask_l: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Hebb non supervisionato su client l con rinormalizzazione per N_obs e peso pairwise.
    Ritorna J_l_masked (NxN) e W_l = mask_l ⊗ mask_l (NxN) per l'aggregazione.
    """
    N = E_l.shape[1]
    N_obs = int(mask_l.sum())
    M_eff_param = max(1, E_l.shape[0] // K)
    J_l = unsupervised_J(E_l, M_eff_param)  # scala ~ 1/(N * M_eff_param)
    # rinormalizza per il minor numero di feature osservate
    scale = (N / max(1, N_obs))
    J_l *= scale
    # azzera coppie non osservate
    W_l = np.outer(mask_l.astype(np.float32), mask_l.astype(np.float32))
    J_l *= W_l
    return J_l, W_l


def aggregate_clients_J(E: np.ndarray, masks: np.ndarray, K: int) -> Tuple[np.ndarray, float]:
    """Aggrega le J dei client con media pairwise pesata dai W_l.
    E: (L, M_eff, N)
    masks: (L, N)
    """
    L, M_eff, N = E.shape
    sumJ = np.zeros((N, N), dtype=np.float32)
    sumW = np.zeros((N, N), dtype=np.float32)
    for l in range(L):
        J_l, W_l = _client_J_with_mask(E[l], masks[l], K)
        sumJ += J_l
        sumW += W_l
    J = sumJ / np.clip(sumW, 1.0, None)
    # pair coverage (frazione di coppie i,j coperte da almeno un client)
    coverage_pairs = float((sumW > 0.0).mean())
    return J, coverage_pairs

# ---------------------------------------------------------------------
# Core run per seed
# ---------------------------------------------------------------------

def spectral_snr(J: np.ndarray, K: int) -> float:
    vals = np.real(np.linalg.eigvals(J))
    pos = np.sort(vals[vals > 0.0])[::-1]
    if pos.size == 0:
        return 0.0
    s_sig = pos[: min(K, pos.size)].sum()
    s_noise = pos[min(K, pos.size):].sum()
    if s_noise <= 1e-12:
        return float(np.inf)
    return float(s_sig / s_noise)


def run_one_seed(hp: HyperParams, seed: int, *, out_dir: Path) -> SeedResult:
    rng = np.random.default_rng(seed)
    xi_true = gen_patterns(hp.N, hp.K)
    J_star = JK_real(xi_true)

    masks = make_feature_masks(hp.L, hp.N, hp.m_min, hp.m_max, rng)
    ETA, labels = gen_dataset_feature_masking(
        xi_true, hp.M_total, hp.r_ex, hp.n_batch, hp.L, masks, rng
    )

    fro_extend_rounds: List[float] = []
    magn_extend_rounds: List[float] = []
    keff_mp_extend_rounds: List[int] = []
    snr_extend_rounds: List[float] = []

    xi_ref = None

    round_iter = trange(hp.n_batch, desc=f"seed {seed} rounds", leave=False) if (hp.use_tqdm_rounds and _TQDM_AVAILABLE) else range(hp.n_batch)
    for b in round_iter:
        # Costruisci viste single/extend
        ETA_round = ETA[:, b, :, :]
        ETA_extend = ETA[:, : b + 1, :, :].transpose(0, 2, 1, 3).reshape(hp.L, -1, hp.N)

        # Aggregazione pesata delle J
        J_unsup_round, _ = aggregate_clients_J(ETA_round, masks, hp.K)
        J_unsup_extend, pair_cov = aggregate_clients_J(ETA_extend, masks, hp.K)

        # Blending con Hebb sugli archetipi precedenti (se disponibili)
        if b == 0 or xi_ref is None:
            J_rec_extend = J_unsup_extend.copy()
        else:
            J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
            J_rec_extend = hp.w * J_unsup_extend + (1.0 - hp.w) * J_hebb_prev

        # Propagazione + disentangling TAM
        JKS_iter_ext = propagate_J(J_rec_extend, iters=1, verbose=False)
        vals_ext, vecs_ext = np.linalg.eig(JKS_iter_ext)
        order = np.argsort(np.real(vals_ext))[::-1]
        autov_ext = np.real(vecs_ext[:, order[: hp.K]]).T

        Net = TAM_Network()
        Net.prepare(J_rec_extend, hp.L)
        xi_r_ext, magn_ext = dis_check(
            autov_ext, hp.K, hp.L, J_rec_extend, JKS_iter_ext,
            ξ=xi_true, updates=hp.updates,
            show_bar=(hp.use_tqdm_updates and _TQDM_AVAILABLE)
        )
        xi_ref = xi_r_ext

        # Metriche
        fro_rel = float(np.linalg.norm(JKS_iter_ext - J_star, ord='fro') / (np.linalg.norm(J_star, ord='fro') + 1e-9))
        fro_extend_rounds.append(fro_rel)
        magn_extend_rounds.append(float(np.mean(magn_ext)))
        try:
            K_eff_mp_ext, _, _ = estimate_K_eff_from_J(JKS_iter_ext, method='shuffle', M_eff=ETA_extend.shape[1])
        except Exception:
            K_eff_mp_ext = int(hp.K)
        keff_mp_extend_rounds.append(int(K_eff_mp_ext))
        snr_extend_rounds.append(spectral_snr(JKS_iter_ext, hp.K))

    # Confronto first vs final con matching Hungarian
    def _match_and_overlap(estimated: np.ndarray, true: np.ndarray) -> float:
        from scipy.optimize import linear_sum_assignment
        K_hat, Nloc = estimated.shape
        Kt, _ = true.shape
        M = np.abs(estimated @ true.T / Nloc)
        cost = 1.0 - M
        rI, cI = linear_sum_assignment(cost)
        overlaps = M[rI, cI]
        if K_hat < Kt:
            return float(overlaps.sum() / Kt)
        return float(overlaps.mean())

    # First (extend@b=0)
    ETA_first = ETA[:, :1, :, :].transpose(0, 2, 1, 3).reshape(hp.L, -1, hp.N)
    J_first, _ = aggregate_clients_J(ETA_first, masks, hp.K)
    JKS_first = propagate_J(J_first, iters=1, verbose=False)
    vals0, vecs0 = np.linalg.eig(JKS_first)
    order0 = np.argsort(np.real(vals0))[::-1]
    autov0 = np.real(vecs0[:, order0[: hp.K]]).T
    xi_r_first, magn_first = dis_check(autov0, hp.K, hp.L, J_first, JKS_first,
                                       ξ=xi_true, updates=hp.updates,
                                       show_bar=(hp.use_tqdm_updates and _TQDM_AVAILABLE))

    m_first = _match_and_overlap(xi_r_first, xi_true)
    assert xi_ref is not None, "xi_ref should not be None after rounds"
    m_final = _match_and_overlap(xi_ref, xi_true)
    G_ext = m_final - m_first

    series = RoundSeries(
        rounds=list(range(hp.n_batch)),
        m_extend_mean=magn_extend_rounds,
        fro_extend=fro_extend_rounds,
        keff_extend=keff_mp_extend_rounds,
        snr_extend=snr_extend_rounds,
    )

    deltaK = abs(int(keff_mp_extend_rounds[-1]) - hp.K)

    return SeedResult(
        seed=seed,
        m_first=m_first,
        m_final=m_final,
        G_ext=G_ext,
        fro_final=fro_extend_rounds[-1],
        deltaK=deltaK,
        pair_coverage=float((np.any(masks, axis=0).mean())**2),  # proxy coppie
        series=series,
    )

# ---------------------------------------------------------------------
# Aggregazione, logging e grafico
# ---------------------------------------------------------------------

def aggregate_and_plot(hp: HyperParams, results: List[SeedResult], exp_dir: Path) -> None:
    rounds = results[0].series.rounds
    arr_m = np.array([r.series.m_extend_mean for r in results])
    arr_f = np.array([r.series.fro_extend for r in results])
    arr_k = np.array([r.series.keff_extend for r in results])
    arr_s = np.array([r.series.snr_extend for r in results])
    pair_cov = float(np.mean([r.pair_coverage for r in results]))

    def mean_se(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = x.mean(axis=0)
        se = x.std(axis=0, ddof=1) / np.sqrt(max(1, x.shape[0]))
        return m, se

    m_mean, m_se = mean_se(arr_m)
    f_mean, f_se = mean_se(arr_f)
    k_mean, k_se = mean_se(arr_k)
    s_mean, s_se = mean_se(arr_s)

    sns.set_theme(style=hp.style, palette=hp.palette)  # type: ignore[arg-type]
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)

    # Retrieval
    ax = axes[0]
    ax.plot(rounds, m_mean, label="retrieval (extend)")
    ax.fill_between(rounds, m_mean - m_se, m_mean + m_se, alpha=0.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("round")
    ax.set_ylabel("Mattis overlap")
    ax.set_title("Retrieval per round (feature masking per client)")
    ax.grid(True, alpha=0.3)

    # Frobenius
    ax = axes[1]
    ax.plot(rounds, f_mean, label="Frobenius rel.")
    ax.fill_between(rounds, f_mean - f_se, f_mean + f_se, alpha=0.2)
    ax.set_xlabel("round")
    ax.set_ylabel("||J−J*||_F / ||J*||_F")
    ax.set_title("Convergenza strutturale di J")
    ax.grid(True, alpha=0.3)

    # K_eff e SNR
    ax = axes[2]
    ax.plot(rounds, k_mean, label="K_eff (MP)")
    ax.fill_between(rounds, k_mean - k_se, k_mean + k_se, alpha=0.15)
    ax2 = ax.twinx()
    ax2.plot(rounds, s_mean, linestyle='--', label="SNR spettrale")

    # coverage annotazione
    ax.axhline(hp.K, color='tab:green', linestyle=':', linewidth=1.2, label='K (true)')
    ax2.set_ylabel("SNR")
    ax.set_xlabel("round")
    ax.set_ylabel("K_eff")
    ax.set_title(f"Rank efficace e SNR (pair coverage≈{pair_cov:.2f})")
    ax.grid(True, alpha=0.3)

    # Legenda combinata
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    fig_path = exp_dir / "fig_metrics.png"
    fig.savefig(fig_path, dpi=150)
    print(f"[plot] Figure salvata in: {fig_path}")
    plt.show()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    hp = HyperParams(
        L=5,
        K=12,
        N=100,
        n_batch=10,
        M_total=100,
        r_ex=0.6,
        m_min=0.3,
        m_max=0.9,
        updates=60,
        beta_T=2.5,
        lam=0.2,
        h_in=0.1,
        w=0.8,
        n_seeds=6,
        seed_base=321,
    )

    base_dir = ROOT / "stress_tests" / "exp02_feature_masking"
    tag = (
        f"K{hp.K}_N{hp.N}_L{hp.L}_nb{hp.n_batch}_M{hp.M_total}_r{hp.r_ex}_"
        f"m{hp.m_min:.2f}-{hp.m_max:.2f}_w{hp.w}"
    )
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] Esperimento: {exp_dir}")

    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    results: List[SeedResult] = []
    log_path = exp_dir / "log.jsonl"
    with open(log_path, "w") as flog:
        seed_iter = trange(hp.n_seeds, desc="seeds") if (hp.use_tqdm_seeds and _TQDM_AVAILABLE) else range(hp.n_seeds)
        for s in seed_iter:
            seed = hp.seed_base + s
            t0 = time.perf_counter()
            res = run_one_seed(hp, seed, out_dir=exp_dir)
            t1 = time.perf_counter()
            results.append(res)
            row = {
                "seed": res.seed,
                "m_first": res.m_first,
                "m_final": res.m_final,
                "G_ext": res.G_ext,
                "fro_final": res.fro_final,
                "deltaK": res.deltaK,
                "pair_coverage": res.pair_coverage,
                "rounds": res.series.rounds,
                "m_series": res.series.m_extend_mean,
                "fro_series": res.series.fro_extend,
                "keff_series": res.series.keff_extend,
                "snr_series": res.series.snr_extend,
                "elapsed_s": t1 - t0,
            }
            flog.write(json.dumps(row) + "\n")
            if hp.use_tqdm_seeds and _TQDM_AVAILABLE:
                _si = seed_iter  # alias
                if hasattr(_si, 'set_postfix'):
                    _si.set_postfix({  # type: ignore[attr-defined]
                        "m_final": f"{res.m_final:.2f}",
                        "fro": f"{res.fro_final:.2f}",
                    })
            else:
                print(
                    f"[seed {seed}] m_final={res.m_final:.3f} G_ext={res.G_ext:.3f} "
                    f"fro_final={res.fro_final:.3f} ΔK={res.deltaK} cov≈{res.pair_coverage:.2f}"
                )

    aggregate_and_plot(hp, results, exp_dir)


if __name__ == "__main__":
    main()
