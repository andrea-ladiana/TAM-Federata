#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 02.1 — Sweep di m (feature masking) con m uguale per tutti i client
==============================================================================
Scenario: ogni client osserva la stessa frazione m di feature (neuroni), ma le maschere sono
campionate indipendentemente tra client (stessa copertura attesa, insiemi diversi). Si scandisce m
in [m_min, m_max] e si misura l'effetto sui risultati finali.

Metriche finali (per ciascun m, aggregate su più seed):
- m_final: overlap medio di Mattis (con matching Hungarian) tra archetipi ricostruiti e veri
- fro_final: Frobenius relativa ||J - J*||_F / ||J*||_F
- K_eff_final (MP): stima del rank efficace via Marchenko–Pastur
- SNR_spettrale_finale
- pair_coverage: frazione di coppie (i,j) coperte da almeno un client (proxy)

Output per configurazione:
- hyperparams.json
- log.jsonl (una riga per seed e per m)
- fig_summary.png (grafico Seaborn con bande SE vs m)

Dipendenze: numpy, matplotlib, seaborn, scipy, + i tuoi file Functions.py, Networks.py, Dynamics.py.
Acronimi espansi alla prima occorrenza: FL = Federated Learning (apprendimento federato),
TAM = Tripartite Associative Memory, MP = Marchenko–Pastur, SNR = Signal-to-Noise Ratio (rapporto segnale/rumore).
"""
from __future__ import annotations
import os
import sys
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# tqdm (opzionale) per barre di avanzamento
# ---------------------------------------------------------------------
try:  # uso auto per compatibilità notebook/console
    from tqdm.auto import trange, tqdm  # type: ignore
    _TQDM_AVAILABLE = True
except Exception:  # pragma: no cover
    _TQDM_AVAILABLE = False
    def trange(*args, **kwargs):  # type: ignore
        return range(*args)
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return iterable if iterable is not None else range(0)

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
# Import moduli di progetto (assumiamo questo file in: <ROOT>/stress_tests/exp02_feature_masking/exp21_sweep_m/)
# ---------------------------------------------------------------------
# Root progetto (cartella UNSUP) = parent della cartella 'stress_tests'
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
# Dataclass iperparametri
# ---------------------------------------------------------------------
@dataclass
class HyperParams:
    # Modello / dataset
    L: int = 3
    K: int = 6
    N: int = 500
    n_batch: int = 10
    M_total: int = 3000  # totale esempi su tutti i client e round
    r_ex: float = 0.6

    # Sweep di m
    m_min: float = 0.3
    m_max: float = 0.9
    m_steps: int = 7  # numero di valori equispaziati tra m_min e m_max

    # Dinamica TAM (Tripartite Associative Memory)
    updates: int = 60
    beta_T: float = 2.5
    lam: float = 0.2
    h_in: float = 0.1

    # Blending federato (peso sulla componente unsupervised)
    w: float = 0.9

    # Esperimento
    n_seeds: int = 6
    seed_base: int = 1001

    # Plot
    palette: str = "deep"
    style: str = "whitegrid"
    # tqdm flags
    use_tqdm_m: bool = True       # barra sui valori di m
    use_tqdm_seeds: bool = True   # barra interna sui seed per ciascun m
    use_tqdm_rounds: bool = True  # barra sui round intra-seed
    use_tqdm_updates: bool = True # riusa la stessa barra interna di dis_check se disponibile

# ---------------------------------------------------------------------
# Utilità per dataset e aggregazione
# ---------------------------------------------------------------------

def make_feature_masks_equal_fraction(L: int, N: int, m: float, rng: np.random.Generator) -> np.ndarray:
    """Maschere (L, N) booleane: ogni client osserva esattamente ~m*N neuroni (senza rimpiazzo)."""
    masks = np.zeros((L, N), dtype=bool)
    n_obs = max(1, int(round(m * N)))
    for l in range(L):
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
    """Genera ETA con feature masking: E[l,b,·,i]=0 se mask[l,i]=False.
    Ritorna ETA: (L, n_batch, M_c, N) e labels (indici archetipi usati) per analisi a valle.
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
            # Costruisci esempi con flips indipendenti
            probs = rng.uniform(size=(M_c, N))
            chi = np.where(probs <= p_keep, 1.0, -1.0)
            xi_sel = xi_true[mus]
            E = (chi * xi_sel).astype(np.float32)
            E[:, ~mask_l] = 0.0
            ETA[l, b] = E
    return ETA, labels


def _client_J_with_mask(E_l: np.ndarray, mask_l: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Hebb non supervisionato su client l con rinormalizzazione per N_obs e mascheratura pairwise."""
    N = E_l.shape[1]
    N_obs = int(mask_l.sum())
    M_eff_param = max(1, E_l.shape[0] // K)
    J_l = unsupervised_J(E_l, M_eff_param)  # scala ~ 1/(N*M_eff_param)
    J_l *= (N / max(1, N_obs))  # compensa il minor numero di feature
    W_l = np.outer(mask_l.astype(np.float32), mask_l.astype(np.float32))
    J_l *= W_l
    return J_l, W_l


def aggregate_clients_J(E: np.ndarray, masks: np.ndarray, K: int) -> Tuple[np.ndarray, float]:
    """Media pairwise pesata su (i,j): J = (∑_l J_l) / (∑_l W_l), con W_l=mask_l⊗mask_l.
    Ritorna J e pair_coverage = frazione di coppie osservate da almeno un client.
    """
    L, M_eff, N = E.shape
    sumJ = np.zeros((N, N), dtype=np.float32)
    sumW = np.zeros((N, N), dtype=np.float32)
    for l in range(L):
        J_l, W_l = _client_J_with_mask(E[l], masks[l], K)
        sumJ += J_l
        sumW += W_l
    J = sumJ / np.clip(sumW, 1.0, None)
    pair_cov = float((sumW > 0).mean())
    return J, pair_cov


def spectral_snr(J: np.ndarray, K: int) -> float:
    """SNR spettrale: rapporto tra somma dei primi K autovalori positivi e la somma del resto."""
    vals = np.real(np.linalg.eigvals(J))
    pos = np.sort(vals[vals > 0.0])[::-1]
    if pos.size == 0:
        return 0.0
    s_sig = pos[: min(K, pos.size)].sum()
    s_noise = pos[min(K, pos.size):].sum()
    if s_noise <= 1e-12:
        return float(np.inf)
    return float(s_sig / s_noise)

# ---------------------------------------------------------------------
# Core run per un singolo seed e valore di m
# ---------------------------------------------------------------------

def run_one_seed_one_m(hp: HyperParams, seed: int, m: float) -> dict:
    rng = np.random.default_rng(seed)
    xi_true = gen_patterns(hp.N, hp.K)
    J_star = JK_real(xi_true)

    masks = make_feature_masks_equal_fraction(hp.L, hp.N, m, rng)
    ETA, _ = gen_dataset_feature_masking(xi_true, hp.M_total, hp.r_ex, hp.n_batch, hp.L, masks, rng)

    xi_ref = None
    round_iter = trange(hp.n_batch, desc=f"m={m:.2f} seed {seed}", leave=False) if (hp.use_tqdm_rounds and _TQDM_AVAILABLE) else range(hp.n_batch)
    for b in round_iter:
        ETA_extend = ETA[:, : b + 1, :, :].transpose(0, 2, 1, 3).reshape(hp.L, -1, hp.N)
        J_unsup_extend, pair_cov = aggregate_clients_J(ETA_extend, masks, hp.K)

        if b == 0 or xi_ref is None:
            J_rec_extend = J_unsup_extend.copy()
        else:
            J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
            J_rec_extend = hp.w * J_unsup_extend + (1.0 - hp.w) * J_hebb_prev

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

    # metriche finali @ ultimo round
    fro_final = float(np.linalg.norm(JKS_iter_ext - J_star, ord='fro') / (np.linalg.norm(J_star, ord='fro') + 1e-9))
    try:
        K_eff_final, _, _ = estimate_K_eff_from_J(JKS_iter_ext, method='shuffle', M_eff=ETA_extend.shape[1])
    except Exception:
        K_eff_final = hp.K
    snr_final = spectral_snr(JKS_iter_ext, hp.K)

    # Matching Hungarian per m_final
    from scipy.optimize import linear_sum_assignment
    assert xi_ref is not None, "xi_ref None: nessun round eseguito?"
    K_hat, Nloc = xi_ref.shape
    M = np.abs(xi_ref @ xi_true.T / Nloc)
    cost = 1.0 - M
    rI, cI = linear_sum_assignment(cost)
    overlaps = M[rI, cI]
    if K_hat < hp.K:
        m_final = float(overlaps.sum() / hp.K)
    else:
        m_final = float(overlaps.mean())

    return {
        "m": float(m),
        "seed": int(seed),
        "m_final": m_final,
        "fro_final": fro_final,
        "K_eff_final": int(K_eff_final),
        "snr_final": float(snr_final),
        "pair_coverage": float(pair_cov),
    }

# ---------------------------------------------------------------------
# Aggregazione e grafici
# ---------------------------------------------------------------------

def aggregate_and_plot(hp: HyperParams, rows: List[dict], out_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.sort_values(["m", "seed"], inplace=True)

    # Salva un csv comodo per analisi future (opzionale oltre al log.jsonl)
    df.to_csv(out_dir / "results_table.csv", index=False)

    sns.set_theme(style=hp.style, palette=hp.palette)  # type: ignore[arg-type]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    # 1) m_final vs m
    ax = axes[0, 0]
    sns.pointplot(data=df, x="m", y="m_final", errorbar=("se", 1), dodge=True, ax=ax)  # type: ignore[arg-type]
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Retrieval finale vs m")
    ax.set_xlabel("frazione di feature osservate (m)")
    ax.set_ylabel("Mattis overlap (finale)")
    ax.grid(True, alpha=0.3)

    # 2) Frobenius finale vs m
    ax = axes[0, 1]
    sns.pointplot(data=df, x="m", y="fro_final", errorbar=("se", 1), dodge=True, ax=ax)  # type: ignore[arg-type]
    ax.set_title("Frobenius relativa finale vs m")
    ax.set_xlabel("m")
    ax.set_ylabel("||J−J*||_F / ||J*||_F")
    ax.grid(True, alpha=0.3)

    # 3) K_eff finale (MP) vs m
    ax = axes[1, 0]
    sns.pointplot(data=df, x="m", y="K_eff_final", errorbar=("se", 1), dodge=True, ax=ax)  # type: ignore[arg-type]
    ax.axhline(hp.K, color="tab:green", linestyle=":", linewidth=1.2, label="K (vero)")
    ax.set_title("K_eff (MP) finale vs m")
    ax.set_xlabel("m")
    ax.set_ylabel("K_eff")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # 4) SNR spettrale finale vs m
    ax = axes[1, 1]
    sns.pointplot(data=df, x="m", y="snr_final", errorbar=("se", 1), dodge=True, ax=ax)  # type: ignore[arg-type]
    ax.set_title("SNR spettrale finale vs m")
    ax.set_xlabel("m")
    ax.set_ylabel("SNR")
    ax.grid(True, alpha=0.3)

    fig_path = out_dir / "fig_summary.png"
    fig.savefig(fig_path, dpi=150)
    print(f"[plot] Figura salvata in: {fig_path}")
    plt.show()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    hp = HyperParams(
        L=3,
        K=6,
        N=200,
        n_batch=10,
        M_total=200,
        r_ex=0.6,
        m_min=0.3,
        m_max=0.9,
        m_steps=7,
        updates=60,
        beta_T=2.5,
        lam=0.2,
        h_in=0.1,
        w=0.8,
        n_seeds=3,
        seed_base=777,
    )

    base_dir = ROOT / "stress_tests" / "exp021_sweep_m"
    m_grid = np.linspace(hp.m_min, hp.m_max, hp.m_steps)
    tag = (
        f"K{hp.K}_N{hp.N}_L{hp.L}_nb{hp.n_batch}_M{hp.M_total}_r{hp.r_ex}_" 
        f"m{hp.m_min:.2f}-{hp.m_max:.2f}_steps{hp.m_steps}_w{hp.w}"
    )
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Salva hyperparams
    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    # Loop su m e seed, log jsonl
    rows: List[dict] = []
    log_path = exp_dir / "log.jsonl"
    with open(log_path, "w") as flog:
        m_iter = trange(len(m_grid), desc="m values") if (hp.use_tqdm_m and _TQDM_AVAILABLE) else range(len(m_grid))
        for mi in m_iter:
            m = float(m_grid[mi])
            seed_iter = trange(hp.n_seeds, desc=f"m={m:.2f} seeds", leave=False) if (hp.use_tqdm_seeds and _TQDM_AVAILABLE) else range(hp.n_seeds)
            for s in seed_iter:
                seed = hp.seed_base + s
                t0 = time.perf_counter()
                row = run_one_seed_one_m(hp, seed, m)
                t1 = time.perf_counter()
                row["elapsed_s"] = t1 - t0
                rows.append(row)
                flog.write(json.dumps(row) + "\n")
                if hp.use_tqdm_seeds and _TQDM_AVAILABLE:
                    if hasattr(seed_iter, 'set_postfix'):
                        seed_iter.set_postfix({  # type: ignore[attr-defined]
                            'm_final': f"{row['m_final']:.2f}",
                            'fro': f"{row['fro_final']:.2f}"
                        })
                else:
                    print(
                        f"[m={m:.2f} seed={seed}] m_final={row['m_final']:.3f} "
                        f"fro_final={row['fro_final']:.3f} K_eff={row['K_eff_final']} SNR={row['snr_final']:.2f}"
                    )

    # Grafico riassuntivo
    aggregate_and_plot(hp, rows, exp_dir)


if __name__ == "__main__":
    main()
