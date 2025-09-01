#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 03 — DP-like Gaussian noise direttamente su J locali
==============================================================
Scenario: ogni client costruisce J_l non supervisionata (Hebb dagli esempi locali) e vi aggiunge
rumore gaussiano simmetrico (DP-like) prima dell'aggregazione sul server. L'obiettivo è far sì che
il singolo round sia sotto-informato, ma che l'accumulo di round (extend) e la media sui client
recuperino il segnale.

Output per run:
- hyperparams.json
- log.jsonl (una riga per seed con serie round-wise)
- fig_metrics.png (Seaborn): retrieval, Frobenius, K_eff(shuffle) vs round

Sigle spiegate:
- FL = Federated Learning (apprendimento federato)
- TAM = Tripartite Associative Memory
- DP = Differential Privacy (privacy differenziale) — qui simulata come rumore Gaussiano su J
- shuffle = stimatore K_eff basato su baseline a permutazione
"""

from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # sopprime log TF se presenti
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import json
import math
import time
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Import moduli di progetto
# Nota: questo file risiede in <UNSUP>/stress_tests/, quindi il root del progetto
# (dove vivono Files come Functions.py, Dynamics.py, Networks.py) è il parent
# immediato della cartella stress_tests. In precedenza si usava parents[2], che
# puntava una directory troppo in alto (es. .../TAMFed) causando
# ModuleNotFoundError per 'Functions'. Usiamo quindi parents[1].
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # == <UNSUP>
if str(ROOT) not in sys.path:
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
    n_batch: int = 12
    M_total: int = 2400  # totale esempi su tutti i client e round
    r_ex: float = 0.6

    # Rumore DP-like su J locali
    noise_std0: float = 0.8   # scala base del rumore per round 0
    noise_schedule: str = "1/sqrt(round*M)"  # oppure "const", "1/sqrt(round)", "1/sqrt(M)"
    symmetrize: bool = True
    zero_diag: bool = True

    # Dinamica TAM
    updates: int = 60
    beta_T: float = 2.5
    lam: float = 0.2
    h_in: float = 0.1

    # Blending federato (peso sulla componente unsupervised)
    w: float = 0.9

    # Esperimento
    n_seeds: int = 6
    seed_base: int = 9001

    # Plot
    palette: str = "deep"
    style: str = "whitegrid"
    # Progress
    show_round_bar: bool = True

# ---------------------------------------------------------------------
# Dataset unsupervised semplice (tutti i client vedono tutti gli archetipi/feature)
# ---------------------------------------------------------------------

def gen_dataset_unsup(
    xi_true: np.ndarray,
    M_total: int,
    r_ex: float,
    n_batch: int,
    L: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Ritorna ETA di shape (L, n_batch, M_c, N) con distribuzione uniforme su μ."""
    K, N = xi_true.shape
    M_c = math.ceil(M_total / (L * n_batch))
    p_keep = 0.5 * (1.0 + r_ex)
    ETA = np.zeros((L, n_batch, M_c, N), dtype=np.float32)
    for l in range(L):
        for b in range(n_batch):
            mus = rng.integers(0, K, size=M_c)
            probs = rng.uniform(size=(M_c, N))
            chi = np.where(probs <= p_keep, 1.0, -1.0)
            xi_sel = xi_true[mus]
            ETA[l, b] = (chi * xi_sel).astype(np.float32)
    return ETA

# ---------------------------------------------------------------------
# J non supervisionata per client con rumore DP-like
# ---------------------------------------------------------------------

def noise_scale(hp: HyperParams, M_eff: int, round_idx: int) -> float:
    r = max(1, round_idx + 1)
    if hp.noise_schedule == "const":
        return hp.noise_std0
    elif hp.noise_schedule == "1/sqrt(round)":
        return hp.noise_std0 / math.sqrt(r)
    elif hp.noise_schedule == "1/sqrt(M)":
        return hp.noise_std0 / math.sqrt(max(1, M_eff))
    else:  # default "1/sqrt(round*M)"
        return hp.noise_std0 / math.sqrt(max(1, r * M_eff))


def client_J_with_dp_noise(E_l: np.ndarray, K: int, hp: HyperParams, round_idx: int, rng: np.random.Generator) -> np.ndarray:
    """Calcola J_l da esempi E_l (M_eff,N) via unsupervised_J e aggiunge rumore gaussiano simmetrico.
    La varianza del rumore decresce con round e/o con M_eff secondo noise_schedule.
    """
    N = E_l.shape[1]
    M_eff_param = max(1, E_l.shape[0] // K)
    J_l = unsupervised_J(E_l, M_eff_param)
    sigma = noise_scale(hp, M_eff_param, round_idx)
    Z = rng.normal(loc=0.0, scale=sigma, size=(N, N)).astype(np.float32)
    if hp.symmetrize:
        Z = 0.5 * (Z + Z.T)
    J_l = J_l + Z
    if hp.zero_diag:
        np.fill_diagonal(J_l, 0.0)
    return J_l


def aggregate_clients_J_with_noise(E: np.ndarray, K: int, hp: HyperParams, round_idx: int, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """Media semplice delle J_l rumorose sui client. Ritorna J e sigma_eff usato al round.
    """
    L, M_eff, N = E.shape
    Js = []
    sigmas = []
    for l in range(L):
        J_l = client_J_with_dp_noise(E[l], K, hp, round_idx, rng)
        Js.append(J_l)
        M_eff_param = max(1, E[l].shape[0] // K)
        sigmas.append(noise_scale(hp, M_eff_param, round_idx))
    J = np.mean(Js, axis=0)
    sigma_eff = float(np.mean(sigmas))
    return J, sigma_eff

# ---------------------------------------------------------------------
# Core run (seed)
# ---------------------------------------------------------------------

def run_one_seed(hp: HyperParams, seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    xi_true = gen_patterns(hp.N, hp.K)
    J_star = JK_real(xi_true)
    ETA = gen_dataset_unsup(xi_true, hp.M_total, hp.r_ex, hp.n_batch, hp.L, rng)

    fro_extend_rounds: List[float] = []
    magn_extend_rounds: List[float] = []
    keff_shuf_extend_rounds: List[int] = []
    sigma_rounds: List[float] = []

    xi_ref = None

    round_iter = range(hp.n_batch)
    if hp.show_round_bar:
        round_iter = tqdm(round_iter, desc=f"seed {seed} rounds", leave=False, dynamic_ncols=True)
    for b in round_iter:
        ETA_extend = ETA[:, : b + 1, :, :].transpose(0, 2, 1, 3).reshape(hp.L, -1, hp.N)
        # J aggregata con rumore su client
        J_unsup_ext, sigma_eff = aggregate_clients_J_with_noise(ETA_extend, hp.K, hp, b, rng)
        sigma_rounds.append(sigma_eff)

        # Blending con Hebb sugli archetipi precedenti
        if b == 0 or xi_ref is None:
            J_rec_ext = J_unsup_ext.copy()
        else:
            J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
            J_rec_ext = hp.w * J_unsup_ext + (1.0 - hp.w) * J_hebb_prev

        # Propagazione + TAM
        JKS_ext = propagate_J(J_rec_ext, iters=1, verbose=False)
        # Robustezza numerica: forza finitezza e simmetrizza
        if not np.all(np.isfinite(JKS_ext)):
            JKS_ext = np.nan_to_num(JKS_ext, nan=0.0, posinf=0.0, neginf=0.0)
        # Clipping soft per evitare esplosioni (range empirico)
        JKS_ext = np.clip(JKS_ext, -1e3, 1e3)
        JKS_ext = 0.5 * (JKS_ext + JKS_ext.T)
        vals, vecs = np.linalg.eig(JKS_ext)
        order = np.argsort(np.real(vals))[::-1]
        autov = np.real(vecs[:, order[: hp.K]]).T

        Net = TAM_Network()
        Net.prepare(J_rec_ext, hp.L)
        xi_r, magn = dis_check(autov, hp.K, hp.L, J_rec_ext, JKS_ext, ξ=xi_true, updates=hp.updates, show_bar=False)
        xi_ref = xi_r

        # Metriche
        fro_rel = float(np.linalg.norm(JKS_ext - J_star, ord='fro') / (np.linalg.norm(J_star, ord='fro') + 1e-9))
        fro_extend_rounds.append(fro_rel)
        magn_extend_rounds.append(float(np.mean(magn)))
        try:
            K_eff_shuf, _, _ = estimate_K_eff_from_J(JKS_ext, method='shuffle', M_eff=ETA_extend.shape[1])
        except Exception:
            K_eff_shuf = hp.K
        keff_shuf_extend_rounds.append(int(K_eff_shuf))

    # Overlap finale (Hungarian) con fallback se SciPy non disponibile
    if xi_ref is None or xi_ref.size == 0:
        m_final = 0.0
    else:
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore
            K_hat, Nloc = xi_ref.shape
            M = np.abs(xi_ref @ xi_true.T / Nloc)
            rI, cI = linear_sum_assignment(1.0 - M)
            m_final = float(M[rI, cI].mean())
        except Exception:
            # Fallback: media dei massimi (approssimazione, no matching uno-a-uno garantito)
            K_hat, Nloc = xi_ref.shape
            M = np.abs(xi_ref @ xi_true.T / Nloc)
            m_final = float(np.mean(np.max(M, axis=1)))

    return {
        "seed": seed,
        "m_final": m_final,
        "fro_final": fro_extend_rounds[-1],
        "deltaK": int(abs(keff_shuf_extend_rounds[-1] - hp.K)),
        "rounds": list(range(hp.n_batch)),
        "m_series": magn_extend_rounds,
        "fro_series": fro_extend_rounds,
        "keff_series": keff_shuf_extend_rounds,
        "sigma_series": sigma_rounds,
    }

# ---------------------------------------------------------------------
# Aggregazione e grafici
# ---------------------------------------------------------------------

def aggregate_and_plot(hp: HyperParams, rows: List[dict], exp_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    # Serie aggregate
    rounds = rows[0]["rounds"]
    arr_m = np.stack(df["m_series"].to_list(), axis=0)
    arr_f = np.stack(df["fro_series"].to_list(), axis=0)
    arr_k = np.stack(df["keff_series"].to_list(), axis=0)
    arr_s = np.stack(df["sigma_series"].to_list(), axis=0)

    def mean_se(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = x.mean(axis=0)
        se = x.std(axis=0, ddof=1) / np.sqrt(max(1, x.shape[0]))
        return m, se

    m_mean, m_se = mean_se(arr_m)
    f_mean, f_se = mean_se(arr_f)
    k_mean, k_se = mean_se(arr_k)
    s_mean, s_se = mean_se(arr_s)

    # Garantisce che style sia uno dei valori permessi da seaborn.set_theme
    _allowed_styles = {'white', 'dark', 'whitegrid', 'darkgrid', 'ticks'}
    style_use = hp.style if hp.style in _allowed_styles else 'whitegrid'
    sns.set_theme(style=style_use, palette=hp.palette)
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)

    # 1) Retrieval vs round
    ax = axes[0]
    ax.plot(rounds, m_mean, label="retrieval (extend)")
    ax.fill_between(rounds, m_mean - m_se, m_mean + m_se, alpha=0.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("round")
    ax.set_ylabel("Mattis overlap")
    ax.set_title("Retrieval per round con rumore su J locali")
    ax.grid(True, alpha=0.3)

    # 2) Frobenius vs round
    ax = axes[1]
    ax.plot(rounds, f_mean, label="Frobenius rel.")
    ax.fill_between(rounds, f_mean - f_se, f_mean + f_se, alpha=0.2)
    ax.set_xlabel("round")
    ax.set_ylabel("||J−J*||_F / ||J*||_F")
    ax.set_title("Convergenza strutturale di J (extend)")
    ax.grid(True, alpha=0.3)

    # 3) K_eff(shuffle) e sigma_eff vs round (doppio asse)
    ax = axes[2]
    ax.plot(rounds, k_mean, label="K_eff (shuffle)")
    ax.fill_between(rounds, k_mean - k_se, k_mean + k_se, alpha=0.15)
    ax2 = ax.twinx()
    ax2.plot(rounds, s_mean, linestyle='--', label="sigma_eff")
    ax.set_xlabel("round")
    ax.set_ylabel("K_eff")
    ax2.set_ylabel("sigma_eff")
    ax.set_title("Rank efficace e intensità del rumore")
    ax.grid(True, alpha=0.3)

    # legenda combinata
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    fig_path = exp_dir / "fig_metrics.png"
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
        N=400,
        n_batch=12,
        M_total=800,
        r_ex=0.7,
        noise_std0=0.8,
        noise_schedule="1/sqrt(round*M)",
        updates=60,
        beta_T=2.5,
        lam=0.2,
        h_in=0.1,
        w=0.8,
        n_seeds=6,
        seed_base=9001,
    )

    base_dir = ROOT / "stress_tests" / "exp03_dp_noise_on_J"
    # Sanitizzazione della noise_schedule per path Windows-safe
    def slug(s: str) -> str:
        # minuscole, sostituisce tutto ciò che non è alfanumerico con '-'
        s2 = s.lower()
        s2 = re.sub(r"[^a-z0-9]+", "-", s2).strip('-')
        return s2 or "sched"

    sched_slug = slug(hp.noise_schedule)
    tag = (
        f"K{hp.K}_N{hp.N}_L{hp.L}_nb{hp.n_batch}_M{hp.M_total}_r{hp.r_ex}_"
        f"noise{hp.noise_std0}_sched{sched_slug}_w{hp.w}"
    )
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] Esperimento: {exp_dir}")

    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    rows: List[dict] = []
    log_path = exp_dir / "log.jsonl"
    with open(log_path, "w") as flog:
        for s in tqdm(range(hp.n_seeds), desc="seeds", dynamic_ncols=True):
            seed = hp.seed_base + s
            t0 = time.perf_counter()
            row = run_one_seed(hp, seed, exp_dir)
            t1 = time.perf_counter()
            row["elapsed_s"] = t1 - t0
            rows.append(row)
            flog.write(json.dumps(row) + "\n")
            tqdm.write(
                f"[seed {seed}] m_final={row['m_final']:.3f} "
                f"fro_final={row['fro_final']:.3f} ΔK={row['deltaK']}"
            )

    aggregate_and_plot(hp, rows, exp_dir)


if __name__ == "__main__":
    main()
