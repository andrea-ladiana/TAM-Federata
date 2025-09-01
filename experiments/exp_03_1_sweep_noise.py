#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 03.1 â€” Sweep dell'intensitÃ  del rumore su J locali (DP-like)
======================================================================
Obiettivo: mostrare come variano le metriche **finali** (ultimo round) al variare della scala di
rumore iniettato direttamente nelle J locali dei client.

Sigle (espansione alla prima occorrenza):
- FL = Federated Learning (apprendimento federato)
- TAM = Tripartite Associative Memory
- SNR = Signal-to-Noise Ratio (rapporto segnale/rumore)
- "shuffle" = stima di K_eff tramite baseline a permutazione (non Marchenkoâ€“Pastur)

Output:
- hyperparams.json
- log.jsonl (una riga per seed e per valore di rumore)
- results_table.csv (tabella comoda)
- fig_summary.png (grafico Seaborn 2Ã—2 con bande SE)

Note implementative richieste:
- uso estensivo di tqdm
- soppressione log TF
- K_eff via metodo "shuffle"
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # sopprimi rumore TF
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

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
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Import dei moduli di progetto
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # progetto UNSUP
if str(ROOT) not in sys.path:
    pass
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from unsup.functions import (
    gen_patterns,
    JK_real,
    unsupervised_J,
    propagate_J,
    estimate_K_eff_from_J,
)
from unsup.networks import TAM_Network
from unsup.dynamics import dis_check

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

    # Sweep della scala di rumore iniziale (pre-schedule)
    noise_min: float = 0.0
    noise_max: float = 1.0
    noise_steps: int = 6  # es. [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]

    # Schedule del rumore su round / M_eff
    noise_schedule: str = "1/sqrt(round*M)"  # anche: "const", "1/sqrt(round)", "1/sqrt(M)"
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
    seed_base: int = 91001

    # Plot
    palette: str = "deep"
    style: str = "whitegrid"
    # Progress bars
    show_noise_bar: bool = True
    show_seed_bar: bool = True
    show_round_bar: bool = True

# ---------------------------------------------------------------------
# Utility robusta di propagazione (evita NaN/Inf da explode numerico)
# ---------------------------------------------------------------------
def safe_propagate(J: np.ndarray, *, iters: int = 1, max_steps: int = 300, clip: float = 1e3) -> np.ndarray:
    # Pre-clipping hard se fuori scala enorme
    if not np.all(np.isfinite(J)):
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
    J = np.clip(J, -clip, clip)
    # Se raggio spettrale troppo grande, normalizza per stabilitÃ 
    try:
        vals = np.linalg.eigvalsh(0.5 * (J + J.T))
        rho = np.max(np.abs(vals)) if vals.size else 0.0
        if rho > 50:  # soglia euristica
            J = J / (rho / 50.0)
    except Exception:
        pass
    try:
        JKS = propagate_J(J, iters=iters, verbose=False, max_steps=max_steps)
    except Exception:
        # Fallback: niente propagazione
        JKS = J.copy()
    if not np.all(np.isfinite(JKS)):
        JKS = np.nan_to_num(JKS, nan=0.0, posinf=0.0, neginf=0.0)
    JKS = np.clip(JKS, -clip, clip)
    # Simmetrizza leggermente per mitigare drift
    JKS = 0.5 * (JKS + JKS.T)
    return JKS

# ---------------------------------------------------------------------
# Dataset unsupervised (IID su Î¼)
# ---------------------------------------------------------------------

def gen_dataset_unsup(xi_true: np.ndarray, M_total: int, r_ex: float, n_batch: int, L: int,
                      rng: np.random.Generator) -> np.ndarray:
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
# Rumore su J locali
# ---------------------------------------------------------------------

def noise_scale(noise_std0: float, schedule: str, M_eff: int, round_idx: int) -> float:
    r = max(1, round_idx + 1)
    if schedule == "const":
        return noise_std0
    elif schedule == "1/sqrt(round)":
        return noise_std0 / math.sqrt(r)
    elif schedule == "1/sqrt(M)":
        return noise_std0 / math.sqrt(max(1, M_eff))
    else:  # default "1/sqrt(round*M)"
        return noise_std0 / math.sqrt(max(1, r * M_eff))


def client_J_with_noise(E_l: np.ndarray, K: int, *, noise_std0: float, schedule: str,
                         symmetrize: bool, zero_diag: bool,
                         round_idx: int, rng: np.random.Generator) -> np.ndarray:
    N = E_l.shape[1]
    M_eff_param = max(1, E_l.shape[0] // K)
    J_l = unsupervised_J(E_l, M_eff_param)
    sigma = noise_scale(noise_std0, schedule, M_eff_param, round_idx)
    Z = rng.normal(loc=0.0, scale=sigma, size=(N, N)).astype(np.float32)
    if symmetrize:
        Z = 0.5 * (Z + Z.T)
    J_l = J_l + Z
    if zero_diag:
        np.fill_diagonal(J_l, 0.0)
    return J_l


def aggregate_clients_J_with_noise(E: np.ndarray, K: int, *, noise_std0: float, schedule: str,
                                   symmetrize: bool, zero_diag: bool,
                                   round_idx: int, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    L, M_eff, N = E.shape
    Js = []
    sigmas = []
    for l in range(L):
        J_l = client_J_with_noise(E[l], K, noise_std0=noise_std0, schedule=schedule,
                                  symmetrize=symmetrize, zero_diag=zero_diag,
                                  round_idx=round_idx, rng=rng)
        Js.append(J_l)
        M_eff_param = max(1, E[l].shape[0] // K)
        sigmas.append(noise_scale(noise_std0, schedule, M_eff_param, round_idx))
    J = np.mean(Js, axis=0)
    sigma_eff = float(np.mean(sigmas))
    return J, sigma_eff

# ---------------------------------------------------------------------
# Metriche helper
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


def spectral_radius(J: np.ndarray) -> float:
    vals = np.linalg.eigvals(J)
    return float(np.max(np.abs(vals)))

# ---------------------------------------------------------------------
# Core run (per seed e valore di rumore)
# ---------------------------------------------------------------------

def run_one_seed_one_noise(hp: HyperParams, seed: int, noise_std0: float) -> dict:
    rng = np.random.default_rng(seed)
    xi_true = gen_patterns(hp.N, hp.K)
    J_star = JK_real(xi_true)
    ETA = gen_dataset_unsup(xi_true, hp.M_total, hp.r_ex, hp.n_batch, hp.L, rng)

    xi_ref = None
    J_rec_pre = None  # per Frobenius pre-propagation
    round_iter = range(hp.n_batch)
    if hp.show_round_bar:
        round_iter = tqdm(round_iter, desc=f"rounds n={noise_std0:.2f} seed={seed}", leave=False, dynamic_ncols=True)
    for b in round_iter:
        ETA_extend = ETA[:, : b + 1, :, :].transpose(0, 2, 1, 3).reshape(hp.L, -1, hp.N)
        J_unsup_ext, sigma_eff = aggregate_clients_J_with_noise(
            ETA_extend, hp.K,
            noise_std0=noise_std0, schedule=hp.noise_schedule,
            symmetrize=hp.symmetrize, zero_diag=hp.zero_diag,
            round_idx=b, rng=rng,
        )
        if b == 0 or xi_ref is None:
            J_rec_ext = J_unsup_ext.copy()
        else:
            J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
            J_rec_ext = hp.w * J_unsup_ext + (1.0 - hp.w) * J_hebb_prev
        J_rec_pre = J_rec_ext.copy()

        # Propagazione + TAM
        JKS_ext = safe_propagate(J_rec_ext, iters=1, max_steps=300)
        if not np.all(np.isfinite(JKS_ext)):
            # Ultimo fallback: usa J_rec_ext diretto
            JKS_ext = J_rec_ext.copy()
        vals, vecs = np.linalg.eig(JKS_ext)
        order = np.argsort(np.real(vals))[::-1]
        autov = np.real(vecs[:, order[: hp.K]]).T

        Net = TAM_Network()
        Net.prepare(J_rec_ext, hp.L)
        xi_r, magn = dis_check(autov, hp.K, hp.L, J_rec_ext, JKS_ext,
                               Î¾=xi_true, updates=hp.updates, show_bar=False)
        xi_ref = xi_r

    # Metriche finali @ ultimo round
    fro_post = float(np.linalg.norm(JKS_ext - J_star, ord='fro') / (np.linalg.norm(J_star, ord='fro') + 1e-9))
    fro_pre = float(np.linalg.norm(J_rec_pre - J_star, ord='fro') / (np.linalg.norm(J_star, ord='fro') + 1e-9))
    try:
        K_eff_final, _, _ = estimate_K_eff_from_J(JKS_ext, method='shuffle', M_eff=ETA_extend.shape[1])
    except Exception:
        K_eff_final = hp.K
    snr_final = spectral_snr(JKS_ext, hp.K)
    rho_final = spectral_radius(JKS_ext)

    # Matching Hungarian per m_final
    if xi_ref is None or xi_ref.size == 0:
        m_final = 0.0
    else:
        from scipy.optimize import linear_sum_assignment
        K_hat, Nloc = xi_ref.shape
        M = np.abs(xi_ref @ xi_true.T / Nloc)
        rI, cI = linear_sum_assignment(1.0 - M)
        m_final = float(M[rI, cI].mean())

    return {
        "noise_std0": float(noise_std0),
        "seed": int(seed),
        "m_final": m_final,
        "fro_final_post": fro_post,
        "fro_final_pre": fro_pre,
        "K_eff_final": int(K_eff_final),
        "snr_final": float(snr_final),
        "rho_final": float(rho_final),
    }

# ---------------------------------------------------------------------
# Aggregazione e grafico
# ---------------------------------------------------------------------

def aggregate_and_plot(hp: HyperParams, rows: List[dict], out_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.sort_values(["noise_std0", "seed"], inplace=True)
    df.to_csv(out_dir / "results_table.csv", index=False)

    _allowed_styles = {"white", "dark", "whitegrid", "darkgrid", "ticks"}
    style_use = hp.style if hp.style in _allowed_styles else "whitegrid"
    sns.set_theme(style=style_use, palette=hp.palette)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    # 1) Retrieval finale vs rumore
    ax = axes[0, 0]
    sns.pointplot(data=df, x="noise_std0", y="m_final", errorbar=("se", 1), dodge=True, ax=ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Retrieval finale vs rumore iniziale")
    ax.set_xlabel("noise_std0 (scala iniziale)")
    ax.set_ylabel("Mattis overlap (finale)")
    ax.grid(True, alpha=0.3)

    # 2) Frobenius finale (pre/post) vs rumore
    ax = axes[0, 1]
    df_f = df.melt(id_vars=["noise_std0", "seed"], value_vars=["fro_final_pre", "fro_final_post"],
                   var_name="kind", value_name="fro")
    sns.pointplot(data=df_f, x="noise_std0", y="fro", hue="kind", errorbar=("se", 1), dodge=True, ax=ax)
    ax.set_title("Frobenius finale: pre vs post-propagation")
    ax.set_xlabel("noise_std0")
    ax.set_ylabel("||Jâˆ’J*||_F / ||J*||_F")
    ax.grid(True, alpha=0.3)
    ax.legend(title="")

    # 3) K_eff (shuffle) finale vs rumore
    ax = axes[1, 0]
    sns.pointplot(data=df, x="noise_std0", y="K_eff_final", errorbar=("se", 1), dodge=True, ax=ax)
    ax.axhline(hp.K, color="tab:green", linestyle=":", linewidth=1.2, label="K (vero)")
    ax.set_title("K_eff (shuffle) finale vs rumore")
    ax.set_xlabel("noise_std0")
    ax.set_ylabel("K_eff")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # 4) StabilitÃ  spettrale (SNR e raggio) â€” mostro SNR
    ax = axes[1, 1]
    sns.pointplot(data=df, x="noise_std0", y="snr_final", errorbar=("se", 1), dodge=True, ax=ax)
    ax.set_title("SNR spettrale finale vs rumore")
    ax.set_xlabel("noise_std0")
    ax.set_ylabel("SNR finale")
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
        n_batch=12, 
        M_total=600, 
        r_ex=0.7,
        noise_min=0.0, 
        noise_max=1.0, 
        noise_steps=5,
        noise_schedule="1/sqrt(round*M)", 
        symmetrize=True, 
        zero_diag=True,
        updates=40, 
        beta_T=2.5, 
        lam=0.2, 
        h_in=0.1,
        w=0.9, 
        n_seeds=2, 
        seed_base=91001,
    )

    import re
    base_dir = ROOT / "stress_tests" / "exp031_sweep_noise"
    noise_grid = np.linspace(hp.noise_min, hp.noise_max, hp.noise_steps)
    def slug(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip('-').lower() or "sched"
    sched_slug = slug(hp.noise_schedule)
    tag = (
        f"K{hp.K}_N{hp.N}_L{hp.L}_nb{hp.n_batch}_M{hp.M_total}_r{hp.r_ex}_"
        f"noise{hp.noise_min:.2f}-{hp.noise_max:.2f}_steps{hp.noise_steps}_sched{sched_slug}_w{hp.w}"
    )
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] Esperimento: {exp_dir}")

    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    rows: List[dict] = []
    log_path = exp_dir / "log.jsonl"
    with open(log_path, "w") as flog:
        noise_iter = noise_grid
        if hp.show_noise_bar:
            noise_iter = tqdm(noise_iter, desc="noise grid", dynamic_ncols=True)
        for noise in noise_iter:
            seed_iter = range(hp.n_seeds)
            if hp.show_seed_bar:
                seed_iter = tqdm(seed_iter, desc="seeds", leave=False, dynamic_ncols=True)
            for s in seed_iter:
                seed = hp.seed_base + s
                t0 = time.perf_counter()
                row = run_one_seed_one_noise(hp, int(seed), float(noise))
                t1 = time.perf_counter()
                row["elapsed_s"] = t1 - t0
                rows.append(row)
                flog.write(json.dumps(row) + "\n")
                tqdm.write(
                    f"[noise={noise:.2f} seed={seed}] m_final={row['m_final']:.3f} "
                    f"fro_pre={row['fro_final_pre']:.3f} fro_post={row['fro_final_post']:.3f} "
                    f"K_eff={row['K_eff_final']} SNR={row['snr_final']:.2f}"
                )

    aggregate_and_plot(hp, rows, exp_dir)


if __name__ == "__main__":
    main()

