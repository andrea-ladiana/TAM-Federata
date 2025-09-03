#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 05 — Client bizantini e aggregazione robusta a confronto
===================================================================
Confronto tra diversi **aggregatori lato server** in presenza di una frazione di client **bizantini**.
Si misura l'effetto sul recupero degli archetipi e sulla struttura di J.

Acronimi (espansione alla prima occorrenza):
- FL = Federated Learning (apprendimento federato)
- TAM = Tripartite Associative Memory
- SNR = Signal-to-Noise Ratio (rapporto segnale/rumore)
- GM = Geometric Median (mediana geometrica)

Aggregatori implementati:
- mean (media semplice)
- median (mediana coordinata)
- trimmed (trimmed-mean coordinato, con trim α per lato)
- gmedian (GM su vettore upper-tri, Weiszfeld)
- mkrum (Multi-Krum su vettore upper-tri)

Attacchi bizantini implementati (selezionabili):
- signflip: \tilde{J} = -a J
- rank1: \tilde{J} = J + b v v^T (v scelto per non allinearsi agli archetipi veri)
- collude: tutti i malevoli inviano la stessa matrice estrema
- permute: archetipi permutati e con flip di segno nelle costruzioni locali
- mix: mix random dei precedenti per i malevoli

Output per esecuzione:
- hyperparams.json
- log.jsonl (una riga per combinazione (fb, aggregator, seed) con serie per round)
- results_table.csv (riassunto finale per grafici)
- fig_robustness.png (Seaborn, 2×2)

Note richieste:
- uso estensivo di tqdm
- soppressione log TF
- stima K_eff via metodo "shuffle" (baseline a permutazione)
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # sopprimi log TF
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Import moduli progetto (file in: <UNSUP>/stress_tests/)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
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
# Iperparametri
# ---------------------------------------------------------------------
@dataclass
class HyperParams:
    # Modello / dataset
    L: int = 9               # più client per senso statistico robusto
    K: int = 6
    N: int = 400
    n_batch: int = 12
    M_total: int = 2160      # ≈ 20 esempi/client/round
    r_ex: float = 0.6

    # Federated blending
    w: float = 0.9

    # Attacco bizantino
    attack_type: str = "mix"  # {signflip, rank1, collude, permute, mix}
    a_scale: float = 1.0      # intensità per signflip
    b_rank1: float = 2.0      # intensità per rank1
    fb_grid: Optional[List[float]] = field(default=None)  # frazioni di malevoli (impostato nel main)

    # Aggregatori
    aggregators: Optional[List[str]] = field(default=None)   # ["mean","median","trimmed","gmedian","mkrum"]
    trim_alpha: float = 0.2         # trimmed-mean: frazione per lato
    gm_maxit: int = 50
    gm_eps: float = 1e-5
    mkrum_m: Optional[int] = None            # numero di update selezionati in Multi-Krum (di default L - f_b*L - 2)

    # Clipping (facoltativo, abilitato per sicurezza)
    spectral_clip_tau: float = 1.2  # clip raggio spettrale dei J locali
    coord_clip_k: float = 3.5       # k * MAD coordinata

    # Dinamica TAM
    updates: int = 60
    beta_T: float = 2.5
    lam: float = 0.2
    h_in: float = 0.1

    # Esperimento
    n_seeds: int = 6
    seed_base: int = 101001

    # Plot
    palette: str = "deep"
    style: str = "whitegrid"
    # Progress flags
    show_fb_bar: bool = True
    show_agg_bar: bool = True
    show_seed_bar: bool = True
    show_round_bar: bool = True
    show_client_bar: bool = False

    # Propagazione robusta
    max_propagate_steps: int = 300

# ---------------------------------------------------------------------
# Propagazione sicura (evita NaN/Inf) riusata per ogni round
# ---------------------------------------------------------------------
def safe_propagate(J: np.ndarray, *, iters: int = 1, max_steps: int = 300, clip: float = 1e3) -> np.ndarray:
    if not np.all(np.isfinite(J)):
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
    J = np.clip(J, -clip, clip)
    # normalizza se raggio spettrale troppo grande (stabilizzazione)
    try:
        vals = np.linalg.eigvalsh(0.5 * (J + J.T))
        if vals.size:
            rho = np.max(np.abs(vals))
            if rho > 50:
                J = J / (rho / 50.0)
    except Exception:
        pass
    try:
        JKS = propagate_J(J, iters=iters, verbose=False, max_steps=max_steps)
    except Exception:
        JKS = J.copy()
    if not np.all(np.isfinite(JKS)):
        JKS = np.nan_to_num(JKS, nan=0.0, posinf=0.0, neginf=0.0)
    JKS = np.clip(JKS, -clip, clip)
    return 0.5 * (JKS + JKS.T)

# ---------------------------------------------------------------------
# Utilità: vettorizzazione upper-tri e ricostruzione
# ---------------------------------------------------------------------

def upper_tri_indices(N: int) -> Tuple[np.ndarray, np.ndarray]:
    iu = np.triu_indices(N, k=1)
    return iu


def to_upper_vec(J: np.ndarray, iu: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return J[iu]


def from_upper_vec(v: np.ndarray, iu: Tuple[np.ndarray, np.ndarray], N: int) -> np.ndarray:
    J = np.zeros((N, N), dtype=np.float32)
    J[iu] = v
    J[(iu[1], iu[0])] = v
    return J

# ---------------------------------------------------------------------
# Dataset
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
# Costruzione J locale e varianti bizantine
# ---------------------------------------------------------------------

def spectral_radius(J: np.ndarray) -> float:
    vals = np.linalg.eigvals(J)
    return float(np.max(np.abs(vals)))


def spectral_clip(J: np.ndarray, tau: float) -> np.ndarray:
    rho = spectral_radius(J)
    if rho > tau and rho > 1e-12:
        J = (tau / rho) * J
    return J


def coord_clip(Js: List[np.ndarray], k: float) -> List[np.ndarray]:
    """Clip coordinato rispetto a median±k*MAD sulle coordinate (upper-tri) tra client.
    Attento: costo O(L*N^2). Usare su L moderati e N<=400.
    """
    if len(Js) <= 2:
        return Js
    N = Js[0].shape[0]
    iu = upper_tri_indices(N)
    X = np.stack([to_upper_vec(J, iu) for J in Js], axis=0)  # (L, D)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) + 1e-12
    lo = med - k * mad
    hi = med + k * mad
    Xc = np.clip(X, lo, hi)
    Js_clip = [from_upper_vec(Xc[l], iu, N) for l in range(X.shape[0])]
    return Js_clip


def local_J(E_l: np.ndarray, K: int, *, symmetrize: bool = True, zero_diag: bool = True) -> np.ndarray:
    M_eff_param = max(1, E_l.shape[0] // K)
    J = unsupervised_J(E_l, M_eff_param)
    if symmetrize:
        J = 0.5 * (J + J.T)
    if zero_diag:
        np.fill_diagonal(J, 0.0)
    return J


def orthogonal_noise_direction(xi_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r"""Costruisce un vettore v \\in {±1}^N e lo ortogonalizza rispetto allo span degli archetipi veri."""
    K, N = xi_true.shape
    v = rng.choice([-1.0, 1.0], size=(N,), replace=True).astype(np.float32)
    # proietta fuori dallo span delle xi
    B = xi_true.T  # (N,K)
    try:
        proj = B @ np.linalg.pinv(B.T @ B + 1e-6 * np.eye(K)) @ (B.T @ v)
        v = v - proj
    except Exception:
        pass
    nrm = np.linalg.norm(v) + 1e-12
    return (v / nrm).astype(np.float32)


def corrupt_J(J: np.ndarray, *, attack_type: str, a_scale: float, b_rank1: float,
              collude_J: np.ndarray | None, v_backdoor: np.ndarray | None,
              rng: np.random.Generator) -> np.ndarray:
    if attack_type == "signflip":
        return (-a_scale) * J
    elif attack_type == "rank1":
        v = v_backdoor if v_backdoor is not None else rng.standard_normal(J.shape[0]).astype(np.float32)
        return J + b_rank1 * np.outer(v, v)
    elif attack_type == "collude":
        if collude_J is None:
            return (-a_scale) * J
        return collude_J
    elif attack_type == "permute":
        # permuta righe/colonne con la stessa permutazione e flip casuali
        N = J.shape[0]
        perm = rng.permutation(N)
        Jp = J[perm][:, perm]
        s = rng.choice([-1.0, 1.0], size=(N,)).astype(np.float32)
        Jp = (s[:, None] * Jp) * s[None, :]
        return Jp
    else:  # mix
        choice = rng.choice(["signflip", "rank1", "permute"], p=[0.4, 0.4, 0.2])
        return corrupt_J(J, attack_type=choice, a_scale=a_scale, b_rank1=b_rank1,
                         collude_J=collude_J, v_backdoor=v_backdoor, rng=rng)

# ---------------------------------------------------------------------
# Aggregatori robusti
# ---------------------------------------------------------------------

def agg_mean(Js: List[np.ndarray]) -> np.ndarray:
    return np.mean(Js, axis=0)


def agg_median(Js: List[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(Js, axis=0), axis=0)


def agg_trimmed(Js: List[np.ndarray], alpha: float) -> np.ndarray:
    X = np.stack(Js, axis=0)
    L = X.shape[0]
    k = int(math.floor(alpha * L))
    if k == 0:
        return np.mean(X, axis=0)
    Xs = np.sort(X, axis=0)
    return np.mean(Xs[k:L - k], axis=0)


def weiszfeld_geometric_median(V: np.ndarray, maxit: int = 50, eps: float = 1e-5) -> np.ndarray:
    """GM in R^D per punti V (L,D). Ritorna x*.
    Implementazione protetta da divisioni per zero.
    """
    x = np.median(V, axis=0)
    for _ in range(maxit):
        d = np.linalg.norm(V - x, axis=1) + 1e-12
        w = 1.0 / d
        x_new = (w[:, None] * V).sum(axis=0) / (w.sum() + 1e-12)
        if np.linalg.norm(x_new - x) <= eps * (np.linalg.norm(x) + 1e-12):
            x = x_new
            break
        x = x_new
    return x


def agg_gmedian(Js: List[np.ndarray]) -> np.ndarray:
    N = Js[0].shape[0]
    iu = upper_tri_indices(N)
    V = np.stack([to_upper_vec(J, iu) for J in Js], axis=0)
    x = weiszfeld_geometric_median(V)
    return from_upper_vec(x.astype(np.float32), iu, N)


def pairwise_sq_dists(V: np.ndarray) -> np.ndarray:
    # V: (L,D)
    G = V @ V.T
    nrm = np.diag(G)
    D2 = nrm[:, None] + nrm[None, :] - 2 * G
    D2[D2 < 0] = 0.0
    return D2


def agg_mkrum(Js: List[np.ndarray], m: int | None) -> np.ndarray:
    """Multi-Krum: seleziona m vettori con punteggio minore e fa media.
    Se m è None: m = max(1, L - f_b_est - 2) con f_b_est = floor((L-2)/2)
    (scelta conservativa se non noto il numero di malevoli esatto).
    """
    N = Js[0].shape[0]
    iu = upper_tri_indices(N)
    V = np.stack([to_upper_vec(J, iu) for J in Js], axis=0)
    L = V.shape[0]
    D2 = pairwise_sq_dists(V)
    # per ciascun l, somma le distanze ai L-2 più vicini
    k = max(1, L - 2)
    scores = np.partition(D2, kth=k-1, axis=1)[:, :k].sum(axis=1)
    idx = np.argsort(scores)
    m_sel = m if m is not None else max(1, L - 3)
    sel = idx[:m_sel]
    x = V[sel].mean(axis=0)
    return from_upper_vec(x.astype(np.float32), iu, N)

# ---------------------------------------------------------------------
# SNR, gap, backdoor score, K_eff(shuffle)
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


def spectral_gap(J: np.ndarray, K: int) -> float:
    vals = np.real(np.linalg.eigvals(J))
    pos = np.sort(vals)[::-1]
    if len(pos) <= K:
        return 0.0
    lamK = pos[K - 1]
    lamK1 = pos[K]
    denom = max(abs(lamK), 1e-12)
    return float((lamK - lamK1) / denom)


def backdoor_score(xi_hat: np.ndarray, v: np.ndarray) -> float:
    # media overlap |<xi_hat^mu, v>|/N sui K archetipi stimati
    K_hat, N = xi_hat.shape
    return float(np.mean(np.abs(xi_hat @ v) / N))

# ---------------------------------------------------------------------
# Core run per (fb, aggregator, seed)
# ---------------------------------------------------------------------

def run_one_case(hp: HyperParams, seed: int, fb: float, aggregator: str, *, out_dir: Path) -> Dict:
    rng = np.random.default_rng(seed)
    xi_true = gen_patterns(hp.N, hp.K)
    J_star = JK_real(xi_true)
    ETA = gen_dataset_unsup(xi_true, hp.M_total, hp.r_ex, hp.n_batch, hp.L, rng)

    # scegli v_backdoor e (per collusion) una matrice estrema fissa
    v_bd = orthogonal_noise_direction(xi_true, rng)
    collude_J = None

    # indici malevoli
    Lb = int(round(fb * hp.L))
    malevoli_idx = set(rng.choice(hp.L, size=Lb, replace=False)) if Lb > 0 else set()

    # metriche round-wise
    magn_rounds: List[float] = []
    fro_post_rounds: List[float] = []
    keff_shuf_rounds: List[int] = []
    snr_rounds: List[float] = []

    t_server_agg_sum = 0.0

    xi_ref = None

    round_iter = range(hp.n_batch)
    if hp.show_round_bar:
        round_iter = tqdm(round_iter, desc=f"rounds fb={fb:.1f} agg={aggregator} seed={seed}", leave=False, dynamic_ncols=True)
    for b in round_iter:
        # Vista extend
        ETA_extend = ETA[:, : b + 1, :, :].transpose(0, 2, 1, 3).reshape(hp.L, -1, hp.N)

        # Costruisci J locali (onesti), poi applica attacco dove necessario
        Js_local = []
        client_iter = range(hp.L)
        if hp.show_client_bar:
            client_iter = tqdm(client_iter, desc=f"clients r{b}", leave=False, dynamic_ncols=True)
        for l in client_iter:
            Jl = local_J(ETA_extend[l], hp.K, symmetrize=True, zero_diag=True)
            Jl = spectral_clip(Jl, hp.spectral_clip_tau)
            Js_local.append(Jl)

        # Definisci collude_J una volta (se necessario)
        if hp.attack_type == "collude" and collude_J is None and len(malevoli_idx) > 0:
            # matrice estrema: rank-1 con v_bd
            collude_J = b_rank = hp.b_rank1 * np.outer(v_bd, v_bd)

        # Applica attacchi
        for l in range(hp.L):
            if l in malevoli_idx:
                Js_local[l] = corrupt_J(Js_local[l], attack_type=hp.attack_type,
                                        a_scale=hp.a_scale, b_rank1=hp.b_rank1,
                                        collude_J=collude_J, v_backdoor=v_bd, rng=rng)

        # Clipping coordinato robusto tra client
        Js_local = coord_clip(Js_local, hp.coord_clip_k)

        # Aggregazione
        t0 = time.perf_counter()
        if aggregator == "mean":
            J_agg = agg_mean(Js_local)
        elif aggregator == "median":
            J_agg = agg_median(Js_local)
        elif aggregator == "trimmed":
            J_agg = agg_trimmed(Js_local, hp.trim_alpha)
        elif aggregator == "gmedian":
            J_agg = agg_gmedian(Js_local)
        elif aggregator == "mkrum":
            m_sel = hp.mkrum_m if hp.mkrum_m is not None else max(1, hp.L - int(round(fb * hp.L)) - 2)
            J_agg = agg_mkrum(Js_local, m_sel)
        else:
            raise ValueError(f"Aggregatore sconosciuto: {aggregator}")
        t1 = time.perf_counter()
        t_server_agg_sum += (t1 - t0)

        # Blend con memoria (Hebb su archetipi precedenti)
        if b == 0 or xi_ref is None:
            J_rec = J_agg.copy()
        else:
            J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
            J_rec = hp.w * J_agg + (1.0 - hp.w) * J_hebb_prev

        # Propagazione e TAM
        JKS = safe_propagate(J_rec, iters=1, max_steps=hp.max_propagate_steps)
        if not np.all(np.isfinite(JKS)):
            JKS = J_rec.copy()
        vals, vecs = np.linalg.eig(JKS)
        order = np.argsort(np.real(vals))[::-1]
        autov = np.real(vecs[:, order[: hp.K]]).T

        Net = TAM_Network()
        Net.prepare(J_rec, hp.L)
        xi_r, magn = dis_check(autov, hp.K, hp.L, J_rec, JKS, ξ=xi_true, updates=hp.updates, show_bar=False)
        xi_ref = xi_r

        # Metriche
        fro_rel_post = float(np.linalg.norm(JKS - J_star, ord='fro') / (np.linalg.norm(J_star, ord='fro') + 1e-9))
        fro_post_rounds.append(fro_rel_post)
        magn_rounds.append(float(np.mean(magn)))
        try:
            K_eff_shuf, _, _ = estimate_K_eff_from_J(JKS, method='shuffle', M_eff=ETA_extend.shape[1])
        except Exception:
            K_eff_shuf = hp.K
        keff_shuf_rounds.append(int(K_eff_shuf))
        snr_rounds.append(spectral_snr(JKS, hp.K))

    # metriche finali
    from scipy.optimize import linear_sum_assignment
    if xi_ref is not None:
        K_hat, Nloc = xi_ref.shape
        Mmat = np.abs(xi_ref @ xi_true.T / Nloc)
        rI, cI = linear_sum_assignment(1.0 - Mmat)
        m_final = float(Mmat[rI, cI].mean())
        bd_score = backdoor_score(xi_ref, v_bd)
    else:
        m_final = 0.0
        bd_score = 0.0
    gap_final = spectral_gap(JKS, hp.K)

    return {
        "seed": seed,
        "fb": fb,
        "aggregator": aggregator,
        "m_final": m_final,
        "fro_final_post": fro_post_rounds[-1],
        "K_eff_final": keff_shuf_rounds[-1],
        "snr_final": snr_rounds[-1],
        "gap_final": gap_final,
        "backdoor_final": bd_score,
        "server_agg_time_per_round": t_server_agg_sum / hp.n_batch,
        "rounds": list(range(hp.n_batch)),
        "m_series": magn_rounds,
        "fro_post_series": fro_post_rounds,
        "keff_series": keff_shuf_rounds,
        "snr_series": snr_rounds,
    }

# ---------------------------------------------------------------------
# Aggregazione e grafico
# ---------------------------------------------------------------------

def aggregate_and_plot(hp: HyperParams, rows: List[Dict], out_dir: Path) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    df.sort_values(["fb", "aggregator", "seed"], inplace=True)
    df.to_csv(out_dir / "results_table.csv", index=False)

    _allowed_styles = {"white","dark","whitegrid","darkgrid","ticks"}
    style_use = hp.style if hp.style in _allowed_styles else "whitegrid"
    # type: ignore to silence potential type checker complaint (runtime accepts string literal subset)
    sns.set_theme(style=style_use, palette=hp.palette)  # type: ignore
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    # 1) Retrieval finale vs fb (per aggregatore)
    ax = axes[0, 0]
    sns.pointplot(data=df, x="fb", y="m_final", hue="aggregator", errorbar=("se", 1), dodge=True, ax=ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Retrieval finale vs frazione bizantina (fb)")
    ax.set_xlabel("frazione bizantina fb")
    ax.set_ylabel("Mattis overlap (finale)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="aggregatore", bbox_to_anchor=(1.02, 1), loc="upper left")

    # 2) Backdoor score vs fb
    ax = axes[0, 1]
    sns.pointplot(data=df, x="fb", y="backdoor_final", hue="aggregator", errorbar=("se", 1), dodge=True, ax=ax)
    ax.set_title("Backdoor score vs fb (più basso = meglio)")
    ax.set_xlabel("fb")
    ax.set_ylabel("|<xi_hat, v>|/N (finale)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")

    # 3) Stabilità spettrale: SNR e gap vs fb (mostro gap)
    ax = axes[1, 0]
    sns.pointplot(data=df, x="fb", y="gap_final", hue="aggregator", errorbar=("se", 1), dodge=True, ax=ax)
    ax.set_title("Gap spettrale finale vs fb")
    ax.set_xlabel("fb")
    ax.set_ylabel("(λ_K−λ_{K+1})/λ_K")
    ax.grid(True, alpha=0.3)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")

    # 4) Overhead server: tempo aggregazione per round
    ax = axes[1, 1]
    df_over = df.groupby(["fb", "aggregator"], as_index=False)["server_agg_time_per_round"].mean()
    # type: ignore below due to seaborn stub limitations
    sns.pointplot(data=df_over, x="fb", y="server_agg_time_per_round", hue="aggregator", dodge=True, ax=ax)  # type: ignore
    ax.set_title("Overhead server: tempo aggregazione/round")
    ax.set_xlabel("fb")
    ax.set_ylabel("secondi / round")
    ax.grid(True, alpha=0.3)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig_path = out_dir / "fig_robustness.png"
    fig.savefig(fig_path, dpi=150)
    print(f"[plot] Figura salvata in: {fig_path}")
    plt.show()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    hp = HyperParams(
        L=9, 
        K=6, 
        N=200, 
        n_batch=12, 
        M_total=1000, 
        r_ex=0.7,
        w=0.9,
        attack_type="mix", 
        a_scale=1.0, 
        b_rank1=2.0,
        fb_grid=[0.0, 0.1, 0.2, 0.3],
        aggregators=["mean", "median", "trimmed", "gmedian", "mkrum"],
        trim_alpha=0.2, 
        gm_maxit=50, 
        gm_eps=1e-5, 
        mkrum_m=None,
        spectral_clip_tau=1.2, 
        coord_clip_k=3.5,
        updates=60, 
        beta_T=2.5, 
        lam=0.2, 
        h_in=0.1,
        n_seeds=3, 
        seed_base=101001,
    )
    # Ensure non-None lists (defensive if user customizes later)
    if hp.fb_grid is None:
        hp.fb_grid = [0.0]
    if hp.aggregators is None:
        hp.aggregators = ["mean"]

    base_dir = ROOT / "stress_tests" / "exp05_byzantine_robust_agg"
    tag = (
        f"K{hp.K}_N{hp.N}_L{hp.L}_nb{hp.n_batch}_M{hp.M_total}_w{hp.w}_"
        f"attack{hp.attack_type}_fb{min(hp.fb_grid):.1f}-{max(hp.fb_grid):.1f}_aggs{len(hp.aggregators)}"
    )
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] Esperimento: {exp_dir}")

    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    rows: List[Dict] = []
    log_path = exp_dir / "log.jsonl"
    with open(log_path, "w") as flog:
        fb_iter = hp.fb_grid
        if hp.show_fb_bar:
            fb_iter = tqdm(fb_iter, desc="fb grid", dynamic_ncols=True)
        for fb in fb_iter:
            agg_iter = hp.aggregators
            if hp.show_agg_bar:
                agg_iter = tqdm(agg_iter, desc="aggregators", leave=False, dynamic_ncols=True)
            for agg in agg_iter:
                seed_iter = range(hp.n_seeds)
                if hp.show_seed_bar:
                    seed_iter = tqdm(seed_iter, desc="seeds", leave=False, dynamic_ncols=True)
                for s in seed_iter:
                    seed = hp.seed_base + s
                    t0 = time.perf_counter()
                    row = run_one_case(hp, int(seed), float(fb), agg, out_dir=exp_dir)
                    t1 = time.perf_counter()
                    row["elapsed_s"] = t1 - t0
                    rows.append(row)
                    flog.write(json.dumps(row) + "\n")
                    tqdm.write(
                        f"[fb={fb:.1f} agg={agg} seed={seed}] m_final={row['m_final']:.3f} "
                        f"gap={row['gap_final']:.3f} backdoor={row['backdoor_final']:.3f} "
                        f"t_srv={row['server_agg_time_per_round']*1e3:.2f} ms/round"
                    )

    aggregate_and_plot(hp, rows, exp_dir)


if __name__ == "__main__":
    main()
