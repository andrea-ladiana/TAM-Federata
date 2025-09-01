#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 06 (single) â€” Mixing drift (K=3) con grafico sul simplesso
=====================================================================
Versione SINGLE: J viene stimata ad ogni round solo dagli esempi del round corrente
(no extend cumulativo). Mantiene output e figure come la versione originale.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
from tqdm.auto import tqdm

# Root import
_THIS = Path(__file__).resolve()
ROOT = _THIS
while ROOT != ROOT.parent and not (ROOT / "Functions.py").exists():
    ROOT = ROOT.parent

SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from src.unsup.functions import (
    gen_patterns,
    JK_real,
    unsupervised_J,
    propagate_J,
)
from src.unsup.dynamics import dis_check


@dataclass
class HyperParams:
    # Modello / dataset
    L: int = 3
    K: int = 3
    N: int = 300
    n_batch: int = 24
    M_total: int = 2400
    r_ex: float = 0.8

    # Mixing drift (scheduler globale su round)
    drift_type: Literal["cyclic", "piecewise_dirichlet", "random_walk"] = "cyclic"
    drift_strength: float = 1.0
    period: int = 12

    # Dinamica
    w: float = 0.0
    updates: int = 60
    beta_T: float = 2.5
    lam: float = 0.2
    h_in: float = 0.1

    # Esecuzione
    n_seeds: int = 3
    seed_base: int = 120001
    progress_seeds: bool = True
    progress_rounds: bool = True
    progress_layers: bool = False


# ----------------------------- Mixing + dataset ------------------------------
def schedule_mixing(hp: HyperParams, rng: np.random.Generator) -> np.ndarray:
    T = hp.n_batch
    if hp.drift_type == "cyclic":
        A = float(hp.drift_strength)
        t = np.arange(T)
        phi = 2 * np.pi * t / max(1, hp.period)
        # traiettoria liscia sul simplesso per K=3
        x = 1.0/3 + (A/3) * np.cos(phi)
        y = 1.0/3 + (A/3) * np.cos(phi + 2*np.pi/3)
        z = 1.0 - x - y
        pis = np.stack([x, y, z], axis=1)
    elif hp.drift_type == "piecewise_dirichlet":
        pis = []
        seg = max(1, hp.period)
        for t in range(T):
            if t % seg == 0:
                base = rng.dirichlet(alpha=[1.0, 1.0, 1.0])
            pis.append(base)
        pis = np.array(pis)
    else:  # random_walk
        p = np.array([1/3, 1/3, 1/3], dtype=float)
        pis = []
        for _ in range(T):
            p = p + 0.05 * rng.normal(size=3)
            p = np.maximum(1e-3, p); p = p / p.sum()
            pis.append(p)
        pis = np.array(pis)
    return pis.astype(float)


def gen_dataset_mixing(xi_true: np.ndarray, pis: np.ndarray, M_total: int, r_ex: float,
                       L: int, rng: np.random.Generator) -> np.ndarray:
    K, N = xi_true.shape
    T = pis.shape[0]
    M_c = math.ceil(M_total / (L * T))
    ETA = np.zeros((L, T, M_c, N), dtype=np.float32)
    p_keep = 0.5 * (1.0 + r_ex)
    for l in range(L):
        for t in range(T):
            mus = rng.choice(K, size=M_c, p=pis[t])
            probs = rng.uniform(size=(M_c, N))
            chi = np.where(probs <= p_keep, 1.0, -1.0)
            xi_sel = xi_true[mus]
            ETA[l, t] = (chi * xi_sel).astype(np.float32)
    return ETA


# -------------------------- Metriche e allineamento --------------------------
def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    return 0.5 * float(np.sum(np.abs(p - q)))


def retrieval_and_align(xi_hat: np.ndarray, xi_true: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    from scipy.optimize import linear_sum_assignment
    K, N = xi_true.shape
    if xi_hat.ndim != 2 or xi_hat.size == 0:
        return np.zeros(K, dtype=np.float32), 0.0, np.zeros_like(xi_true)
    if xi_hat.shape[1] != N:
        if xi_hat.shape[1] > N:
            xi_hat = xi_hat[:, :N]
        else:
            pad = np.zeros((xi_hat.shape[0], N - xi_hat.shape[1]), dtype=xi_hat.dtype)
            xi_hat = np.concatenate([xi_hat, pad], axis=1)
    M = (xi_hat @ xi_true.T) / float(N)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    cost = 1.0 - np.abs(M)
    rI, cI = linear_sum_assignment(cost)
    mapping = np.full(K, -1, dtype=int)
    for r, c in zip(rI, cI):
        if 0 <= c < K:
            mapping[c] = r
    used = set([r for r in mapping if r >= 0])
    for k_true in range(K):
        if mapping[k_true] >= 0:
            continue
        col = np.abs(M[:, k_true])
        if np.all(col == 0):
            mapping[k_true] = 0
            continue
        candidates = [i for i in range(M.shape[0]) if i not in used]
        if candidates:
            r_star = candidates[int(np.argmax(col[candidates]))]
        else:
            r_star = int(np.argmax(col))
        mapping[k_true] = r_star
        used.add(r_star)
    xi_al = np.zeros_like(xi_true)
    m_per = np.zeros(K, dtype=np.float32)
    for k_true in range(K):
        r = mapping[k_true]
        val = M[r, k_true]
        sgn = 1.0 if val >= 0 else -1.0
        xi_al[k_true] = sgn * xi_hat[r]
        m_per[k_true] = float(abs(val))
    return m_per, float(np.mean(m_per)), xi_al


def estimate_pi_hat_from_examples(xi_aligned: np.ndarray, E_round: np.ndarray) -> np.ndarray:
    K, N = xi_aligned.shape
    X = np.asarray(E_round, dtype=np.float32).reshape(-1, N)
    scores = X @ xi_aligned.T
    labels = np.argmax(scores, axis=1)
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    return (counts / max(1.0, counts.sum())).astype(np.float64)


# ---------------------------------- Run -------------------------------------
def run_one_seed(hp: HyperParams, seed: int, out_dir: Path) -> Dict:
    rng = np.random.default_rng(seed)
    xi_true = gen_patterns(hp.N, hp.K)
    J_star = JK_real(xi_true)
    pis = schedule_mixing(hp, rng)  # (T,3)
    ETA = gen_dataset_mixing(xi_true, pis, hp.M_total, hp.r_ex, hp.L, rng)

    pis_true_set: List[np.ndarray] = []
    pis_hat_set: List[np.ndarray] = []
    tv_series: List[float] = []
    m_series: List[np.ndarray] = []

    xi_ref = None

    round_iter = range(hp.n_batch)
    if hp.progress_rounds:
        round_iter = tqdm(round_iter, desc=f"seed {seed} | rounds", leave=False, dynamic_ncols=True)
    for t in round_iter:
        # SINGLE: usa solo i dati del round t
        E_t = ETA[:, t, :, :]  # (L, M_c, N)
        L_loc, M_eff, N_loc = E_t.shape
        M_eff_param = max(1, M_eff // hp.K)
        Js = []
        layer_iter = range(hp.L)
        if hp.progress_layers:
            layer_iter = tqdm(layer_iter, desc=f"t{t} layers", leave=False, dynamic_ncols=True)
        for l in layer_iter:
            Jl = unsupervised_J(E_t[l], M_eff_param)
            Jl = 0.5 * (Jl + Jl.T)
            np.fill_diagonal(Jl, 0.0)
            Js.append(Jl)
        J_unsup = np.mean(Js, axis=0)

        if t == 0:
            J_rec = J_unsup.copy()
        else:
            J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
            J_rec = hp.w * J_unsup + (1.0 - hp.w) * J_hebb_prev

        JKS_iter = propagate_J(J_rec, iters=1, verbose=False)
        # Disentangle
        vals, vecs = np.linalg.eig(JKS_iter)
        mask = (np.real(vals) > 0.5)
        autov = np.real(vecs[:, mask]).T
        xi_hat, m_per = dis_check(autov, hp.K, hp.L, J_rec, JKS_iter, ξ=xi_true, updates=hp.updates, show_bar=False)
        xi_ref = xi_hat

        # Serie
        m_series.append(np.asarray(m_per, dtype=float))
        # mixing stimato classificando SOLO round t
        _, _, xi_aligned = retrieval_and_align(xi_ref, xi_true)
        pi_hat = estimate_pi_hat_from_examples(xi_aligned, E_t)
        pis_hat_set.append(pi_hat)
        pis_true_set.append(pis[t])
        if t == 0:
            tv_series.append(0.0)
        else:
            tv_series.append(tv_distance(pis[t], pis[t-1]))

    out = dict(
        pis_true=np.array(pis_true_set).tolist(),
        pis_hat=np.array(pis_hat_set).tolist(),
        tv_series=np.array(tv_series).tolist(),
        m_series=np.array(m_series).tolist(),
        m_mean_series=np.mean(np.array(m_series), axis=1).tolist(),
    )
    return out


# --------------------------------- Plotting ---------------------------------
def plot_simplex(hp: HyperParams, pis_true: np.ndarray, pis_hat: np.ndarray, out_path: Path) -> None:
    def b2xy(p):
        a, b, c = p[:, 0], p[:, 1], p[:, 2]
        x = 0.5 * (2*b + c) / (a + b + c)
        y = (np.sqrt(3)/2) * c / (a + b + c)
        return np.stack([x, y], axis=1)
    xy_true = b2xy(pis_true)
    xy_hat = b2xy(pis_hat)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xy_true[:,0], xy_true[:,1], lw=2.0, alpha=0.95, label="mixing vero")
    ax.plot(xy_hat[:,0],  xy_hat[:,1],  lw=2.0, alpha=0.95, linestyle="--", label="mixing stimato")
    ax.set_xlim(0, 1); ax.set_ylim(0, np.sqrt(3)/2)
    ax.set_aspect('equal', 'box')
    ax.set_title("Evoluzione dei pesi di mixing sul simplesso (K=3) â€” SINGLE")
    ax.legend(loc='best')
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"[plot] Salvato: {out_path}")
    plt.show()


def plot_time_panels(hp: HyperParams, tv: np.ndarray, m_per: np.ndarray, out_path: Path) -> None:
    T = hp.n_batch; rounds = np.arange(T)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    ax = axes[0]
    ax.plot(rounds, tv, lw=2.0)
    ax.set_title("IntensitÃ  del drift: TV(Ï€_t, Ï€_{t-1})")
    ax.set_xlabel("round"); ax.set_ylabel("TV")
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    for k in range(hp.K):
        ax.plot(rounds, m_per[:,k], lw=1.8, label=f"m_Î¾{k+1}")
    ax.set_title("Retrieval per archetipo vs round (single)")
    ax.set_xlabel("round"); ax.set_ylabel("Mattis overlap")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncols=min(3, hp.K))
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"[plot] Salvato: {out_path}")
    plt.show()


# ---------------------------------- Main ------------------------------------
def main():
    hp = HyperParams()
    base_dir = ROOT / "stress_tests" / "exp06_mixing_drift_single"
    tag = (
        f"K{hp.K}_N{hp.N}_L{hp.L}_R{hp.n_batch}_M{hp.M_total}_r{hp.r_ex}_"
        f"drift{hp.drift_type}_A{hp.drift_strength}_per{hp.period}_w{hp.w}"
    )
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] Esperimento: {exp_dir}")

    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    rows: List[Dict] = []
    pis_true_all, pis_hat_all, tv_all, m_per_all = [], [], [], []
    log_path = exp_dir / "log.jsonl"
    with open(log_path, "w") as flog:
        seed_iter = range(hp.n_seeds)
        if hp.progress_seeds:
            seed_iter = tqdm(seed_iter, desc="seeds", dynamic_ncols=True)
        for s in seed_iter:
            seed = hp.seed_base + s
            out = run_one_seed(hp, seed, exp_dir)
            rows.append({"seed": seed, "m_mean_last": out["m_mean_series"][-1]})
            pis_true_all.append(np.array(out["pis_true"]))
            pis_hat_all.append(np.array(out["pis_hat"]))
            tv_all.append(np.array(out["tv_series"]))
            m_per_all.append(np.array(out["m_series"]))
            flog.write(json.dumps({"seed": seed, **out}) + "\n")

    pis_true_mean = np.mean(np.stack(pis_true_all, axis=0), axis=0)
    pis_hat_mean  = np.mean(np.stack(pis_hat_all,  axis=0), axis=0)
    tv_mean       = np.mean(np.stack(tv_all,       axis=0), axis=0)
    m_per_mean    = np.mean(np.stack(m_per_all,    axis=0), axis=0)

    import pandas as pd
    pd.DataFrame(rows).to_csv(exp_dir / "results_table.csv", index=False)

    plot_simplex(hp, pis_true_mean, pis_hat_mean, exp_dir / "fig_mixing_simplex.png")
    plot_time_panels(hp, tv_mean, m_per_mean, exp_dir / "fig_time_panels.png")


if __name__ == "__main__":
    main()

