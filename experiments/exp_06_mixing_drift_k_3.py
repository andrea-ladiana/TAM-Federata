#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 06 (fix) â€” Drift dei pesi di mixing (K=3) con grafico sul simplesso
==============================================================================
Correzione e riscrittura robusta dell'esperimento K=3 con **mixing non-stazionario**.

Cosa cambia rispetto alla versione modificata dall'utente:
- **Bug di indentazione**: tutta la pipeline (propagazione, TAM e metriche) torna **dentro** il ciclo dei round.
- **Allineamento stabile**: ogni round allineiamo i prototipi \(\hat\Xi\) a \(\Xi\) tramite **Hungarian** e **flip di segno**, cosÃ¬ le componenti di \(\hat\boldsymbol{\pi}\) sono coerenti con l'ordinamento "vero".
- **Stima di \(\hat\boldsymbol{\pi}\)**: classificazione degli esempi del **round corrente** con i prototipi **allineati**.
- **tqdm** parametrica (seed/round/layer), log TF soppressi, codice pulito e sicuro.

Output:
- `hyperparams.json`, `log.jsonl` (una riga per round), `results_table.csv`
- `fig_mixing_simplex.png` (traiectorie vero vs stimato + frecce), `fig_time_panels.png` (TV e retrieval per archetipo)
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # sopprimi log TF
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

# ---------------------------------------------------------------------
# Import moduli di progetto: trova dinamicamente la root con Functions.py
# ---------------------------------------------------------------------
_THIS = Path(__file__).resolve()
ROOT = _THIS
while ROOT != ROOT.parent and not (ROOT / "Functions.py").exists():
    ROOT = ROOT.parent

SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from unsup.functions import (
    gen_patterns,
    JK_real,
    unsupervised_J,
    propagate_J,
)
from unsup.networks import TAM_Network
from unsup.dynamics import dis_check

# ---------------------------------------------------------------------
# Iperparametri
# ---------------------------------------------------------------------
@dataclass
class HyperParams:
    # Modello / dataset
    L: int = 3
    K: int = 3
    N: int = 300
    n_batch: int = 18           # round
    M_total: int = 1620         # â‰ˆ 30 esempi/client/round
    r_ex: float = 0.6

    # Mixing drift (scheduler globale su round)
    drift_type: Literal["cyclic", "piecewise_dirichlet", "random_walk"] = "cyclic"
    drift_strength: float = 1.0
    period: int = 12
    dirichlet_alpha: float = 0.6
    piece_len: int = 6
    rw_sigma: float = 0.25

    # Federated blending
    w: float = 0.9               # peso sulla componente unsupervised

    # Dinamica TAM
    updates: int = 60
    beta_T: float = 2.5
    lam: float = 0.2
    h_in: float = 0.1

    # Esperimento
    n_seeds: int = 5
    seed_base: int = 120001

    # Plot
    palette: str = "deep"
    style: Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "whitegrid"

    # Progress bars
    progress_seeds: bool = True
    progress_rounds: bool = True
    progress_layers: bool = False
    progress_updates: bool = False

# ---------------------------------------------------------------------
# UtilitÃ  mixing + dataset
# ---------------------------------------------------------------------

def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = (x - np.max(x)) / max(1e-12, tau)
    e = np.exp(x)
    return e / np.sum(e)


def schedule_mixing(hp: HyperParams, rng: np.random.Generator) -> np.ndarray:
    """Ritorna array (n_batch, 3) con pi_true per round."""
    T = hp.n_batch
    if hp.drift_type == "cyclic":
        w = 2 * math.pi / max(1, hp.period)
        A = float(hp.drift_strength)
        base = np.array([0.0, 2.0, 4.0], dtype=np.float32)
        S = np.stack([A * np.sin(w * t + base) for t in range(T)], axis=0)
        pis = np.stack([softmax(S[t], tau=0.7) for t in range(T)], axis=0)
        return pis
    elif hp.drift_type == "piecewise_dirichlet":
        pis = []
        for t in range(T):
            if t % max(1, hp.piece_len) == 0:
                pi = np.random.default_rng(rng.integers(0, 2**31-1)).dirichlet([hp.dirichlet_alpha]*hp.K)
            pis.append(pi.astype(np.float32))
        return np.stack(pis, axis=0)
    else:  # random walk su logits
        z = np.zeros((T, hp.K), dtype=np.float32)
        for t in range(1, T):
            z[t] = z[t-1] + rng.normal(0.0, hp.rw_sigma, size=(hp.K,)).astype(np.float32)
        return np.stack([softmax(z[t], tau=0.8) for t in range(T)], axis=0)


def gen_dataset_mixing(xi_true: np.ndarray, pis: np.ndarray, M_total: int, r_ex: float,
                       L: int, rng: np.random.Generator) -> np.ndarray:
    """Genera ETA shape (L, T, M_c, N) campionando Î¼ ~ Categorical(pi_t)."""
    K, N = xi_true.shape
    T = pis.shape[0]
    M_c = math.ceil(M_total / (L * T))
    p_keep = 0.5 * (1.0 + r_ex)
    ETA = np.zeros((L, T, M_c, N), dtype=np.float32)
    for l in range(L):
        for t in range(T):
            mus = rng.choice(K, size=M_c, p=pis[t])
            probs = rng.uniform(size=(M_c, N))
            chi = np.where(probs <= p_keep, 1.0, -1.0)
            xi_sel = xi_true[mus]
            ETA[l, t] = (chi * xi_sel).astype(np.float32)
    return ETA

# ---------------------------------------------------------------------
# Metriche e allineamento
# ---------------------------------------------------------------------

def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    return 0.5 * float(np.sum(np.abs(p - q)))


def retrieval_and_align(xi_hat: np.ndarray, xi_true: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Hungarian + flip di segno, robusto se alcune colonne non vengono assegnate.
    Ritorna (m_per, m_mean, xi_aligned) con xi_aligned ordinato come xi_true.
    """
    from scipy.optimize import linear_sum_assignment

    K, N = xi_true.shape
    # Sanity sui shape (taglia o pad a N colonne)
    if xi_hat.ndim != 2 or xi_hat.size == 0:
        # nessun prototipo stimato: ritorna zeri
        return np.zeros(K, dtype=np.float32), 0.0, np.zeros_like(xi_true)
    if xi_hat.shape[1] != N:
        if xi_hat.shape[1] > N:
            xi_hat = xi_hat[:, :N]
        else:
            pad = np.zeros((xi_hat.shape[0], N - xi_hat.shape[1]), dtype=xi_hat.dtype)
            xi_hat = np.concatenate([xi_hat, pad], axis=1)

    # SimilaritÃ  (puÃ² essere negativa): righe = stimati, colonne = veri
    M = (xi_hat @ xi_true.T) / float(N)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    # Hungarian su costo = 1 - |M|
    cost = 1.0 - np.abs(M)
    rI, cI = linear_sum_assignment(cost)

    # Mappa colonna-â†’riga; alcune colonne possono restare scoperte
    mapping = np.full(K, -1, dtype=int)
    for r, c in zip(rI, cI):
        if 0 <= c < K:
            mapping[c] = r

    # Per le colonne non assegnate scegli il miglior r (greedy), evitando riusi quando possibile
    used = set([r for r in mapping if r >= 0])
    for k_true in range(K):
        if mapping[k_true] >= 0:
            continue
        # r* = argmax_r |M[r, k_true]|
        col = np.abs(M[:, k_true])
        if np.all(col == 0):
            mapping[k_true] = 0  # fallback innocuo
            continue
        # cerca tra righe non usate, altrimenti consenti il riuso
        candidates = [i for i in range(M.shape[0]) if i not in used]
        if candidates:
            r_star = candidates[int(np.argmax(col[candidates]))]
        else:
            r_star = int(np.argmax(col))
        mapping[k_true] = r_star
        used.add(r_star)

    # Costruisci xi_aligned e gli overlap per archetipo
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
    # classifica SOLO gli esempi del round corrente con i prototipi allineati
    K, N = xi_aligned.shape
    X = np.asarray(E_round, dtype=np.float32).reshape(-1, N)
    scores = X @ xi_aligned.T
    labels = np.argmax(scores, axis=1)
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    return (counts / max(1.0, counts.sum())).astype(np.float64)


# ---------------------------------------------------------------------
# Ternary utilities (equilateral mapping)
# ---------------------------------------------------------------------

def ternary_to_xy(p: np.ndarray) -> Tuple[float, float]:
    p1, p2, p3 = float(p[0]), float(p[1]), float(p[2])
    x = p2 + 0.5 * p3
    y = (math.sqrt(3) / 2.0) * p3
    return x, y


def draw_ternary_axes(ax, labels=("Î¾1", "Î¾2", "Î¾3")):
    V1 = (0.0, 0.0); V2 = (1.0, 0.0); V3 = (0.5, math.sqrt(3)/2.0)
    ax.plot([V1[0], V2[0], V3[0], V1[0]], [V1[1], V2[1], V3[1], V1[1]], lw=1.2, color="0.3")
    ax.text(V1[0]-0.03, V1[1]-0.03, labels[0], ha="right", va="top")
    ax.text(V2[0]+0.03, V2[1]-0.03, labels[1], ha="left", va="top")
    ax.text(V3[0], V3[1]+0.03, labels[2], ha="center", va="bottom")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, math.sqrt(3)/2.0 + 0.05)
    ax.set_aspect("equal"); ax.axis("off")

# ---------------------------------------------------------------------
# Core run (un seed)
# ---------------------------------------------------------------------

def run_one_seed(hp: HyperParams, seed: int, out_dir: Path) -> Dict:
    rng = np.random.default_rng(seed)
    xi_true = gen_patterns(hp.N, hp.K)
    _ = JK_real(xi_true)  # manteniamo per completezza; non usato nel plotting

    # schedule mixing + dataset
    pis = schedule_mixing(hp, rng)              # (T,3)
    ETA = gen_dataset_mixing(xi_true, pis, hp.M_total, hp.r_ex, hp.L, rng)

    # storage
    tv_to_prev: List[float] = []
    pi_true_series: List[np.ndarray] = []
    pi_hat_series: List[np.ndarray] = []
    m_per_series: List[np.ndarray] = []
    m_mean_series: List[float] = []

    xi_ref = None

    round_iter = range(hp.n_batch)
    if hp.progress_rounds:
        round_iter = tqdm(round_iter, desc=f"seed {seed} | rounds", leave=False, dynamic_ncols=True)

    for t in round_iter:
        # Estendi i dati fino al round t
        ETA_extend = ETA[:, : t + 1, :, :].transpose(0, 2, 1, 3).reshape(hp.L, -1, hp.N)

        # Costruisci e media le J locali (unsupervised)
        Js = []
        layer_iter = range(hp.L)
        if hp.progress_layers:
            layer_iter = tqdm(layer_iter, desc=f"t{t} layers", leave=False, dynamic_ncols=True)
        for l in layer_iter:
            M_eff_param = max(1, ETA_extend[l].shape[0] // hp.K)
            Jl = unsupervised_J(ETA_extend[l], M_eff_param)
            Jl = 0.5 * (Jl + Jl.T)
            np.fill_diagonal(Jl, 0.0)
            Js.append(Jl)
        J_unsup = np.mean(Js, axis=0)

        # Blend con memoria (Hebb sulle \hat\Xi previe)
        if t == 0 or xi_ref is None:
            J_rec = J_unsup.copy()
        else:
            J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
            J_rec = hp.w * J_unsup + (1.0 - hp.w) * J_hebb_prev

        # Propagazione + autovettori top-K
        JKS = propagate_J(J_rec, iters=1, verbose=False)
        vals, vecs = np.linalg.eig(JKS)
        order = np.argsort(np.real(vals))[::-1]
        autov = np.real(vecs[:, order[: hp.K]]).T

        # TAM
        Net = TAM_Network()
        Net.prepare(J_rec, hp.L)
        xi_r, magn = dis_check(autov, hp.K, hp.L, J_rec, JKS,
                               Î¾=xi_true, updates=hp.updates, show_bar=hp.progress_updates)
        xi_ref = xi_r

        # Metriche + allineamento
        m_per, m_mean, xi_aligned = retrieval_and_align(xi_ref, xi_true)
        m_per_series.append(m_per)
        m_mean_series.append(m_mean)

        # Stima \hat{pi} usando SOLO dati del round t (tutti i client)
        E_t = ETA[:, t].reshape(-1, hp.N)
        pi_hat = estimate_pi_hat_from_examples(xi_aligned, E_t)

        # Serie mixing
        pi_true = pis[t].astype(np.float64)
        pi_true_series.append(pi_true)
        pi_hat_series.append(pi_hat)
        tv_to_prev.append(tv_distance(pis[t], pis[t-1]) if t > 0 else 0.0)

    return {
        "seed": seed,
        "pis_true": np.stack(pi_true_series, axis=0).tolist(),
        "pis_hat": np.stack(pi_hat_series, axis=0).tolist(),
        "tv_series": tv_to_prev,
        "m_series": np.stack(m_per_series, axis=0).tolist(),
        "m_mean_series": m_mean_series,
    }

# ---------------------------------------------------------------------
# Grafici
# ---------------------------------------------------------------------

def plot_simplex(hp: HyperParams, pis_true: np.ndarray, pis_hat: np.ndarray, out_path: Path) -> None:
    sns.set_theme(style=hp.style, palette=hp.palette)
    T = pis_true.shape[0]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(t / max(1, T-1)) for t in range(T)]

    fig, ax = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    draw_ternary_axes(ax, labels=("Î¾1", "Î¾2", "Î¾3"))

    xy_true = np.array([ternary_to_xy(pis_true[t]) for t in range(T)])
    xy_hat  = np.array([ternary_to_xy(pis_hat[t])  for t in range(T)])

    ax.plot(xy_true[:,0], xy_true[:,1], lw=2.0, alpha=0.95, label="mixing vero")
    ax.plot(xy_hat[:,0],  xy_hat[:,1],  lw=2.0, alpha=0.95, linestyle="--", label="mixing stimato")

    for t in range(T):
        ax.scatter(xy_true[t,0], xy_true[t,1], s=42, color=colors[t], edgecolor="white", zorder=3)
        ax.scatter(xy_hat[t,0],  xy_hat[t,1],  s=28, color=colors[t], edgecolor="none", zorder=3)
        ax.annotate("", xy=(xy_hat[t,0], xy_hat[t,1]), xytext=(xy_true[t,0], xy_true[t,1]),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=colors[t], alpha=0.9))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=T-1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("round")

    ax.set_title("Evoluzione dei pesi di mixing sul simplesso (K=3)")
    ax.legend(loc="upper right")

    fig.savefig(out_path, dpi=150)
    print(f"[plot] Salvato: {out_path}")
    plt.show()


def plot_time_panels(hp: HyperParams, tv: np.ndarray, m_per: np.ndarray, out_path: Path) -> None:
    sns.set_theme(style=hp.style, palette=hp.palette)
    T = tv.shape[0]
    rounds = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(rounds, tv, lw=2.0)
    ax.set_title("IntensitÃ  del drift: TV(Ï€_t, Ï€_{t-1})")
    ax.set_xlabel("round"); ax.set_ylabel("TV")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for k in range(hp.K):
        ax.plot(rounds, m_per[:,k], lw=1.8, label=f"m_Î¾{k+1}")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Retrieval per archetipo vs round")
    ax.set_xlabel("round"); ax.set_ylabel("Mattis overlap")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.savefig(out_path, dpi=150)
    print(f"[plot] Salvato: {out_path}")
    plt.show()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    hp = HyperParams(
        L=3, K=3, N=300, n_batch=24, M_total=2400, r_ex=0.8,
        drift_type="cyclic", drift_strength=1.0, period=12,
        w=0.0,
        updates=60, beta_T=2.5, lam=0.2, h_in=0.1,
        n_seeds=3, seed_base=120001,
    )

    base_dir = ROOT / "stress_tests" / "exp06_mixing_drift"
    tag = (
        f"K{hp.K}_N{hp.N}_L{hp.L}_R{hp.n_batch}_M{hp.M_total}_r{hp.r_ex}_"
        f"drift{hp.drift_type}_A{hp.drift_strength}_per{hp.period}_w{hp.w}"
    )
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] Esperimento: {exp_dir}")

    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    # Esegui piÃ¹ seed e aggrega (per i pannelli temporali uso medie su seed)
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

    # Medie su seed per i grafici
    pis_true_mean = np.mean(np.stack(pis_true_all, axis=0), axis=0)
    pis_hat_mean  = np.mean(np.stack(pis_hat_all,  axis=0), axis=0)
    tv_mean       = np.mean(np.stack(tv_all,       axis=0), axis=0)
    m_per_mean    = np.mean(np.stack(m_per_all,    axis=0), axis=0)

    # Tabelle
    import pandas as pd
    pd.DataFrame(rows).to_csv(exp_dir / "results_table.csv", index=False)

    # Grafici
    plot_simplex(hp, pis_true_mean, pis_hat_mean, exp_dir / "fig_mixing_simplex.png")
    plot_time_panels(hp, tv_mean, m_per_mean, exp_dir / "fig_time_panels.png")


if __name__ == "__main__":
    main()
