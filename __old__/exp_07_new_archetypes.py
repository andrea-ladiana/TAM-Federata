#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 07 — Comparsa di nuovi archetipi a metà training (Grafico 7)
=====================================================================
Obiettivo: simulare l'introduzione di **nuovi archetipi** durante il training federato e
confrontare una **baseline lenta** con una variante **reattiva** (EMA + w adattivo).

Acronimi (espansi alla prima occorrenza):
- FL = Federated Learning (apprendimento federato)
- TAM = Tripartite Associative Memory
- EMA = Exponential Moving Average (media mobile esponenziale)
- SNR = Signal-to-Noise Ratio (rapporto segnale/rumore)

Output nella cartella dell'esperimento:
- hyperparams.json
- log.jsonl (una riga per seed/strategia/round con metriche complete)
- results_table.csv (riassunto finale per seed/strategia)
- fig_grafico7.png (pannello 2×2: K_eff, retrieval old/new, gap spettrale, errore di mixing)

Richieste rispettate:
- uso estensivo di tqdm
- soppressione dei log di TensorFlow
- grafici puliti con Seaborn (etichette chiare, legende concise)
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Import moduli di progetto: risali finché non trovi Functions.py
# ---------------------------------------------------------------------
_THIS = Path(__file__).resolve()
ROOT = _THIS
while ROOT != ROOT.parent and not (ROOT / "Functions.py").exists():
    ROOT = ROOT.parent
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
    L: int = 5                 # più client per statistica
    K_old: int = 3
    K_new: int = 3
    N: int = 400
    n_batch: int = 24          # round totali (T)
    M_total: int = 2400        # ≈ 20 esempi/client/round
    r_ex: float = 0.6

    # Introduzione nuovi archetipi
    t_intro: int = 12          # round di introduzione (0-indexed)
    ramp_len: int = 4          # lunghezza rampa d'introduzione (round)
    alpha_max: float = 0.45    # quota massima assegnata ai "nuovi" dopo la rampa
    new_visibility_frac: float = 1.0  # frazione di client che vede i nuovi subito (non-IID se <1)

    # Federated blending
    w_base: float = 0.9        # peso fisso baseline

    # Strategia reattiva (EMA + w adattivo)
    ema_alpha: float = 0.4     # coefficiente EMA per J_unsup
    w_min: float = 0.6
    w_max: float = 0.98
    w_tau: float = 0.10        # soglia per TV stimata (pi_hat) per aumentare w
    w_scale: float = 0.05      # pendenza della sigmoide per w_t

    # Rilevazione novità
    detect_patience: int = 2   # round consecutivi con K_eff >= K_old+1 per espandere

    # Dinamica TAM
    updates: int = 60
    beta_T: float = 2.5
    lam: float = 0.2
    h_in: float = 0.1

    # Esperimento
    n_seeds: int = 6
    seed_base: int = 200001

    # Plot
    palette: str = "deep"
    style: Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "whitegrid"

    # Barre di progresso
    progress_seeds: bool = True
    progress_rounds: bool = True
    progress_strategies: bool = True   # barra sul loop delle strategie per seed
    progress_layers: bool = False      # barra interna sul calcolo delle J_l per round
    progress_updates: bool = False     # mostra barra interna in dis_check (se supportato)

# ---------------------------------------------------------------------
# Utility: mixing schedule con nuovi archetipi
# ---------------------------------------------------------------------

def build_mixing_schedule(hp: HyperParams, rng: np.random.Generator) -> np.ndarray:
    """Costruisce pi_true(t) ∈ R^{T×K_tot}.
    Prima di t_intro: massa solo su K_old, bilanciata (o piccole fluttuazioni). Dopo:
    rampa di lunghezza ramp_len che porta una quota alpha_max ai nuovi, bilanciata tra loro.
    I vecchi vengono riscalati a 1-alpha(t).
    """
    K_tot = hp.K_old + hp.K_new
    T = hp.n_batch
    pi = np.zeros((T, K_tot), dtype=np.float64)

    # base sui vecchi (leggera randomizzazione per evitare perfetto bilanciamento)
    base_old = rng.dirichlet(alpha=np.ones(hp.K_old))  # distribuzione sui vecchi
    for t in range(T):
        if t <= hp.t_intro:
            pi[t, :hp.K_old] = base_old
            pi[t, hp.K_old:] = 0.0
        else:
            # rampa
            dt = t - hp.t_intro
            a = min(1.0, dt / max(1, hp.ramp_len)) * hp.alpha_max
            # quota per i nuovi (bilanciata tra K_new)
            new_share = np.ones(hp.K_new, dtype=np.float64) / max(1, hp.K_new)
            # vecchi riscalati su (1-a) mantenendo le proporzioni base_old
            pi[t, :hp.K_old] = (1.0 - a) * base_old
            pi[t, hp.K_old:] = a * new_share
    # normalizza numericamente
    pi /= pi.sum(axis=1, keepdims=True)
    return pi

# ---------------------------------------------------------------------
# Metriche & allineamento
# ---------------------------------------------------------------------

def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    return 0.5 * float(np.sum(np.abs(p - q)))


def retrieval_and_align(xi_hat: np.ndarray, xi_true: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Hungarian + flip di segno robusto. Ritorna (m_per, m_mean, xi_aligned).
    xi_aligned è ordinato come xi_true (dimensione K_true × N).
    """
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
    r"""Stima \hat{pi} classificando SOLO gli esempi del round corrente con i prototipi allineati."""
    K, N = xi_aligned.shape
    X = np.asarray(E_round, dtype=np.float32).reshape(-1, N)
    scores = X @ xi_aligned.T
    labels = np.argmax(scores, axis=1)
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    return (counts / max(1.0, counts.sum())).astype(np.float64)

# ---------------------------------------------------------------------
# Core: una strategia su un seed
# ---------------------------------------------------------------------

def run_one_seed_strategy(
    hp: HyperParams,
    seed: int,
    strategy: Literal["baseline", "ema_adapt"],
    out_dir: Path,
) -> Dict:
    rng = np.random.default_rng(seed)

    K_tot = hp.K_old + hp.K_new
    xi_true = gen_patterns(hp.N, K_tot)
    _ = JK_real(xi_true)  # per coerenza, non usato nei grafici

    # schedule mixing vero
    pi_true = build_mixing_schedule(hp, rng)  # (T, K_tot)

    # per-client visibility dei nuovi
    L_new = int(round(hp.new_visibility_frac * hp.L))
    client_has_new = np.array([1]*L_new + [0]*(hp.L - L_new), dtype=int)
    rng.shuffle(client_has_new)

    # Genera dataset round-wise (no pre-accumulo per facilitare EMA/extend)
    # shape: (L, T, M_c, N)
    T = hp.n_batch
    M_c = math.ceil(hp.M_total / (hp.L * hp.n_batch))
    p_keep = 0.5 * (1.0 + hp.r_ex)
    ETA = np.zeros((hp.L, T, M_c, hp.N), dtype=np.float32)
    for l in range(hp.L):
        for t in range(T):
            # visibilità non-IID: se client non ha ancora "nuovi", azzera le componenti nuove
            p_t = pi_true[t].copy()
            if t <= hp.t_intro or client_has_new[l] == 0:
                p_t[hp.K_old:] = 0.0
                p_t[:hp.K_old] /= max(1e-12, p_t[:hp.K_old].sum())
            mus = rng.choice(K_tot, size=M_c, p=p_t)
            probs = rng.uniform(size=(M_c, hp.N))
            chi = np.where(probs <= p_keep, 1.0, -1.0)
            xi_sel = xi_true[mus]
            ETA[l, t] = (chi * xi_sel).astype(np.float32)

    # stato per strategia
    K_work = hp.K_old  # dimensione TAM corrente
    detect_count = 0

    # serie loggate
    m_old, m_new = [], []
    k_eff_list, gap_list, mix_err_list = [], [], []

    # per EMA e w adattivo
    J_ema_prev = None
    w_prev = hp.w_base
    pi_hat_prev = None
    pi_hat_prev2 = None
    xi_ref = None  # inizializzato prima dell'uso in Hebb

    # loop round
    round_iter = range(T)
    if hp.progress_rounds:
        round_iter = tqdm(round_iter, desc=f"seed {seed} | {strategy}", leave=False, dynamic_ncols=True)

    for t in round_iter:
        # Costruisci J_unsup da dati del round t
        Js_curr = []
        layer_iter = range(hp.L)
        if hp.progress_layers:
            layer_iter = tqdm(layer_iter, desc=f"{strategy} t{t} layers", leave=False, dynamic_ncols=True)
        for l in layer_iter:
            E_lt = ETA[l, t]
            M_eff_param = max(1, E_lt.shape[0] // max(1, K_work))
            Jl = unsupervised_J(E_lt, M_eff_param)
            Jl = 0.5 * (Jl + Jl.T)
            np.fill_diagonal(Jl, 0.0)
            Js_curr.append(Jl)
        J_unsup_curr = np.mean(Js_curr, axis=0)

        if strategy == "baseline":
            # extend: media cumulativa dei J_unsup_curr
            if t == 0:
                J_unsup = J_unsup_curr.copy()
            else:
                J_unsup = (t/(t+1))*J_unsup + (1/(t+1))*J_unsup_curr
            w_t = hp.w_base
        else:  # ema_adapt
            if J_ema_prev is None:
                J_unsup = J_unsup_curr.copy()
            else:
                J_unsup = (1.0 - hp.ema_alpha) * J_ema_prev + hp.ema_alpha * J_unsup_curr
            J_ema_prev = J_unsup
            # w adattivo in base alla TV stimata sui mixing (con pi_hat disponibili)
            if pi_hat_prev is not None and pi_hat_prev2 is not None:
                tv_hat = tv_distance(pi_hat_prev, pi_hat_prev2)
                # sigmoide squashed tra w_min e w_max
                z = (tv_hat - hp.w_tau) / max(1e-6, hp.w_scale)
                w_t = hp.w_min + (hp.w_max - hp.w_min) / (1.0 + math.exp(-z))
            else:
                w_t = hp.w_base

        # Blend con memoria (Hebb su prototipi precedenti)
        if t == 0 or K_work <= 0:
            J_rec = J_unsup.copy()
        else:
            # Nota: usiamo i prototipi dell'iterazione precedente (se esistono)
            try:
                J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
                J_rec = w_t * J_unsup + (1.0 - w_t) * J_hebb_prev
            except NameError:
                J_rec = J_unsup.copy()

        # Propaga e calcola spettro
        JKS = propagate_J(J_rec, iters=1, verbose=False)
        vals = np.real(np.linalg.eigvals(JKS))
        lam = np.sort(vals)[::-1]

        # K_eff (shuffle) e gap al confine K_old
        try:
            K_eff, _, _ = estimate_K_eff_from_J(JKS, method='shuffle', M_eff=ETA[:, :t+1].reshape(hp.L, -1, hp.N).shape[1])
        except Exception:
            K_eff = K_work
        k_eff_list.append(int(K_eff))
        if len(lam) > hp.K_old:
            lamK = lam[hp.K_old - 1]
            lamK1 = lam[hp.K_old]
            gap = float((lamK - lamK1) / (abs(lamK) + 1e-12))
        else:
            gap = float('nan')
        gap_list.append(gap)

        # TAM: costruisci autovettori e ricostruisci prototipi (dimensione K_work)
        # Se serve espandere, fallo (detection con pazienza)
        if t > hp.t_intro and K_eff >= hp.K_old + 1:
            detect_count += 1
        else:
            detect_count = 0
        if detect_count >= hp.detect_patience and K_work < hp.K_old + hp.K_new:
            K_work = min(hp.K_old + hp.K_new, K_tot)

        # Autovettori top-K_work
        try:
            evals, evecs = np.linalg.eig(JKS)
            order = np.argsort(np.real(evals))[::-1]
            autov = np.real(evecs[:, order[: K_work]]).T
        except Exception:
            # fallback sicuro
            autov = np.eye(hp.N, K_work, dtype=np.float32).T

        Net = TAM_Network()
        Net.prepare(J_rec, hp.L)
        xi_r, magn = dis_check(autov, K_work, hp.L, J_rec, JKS, ξ=xi_true, updates=hp.updates, show_bar=hp.progress_updates)
        # mantieni riferimento per il prossimo round (Hebb memoria)
        xi_ref = xi_r

        # Allinea ai veri archetipi (sempre K_tot × N)
        m_per, m_mean, xi_aligned = retrieval_and_align(xi_ref, xi_true)
        # Media separata su vecchi e nuovi
        m_old.append(float(np.mean(m_per[:hp.K_old])))
        if t <= hp.t_intro:
            m_new.append(np.nan)
        else:
            # se K_work < K_tot, l'allineamento può usare prototipi riutilizzati → trend comunque informativo
            m_new.append(float(np.mean(m_per[hp.K_old:])))

        # Stima mixing dal round corrente (tutti i client)
        E_t = ETA[:, t].reshape(-1, hp.N)
        pi_hat_t = estimate_pi_hat_from_examples(xi_aligned, E_t)
        pi_hat_prev2 = pi_hat_prev
        pi_hat_prev = pi_hat_t

        l1_err = float(np.sum(np.abs(pi_hat_t - pi_true[t])))  # ||.||_1
        mix_err_list.append(l1_err)

    # Costruisci dizionario di serie
    true_K_series = [hp.K_old if t <= hp.t_intro else hp.K_old + hp.K_new for t in range(T)]
    return {
        "seed": seed,
        "strategy": strategy,
        "K_true": true_K_series,
        "K_eff": k_eff_list,
        "gap": gap_list,
        "m_old": m_old,
        "m_new": m_new,
        "mix_err": mix_err_list,
    }

# ---------------------------------------------------------------------
# Aggregazione e grafico 7 (2×2)
# ---------------------------------------------------------------------

def aggregate_and_plot(hp: HyperParams, rows: List[Dict], out_dir: Path) -> None:
    # Tabella long per plotting facile
    recs = []
    for r in rows:
        T = len(r["K_eff"])
        for t in range(T):
            recs.append({
                "strategy": r["strategy"],
                "seed": r["seed"],
                "round": t,
                "K_true": r["K_true"][t],
                "K_eff": r["K_eff"][t],
                "gap": r["gap"][t],
                "m_old": r["m_old"][t],
                "m_new": r["m_new"][t],
                "mix_err": r["mix_err"][t],
            })
    df = pd.DataFrame.from_records(recs)
    df.sort_values(["round", "strategy", "seed"], inplace=True)
    df.to_csv(out_dir / "results_table.csv", index=False)

    # medie ed errore standard per round/strategia
    def agg_se(x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if x.size == 0:
            return np.nan
        return float(np.std(x, ddof=1) / max(1, math.isfinite(x.size) and x.size) ** 0.5)

    g = df.groupby(["strategy", "round"], as_index=False).agg(
        K_true_mean=("K_true", "mean"),
        K_eff_mean=("K_eff", "mean"),
        K_eff_se=("K_eff", lambda s: float(np.std(s, ddof=1)/max(1, len(s))**0.5)),
        gap_mean=("gap", np.nanmean),
        gap_se=("gap", lambda s: float(np.nanstd(s, ddof=1)/max(1, s.notna().sum())**0.5)),
        m_old_mean=("m_old", np.nanmean),
        m_old_se=("m_old", lambda s: float(np.nanstd(s, ddof=1)/max(1, s.notna().sum())**0.5)),
        m_new_mean=("m_new", np.nanmean),
        m_new_se=("m_new", lambda s: float(np.nanstd(s, ddof=1)/max(1, s.notna().sum())**0.5)),
        mix_err_mean=("mix_err", np.nanmean),
        mix_err_se=("mix_err", lambda s: float(np.nanstd(s, ddof=1)/max(1, s.notna().sum())**0.5)),
    )

    # Plot
    sns.set_theme(style=hp.style, palette=hp.palette)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), constrained_layout=True)

    # A) K_eff vs round + K_true
    ax = axes[0, 0]
    for strat, df_s in g.groupby("strategy"):
        ax.plot(df_s["round"], df_s["K_eff_mean"], label=f"K_eff — {strat}", linewidth=2.0)
    # verità a pezzi (media identica su strategie)
    df_true = g[g["strategy"] == g["strategy"].unique()[0]]
    ax.plot(df_true["round"], df_true["K_true_mean"], linestyle=":", linewidth=2.0, color="black", label="K (vero)")
    ax.axvline(hp.t_intro, color="0.4", linestyle="--", linewidth=1.2)
    ax.axvspan(hp.t_intro, hp.t_intro + hp.ramp_len, color="0.6", alpha=0.08)
    ax.set_title("A) Rilevazione di novità: K_eff vs round")
    ax.set_xlabel("round"); ax.set_ylabel("K_eff")
    ax.set_ylim(bottom=hp.K_old - 0.8, top=hp.K_old + hp.K_new + 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # B) Retrieval old/new vs round
    ax = axes[0, 1]
    for strat, df_s in g.groupby("strategy"):
        ax.plot(df_s["round"], df_s["m_old_mean"], linewidth=2.0, label=f"old — {strat}")
        ax.plot(df_s["round"], df_s["m_new_mean"], linewidth=2.0, linestyle="--", label=f"new — {strat}")
    ax.axvline(hp.t_intro, color="0.4", linestyle="--", linewidth=1.2)
    ax.axvspan(hp.t_intro, hp.t_intro + hp.ramp_len, color="0.6", alpha=0.08)
    ax.set_title("B) Retrieval vecchi vs nuovi archetipi")
    ax.set_xlabel("round"); ax.set_ylabel("Mattis overlap")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # C) Gap spettrale al confine K_old
    ax = axes[1, 0]
    for strat, df_s in g.groupby("strategy"):
        ax.plot(df_s["round"], df_s["gap_mean"], linewidth=2.0, label=strat)
    ax.axvline(hp.t_intro, color="0.4", linestyle="--", linewidth=1.2)
    ax.axvspan(hp.t_intro, hp.t_intro + hp.ramp_len, color="0.6", alpha=0.08)
    ax.set_title("C) Gap spettrale (λ_K−λ_{K+1})/λ_K")
    ax.set_xlabel("round"); ax.set_ylabel("gap")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # D) Errore di mixing L1
    ax = axes[1, 1]
    for strat, df_s in g.groupby("strategy"):
        ax.plot(df_s["round"], df_s["mix_err_mean"], linewidth=2.0, label=strat)
    ax.axvline(hp.t_intro, color="0.4", linestyle="--", linewidth=1.2)
    ax.axvspan(hp.t_intro, hp.t_intro + hp.ramp_len, color="0.6", alpha=0.08)
    ax.set_title("D) Errore di mixing ∥π_t−\u005Chat{π}_t∥_1")
    ax.set_xlabel("round"); ax.set_ylabel("||·||_1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig_path = out_dir / "fig_grafico7.png"
    fig.savefig(fig_path, dpi=150)
    print(f"[plot] Salvato: {fig_path}")
    plt.show()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # Istanziazione esplicita degli iperparametri (stile exp_06) — modifica liberamente.
    hp = HyperParams(
        # Modello / dataset
        L=3,
        K_old=3,
        K_new=3,
        N=400,
        n_batch=24,
        M_total=2400,
        r_ex=0.8,

        # Introduzione nuovi archetipi
        t_intro=12,          # round di introduzione (0-index)
        ramp_len=2,          # lunghezza rampa (round)
        alpha_max=0.5,      # quota finale dei nuovi archetipi
        new_visibility_frac=1.0,  # frazione client che vede subito i nuovi

        # Federated blending (baseline)
        w_base=0.8,

        # Strategia reattiva EMA + w adattivo
        ema_alpha=0.4,
        w_min=0.6,
        w_max=0.98,
        w_tau=0.10,
        w_scale=0.05,

        # Rilevazione novità
        detect_patience=2,

        # Dinamica TAM
        updates=60,
        beta_T=2.5,
        lam=0.2,
        h_in=0.1,

        # Esperimento
        n_seeds=3,
        seed_base=200001,

        # Plot / progresso
        palette="deep",
        style="whitegrid",
        progress_seeds=True,
        progress_rounds=True,
    )

    base_dir = ROOT / "stress_tests" / "exp07_new_archetypes"
    tag = (
        f"Kold{hp.K_old}_Knew{hp.K_new}_N{hp.N}_L{hp.L}_R{hp.n_batch}_M{hp.M_total}_"
        f"intro{hp.t_intro}_ramp{hp.ramp_len}_alpha{hp.alpha_max}_vis{hp.new_visibility_frac}_"
        f"w{hp.w_base}_ema{hp.ema_alpha}"
    )
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] Esperimento: {exp_dir}")

    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    strategies = ["baseline", "ema_adapt"]
    rows: List[Dict] = []

    log_path = exp_dir / "log.jsonl"
    with open(log_path, "w") as flog:
        seed_iter = range(hp.n_seeds)
        if hp.progress_seeds:
            seed_iter = tqdm(seed_iter, desc="seeds", dynamic_ncols=True)
        for s in seed_iter:
            seed = hp.seed_base + s
            strat_iter = strategies
            if hp.progress_strategies:
                strat_iter = tqdm(strategies, desc=f"seed {seed} strategies", leave=False, dynamic_ncols=True)
            for strat in strat_iter:
                out = run_one_seed_strategy(hp, seed, strat, exp_dir)  # type: ignore[arg-type]
                rows.append(out | {"seed": seed, "strategy": strat})
                flog.write(json.dumps(out | {"seed": seed, "strategy": strat}) + "\n")

    # Riepilogo per seed/strategia (ultimo round)
    recs = []
    for r in rows:
        T = len(r["K_eff"])
        recs.append({
            "seed": r["seed"],
            "strategy": r["strategy"],
            "K_eff_last": r["K_eff"][T-1],
            "m_old_last": r["m_old"][T-1],
            "m_new_last": r["m_new"][T-1],
            "mix_err_last": r["mix_err"][T-1],
        })
    pd.DataFrame.from_records(recs).to_csv(exp_dir / "results_table.csv", index=False)

    aggregate_and_plot(hp, rows, exp_dir)


if __name__ == "__main__":
    main()
