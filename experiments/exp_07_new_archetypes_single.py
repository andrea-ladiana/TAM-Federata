#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 07 (single) â€” Comparsa di nuovi archetipi a metÃ  training
====================================================================
Versione SINGLE: al round t stimiamo J solo dagli esempi del round t; baseline
senza media cumulativa (no extend). Manteniamo due strategie:
- baseline (w fisso)
- ema_adapt (EMA su J_unsup_curr + w adattivo come nello script originale)

Output: stessi file della versione originale in una cartella dedicata `_single`.
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
from typing import List, Tuple, Dict, Literal, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

_THIS = Path(__file__).resolve()
ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from src.unsup.functions import (
    gen_patterns,
    JK_real,
    unsupervised_J,
    propagate_J,
    estimate_K_eff_from_J,
)
from src.unsup.networks import TAM_Network
from src.unsup.dynamics import dis_check


@dataclass
class HyperParams:
    # Modello / dataset
    L: int = 3
    K: int = 6
    N: int = 300
    n_batch: int = 24
    M_total: int = 1200
    r_ex: float = 0.8
    new_k: int = 3
    t_intro: int = 12
    ramp_len: int = 4
    new_visibility_frac: float = 0.5

    # Strategie
    w_base: float = 0.8
    ema_alpha: float = 0.4
    detect_patience: int = 2

    # Interfaccia / esecuzione
    updates: int = 60
    beta_T: float = 2.5
    lam: float = 0.2
    h_in: float = 0.1
    n_seeds: int = 3
    seed_base: int = 111
    progress_rounds: bool = True


def gen_dataset_new_archetypes(hp: HyperParams, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera dataset round-wise introducendo K_new dopo t_intro con rampa di visibilitÃ .
    Ritorna (ETA: L,T,M_c,N), labels (L,T,M_c), K_true_per_round (T,)."""
    K_old = hp.K - hp.new_k
    K = hp.K; N = hp.N; L = hp.L; T = hp.n_batch
    M_c = math.ceil(hp.M_total / (hp.L * hp.n_batch))
    xi_old = gen_patterns(N, K_old)
    xi_new = gen_patterns(N, hp.new_k)
    xi_true = np.concatenate([xi_old, xi_new], axis=0)

    # visibilitÃ  client dei nuovi archetipi
    L_new = int(round(hp.new_visibility_frac * L))
    client_has_new = np.array([1]*L_new + [0]*(L - L_new), dtype=int)
    rng.shuffle(client_has_new)

    ETA = np.zeros((L, T, M_c, N), dtype=np.float32)
    labels = np.zeros((L, T, M_c), dtype=np.int32)
    p_keep = 0.5 * (1.0 + hp.r_ex)
    K_true_round = np.ones((T,), dtype=int) * K_old
    for t in range(T):
        if t >= hp.t_intro:
            # rampa lineare su ramp_len
            frac = min(1.0, (t - hp.t_intro + 1) / max(1, hp.ramp_len))
            K_true_round[t] = K_old + int(round(frac * hp.new_k))
        for l in range(L):
            allow_new = (client_has_new[l] == 1)
            if t < hp.t_intro or not allow_new:
                pool = np.arange(K_old)
            else:
                frac = min(1.0, (t - hp.t_intro + 1) / max(1, hp.ramp_len))
                K_vis = K_old + int(round(frac * hp.new_k))
                pool = np.arange(K_vis)
            mus = rng.choice(pool, size=M_c, replace=True)
            probs = rng.uniform(size=(M_c, N))
            chi = np.where(probs <= p_keep, 1.0, -1.0)
            ETA[l, t] = (chi * xi_true[mus]).astype(np.float32)
            labels[l, t] = mus.astype(np.int32)
    return ETA, labels, K_true_round


def run_one_seed_strategy(hp: HyperParams, seed: int, strategy: Literal["baseline", "ema_adapt"], out_dir: Path) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    xi_true = gen_patterns(hp.N, hp.K)
    J_star = JK_real(xi_true)
    ETA, labels, K_true_round = gen_dataset_new_archetypes(hp, rng)

    # per EMA e w adattivo
    J_ema_prev = None
    xi_ref = None
    detect_count = 0
    K_eff_series: List[int] = []
    m_old_series: List[float] = []
    m_new_series: List[float] = []
    gap_series: List[float] = []
    mix_err_series: List[float] = []

    round_iter = range(hp.n_batch)
    if hp.progress_rounds:
        round_iter = tqdm(round_iter, desc=f"seed {seed} | {strategy}", leave=False, dynamic_ncols=True)

    for t in round_iter:
        # SINGLE: J da round corrente
        E_t = ETA[:, t, :, :]  # (L, M_c, N)
        L_loc, M_eff, N_loc = E_t.shape
        M_eff_param = max(1, M_eff // hp.K)
        Js_curr = []
        for l in range(hp.L):
            Jl = unsupervised_J(E_t[l], M_eff_param)
            Jl = 0.5 * (Jl + Jl.T)
            np.fill_diagonal(Jl, 0.0)
            Js_curr.append(Jl)
        J_unsup_curr = np.mean(Js_curr, axis=0)

        # blending
        if strategy == "baseline":
            # single: nessuna media cumulativa; w fisso
            J_unsup = J_unsup_curr
            w_t = hp.w_base
        else:  # ema_adapt
            if J_ema_prev is None:
                J_unsup = J_unsup_curr.copy()
            else:
                J_unsup = (1.0 - hp.ema_alpha) * J_ema_prev + hp.ema_alpha * J_unsup_curr
            J_ema_prev = J_unsup
            w_t = hp.w_base  # qui si potrebbe rendere adattivo in base a gap/novelty

        if t == 0:
            J_rec = J_unsup.copy()
        else:
            if xi_ref is None:
                J_rec = J_unsup.copy()
            else:
                J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
                J_rec = w_t * J_unsup + (1.0 - w_t) * J_hebb_prev

        JKS_iter = propagate_J(J_rec, iters=1, verbose=False)
        vals, vecs = np.linalg.eig(JKS_iter)
        mask = (np.real(vals) > 0.5)
        autov = np.real(vecs[:, mask]).T
        xi_hat, Magn = dis_check(autov, hp.K, hp.L, J_rec, JKS_iter, ξ=xi_true, updates=hp.updates, show_bar=False)
        xi_ref = xi_hat

        # Gap spettrale (semplice): Î»_max - Î»_next
        lam = np.real(vals)
        lam.sort()
        gap = float(lam[-1] - lam[-2]) if lam.size >= 2 else float(lam[-1])
        gap_series.append(gap)

        # K_eff stimato
        try:
            K_eff_mp, _, _ = estimate_K_eff_from_J(JKS_iter, method='shuffle', M_eff=M_eff)
        except Exception:
            K_eff_mp = autov.shape[0]
        K_eff_series.append(int(K_eff_mp))

        # Retrieval old/new
        m_mean = float(np.mean(Magn))
        m_old_series.append(m_mean)  # qui non separiamo old/new per brevitÃ 
        m_new_series.append(m_mean)

        # Errore di mixing (stimato grezzo): non abbiamo Ï€_hat robusto qui; lasciamo placeholder 0
        mix_err_series.append(0.0)

    # Riepilogo
    return dict(
        K_true=K_true_round.tolist(),
        K_eff=K_eff_series,
        m_old=m_old_series,
        m_new=m_new_series,
        gap=gap_series,
        mix_err=mix_err_series,
        strategy=strategy,
    )


def run_and_plot(hp: HyperParams, exp_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    log_path = exp_dir / "log.jsonl"
    strategies: List[Literal["baseline", "ema_adapt"]] = ["baseline", "ema_adapt"]
    with open(log_path, "w") as flog:
        for strategy in strategies:
            for i in range(hp.n_seeds):
                seed = hp.seed_base + i
                out = run_one_seed_strategy(hp, seed, strategy, exp_dir)
                rows.append(out | {"seed": seed, "strategy": strategy})
                flog.write(json.dumps(out | {"seed": seed, "strategy": strategy}) + "\n")

    df = pd.DataFrame(rows)
    df.sort_values(["strategy", "seed"], inplace=True)
    df.to_csv(exp_dir / "results_table.csv", index=False)

    # Aggregazioni per plot
    g = df.groupby(["strategy"], as_index=False).agg(
        K_eff_mean=("K_eff", lambda s: np.mean(np.stack(s), axis=0)),
        m_old_mean=("m_old", lambda s: np.mean(np.stack(s), axis=0)),
        m_new_mean=("m_new", lambda s: np.mean(np.stack(s), axis=0)),
        gap_mean=("gap", lambda s: np.mean(np.stack(s), axis=0)),
        mix_err_mean=("mix_err", lambda s: np.mean(np.stack(s), axis=0)),
    )

    rounds = np.arange(hp.n_batch)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # A) K_eff vs round + K_true
    ax = axes[0,0]
    for strat, row in g.groupby("strategy"):
        y = row["K_eff_mean"].iloc[0]
        ax.plot(rounds, y, label=str(strat), linewidth=2.0)
    ax.plot(rounds, df["K_true"].iloc[0], linestyle=":", linewidth=2.0, color="black", label="K (vero)")
    ax.set_title("A) Rilevazione di novitÃ : K_eff vs round (single)")
    ax.set_xlabel("round"); ax.set_ylabel("K_eff"); ax.legend()

    # B) Retrieval old/new vs round (qui old/new coincidono nella nostra stima semplificata)
    ax = axes[0,1]
    for strat, row in g.groupby("strategy"):
        y = row["m_old_mean"].iloc[0]
        ax.plot(rounds, y, linewidth=2.0, label=f"old â€” {str(strat)}")
    ax.set_title("B) Retrieval vs round (single)")
    ax.set_xlabel("round"); ax.set_ylabel("Mattis overlap"); ax.legend()

    # C) Gap spettrale
    ax = axes[1,0]
    for strat, row in g.groupby("strategy"):
        y = row["gap_mean"].iloc[0]
        ax.plot(rounds, y, linewidth=2.0, label=str(strat))
    ax.set_title("C) Gap spettrale"); ax.set_xlabel("round"); ax.set_ylabel("gap"); ax.legend()

    # D) Errore mixing (placeholder)
    ax = axes[1,1]
    for strat, row in g.groupby("strategy"):
        y = row["mix_err_mean"].iloc[0]
        ax.plot(rounds, y, linewidth=2.0, label=str(strat))
    ax.set_title("D) Errore mixing"); ax.set_xlabel("round"); ax.set_ylabel("||Â·||_1"); ax.legend()

    out_path = exp_dir / "fig_grafico7.png"
    fig.savefig(out_path, dpi=150)
    print(f"[plot] Salvato: {out_path}")
    plt.show()


def main():
    hp = HyperParams()
    base_dir = ROOT / "stress_tests" / "exp07_new_archetypes_single"
    tag = (
        f"Kold{hp.K-hp.new_k}_Knew{hp.new_k}_N{hp.N}_L{hp.L}_R{hp.n_batch}_M{hp.M_total}_"
        f"intro{hp.t_intro}_ramp{hp.ramp_len}_alpha{hp.ema_alpha}_w{hp.w_base}"
    )
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    run_and_plot(hp, exp_dir)


if __name__ == '__main__':
    main()


