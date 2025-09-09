#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp07_novelty_emergence.py — Experiment 07: Emergence of New Archetypes

Segue l'architettura standard degli altri esperimenti per simulare l'introduzione
di nuovi archetipi durante il training federato in modalità SINGLE.

Obiettivi:
- Simulare l'emergenza graduale di nuovi archetipi (K_old -> K_old + K_new)
- Confrontare diverse strategie di adattamento (baseline vs reattiva)
- Generare plot significativi: K_eff, retrieval, gap spettrale, mixing error
- Utilizzare la pipeline TAM standard della codebase
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Setup project paths
_THIS = Path(__file__).resolve()
ROOT = _THIS.parent.parent  # Go to project root

# Add project root to path so that 'import src.*' works
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / 'src'

# Core imports from codebase
from src.unsup.functions import (
    gen_patterns,
    JK_real,
    unsupervised_J,
    propagate_J,
    estimate_K_eff_from_J,
)
from src.unsup.dynamics import dis_check
from src.narch.novelty import novelty_schedule

# ---------------------------------------------------------------------
# Hyperparameters following the pattern of other experiments
# ---------------------------------------------------------------------
@dataclass
class HyperParams:
    # Model / dataset
    L: int = 3                 # clients
    K_old: int = 3             # initial archetypes
    K_new: int = 3             # new archetypes to introduce
    N: int = 400               # pattern dimension
    n_batch: int = 24          # total rounds
    M_total: int = 2400        # total examples budget
    r_ex: float = 0.8          # signal-to-noise ratio

    # Novelty introduction schedule
    t_intro: int = 12          # round when new archetypes appear
    ramp_len: int = 4          # ramp length for gradual introduction
    alpha_max: float = 0.5     # maximum allocation to new archetypes
    new_visibility_frac: float = 1.0  # fraction of clients that see new archetypes

    # Federated learning parameters
    w_baseline: float = 0.8    # baseline memory weight
    w_adaptive_min: float = 0.6
    w_adaptive_max: float = 0.95
    ema_alpha: float = 0.3     # EMA coefficient for adaptive strategy

    # TAM dynamics
    updates: int = 60
    beta_T: float = 2.5
    lam: float = 0.2
    h_in: float = 0.1

    # Detection parameters
    detect_patience: int = 2   # rounds to wait before expanding K

    # Experiment settings
    n_seeds: int = 6
    seed_base: int = 200001
    progress_seeds: bool = True
    progress_rounds: bool = True

    # Plot settings
    palette: str = "deep"
    style: str = "whitegrid"

# ---------------------------------------------------------------------
# Mixing schedule and dataset generation
# ---------------------------------------------------------------------
def generate_dataset_with_novelty(
    xi_true: np.ndarray,
    pi_schedule: np.ndarray,
    hp: HyperParams,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate dataset following the novelty schedule.
    
    Returns:
        ETA: (L, T, M_c, N) - examples per client/round
        labels: (L, T, M_c) - true class labels
    """
    K, N = xi_true.shape
    T = hp.n_batch
    M_c = math.ceil(hp.M_total / (hp.L * hp.n_batch))
    
    ETA = np.zeros((hp.L, T, M_c, N), dtype=np.float32)
    labels = np.zeros((hp.L, T, M_c), dtype=np.int32)
    
    p_keep = 0.5 * (1.0 + hp.r_ex)
    
    # Client visibility of new archetypes
    n_clients_with_new = int(hp.L * hp.new_visibility_frac)
    clients_see_new = np.array([1] * n_clients_with_new + [0] * (hp.L - n_clients_with_new))
    rng.shuffle(clients_see_new)
    
    for l in range(hp.L):
        for t in range(T):
            # Determine which classes this client can see at this round
            pi_t = pi_schedule[t].copy()
            
            # If client can't see new classes or they haven't been introduced yet
            if t < hp.t_intro or not clients_see_new[l]:
                pi_t[hp.K_old:] = 0.0
                if pi_t[:hp.K_old].sum() > 0:
                    pi_t[:hp.K_old] /= pi_t[:hp.K_old].sum()
                else:
                    pi_t[:hp.K_old] = 1.0 / hp.K_old
            
            # Sample class assignments for this client/round
            mus = rng.choice(K, size=M_c, p=pi_t)
            
            # Generate examples with noise
            for m in range(M_c):
                mu = mus[m]
                chi = (rng.uniform(size=N) <= p_keep).astype(np.float32) * 2.0 - 1.0
                ETA[l, t, m] = chi * xi_true[mu].astype(np.float32)
                labels[l, t, m] = mu
    
    return ETA, labels

# ---------------------------------------------------------------------
# Metrics and alignment utilities
# ---------------------------------------------------------------------
def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.sum(np.abs(p - q)))

def retrieval_and_align(xi_hat: np.ndarray, xi_true: np.ndarray) -> Tuple[np.ndarray, float]:
    """Hungarian matching with sign alignment."""
    if xi_hat.size == 0 or xi_hat.ndim != 2:
        return np.zeros(xi_true.shape[0]), 0.0
    
    from scipy.optimize import linear_sum_assignment
    
    K_true, N = xi_true.shape
    K_hat = xi_hat.shape[0]
    
    # Compute overlap matrix
    M = np.abs(xi_hat @ xi_true.T) / float(N)
    
    # Hungarian assignment
    if K_hat >= K_true:
        cost = 1.0 - M[:K_true, :]
        rI, cI = linear_sum_assignment(cost)
    else:
        cost = 1.0 - M
        rI, cI = linear_sum_assignment(cost)
    
    # Compute aligned overlaps
    overlaps = np.zeros(K_true)
    for i, (r, c) in enumerate(zip(rI, cI)):
        if c < K_true and r < K_hat:
            overlaps[c] = M[r, c]
    
    return overlaps, float(np.mean(overlaps))

def estimate_pi_hat_from_examples(xi_aligned: np.ndarray, E_round: np.ndarray) -> np.ndarray:
    """Estimate mixing probabilities from current round examples."""
    K, N = xi_aligned.shape
    X = E_round.reshape(-1, N)
    
    # Classify examples
    scores = X @ xi_aligned.T
    labels = np.argmax(scores, axis=1)
    
    # Count occurrences
    counts = np.bincount(labels, minlength=K).astype(float)
    pi_hat = counts / (counts.sum() + 1e-9)
    
    return pi_hat

# ---------------------------------------------------------------------
# Core simulation for one seed and strategy
# ---------------------------------------------------------------------
def run_one_seed_strategy(
    hp: HyperParams,
    seed: int,
    strategy: Literal["baseline", "adaptive"],
    exp_dir: Path,
) -> Dict[str, Any]:
    """Run simulation for one seed and strategy."""
    rng = np.random.default_rng(seed)
    
    # Generate true archetypes
    K_total = hp.K_old + hp.K_new
    xi_true = gen_patterns(hp.N, K_total)
    
    # Generate novelty schedule
    pi_schedule = novelty_schedule(
        T=hp.n_batch,
        K_old=hp.K_old,
        K_new=hp.K_new,
        t_intro=hp.t_intro,
        ramp_len=hp.ramp_len,
        alpha_max=hp.alpha_max,
    )
    
    # Generate dataset
    ETA, labels = generate_dataset_with_novelty(xi_true, pi_schedule, hp, rng)
    
    # Initialize tracking variables
    K_work = hp.K_old  # Current working dimension
    detect_count = 0
    xi_ref = None
    
    # For adaptive strategy
    J_ema = None
    pi_hat_prev = None
    
    # Series to track
    keff_series = []
    gap_series = []
    m_old_series = []
    m_new_series = []
    tv_series = []
    w_series = []
    
    # Main training loop
    round_iter = range(hp.n_batch)
    if hp.progress_rounds:
        round_iter = tqdm(round_iter, desc=f"seed {seed} | {strategy}", leave=False)
    
    for t in round_iter:
        # Get current round data (SINGLE mode)
        E_t = ETA[:, t, :, :]  # (L, M_c, N)
        
        # Estimate J_unsup from current round only
        M_c = E_t.shape[1]
        M_eff_param = max(1, M_c // K_work)
        
        Js = []
        for l in range(hp.L):
            J_l = unsupervised_J(E_t[l], M_eff_param)
            J_l = 0.5 * (J_l + J_l.T)  # Symmetrize
            np.fill_diagonal(J_l, 0.0)
            Js.append(J_l)
        
        J_unsup = np.mean(Js, axis=0)
        
        # Strategy-specific processing
        if strategy == "baseline":
            # Simple weighted combination with previous memory
            if t == 0 or xi_ref is None:
                J_rec = J_unsup.copy()
            else:
                J_hebb = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
                J_rec = hp.w_baseline * J_unsup + (1.0 - hp.w_baseline) * J_hebb
            
            w_current = hp.w_baseline
            
        else:  # adaptive strategy
            # EMA on J_unsup
            if J_ema is None:
                J_ema = J_unsup.copy()
            else:
                J_ema = (1.0 - hp.ema_alpha) * J_ema + hp.ema_alpha * J_unsup
            
            # Adaptive weight based on stability
            if pi_hat_prev is not None:
                # Estimate current pi_hat
                if xi_ref is not None:
                    pi_hat_current = estimate_pi_hat_from_examples(xi_ref, E_t)
                    tv_change = tv_distance(pi_hat_current, pi_hat_prev)
                    
                    # Adaptive weight: higher w when distribution is changing
                    w_adaptive = hp.w_adaptive_min + (hp.w_adaptive_max - hp.w_adaptive_min) / (1.0 + np.exp(-10 * (tv_change - 0.1)))
                else:
                    w_adaptive = hp.w_baseline
            else:
                w_adaptive = hp.w_baseline
            
            # Combine with memory
            if t == 0 or xi_ref is None:
                J_rec = J_ema.copy()
            else:
                J_hebb = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
                J_rec = w_adaptive * J_ema + (1.0 - w_adaptive) * J_hebb
            
            w_current = w_adaptive
        
        w_series.append(w_current)
        
        # Propagation
        JKS = propagate_J(J_rec, iters=1, verbose=False)
        
        # Spectral analysis
        eigenvals, eigenvecs = np.linalg.eig(JKS)
        eigenvals = np.real(eigenvals)
        order = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]
        
        # K_eff estimation
        try:
            K_eff, _, _ = estimate_K_eff_from_J(JKS, method='shuffle', M_eff=M_c)
        except:
            K_eff = K_work
        
        keff_series.append(int(K_eff))
        
        # Spectral gap at K_old boundary
        if len(eigenvals) > hp.K_old:
            lam_k = eigenvals[hp.K_old - 1]
            lam_k1 = eigenvals[hp.K_old]
            gap = (lam_k - lam_k1) / (abs(lam_k) + 1e-12)
        else:
            gap = np.nan
        gap_series.append(gap)
        
        # Novelty detection
        if t > hp.t_intro and K_eff >= hp.K_old + 1:
            detect_count += 1
        else:
            detect_count = 0
        
        # Expand working dimension if novelty detected
        if detect_count >= hp.detect_patience and K_work < K_total:
            K_work = min(K_total, hp.K_old + hp.K_new)
        
        # TAM reconstruction with current working dimension
        mask = eigenvals > 0.5
        if np.sum(mask) < K_work:
            mask[:K_work] = True
        
        autov = np.real(eigenvecs[:, mask]).T
        if autov.shape[0] > K_work:
            autov = autov[:K_work]
        elif autov.shape[0] < K_work:
            # Pad with random vectors if needed
            extra = K_work - autov.shape[0]
            random_vecs = rng.normal(size=(extra, hp.N))
            autov = np.vstack([autov, random_vecs])
        
        # TAM dynamics - simplified call matching other experiments
        try:
            # Use a simple approach similar to other experiments
            from src.unsup.networks import TAM_Network
            Net = TAM_Network()
            Net.prepare(J_rec, hp.L)
            
            # Initialize candidates and run TAM
            sigma_init = []
            for k in range(K_work):
                s0 = np.sign(autov[k] + 0.1 * rng.normal(size=hp.N))
                s0[s0 == 0] = 1
                sigma_init.append(s0)
            
            xi_hat = []
            m_per = []
            
            for k in range(K_work):
                s = sigma_init[k].copy()
                # Simple TAM dynamics
                for _ in range(hp.updates):
                    h = J_rec @ s
                    s = np.sign(h)
                    s[s == 0] = 1
                
                xi_hat.append(s)
                # Compute overlap with true archetype if available
                if k < K_total:
                    overlap = abs(np.dot(s, xi_true[k])) / float(hp.N)
                    m_per.append(overlap)
                else:
                    m_per.append(0.5)
            
            xi_hat = np.array(xi_hat)
            xi_ref = xi_hat
            
        except Exception as e:
            # Fallback if TAM fails
            xi_ref = autov
            m_per = np.ones(K_work) * 0.5
        
        # Align to true archetypes and compute retrieval
        overlaps, _ = retrieval_and_align(xi_ref, xi_true)
        
        m_old = float(np.mean(overlaps[:hp.K_old])) if hp.K_old > 0 else np.nan
        m_new = float(np.mean(overlaps[hp.K_old:])) if hp.K_new > 0 else np.nan
        
        m_old_series.append(m_old)
        m_new_series.append(m_new)
        
        # Estimate current mixing and compute TV distance
        if xi_ref is not None and xi_ref.shape[0] >= K_total:
            pi_hat = estimate_pi_hat_from_examples(xi_ref[:K_total], E_t)
            pi_true = pi_schedule[t] / (pi_schedule[t].sum() + 1e-9)
            tv_err = tv_distance(pi_hat, pi_true)
            
            pi_hat_prev = pi_hat
        else:
            tv_err = np.nan
        
        tv_series.append(tv_err)
    
    return {
        "seed": seed,
        "strategy": strategy,
        "keff": keff_series,
        "gap": gap_series,
        "m_old": m_old_series,
        "m_new": m_new_series,
        "tv_error": tv_series,
        "w_values": w_series,
    }

# ---------------------------------------------------------------------
# Plotting and aggregation
# ---------------------------------------------------------------------
def aggregate_and_plot(hp: HyperParams, results: List[Dict], exp_dir: Path) -> None:
    """Create comprehensive plots following the style of other experiments."""
    
    # Convert to long format for easy plotting
    records = []
    for r in results:
        T = len(r["keff"])
        for t in range(T):
            records.append({
                "strategy": r["strategy"],
                "seed": r["seed"],
                "round": t,
                "keff": r["keff"][t],
                "gap": r["gap"][t],
                "m_old": r["m_old"][t],
                "m_new": r["m_new"][t],
                "tv_error": r["tv_error"][t],
                "w_value": r["w_values"][t],
            })
    
    df = pd.DataFrame(records)
    df.to_csv(exp_dir / "results_detailed.csv", index=False)
    
    # Aggregate by strategy and round
    agg_df = df.groupby(["strategy", "round"]).agg({
        "keff": ["mean", "std"],
        "gap": ["mean", "std"],
        "m_old": ["mean", "std"],
        "m_new": ["mean", "std"],
        "tv_error": ["mean", "std"],
        "w_value": ["mean", "std"],
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = ["_".join(col).strip() if col[1] else col[0] for col in agg_df.columns]
    
    # Create comprehensive plot
    sns.set_theme(style="whitegrid", palette=hp.palette)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Exp-07: Novelty Emergence (K_old={hp.K_old}, K_new={hp.K_new})", fontsize=16)
    
    rounds = np.arange(hp.n_batch)
    
    # Panel A: K_eff detection
    ax = axes[0, 0]
    for strategy in ["baseline", "adaptive"]:
        data = agg_df[agg_df["strategy"] == strategy]
        ax.plot(data["round"], data["keff_mean"], linewidth=2, label=f"K_eff ({strategy})")
        ax.fill_between(data["round"], 
                       data["keff_mean"] - data["keff_std"], 
                       data["keff_mean"] + data["keff_std"], 
                       alpha=0.2)
    
    ax.axhline(hp.K_old, color="gray", linestyle=":", linewidth=1.5, label="K_old")
    ax.axhline(hp.K_old + hp.K_new, color="gray", linestyle="--", linewidth=1.5, label="K_total")
    ax.axvline(hp.t_intro, color="red", linestyle="--", alpha=0.7, label="t_intro")
    ax.axvspan(hp.t_intro, hp.t_intro + hp.ramp_len, alpha=0.1, color="red")
    
    ax.set_title("A) Novelty Detection: K_eff vs Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("K_eff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel B: Retrieval performance
    ax = axes[0, 1]
    for strategy in ["baseline", "adaptive"]:
        data = agg_df[agg_df["strategy"] == strategy]
        ax.plot(data["round"], data["m_old_mean"], linewidth=2, label=f"Old archetypes ({strategy})")
        ax.plot(data["round"], data["m_new_mean"], linewidth=2, linestyle="--", label=f"New archetypes ({strategy})")
    
    ax.axvline(hp.t_intro, color="red", linestyle="--", alpha=0.7)
    ax.axvspan(hp.t_intro, hp.t_intro + hp.ramp_len, alpha=0.1, color="red")
    
    ax.set_title("B) Retrieval: Old vs New Archetypes")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mattis Overlap")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel C: Spectral gap
    ax = axes[1, 0]
    for strategy in ["baseline", "adaptive"]:
        data = agg_df[agg_df["strategy"] == strategy]
        valid_gap = ~np.isnan(data["gap_mean"])
        if valid_gap.any():
            ax.plot(data["round"][valid_gap], data["gap_mean"][valid_gap], 
                   linewidth=2, label=f"Gap ({strategy})")
    
    ax.axvline(hp.t_intro, color="red", linestyle="--", alpha=0.7)
    ax.axvspan(hp.t_intro, hp.t_intro + hp.ramp_len, alpha=0.1, color="red")
    
    ax.set_title("C) Spectral Gap at K_old Boundary")
    ax.set_xlabel("Round")
    ax.set_ylabel("Relative Gap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel D: Mixing error
    ax = axes[1, 1]
    for strategy in ["baseline", "adaptive"]:
        data = agg_df[agg_df["strategy"] == strategy]
        valid_tv = ~np.isnan(data["tv_error_mean"])
        if valid_tv.any():
            ax.plot(data["round"][valid_tv], data["tv_error_mean"][valid_tv], 
                   linewidth=2, label=f"TV error ({strategy})")
    
    ax.axvline(hp.t_intro, color="red", linestyle="--", alpha=0.7)
    ax.axvspan(hp.t_intro, hp.t_intro + hp.ramp_len, alpha=0.1, color="red")
    
    ax.set_title("D) Mixing Estimation Error")
    ax.set_xlabel("Round")
    ax.set_ylabel("TV Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    fig_path = exp_dir / "fig_novelty_emergence.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[PLOT] Saved: {fig_path}")
    plt.show()

# ---------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------
def main():
    """Main experiment execution."""
    hp = HyperParams(
        # Model parameters
        L=3,
        K_old=3,
        K_new=3,
        N=400,
        n_batch=24,
        M_total=2400,
        r_ex=0.8,
        
        # Novelty parameters
        t_intro=12,
        ramp_len=4,
        alpha_max=0.5,
        new_visibility_frac=1.0,
        
        # Learning parameters
        w_baseline=0.8,
        ema_alpha=0.3,
        
        # Experiment parameters
        n_seeds=6,
        progress_seeds=True,
        progress_rounds=True,
    )
    
    # Setup experiment directory
    base_dir = ROOT / "stress_tests" / "exp07_novelty_emergence"
    tag = (
        f"Kold{hp.K_old}_Knew{hp.K_new}_N{hp.N}_L{hp.L}_T{hp.n_batch}_"
        f"intro{hp.t_intro}_ramp{hp.ramp_len}_alpha{hp.alpha_max}"
    )
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[SETUP] Experiment directory: {exp_dir}")
    
    # Save hyperparameters
    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)
    
    # Run experiments
    strategies = ["baseline", "adaptive"]
    results = []
    
    with open(exp_dir / "log.jsonl", "w") as flog:
        seed_iter = range(hp.n_seeds)
        if hp.progress_seeds:
            seed_iter = tqdm(seed_iter, desc="Seeds")
        
        for s in seed_iter:
            seed = hp.seed_base + s
            
            for strategy in strategies:
                result = run_one_seed_strategy(hp, seed, strategy, exp_dir)  # type: ignore
                results.append(result)
                
                # Log result
                flog.write(json.dumps(result) + "\n")
                flog.flush()
    
    print(f"[RESULTS] Completed {len(results)} simulations")
    
    # Generate plots and summary
    aggregate_and_plot(hp, results, exp_dir)
    
    # Summary statistics
    summary_records = []
    for r in results:
        final_keff = r["keff"][-1]
        final_m_old = r["m_old"][-1]
        final_m_new = r["m_new"][-1] if not np.isnan(r["m_new"][-1]) else 0.0
        
        summary_records.append({
            "seed": r["seed"],
            "strategy": r["strategy"],
            "final_keff": final_keff,
            "final_m_old": final_m_old,
            "final_m_new": final_m_new,
            "detection_success": int(final_keff >= hp.K_old + 1),
        })
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(exp_dir / "summary.csv", index=False)
    
    # Print summary
    print("\n[SUMMARY]")
    for strategy in strategies:
        strat_data = summary_df[summary_df["strategy"] == strategy]
        print(f"{strategy.capitalize()} Strategy:")
        print(f"  Detection Rate: {strat_data['detection_success'].mean():.2f}")
        print(f"  Final K_eff: {strat_data['final_keff'].mean():.2f} ± {strat_data['final_keff'].std():.2f}")
        print(f"  Final m_old: {strat_data['final_m_old'].mean():.3f} ± {strat_data['final_m_old'].std():.3f}")
        print(f"  Final m_new: {strat_data['final_m_new'].mean():.3f} ± {strat_data['final_m_new'].std():.3f}")
        print()

if __name__ == "__main__":
    main()
