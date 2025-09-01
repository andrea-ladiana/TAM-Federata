#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 04.1 â€” Overhead di quantizzazione vs 1-bit (compute & comunicazione)
==============================================================================
Mostra l'overhead **computazionale** e **di comunicazione** all'ultimo round quando si usa una
quantizzazione a b bit, rispetto alla baseline **1-bit** (sign) sulle matrici locali J_l.

Sigle (espansione alla prima occorrenza):
- FL = Federated Learning (apprendimento federato)
- TAM = Tripartite Associative Memory
- SNR = Signal-to-Noise Ratio (rapporto segnale/rumore)

Uscite:
- hyperparams.json
- log.jsonl (una riga per seed e bitwidth con tempi medi per round)
- results_table.csv
- fig_overhead.png (Seaborn, 2Ã—2: overhead compute (solo encoding), overhead compute (pipeline client),
  overhead comunicazione relativo, e MB/round assoluti)

Note:
- usa tqdm estensivamente
- sopprime log TF
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Import moduli del progetto
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    pass
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from unsup.functions import (
    gen_patterns,
    JK_real,
    unsupervised_J,
)

# ---------------------------------------------------------------------
# Iperparametri
# ---------------------------------------------------------------------
@dataclass
class HyperParams:
    # Modello / dataset
    L: int = 3
    K: int = 6
    N: int = 500
    n_batch: int = 12
    M_total: int = 2400
    r_ex: float = 0.6

    # Sweep bit
    bits_list: List[int] = field(default_factory=list)  # impostato nel main
    stochastic_rounding: bool = True
    symmetrize: bool = True
    zero_diag: bool = True

    # Esperimento
    n_seeds: int = 6
    seed_base: int = 93001

    # Plot
    palette: str = "deep"
    style: str = "whitegrid"
    # Progress flags
    show_bits_bar: bool = True
    show_seed_bar: bool = True
    show_round_bar: bool = True
    show_client_bar: bool = False

# ---------------------------------------------------------------------
# Dataset unsupervised semplice (IID sugli archetipi)
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
# Quantizzazione simmetrica per-entry (come in exp04)
# ---------------------------------------------------------------------

def quantize_symmetric(J: np.ndarray, bits: int, stochastic: bool, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    if bits <= 0:
        raise ValueError("bits deve essere >=1")
    if bits == 1:
        alpha = float(np.mean(np.abs(J))) + 1e-12
        Jq = alpha * np.sign(J)
        return Jq.astype(np.float32), alpha
    Qmax = float(2**(bits - 1) - 1)
    max_abs = float(np.max(np.abs(J))) + 1e-12
    scale = max_abs / Qmax if Qmax > 0 else 1.0
    Y = J / scale
    if stochastic:
        frac = Y - np.floor(Y)
        rnd = rng.random(size=Y.shape)
        Yq = np.floor(Y) + (rnd < frac).astype(Y.dtype)
    else:
        Yq = np.round(Y)
    Yq = np.clip(Yq, -Qmax, Qmax)
    Jq = (Yq * scale).astype(np.float32)
    return Jq, float(scale)


def client_build_and_quantize(E_l: np.ndarray, K: int, bits: int, stochastic: bool,
                              symmetrize: bool, zero_diag: bool,
                              rng: np.random.Generator) -> Tuple[np.ndarray, float, float, float]:
    """Ritorna (J_q, scale, t_unsup, t_quant) e registra il tempo per J Hebb e per la quantizzazione."""
    t0 = time.perf_counter()
    M_eff_param = max(1, E_l.shape[0] // K)
    J = unsupervised_J(E_l, M_eff_param)
    if symmetrize:
        J = 0.5 * (J + J.T)
    if zero_diag:
        np.fill_diagonal(J, 0.0)
    t1 = time.perf_counter()

    Jq, scale = quantize_symmetric(J, bits, stochastic, rng)
    t2 = time.perf_counter()

    return Jq, float(scale), (t1 - t0), (t2 - t1)

# ---------------------------------------------------------------------
# Comunicazione: stima dei bit per round
# ---------------------------------------------------------------------

def comm_bits_per_round(N: int, L: int, bits: int, symmetrize: bool, zero_diag: bool) -> int:
    if symmetrize and zero_diag:
        elems = N * (N - 1) // 2  # triangolare superiore senza diagonale
    else:
        elems = N * N
    payload = elems * bits
    # inviamo anche uno scale per matrice (float32) per client
    scale_bits = 32 * L
    return payload * L + scale_bits

# ---------------------------------------------------------------------
# Core run per seed e bitwidth (misura tempi medi per round)
# ---------------------------------------------------------------------

def run_one_seed_one_bits(hp: HyperParams, seed: int, bits: int) -> dict:
    rng = np.random.default_rng(seed)
    xi_true = gen_patterns(hp.N, hp.K)
    ETA = gen_dataset_unsup(xi_true, hp.M_total, hp.r_ex, hp.n_batch, hp.L, rng)

    t_unsup_sum = 0.0
    t_quant_sum = 0.0
    t_client_pipeline_sum = 0.0

    # misuriamo tempi su tutti i round
    round_iter = range(hp.n_batch)
    if hp.show_round_bar:
        round_iter = tqdm(round_iter, desc=f"rounds bits={bits} seed={seed}", leave=False, dynamic_ncols=True)
    for b in round_iter:
        # Costruisci vista extend
        ETA_extend = ETA[:, : b + 1, :, :].transpose(0, 2, 1, 3).reshape(hp.L, -1, hp.N)
        # per ogni client
        client_iter = range(hp.L)
        if hp.show_client_bar:
            client_iter = tqdm(client_iter, desc=f"clients r{b}", leave=False, dynamic_ncols=True)
        for l in client_iter:
            E_l = ETA_extend[l]
            t0 = time.perf_counter()
            Jq, scale, t_unsup, t_quant = client_build_and_quantize(
                E_l, hp.K, bits, hp.stochastic_rounding, hp.symmetrize, hp.zero_diag, rng
            )
            t1 = time.perf_counter()
            t_unsup_sum += t_unsup
            t_quant_sum += t_quant
            t_client_pipeline_sum += (t1 - t0)

    rounds_total = hp.n_batch
    clients_total = hp.L
    # tempi medi per client per round
    denom = max(1, rounds_total * clients_total)
    enc_time = t_quant_sum / denom
    pipe_time = t_client_pipeline_sum / denom

    # comunicazione per round
    comm_bits = comm_bits_per_round(hp.N, hp.L, bits, hp.symmetrize, hp.zero_diag)

    return {
        "bits": int(bits),
        "seed": int(seed),
        "enc_time_per_client_round": enc_time,
        "pipeline_time_per_client_round": pipe_time,
        "comm_bits_per_round": comm_bits,
    }

# ---------------------------------------------------------------------
# Aggregazione e grafico
# ---------------------------------------------------------------------

def aggregate_and_plot(hp: HyperParams, rows: List[dict], out_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.sort_values(["bits", "seed"], inplace=True)
    df.to_csv(out_dir / "results_table.csv", index=False)

    # baseline 1-bit
    base = df[df["bits"] == 1].groupby("bits").agg(
        enc=("enc_time_per_client_round", "mean"),
        pipe=("pipeline_time_per_client_round", "mean"),
        comm=("comm_bits_per_round", "mean"),
    )
    base_enc = float(base.loc[1, "enc"]) if 1 in base.index else np.nan
    base_pipe = float(base.loc[1, "pipe"]) if 1 in base.index else np.nan
    base_comm = float(base.loc[1, "comm"]) if 1 in base.index else np.nan

    # colonne di overhead relativo
    df["oh_enc_rel"] = df["enc_time_per_client_round"] / base_enc
    df["oh_pipe_rel"] = df["pipeline_time_per_client_round"] / base_pipe
    df["oh_comm_rel"] = df["comm_bits_per_round"] / base_comm

    _allowed_styles = {"white","dark","whitegrid","darkgrid","ticks"}
    style_use = hp.style if hp.style in _allowed_styles else "whitegrid"
    sns.set_theme(style=style_use, palette=hp.palette)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    # 1) Overhead compute (solo encoding)
    ax = axes[0, 0]
    sns.pointplot(data=df, x="bits", y="oh_enc_rel", errorbar=("se", 1), dodge=True, ax=ax)
    ax.axhline(1.0, color="tab:gray", linestyle=":", linewidth=1.2)
    ax.set_title("Overhead compute (encoding) vs 1-bit")
    ax.set_xlabel("bit di quantizzazione")
    ax.set_ylabel("tempo relativo (Ã— baseline 1-bit)")
    ax.grid(True, alpha=0.3)

    # 2) Overhead compute (pipeline client)
    ax = axes[0, 1]
    sns.pointplot(data=df, x="bits", y="oh_pipe_rel", errorbar=("se", 1), dodge=True, ax=ax)
    ax.axhline(1.0, color="tab:gray", linestyle=":", linewidth=1.2)
    ax.set_title("Overhead compute (pipeline client) vs 1-bit")
    ax.set_xlabel("bit")
    ax.set_ylabel("tempo relativo (Ã— baseline 1-bit)")
    ax.grid(True, alpha=0.3)

    # 3) Overhead comunicazione (relativo)
    ax = axes[1, 0]
    # comunicazione Ã¨ deterministica per bit â†’ raggruppo
    df_comm = df.groupby("bits", as_index=False)["oh_comm_rel"].mean()
    sns.pointplot(data=df_comm, x="bits", y="oh_comm_rel", dodge=False, ax=ax)
    ax.axhline(1.0, color="tab:gray", linestyle=":", linewidth=1.2)
    ax.set_title("Overhead comunicazione vs 1-bit")
    ax.set_xlabel("bit")
    ax.set_ylabel("bit trasmessi (Ã— baseline 1-bit)")
    ax.grid(True, alpha=0.3)

    # 4) Payload assoluto (MB/round)
    ax = axes[1, 1]
    df_mb = df.groupby("bits", as_index=False)["comm_bits_per_round"].mean()
    df_mb["MB_per_round"] = df_mb["comm_bits_per_round"] / 8.0 / 1e6
    sns.pointplot(data=df_mb, x="bits", y="MB_per_round", dodge=False, ax=ax)
    ax.set_title("Payload per round (MB)")
    ax.set_xlabel("bit")
    ax.set_ylabel("MB / round")
    ax.grid(True, alpha=0.3)

    fig_path = out_dir / "fig_overhead.png"
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
        M_total=1200, 
        r_ex=0.7,
        bits_list=[1, 2, 3, 4, 8, 16, 32],
        stochastic_rounding=True, 
        symmetrize=True, 
        zero_diag=True,
        n_seeds=32, 
        seed_base=93001,
    )

    base_dir = ROOT / "stress_tests" / "exp041_overhead"
    tag = (
        f"K{hp.K}_N{hp.N}_L{hp.L}_nb{hp.n_batch}_M{hp.M_total}_r{hp.r_ex}_"
        f"bits{min(hp.bits_list)}-{max(hp.bits_list)}_stoch{int(hp.stochastic_rounding)}"
    )
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] Esperimento: {exp_dir}")

    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    rows: List[dict] = []
    log_path = exp_dir / "log.jsonl"
    with open(log_path, "w") as flog:
        bits_iter = hp.bits_list
        if hp.show_bits_bar:
            bits_iter = tqdm(bits_iter, desc="bitwidth grid", dynamic_ncols=True)
        for bits in bits_iter:
            seed_iter = range(hp.n_seeds)
            if hp.show_seed_bar:
                seed_iter = tqdm(seed_iter, desc="seeds", leave=False, dynamic_ncols=True)
            for s in seed_iter:
                seed = hp.seed_base + s
                row = run_one_seed_one_bits(hp, int(seed), int(bits))
                rows.append(row)
                flog.write(json.dumps(row) + "\n")
                tqdm.write(
                    f"[bits={bits} seed={seed}] enc={row['enc_time_per_client_round']*1e3:.3f} ms, "
                    f"pipe={row['pipeline_time_per_client_round']*1e3:.3f} ms, payload={row['comm_bits_per_round']/8/1e6:.3f} MB"
                )

    aggregate_and_plot(hp, rows, exp_dir)


if __name__ == "__main__":
    main()

