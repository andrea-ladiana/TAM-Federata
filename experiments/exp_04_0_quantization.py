#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 04 â€” Quantizzazione di J locali: valori finali vs bitwidth
===================================================================
Obiettivo: studiare l'effetto della **quantizzazione** delle matrici locali J_l (Hebb) prima
dell'aggregazione in **FL** (Federated Learning, apprendimento federato). Valutiamo le metriche
**finali** all'ultimo round in funzione del numero di bit.

Sigle espanse alla prima occorrenza:
- FL = Federated Learning (apprendimento federato)
- TAM = Tripartite Associative Memory
- SNR = Signal-to-Noise Ratio (rapporto segnale/rumore)
- "shuffle" = stima di K_eff tramite baseline a permutazione (no Marchenkoâ€“Pastur)

Output:
- hyperparams.json
- log.jsonl (una riga per seed e per bitwidth)
- results_table.csv (tabella comoda)
- fig_summary.png (Seaborn 2Ã—3 con bande SE)

Note richieste:
- uso estensivo di tqdm
- soppressione log TF
- K_eff via metodo "shuffle"
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
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Import moduli di progetto (file Ã¨ in: <UNSUP>/stress_tests/)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # <UNSUP>
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

    # Sweep dei bit di quantizzazione
    bits_list: List[int] = field(default_factory=list)  # verrÃ  settato nel main o in main()
    stochastic_rounding: bool = True
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
    seed_base: int = 92001

    # Plot
    palette: str = "deep"
    style: str = "whitegrid"
    # Progress bars
    show_bits_bar: bool = True
    show_seed_bar: bool = True
    show_round_bar: bool = True

# ---------------------------------------------------------------------
# Propagazione robusta (evita NaN/Inf)
# ---------------------------------------------------------------------
def safe_propagate(J: np.ndarray, *, iters: int = 1, max_steps: int = 300, clip: float = 1e3) -> np.ndarray:
    from unsup.functions import propagate_J  # import locale per evitare cicli
    if not np.all(np.isfinite(J)):
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
    J = np.clip(J, -clip, clip)
    # Normalizza se raggio spettrale eccessivo
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
# Dataset unsupervised (IID sugli archetipi)
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
# Quantizzazione simmetrica
# ---------------------------------------------------------------------

def quantize_symmetric(J: np.ndarray, bits: int, stochastic: bool, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """Quantizzazione simmetrica per-entry.
    - bits==1: quantizzazione di segno con scala alpha = mean(|J|)
    - bits>=2: livelli uniformi in [-Qmax, Qmax] con Qmax = 2^{bits-1}-1
    Ritorna (J_q, scale), dove 'scale' Ã¨ lo step (o alpha per bits=1).
    """
    if bits <= 0:
        raise ValueError("bits deve essere >=1")
    if bits == 1:
        alpha = float(np.mean(np.abs(J))) + 1e-12
        Jq = alpha * np.sign(J)
        return Jq.astype(np.float32), alpha
    # bits >= 2
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


def client_J_with_quant(E_l: np.ndarray, K: int, bits: int, stochastic: bool,
                        symmetrize: bool, zero_diag: bool,
                        rng: np.random.Generator) -> Tuple[np.ndarray, float, float]:
    """Costruisce J_l (Hebb) e applica quantizzazione simmetrica.
    Ritorna (J_l_q, scale, rel_err) con rel_err = ||J_q - J||_F / (||J||_F + eps).
    """
    N = E_l.shape[1]
    M_eff_param = max(1, E_l.shape[0] // K)
    J_l = unsupervised_J(E_l, M_eff_param)
    if symmetrize:
        J_l = 0.5 * (J_l + J_l.T)
    if zero_diag:
        np.fill_diagonal(J_l, 0.0)
    J_q, scale = quantize_symmetric(J_l, bits, stochastic, rng)
    rel_err = float(np.linalg.norm(J_q - J_l, ord='fro') / (np.linalg.norm(J_l, ord='fro') + 1e-12))
    return J_q, float(scale), rel_err


def aggregate_clients_J_quant(E: np.ndarray, K: int, bits: int, stochastic: bool,
                              symmetrize: bool, zero_diag: bool,
                              rng: np.random.Generator) -> Tuple[np.ndarray, float, float]:
    """Media semplice delle J_l quantizzate; ritorna J, scale medio e quantization error medio."""
    L, M_eff, N = E.shape
    Js = []
    scales = []
    rerrs = []
    for l in range(L):
        J_l_q, scale, rerr = client_J_with_quant(E[l], K, bits, stochastic, symmetrize, zero_diag, rng)
        Js.append(J_l_q)
        scales.append(scale)
        rerrs.append(rerr)
    J = np.mean(Js, axis=0)
    return J, float(np.mean(scales)), float(np.mean(rerrs))

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
# Core run (per seed e bitwidth)
# ---------------------------------------------------------------------

def run_one_seed_one_bits(hp: HyperParams, seed: int, bits: int) -> dict:
    rng = np.random.default_rng(seed)
    xi_true = gen_patterns(hp.N, hp.K)
    J_star = JK_real(xi_true)
    ETA = gen_dataset_unsup(xi_true, hp.M_total, hp.r_ex, hp.n_batch, hp.L, rng)

    xi_ref = None
    q_relerr_last = None
    round_iter = range(hp.n_batch)
    if hp.show_round_bar:
        round_iter = tqdm(round_iter, desc=f"rounds bits={bits} seed={seed}", leave=False, dynamic_ncols=True)
    for b in round_iter:
        ETA_extend = ETA[:, : b + 1, :, :].transpose(0, 2, 1, 3).reshape(hp.L, -1, hp.N)
        J_unsup_ext, q_scale, q_relerr = aggregate_clients_J_quant(
            ETA_extend, hp.K, bits, hp.stochastic_rounding, hp.symmetrize, hp.zero_diag, rng
        )
        q_relerr_last = q_relerr

        if b == 0 or xi_ref is None:
            J_rec_ext = J_unsup_ext.copy()
        else:
            J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
            J_rec_ext = hp.w * J_unsup_ext + (1.0 - hp.w) * J_hebb_prev

        JKS_ext = safe_propagate(J_rec_ext, iters=1, max_steps=300)
        if not np.all(np.isfinite(JKS_ext)):
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
    fro_pre = float(np.linalg.norm(J_rec_ext - J_star, ord='fro') / (np.linalg.norm(J_star, ord='fro') + 1e-9))
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
        "bits": int(bits),
        "seed": int(seed),
        "m_final": m_final,
        "fro_final_pre": fro_pre,
        "fro_final_post": fro_post,
        "K_eff_final": int(K_eff_final),
        "snr_final": float(snr_final),
        "rho_final": float(rho_final),
        "q_relerr_final": float(q_relerr_last if q_relerr_last is not None else np.nan),
    }

# ---------------------------------------------------------------------
# Aggregazione e grafico
# ---------------------------------------------------------------------

def aggregate_and_plot(hp: HyperParams, rows: List[dict], out_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.sort_values(["bits", "seed"], inplace=True)
    df.to_csv(out_dir / "results_table.csv", index=False)

    _allowed_styles = {"white", "dark", "whitegrid", "darkgrid", "ticks"}
    style_use = hp.style if hp.style in _allowed_styles else "whitegrid"
    sns.set_theme(style=style_use, palette=hp.palette)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)

    # 1) Retrieval finale vs bitwidth
    ax = axes[0, 0]
    sns.pointplot(data=df, x="bits", y="m_final", errorbar=("se", 1), dodge=True, ax=ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Retrieval finale vs bitwidth")
    ax.set_xlabel("bit di quantizzazione")
    ax.set_ylabel("Mattis overlap (finale)")
    ax.grid(True, alpha=0.3)

    # 2) Frobenius finale (pre/post)
    ax = axes[0, 1]
    df_f = df.melt(id_vars=["bits", "seed"], value_vars=["fro_final_pre", "fro_final_post"],
                   var_name="kind", value_name="fro")
    sns.pointplot(data=df_f, x="bits", y="fro", hue="kind", errorbar=("se", 1), dodge=True, ax=ax)
    ax.set_title("Frobenius: pre vs post-propagation")
    ax.set_xlabel("bit")
    ax.set_ylabel("||Jâˆ’J*||_F / ||J*||_F")
    ax.grid(True, alpha=0.3)
    ax.legend(title="")

    # 3) K_eff (shuffle) finale
    ax = axes[0, 2]
    sns.pointplot(data=df, x="bits", y="K_eff_final", errorbar=("se", 1), dodge=True, ax=ax)
    ax.axhline(hp.K, color="tab:green", linestyle=":", linewidth=1.2, label="K (vero)")
    ax.set_title("K_eff (shuffle) finale vs bitwidth")
    ax.set_xlabel("bit")
    ax.set_ylabel("K_eff")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # 4) SNR spettrale finale
    ax = axes[1, 0]
    sns.pointplot(data=df, x="bits", y="snr_final", errorbar=("se", 1), dodge=True, ax=ax)
    ax.set_title("SNR spettrale finale vs bitwidth")
    ax.set_xlabel("bit")
    ax.set_ylabel("SNR")
    ax.grid(True, alpha=0.3)

    # 5) Raggio spettrale finale
    ax = axes[1, 1]
    sns.pointplot(data=df, x="bits", y="rho_final", errorbar=("se", 1), dodge=True, ax=ax)
    ax.set_title("Raggio spettrale finale vs bitwidth")
    ax.set_xlabel("bit")
    ax.set_ylabel("rho(J)")
    ax.grid(True, alpha=0.3)

    # 6) Errore relativo di quantizzazione
    ax = axes[1, 2]
    sns.pointplot(data=df, x="bits", y="q_relerr_final", errorbar=("se", 1), dodge=True, ax=ax)
    ax.set_title("Errore relativo di quantizzazione vs bitwidth")
    ax.set_xlabel("bit")
    ax.set_ylabel("||J_qâˆ’J||_F / ||J||_F")
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
        bits_list=[1, 2, 3, 4, 8, 16, 32],
        stochastic_rounding=True, 
        symmetrize=True, 
        zero_diag=True,
        updates=40, 
        beta_T=2.5, 
        lam=0.2, 
        h_in=0.1,
        w=0.9, 
        n_seeds=3, 
        seed_base=92001,
    )

    base_dir = ROOT / "stress_tests" / "exp04_quantization"
    tag = (
        f"K{hp.K}_N{hp.N}_L{hp.L}_nb{hp.n_batch}_M{hp.M_total}_r{hp.r_ex}_"
        f"bits{min(hp.bits_list)}-{max(hp.bits_list)}_stoch{int(hp.stochastic_rounding)}_w{hp.w}"
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
                t0 = time.perf_counter()
                row = run_one_seed_one_bits(hp, int(seed), int(bits))
                t1 = time.perf_counter()
                row["elapsed_s"] = t1 - t0
                rows.append(row)
                flog.write(json.dumps(row) + "\n")
                tqdm.write(
                    f"[bits={bits} seed={seed}] m_final={row['m_final']:.3f} "
                    f"fro_pre={row['fro_final_pre']:.3f} fro_post={row['fro_final_post']:.3f} "
                    f"K_eff={row['K_eff_final']} SNR={row['snr_final']:.2f} qerr={row['q_relerr_final']:.2f}"
                )

    aggregate_and_plot(hp, rows, exp_dir)


if __name__ == "__main__":
    main()

