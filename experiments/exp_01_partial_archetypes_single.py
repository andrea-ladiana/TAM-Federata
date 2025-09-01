#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 01 (single) â€” Partial Archetype Coverage per Client
==============================================================
Versione in modalitÃ  SINGLE: tutte le metriche e la stima di J sono calcolate
sempre e solo sugli esempi del round corrente (no extend cumulativo).

Output (cartella dedicata sotto stress_tests/):
- hyperparams.json
- log.jsonl (una riga per seed con serie per-round)
- results_table.csv (riassunto per seed)
- fig_metrics.png (retrieval, fro, K_eff e coverage vs round)

Nota: file originale non toccato (exp_01_partial_archetypes.py rimane come baseline extend).
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import tensorflow as tf  # type: ignore
    tf.get_logger().setLevel('ERROR')
except Exception:
    tf = None  # noqa: N816

# Import locali di progetto
_THIS = Path(__file__).resolve()
# Robust project root detection: go up until we find a 'src/unsup' package directory.
ROOT = _THIS.parent  # start from experiments/
for _ in range(8):  # limit ascent to avoid infinite loops
    candidate = ROOT / 'src' / 'unsup' / '__init__.py'
    if candidate.exists():
        break
    if ROOT == ROOT.parent:
        break
    ROOT = ROOT.parent
else:
    pass  # loop exhausted (handled below)

SRC = ROOT / 'src'
if not (SRC / 'unsup' / '__init__.py').exists():
    raise RuntimeError(f"Impossibile trovare il package 'src/unsup' a partire da {__file__}. Verificare la struttura del progetto.")
# Per import "from src.unsup..." devo avere in sys.path la directory che CONTIENE 'src', cioè ROOT.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.unsup.functions import (  # type: ignore
    gen_patterns,
    JK_real,
    unsupervised_J,
    propagate_J,
    estimate_K_eff_from_J,
)
from src.unsup.dynamics import dis_check  # type: ignore


# ---------------------------------------------------------------------
# Hyperparams
# ---------------------------------------------------------------------
@dataclass
class HyperParams:
    L: int = 3
    K: int = 6
    N: int = 100
    n_batch: int = 10
    M_total: int = 200
    r_ex: float = 0.6
    K_per_client: int = 3
    updates: int = 60
    beta_T: float = 2.5
    lam: float = 0.2
    h_in: float = 0.1
    w: float = 0.8
    n_seeds: int = 6
    seed_base: int = 123
    pb_seeds: bool = True
    use_mp_keff: bool = True
    pb_dataset: bool = True  # progress bar per generazione dataset
    pb_rounds: bool = True   # progress bar per loop round per seed


# ---------------------------------------------------------------------
# Dataset generator: partial archetypes per client
# ---------------------------------------------------------------------
def make_client_subsets(K: int, L: int, K_per_client: int, rng: np.random.Generator) -> List[List[int]]:
    if K_per_client <= 0:
        raise ValueError("K_per_client deve essere > 0")
    if K_per_client > K:
        raise ValueError("K_per_client non puÃ² superare K")
    subsets: List[List[int]] = [[] for _ in range(L)]
    for i, mu in enumerate(rng.permutation(K)):
        subsets[i % L].append(int(mu))
    for l in range(L):
        need = K_per_client - len(subsets[l])
        pool = [mu for mu in range(K) if mu not in subsets[l]]
        if need > 0:
            choices = rng.choice(pool, size=need, replace=False)
            subsets[l].extend(int(x) for x in choices)
    return [sorted(set(s)) for s in subsets]


def gen_dataset_partial_archetypes(
    xi_true: np.ndarray,
    M_total: int,
    r_ex: float,
    n_batch: int,
    L: int,
    client_subsets: List[List[int]],
    rng: np.random.Generator,
    use_tqdm: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ritorna ETA (L, n_batch, M_c, N) e labels (L, n_batch, M_c)."""
    K, N = xi_true.shape
    M_c = math.ceil(M_total / (L * n_batch))
    ETA = np.zeros((L, n_batch, M_c, N), dtype=np.float32)
    labels = np.zeros((L, n_batch, M_c), dtype=np.int32)
    p_keep = 0.5 * (1.0 + r_ex)
    iter_L = range(L)
    if use_tqdm:
        iter_L = tqdm(iter_L, desc="dataset: client", leave=False)
    for l in iter_L:
        allowed = client_subsets[l]
        iter_T = range(n_batch)
        if use_tqdm:
            iter_T = tqdm(iter_T, desc=f"client {l} batches", leave=False)
        for t in iter_T:
            mus = rng.choice(allowed, size=M_c, replace=True)
            probs = rng.uniform(size=(M_c, N))
            chi = np.where(probs <= p_keep, 1.0, -1.0)
            xi_sel = xi_true[mus]
            ETA[l, t] = (chi * xi_sel).astype(np.float32)
            labels[l, t] = mus.astype(np.int32)
    return ETA, labels


# ---------------------------------------------------------------------
# Metriche/serie per seed (modalitÃ  single)
# ---------------------------------------------------------------------
def run_one_seed_single(hp: HyperParams, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    K, N, L = hp.K, hp.N, hp.L
    xi_true = gen_patterns(N, K)
    J_star = JK_real(xi_true)
    subsets = make_client_subsets(K, L, hp.K_per_client, rng)
    ETA, labels = gen_dataset_partial_archetypes(
        xi_true, hp.M_total, hp.r_ex, hp.n_batch, L, subsets, rng, use_tqdm=hp.pb_dataset
    )

    fro_series: List[float] = []
    m_series: List[float] = []
    keff_series: List[int] = []
    coverage_series: List[float] = []
    xi_ref = None
    K_eff_final = None

    round_iter = range(hp.n_batch)
    if hp.pb_rounds:
        round_iter = tqdm(round_iter, desc=f"rounds seed {seed}", leave=False)
    for t in round_iter:
        # Round corrente ONLY
        ETA_round = ETA[:, t, :, :]  # (L, M_c, N)
        L_loc, M_eff, N_loc = ETA_round.shape
        M_eff_param = max(1, M_eff // K)
        Js = [unsupervised_J(ETA_round[l], M_eff_param) for l in range(L_loc)]
        J_unsup = np.sum(Js, axis=0) / L_loc
        if t == 0:
            J_rec = J_unsup.copy()
        else:
            J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
            J_rec = hp.w * J_unsup + (1.0 - hp.w) * J_hebb_prev

        JKS_iter = propagate_J(J_rec, iters=1, verbose=False)
        vals, vecs = np.linalg.eig(JKS_iter)
        mask = (np.real(vals) > 0.5)
        autov = np.real(vecs[:, mask]).T
        xi_hat, Magn = dis_check(autov, K, L, J_rec, JKS_iter, ξ=xi_true, updates=hp.updates, show_bar=False)
        xi_ref = xi_hat

        fro_series.append(float(np.linalg.norm(JKS_iter - J_star, ord='fro') / (np.linalg.norm(J_star, ord='fro') + 1e-9)))
        m_series.append(float(np.mean(Magn)))
        if hp.use_mp_keff:
            try:
                K_eff_mp, _, _ = estimate_K_eff_from_J(JKS_iter, method='shuffle', M_eff=M_eff)
            except Exception:
                K_eff_mp = autov.shape[0]
        else:
            K_eff_mp = autov.shape[0]
        keff_series.append(int(K_eff_mp))
        K_eff_final = K_eff_mp

        # coverage nel solo round t (tutti i client)
        seen = set(int(mu) for l in range(L) for mu in labels[l, t])
        coverage_series.append(len(seen) / float(K))

    # First/Final retrieval con SINGLE
    ETA_first = ETA[:, 0, :, :]  # (L, M_c, N)
    L_loc, M_eff_f, N_loc = ETA_first.shape
    M_eff_param = max(1, M_eff_f // K)
    Js_f = [unsupervised_J(ETA_first[l], M_eff_param) for l in range(L_loc)]
    J_unsup_first = np.sum(Js_f, axis=0) / L_loc
    JKS_first = propagate_J(J_unsup_first, iters=1, verbose=False)
    vals0, vecs0 = np.linalg.eig(JKS_first)
    mask0 = (np.real(vals0) > 0.5)
    autov0 = np.real(vecs0[:, mask0]).T
    xi_first, Magn_first = dis_check(autov0, K, L, J_unsup_first, JKS_first, ξ=xi_true, updates=hp.updates, show_bar=False)

    def _match_and_overlap(estimated: np.ndarray, true: np.ndarray) -> float:
        from scipy.optimize import linear_sum_assignment
        K_hat, N = estimated.shape
        K_t, N2 = true.shape
        assert N == N2
        M = np.abs(estimated @ true.T / N)
        cost = 1.0 - M
        rI, cI = linear_sum_assignment(cost)
        overlaps = M[rI, cI]
        if K_hat < K_t:
            return float(overlaps.sum() / K_t)
        return float(overlaps.mean())

    m_first = _match_and_overlap(xi_first, xi_true)
    m_final = _match_and_overlap(xi_ref, xi_true) if xi_ref is not None else 0.0
    deltaK = abs(int(K_eff_final) - int(K)) if K_eff_final is not None else int(K)

    return dict(
        seed=seed,
        m_first=m_first,
        m_final=m_final,
        G_single=m_final - m_first,
        fro_final=fro_series[-1],
        deltaK=deltaK,
        series=dict(
            rounds=list(range(hp.n_batch)),
            m_single_mean=m_series,
            fro_single=fro_series,
            keff_single=keff_series,
            coverage_single=coverage_series,
        ),
    )


def aggregate_and_plot(hp: HyperParams, results: List[Dict[str, Any]], exp_dir: Path) -> None:
    rounds = np.arange(hp.n_batch)
    arr_m = np.array([r['series']['m_single_mean'] for r in results])
    arr_f = np.array([r['series']['fro_single'] for r in results])
    arr_k = np.array([r['series']['keff_single'] for r in results])
    arr_c = np.array([r['series']['coverage_single'] for r in results])

    def m_se(a):
        return a.mean(axis=0), a.std(axis=0, ddof=1) / max(1, math.sqrt(a.shape[0]))

    m_mean, m_sev = m_se(arr_m)
    f_mean, f_sev = m_se(arr_f)
    k_mean, k_sev = m_se(arr_k)
    c_mean, c_sev = m_se(arr_c)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)

    ax = axes[0]
    ax.plot(rounds, m_mean, label="retrieval (single)")
    ax.fill_between(rounds, m_mean - m_sev, m_mean + m_sev, alpha=0.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("round"); ax.set_ylabel("Mattis overlap")
    ax.set_title("Retrieval per round (coverage parziale, single)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(rounds, f_mean, label="Frobenius rel.")
    ax.fill_between(rounds, f_mean - f_sev, f_mean + f_sev, alpha=0.2)
    ax.set_xlabel("round"); ax.set_ylabel("||J-J*||_F / ||J*||_F")
    ax.set_title("Convergenza strutturale di J (single)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(rounds, k_mean, label="K_eff (MP)")
    ax.fill_between(rounds, k_mean - k_sev, k_mean + k_sev, alpha=0.15)
    ax2 = ax.twinx()
    ax2.plot(rounds, c_mean, linestyle='--', label="coverage archetipi", color='tab:orange')
    ax2.fill_between(rounds, c_mean - c_sev, c_mean + c_sev, color='tab:orange', alpha=0.15)
    ax.set_xlabel("round"); ax.set_ylabel("K_eff")
    ax2.set_ylabel("coverage (0..1)")
    ax.set_title("Rank efficace vs coverage (single)")
    ax.grid(True, alpha=0.3)

    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels + labels2, loc="lower right")

    fig_path = exp_dir / "fig_metrics.png"
    fig.savefig(fig_path, dpi=150)
    print(f"[plot] Figure salvata in: {fig_path}")
    plt.show()


def main():
    hp = HyperParams()

    base_dir = ROOT / "stress_tests" / "exp01_partial_archetypes_single"
    tag = f"K{hp.K}_N{hp.N}_L{hp.L}_nb{hp.n_batch}_M{hp.M_total}_r{hp.r_ex}_c{hp.K_per_client}_w{hp.w}"
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] Esperimento: {exp_dir}")

    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    results: List[Dict[str, Any]] = []
    log_path = exp_dir / "log.jsonl"
    with open(log_path, "w") as flog:
        seed_iter = range(hp.n_seeds)
        if hp.pb_seeds:
            seed_iter = tqdm(seed_iter, desc="seeds")
        for s in seed_iter:
            seed = hp.seed_base + s
            t0 = time.perf_counter()
            res = run_one_seed_single(hp, seed)
            t1 = time.perf_counter()
            results.append(res)
            row = {
                "mode": "single",
                "seed": res['seed'],
                "m_first": res['m_first'],
                "m_final": res['m_final'],
                "G_single": res['G_single'],
                "fro_final": res['fro_final'],
                "deltaK": res['deltaK'],
                "rounds": res['series']['rounds'],
                "m_series": res['series']['m_single_mean'],
                "fro_series": res['series']['fro_single'],
                "keff_series": res['series']['keff_single'],
                "coverage_series": res['series']['coverage_single'],
                "elapsed_s": t1 - t0,
            }
            flog.write(json.dumps(row) + "\n")
            try:
                tqdm.write(f"[seed {seed}] m_final={res['m_final']:.3f} G_single={res['G_single']:.3f} fro_final={res['fro_final']:.3f} Î”K={res['deltaK']}")
            except Exception:
                print(f"[seed {seed}] m_final={res['m_final']:.3f} G_single={res['G_single']:.3f} fro_final={res['fro_final']:.3f} Î”K={res['deltaK']}")

    # Plot aggregato
    aggregate_and_plot(hp, results, exp_dir)


if __name__ == "__main__":
    main()

