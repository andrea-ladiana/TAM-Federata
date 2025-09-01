#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 01 â€” Partial Archetype Coverage per Client
=====================================================
Scenario (a): ogni client vede solo un sottoinsieme di archetipi; l'unione sui client copre K.

Cosa fa questo script
---------------------
1) Genera K archetipi veri (Î¾) su N neuroni.
2) Associa ad ogni client l un sottoinsieme S_l di archetipi (|S_l| = K_per_client), con \bigcup_l S_l = {1..K}.
3) Genera un dataset unsupervised ETA con solo archetipi in S_l per ciascun client e round.
4) Esegue una dinamica federata in piÃ¹ round con blending J_rec = w*J_unsup + (1-w)*J_hebb_prev (Hebb su archetipi ricostruiti al round precedente, se disponibili), quindi proietta J con "propagate_J".
5) Disentangling tramite TAM e metriche:
   - m_series: retrieval medio (overlap di Mattis) per round (extend)
   - fro_series: Frobenius relativa ||J - J*||_F / ||J*||_F per round
   - K_eff_series: stima del rank efficace con metodo MP (Marchenkoâ€“Pastur) sulla proiezione propagata
   - coverage_series: frazione di archetipi realmente visti nel dataset (extend) fino al round t
6) Ripete per piÃ¹ seed, aggrega le serie (media e SE) e salva grafico Seaborn.
7) Scrive sottocartella dell'esperimento e della specifica scelta di iperparametri, con:
   - hyperparams.json
   - log.jsonl (una riga per seed + dati riassuntivi/serie)
   - fig_metrics.png

Dipendenze: numpy, matplotlib, seaborn, scipy, tqdm (opzionale), + i tuoi file Functions.py, Networks.py, Dynamics.py.
"""

from __future__ import annotations
import os
import sys
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Literal

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

# ---------------------------------------------------------------------
# Silenzia log TensorFlow (INFO / WARNING) se TensorFlow viene importato
# nei moduli locali. Deve essere prima di eventuali import TF.
# ---------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=ALL,1=INFO off,2=INFO+WARNING off,3=ERR off
try:  # configurazione post-import (se giÃ  caricato altrove non crasha)
    import tensorflow as tf  # type: ignore
    tf.get_logger().setLevel('ERROR')
    try:
        tf.autograph.set_verbosity(0)
    except Exception:
        pass
except Exception:
    pass
    


# ---------------------------------------------------------------------
# Import dei tuoi moduli locali
# ---------------------------------------------------------------------
# Assumiamo che questo file sia in: <ROOT>/stress_tests/exp_01_partial_archetypes.py
# <ROOT> Ã¨ la cartella che contiene Functions.py, Networks.py, Dynamics.py.
# Quindi saliamo solo di 1 livello (parent.parent rispetto al file Ã¨ la root UNSUP?).
ROOT = Path(__file__).resolve().parent.parent  # .../UNSUP
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
from unsup.dynamics import (
    new_round,      # lo riutilizziamo per coerenza strutturale del codice
    dis_check,
)

# ---------------------------------------------------------------------
# Utility dataclass per iperparametri e risultati
# ---------------------------------------------------------------------
@dataclass
class HyperParams:
    # Modello e dataset
    L: int = 3
    K: int = 6
    N: int = 500
    n_batch: int = 10
    M_total: int = 3000  # totale su tutti i client e round
    r_ex: float = 0.6
    K_per_client: int = 3  # |S_l| per ciascun client

    # Dinamica TAM
    updates: int = 60
    beta_T: float = 2.5
    lam: float = 0.2
    h_in: float = 0.1

    # Blending federato
    w: float = 0.9

    # Esperimento
    n_seeds: int = 6
    seed_base: int = 42

    # Visual
    palette: str = "deep"
    style: Literal['white', 'dark', 'whitegrid', 'darkgrid', 'ticks'] = "whitegrid"
    # Progress bars
    pb_seeds: bool = True
    pb_rounds: bool = True
    pb_dataset: bool = True
    pb_dynamics: bool = False  # passa show_bar a dis_check se True
    # Deduplica archetipi ricostruiti (overlap assoluto >= soglia considerato duplicato)
    dedup_thresh: float = 0.8

@dataclass
class RoundSeries:
    rounds: List[int]
    m_extend_mean: List[float]
    fro_extend: List[float]
    keff_extend: List[int]
    coverage_extend: List[float]

@dataclass
class SeedResult:
    seed: int
    m_first: float
    m_final: float
    G_ext: float
    fro_final: float
    deltaK: int
    series: RoundSeries

# ---------------------------------------------------------------------
# Generatore dataset con copertura parziale di archetipi per client
# ---------------------------------------------------------------------

def make_client_subsets(K: int, L: int, K_per_client: int, rng: np.random.Generator) -> List[List[int]]:
    """Crea liste S_l (indici archetipi) per ciascun client l, con copertura totale.
    Strategia: assegna prima una permutazione di {0..K-1} garantendo copertura, poi riempie a caso.
    """
    if K_per_client <= 0:
        raise ValueError("K_per_client deve essere > 0")
    if K_per_client > K:
        raise ValueError("K_per_client non puÃ² superare K")
    subsets: List[List[int]] = [[] for _ in range(L)]
    # Fase 1: garantisci copertura
    for i, mu in enumerate(rng.permutation(K)):
        subsets[i % L].append(int(mu))
    # Fase 2: completa fino a K_per_client
    for l in range(L):
        need = K_per_client - len(subsets[l])
        pool = [mu for mu in range(K) if mu not in subsets[l]]
        if need > 0:
            choices = rng.choice(pool, size=need, replace=False)
            subsets[l].extend(int(x) for x in choices)
    # Ordina e deduplica (difensivo)
    subsets = [sorted(set(s)) for s in subsets]
    # Verifica copertura
    covered = set(mu for s in subsets for mu in s)
    if len(covered) < K:
        raise RuntimeError("Copertura incompleta: aumentare K_per_client o rivedere logica")
    return subsets


def gen_dataset_partial_archetypes(
    xi_true: np.ndarray,
    M_total: int,
    r_ex: float,
    n_batch: int,
    L: int,
    client_subsets: List[List[int]],
    rng: np.random.Generator,
    show_bar: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera ETA parziale: ogni client l vede solo archetipi in client_subsets[l].

    Ritorna
    -------
    ETA : array (L, n_batch, M_c, N)  con M_c â‰ˆ ceil(M_total/(L*n_batch))
    labels : array (L, n_batch, M_c)   indice Î¼ usato per ciascun esempio (per coverage)
    """
    K, N = xi_true.shape
    M_c = math.ceil(M_total / (L * n_batch))
    ETA = np.zeros((L, n_batch, M_c, N), dtype=np.float32)
    labels = np.zeros((L, n_batch, M_c), dtype=np.int32)
    p_keep = 0.5 * (1.0 + r_ex)  # stessa logica del tuo gen_dataset_unsup
    iterable_clients = range(L)
    if show_bar:
        iterable_clients = tqdm(iterable_clients, desc="dataset clients", leave=False)
    for l in iterable_clients:
        allowed = client_subsets[l]
        inner_batches = range(n_batch)
        if show_bar:
            inner_batches = tqdm(inner_batches, desc=f"client {l} batches", leave=False)
        for b in inner_batches:
            mus = rng.choice(allowed, size=M_c, replace=True)
            labels[l, b] = mus
            # Costruisci M_c esempi scegliendo Î¼ e flippando bit indipendenti
            probs = rng.uniform(size=(M_c, N))
            chi = np.where(probs <= p_keep, 1.0, -1.0)
            xi_sel = xi_true[mus]  # (M_c, N)
            ETA[l, b] = (chi * xi_sel).astype(np.float32)
    return ETA, labels

# ---------------------------------------------------------------------
# Core run per un seed (usa blending federato e TAM)
# ---------------------------------------------------------------------

def _mean_unsup_J_per_layer(tensor_L_M_N: np.ndarray, K: int) -> Tuple[np.ndarray, int]:
    """Media J non supervisionate sui layer, coerente con Dynamics.py.
    Nota: M_eff_param = M_eff_actual // K per compatibilitÃ  con la normalizzazione del notebook.
    """
    L_loc, M_eff_actual, N_loc = tensor_L_M_N.shape
    M_eff_param = max(1, M_eff_actual // K)
    Js = [unsupervised_J(tensor_L_M_N[l], M_eff_param) for l in range(L_loc)]
    return np.sum(Js, axis=0) / L_loc, M_eff_param


def deduplicate_patterns(xi: np.ndarray, K_target: int, overlap_thresh: float) -> np.ndarray:
    """Deduplica (non-maximum suppression) i pattern in xi usando overlap di Mattis assoluto.

    Strategia greedy: scorri i pattern nell'ordine dato, tieni il primo, elimina quelli
    con overlap >= soglia rispetto a pattern giÃ  tenuti. Se al termine meno di K_target
    pattern restano, riempie con pattern rimossi massimizzando la dissimilaritÃ .
    """
    if xi is None or xi.size == 0:
        return xi
    xi_arr = np.asarray(xi)
    P, N = xi_arr.shape
    kept_idx: List[int] = []
    removed_idx: List[int] = []
    for i in range(P):
        v = xi_arr[i]
        if not kept_idx:
            kept_idx.append(i)
            continue
        overlaps = np.max(np.abs(v @ xi_arr[kept_idx].T) / N)
        if overlaps >= overlap_thresh:
            removed_idx.append(i)
        else:
            kept_idx.append(i)
    # Se troppi pattern, tronca
    if len(kept_idx) > K_target:
        kept_idx = kept_idx[:K_target]
    # Se troppo pochi, reinserisci alcuni rimossi meno ridondanti
    if len(kept_idx) < K_target:
        need = K_target - len(kept_idx)
        for _ in range(need):
            best_j = None
            best_score = 1e9
            for j in removed_idx:
                v = xi_arr[j]
                sim = np.max(np.abs(v @ xi_arr[kept_idx].T) / N)
                if sim < best_score:
                    best_score = sim
                    best_j = j
            if best_j is None:
                break
            kept_idx.append(best_j)
            removed_idx.remove(best_j)
            if len(kept_idx) >= K_target:
                break
    kept = xi_arr[kept_idx]
    return kept


def run_one_seed(hp: HyperParams, seed: int, *, out_dir: Path) -> SeedResult:
    rng = np.random.default_rng(seed)
    # 1) Ground truth
    xi_true = gen_patterns(hp.N, hp.K)  # shape (K, N)
    J_star = JK_real(xi_true)

    # 2) Subset per client e dataset parziale
    client_sets = make_client_subsets(hp.K, hp.L, hp.K_per_client, rng)
    ETA, labels = gen_dataset_partial_archetypes(
        xi_true, hp.M_total, hp.r_ex, hp.n_batch, hp.L, client_sets, rng, show_bar=hp.pb_dataset
    )

    # 3) Loop round con single/extend; manteniamo Î¾r_ref per blending Hebb
    xi_ref = None
    fro_extend_rounds: List[float] = []
    magn_extend_rounds: List[float] = []
    keff_mp_extend_rounds: List[int] = []
    coverage_rounds: List[float] = []

    seen = set()  # archetipi visti nell'extend fin qui

    round_iter = range(hp.n_batch)
    if hp.pb_rounds:
        round_iter = tqdm(round_iter, desc=f"seed {seed} rounds", leave=False)
    for b in round_iter:
        # Costruisci viste single/extend
        ETA_round = ETA[:, b, :, :]                         # (L, M_c, N)
        ETA_extend = ETA[:, : b + 1, :, :]                  # (L, b+1, M_c, N)
        # concatena i batch fino a b sull'asse esempi
        ETA_extend = ETA_extend.transpose(0, 2, 1, 3).reshape(hp.L, -1, hp.N)

        # Coverage reale degli archetipi visti fino a round b
        labs_b = labels[:, : b + 1, :].reshape(hp.L, -1)
        seen.update(int(mu) for mu in np.unique(labs_b))
        coverage_rounds.append(len(seen) / hp.K)

        # Stima J non supervisionate
        J_unsup_single, M_eff_round = _mean_unsup_J_per_layer(ETA_round, hp.K)
        J_unsup_extend, M_eff_extend = _mean_unsup_J_per_layer(ETA_extend, hp.K)

        # Blending
        if b == 0 or xi_ref is None:
            J_rec_extend = J_unsup_extend.copy()
        else:
            J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)  # Hebb su archetipi ricostruiti
            J_rec_extend = hp.w * J_unsup_extend + (1.0 - hp.w) * J_hebb_prev

        # Propagazione (pseudo-inversa iterativa) e disentangling TAM
        # Con max_steps=1 riduciamo il costo rispetto ai ~1000 micro-step default.
        JKS_iter_ext = propagate_J(J_rec_extend, iters=1, max_steps=1, verbose=False)
        vals_ext, vecs_ext = np.linalg.eig(JKS_iter_ext)
        mask_ext = (np.real(vals_ext) > 0.5)
        autov_ext = np.real(vecs_ext[:, mask_ext]).T

        if autov_ext.size == 0:
            order = np.argsort(np.real(vals_ext))[::-1]
            autov_ext = np.real(vecs_ext[:, order[: hp.K]]).T

        Net = TAM_Network()
        Net.prepare(J_rec_extend, hp.L)
        xi_r_ext, magn_ext = dis_check(
            autov_ext, hp.K, hp.L, J_rec_extend, JKS_iter_ext,
            xi=xi_true, updates=hp.updates, show_bar=hp.pb_dynamics
        )
        xi_ref = xi_r_ext

        fro_rel = float(np.linalg.norm(JKS_iter_ext - J_star, ord='fro') / (np.linalg.norm(J_star, ord='fro') + 1e-9))
        fro_extend_rounds.append(fro_rel)
        magn_extend_rounds.append(float(np.mean(magn_ext)))
        try:
            K_eff_mp_ext, _, _ = estimate_K_eff_from_J(JKS_iter_ext, method='shuffle', M_eff=M_eff_extend)
        except Exception:
            K_eff_mp_ext = int(mask_ext.sum())
        keff_mp_extend_rounds.append(int(K_eff_mp_ext))

    # Confronto first vs final
    # First (extend@b=0)
    ETA_first = ETA[:, :1, :, :].transpose(0, 2, 1, 3).reshape(hp.L, -1, hp.N)
    J_unsup_first, M_eff_first = _mean_unsup_J_per_layer(ETA_first, hp.K)
    # Propagazione iniziale con singolo micro-step
    JKS_first = propagate_J(J_unsup_first, iters=1, max_steps=1, verbose=False)
    vals0, vecs0 = np.linalg.eig(JKS_first)
    mask0 = (np.real(vals0) > 0.5)
    autov0 = np.real(vecs0[:, mask0]).T
    if autov0.size == 0:
        order0 = np.argsort(np.real(vals0))[::-1]
        autov0 = np.real(vecs0[:, order0[: hp.K]]).T
    xi_r_first, magn_first = dis_check(autov0, hp.K, hp.L, J_unsup_first, JKS_first,
                                       Î¾=xi_true, updates=hp.updates, show_bar=hp.pb_dynamics)

    # Overlap medio (Hungarian matching) su first/final
    def _match_and_overlap(estimated: np.ndarray, true: np.ndarray) -> float:
        from scipy.optimize import linear_sum_assignment
        K_hat, Nloc = estimated.shape
        Kt, _ = true.shape
        M = np.abs(estimated @ true.T / Nloc)
        cost = 1.0 - M
        rI, cI = linear_sum_assignment(cost)
        overlaps = M[rI, cI]
        if K_hat < Kt:
            return float(overlaps.sum() / Kt)
        return float(overlaps.mean())

    m_first = _match_and_overlap(xi_r_first, xi_true)
    assert xi_ref is not None, "xi_ref should have been set in at least one round"
    # Deduplica prima di calcolare overlap finale
    xi_ref_dedup = deduplicate_patterns(xi_ref, hp.K, hp.dedup_thresh)
    m_final = _match_and_overlap(xi_ref_dedup, xi_true)
    G_ext = m_final - m_first

    series = RoundSeries(
        rounds=list(range(hp.n_batch)),
        m_extend_mean=magn_extend_rounds,
        fro_extend=fro_extend_rounds,
        keff_extend=keff_mp_extend_rounds,
        coverage_extend=coverage_rounds,
    )
    deltaK = abs(int(keff_mp_extend_rounds[-1]) - hp.K)

    return SeedResult(
    seed=seed,
    m_first=m_first,
    m_final=m_final,
    G_ext=G_ext,
    fro_final=fro_extend_rounds[-1],
    deltaK=deltaK,
    series=series,
    )

# ---------------------------------------------------------------------
# Aggregazione, logging e grafico
# ---------------------------------------------------------------------

def aggregate_and_plot(hp: HyperParams, results: List[SeedResult], exp_dir: Path) -> None:
    # Aggrega serie per round (media e SE)
    rounds = results[0].series.rounds
    arr_m = np.array([r.series.m_extend_mean for r in results])
    arr_f = np.array([r.series.fro_extend for r in results])
    arr_k = np.array([r.series.keff_extend for r in results])
    arr_c = np.array([r.series.coverage_extend for r in results])

    def mean_se(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = x.mean(axis=0)
        se = x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
        return m, se

    m_mean, m_se = mean_se(arr_m)
    f_mean, f_se = mean_se(arr_f)
    k_mean, k_se = mean_se(arr_k)
    c_mean, c_se = mean_se(arr_c)

    sns.set_theme(style=hp.style, palette=hp.palette)
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)

    # 1) Retrieval vs round con banda SE
    ax = axes[0]
    ax.plot(rounds, m_mean, label="retrieval (extend)")
    ax.fill_between(rounds, m_mean - m_se, m_mean + m_se, alpha=0.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("round")
    ax.set_ylabel("Mattis overlap")
    ax.set_title("Retrieval per round (coverage parziale per client)")
    ax.grid(True, alpha=0.3)

    # 2) Frobenius relativa vs round
    ax = axes[1]
    ax.plot(rounds, f_mean, label="Frobenius rel.")
    ax.fill_between(rounds, f_mean - f_se, f_mean + f_se, alpha=0.2)
    ax.set_xlabel("round")
    ax.set_ylabel("||Jâˆ’J*||_F / ||J*||_F")
    ax.set_title("Convergenza strutturale di J")
    ax.grid(True, alpha=0.3)

    # 3) K_eff e Coverage vs round (doppio asse)
    ax = axes[2]
    ax.plot(rounds, k_mean, label="K_eff (MP)")
    ax.fill_between(rounds, k_mean - k_se, k_mean + k_se, alpha=0.15)
    ax2 = ax.twinx()
    ax2.plot(rounds, c_mean, linestyle='--', label="coverage archetipi", color='tab:orange')
    ax2.fill_between(rounds, c_mean - c_se, c_mean + c_se, color='tab:orange', alpha=0.15)
    ax.set_xlabel("round")
    ax.set_ylabel("K_eff")
    ax2.set_ylabel("coverage (0..1)")
    ax.set_title("Rank efficace vs coverage reale")
    ax.grid(True, alpha=0.3)

    # legende combinate
    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels + labels2, loc="lower right")

    # Salva e mostra
    fig_path = exp_dir / "fig_metrics.png"
    fig.savefig(fig_path, dpi=150)
    print(f"[plot] Figure salvata in: {fig_path}")
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    hp = HyperParams(
        L=3,
        K=6,
        N=100,
        n_batch=10,
        M_total=200,
        r_ex=0.6,
        K_per_client=3,
        updates=60,
        beta_T=2.5,
        lam=0.2,
        h_in=0.1,
        w=0.8,
        n_seeds=6,
        seed_base=123,
    dedup_thresh=0.8,
    )

    # Cartelle esperimento
    base_dir = ROOT / "stress_tests" / "exp01_partial_archetypes"
    tag = f"K{hp.K}_N{hp.N}_L{hp.L}_nb{hp.n_batch}_M{hp.M_total}_r{hp.r_ex}_c{hp.K_per_client}_w{hp.w}"
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] Esperimento: {exp_dir}")

    # Salva hyperparams
    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    # Esecuzione su piÃ¹ seed
    results: List[SeedResult] = []
    log_path = exp_dir / "log.jsonl"
    with open(log_path, "w") as flog:
        seed_iter = range(hp.n_seeds)
        if hp.pb_seeds:
            seed_iter = tqdm(seed_iter, desc="seeds")
        for s in seed_iter:
            seed = hp.seed_base + s
            t0 = time.perf_counter()
            res = run_one_seed(hp, seed, out_dir=exp_dir)
            t1 = time.perf_counter()
            results.append(res)
            # Logga una riga per seed con serie
            row = {
                "seed": res.seed,
                "m_first": res.m_first,
                "m_final": res.m_final,
                "G_ext": res.G_ext,
                "fro_final": res.fro_final,
                "deltaK": res.deltaK,
                "rounds": res.series.rounds,
                "m_series": res.series.m_extend_mean,
                "fro_series": res.series.fro_extend,
                "keff_series": res.series.keff_extend,
                "coverage_series": res.series.coverage_extend,
                "elapsed_s": t1 - t0,
            }
            flog.write(json.dumps(row) + "\n")
            try:
                tqdm.write(f"[seed {seed}] m_final={res.m_final:.3f} G_ext={res.G_ext:.3f} fro_final={res.fro_final:.3f} Î”K={res.deltaK}")
            except Exception:
                print(f"[seed {seed}] m_final={res.m_final:.3f} G_ext={res.G_ext:.3f} fro_final={res.fro_final:.3f} Î”K={res.deltaK}")

    # Grafico aggregato
    aggregate_and_plot(hp, results, exp_dir)


if __name__ == "__main__":
    main()

