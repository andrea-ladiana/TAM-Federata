# -*- coding: utf-8 -*-
"""
Runner per exp_01 in modalità SINGLE (multi-round, multi-seed).

Pipeline per seed
-----------------
1) Genera archetipi veri ξ_true (K, N).
2) Costruisci subset per client (coverage parziale, non disgiunto).
3) Genera ETA e labels in SINGLE (L, T, M_c, N).
4) Per ogni round t:
   - Estima J_unsup su ETA[:, t, :, :]
   - Blend con memoria ebraica J(ξ_prev) (se t>0) con peso w
   - Propagation pseudo-inversa J -> J_KS
   - Cut spettrale (τ) e disentangling (TAM) → ξ_r^(t), magnetizzazioni
   - Metriche: FRO(J_KS,J*), retrieval (Hungarian), K_eff (shuffle/MP), coverage(t)
   - Aggiorna memoria ξ_prev per round t+1
5) Restituisce serie per-round e oggetti finali (J_server, ξ_ref).

Opzionali
---------
- Salvataggi JSON/CSV/PNG se `out_dir` è fornita.
- Hopfield post-hoc (eval_retrieval_vs_exposure) come step separato (non abilitato di default).

Compatibilità
-------------
Dipende dai moduli:
  - src.unsup.config       : HyperParams
  - src.unsup.data         : make_client_subsets, gen_dataset_partial_archetypes, new_round_single, compute_round_coverage, count_exposures
  - src.unsup.estimators   : build_unsup_J_single, blend_with_memory
  - src.unsup.spectrum     : eigen_cut, estimate_keff
  - src.unsup.metrics      : frobenius_relative, retrieval_mean_hungarian
  - src.unsup.functions    : propagate_J, JK_real
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- imports locali ---
from src.unsup.config import HyperParams
from src.unsup.data import (
    make_client_subsets,
    gen_dataset_partial_archetypes,
    new_round_single,
    compute_round_coverage,
    count_exposures,
)
from src.unsup.estimators import build_unsup_J_single, blend_with_memory
from src.unsup.spectrum import eigen_cut as spectral_cut, estimate_keff
from src.unsup.metrics import frobenius_relative, retrieval_mean_hungarian
from src.unsup.dynamics import dis_check
from src.unsup.functions import gen_patterns, propagate_J, JK_real


__all__ = ["run_exp01_single"]


@dataclass
class RoundLog:
    retrieval: float
    fro: float
    keff: int
    coverage: float


@dataclass
class SeedResult:
    seed: int
    series: List[RoundLog]
    J_server_final: np.ndarray
    xi_ref_final: np.ndarray
    exposure_counts: np.ndarray


def _ensure_outdir(out_dir: Optional[str]) -> Optional[Path]:
    if out_dir is None:
        return None
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _save_hyperparams(path: Path, hp: HyperParams) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(hp.__dict__, f, ensure_ascii=False, indent=2, default=lambda o: o.__dict__)


def _plot_series(path_png: Path, x: np.ndarray, r: np.ndarray, f: np.ndarray, k: np.ndarray, c: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(x, r, marker="o"); plt.title("Retrieval (mean)")
    plt.subplot(2, 2, 2)
    plt.plot(x, f, marker="o"); plt.title("Frobenius (relative)")
    plt.subplot(2, 2, 3)
    plt.plot(x, k, marker="o"); plt.title("K_eff")
    plt.subplot(2, 2, 4)
    plt.plot(x, c, marker="o"); plt.title("Coverage")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def run_exp01_single(
    hp: HyperParams,
    seeds: Optional[List[int]] = None,
    out_dir: Optional[str] = None,
    do_plot: bool = True,
) -> Dict[str, object]:
    """
    Esegue exp_01 in modalità SINGLE su una lista di seed.

    Returns
    -------
    dict con chiavi:
      - 'hp'           : hyperparams (dict)
      - 'per_seed'     : lista SeedResult (serializzabile in parte)
      - 'aggregate'    : medie per round delle serie (retrieval, fro, keff, coverage)
      - 'final_J_list' : lista di J_server_final per seed
      - 'final_xi_list': lista di xi_ref_final per seed
      - 'exposure_list': lista di exposure_counts per seed
    """
    if hp.mode != "single":
        raise ValueError("Runner bloccato alla modalità 'single'.")

    out_path = _ensure_outdir(out_dir)

    # lista seeds
    if seeds is None:
        seeds = [hp.seed_base + i for i in range(hp.n_seeds)]

    per_seed: List[SeedResult] = []

    # loop seeds
    for seed in seeds:
        rng = np.random.default_rng(seed)

        # 1) archetipi veri (K,N) — NB: functions.gen_patterns(N, P)
        xi_true = np.asarray(gen_patterns(hp.N, hp.K), dtype=np.int32)
        # 2) ideal J* (pseudo-inversa)
        J_star = np.asarray(JK_real(xi_true), dtype=np.float32)

        # 3) subset per client
        if hp.K_per_client is None:
            raise ValueError("K_per_client must be specified for partial coverage datasets.")
        subsets = make_client_subsets(K=hp.K, L=hp.L, K_per_client=hp.K_per_client, rng=rng)

        # 4) dataset SINGLE
        ETA, labels = gen_dataset_partial_archetypes(
            xi_true=xi_true,
            M_total=hp.M_total,
            r_ex=hp.r_ex,
            n_batch=hp.n_batch,
            L=hp.L,
            client_subsets=subsets,
            rng=rng,
            use_tqdm=hp.use_tqdm,
        )

        # 5) per-round
        series: List[RoundLog] = []
        xi_ref: Optional[np.ndarray] = None
        J_server_final: Optional[np.ndarray] = None

        for t in range(hp.n_batch):
            # Dati round t
            ETA_t = new_round_single(ETA, t)            # (L, M_c, N)
            labels_t = labels[:, t, :]                  # (L, M_c)

            # Stima J unsup & blending memoria
            J_unsup, M_eff = build_unsup_J_single(ETA_t, K=hp.K)  # (N,N), int
            J_rec = blend_with_memory(J_unsup, xi_prev=xi_ref, w=hp.w)

            # Propagation pseudo-inversa (iterazioni da hp.prop.iters)
            J_KS = np.asarray(propagate_J(J_rec, J_real=-1, verbose=False, iters=hp.prop.iters), dtype=np.float32)

            # Cut spettrale & Keff (coerente con SINGLE ⇒ MP usa M_eff del round)
            V, _k_from_cut, *_ = spectral_cut(J_KS, tau=hp.spec.tau)
            if hp.estimate_keff_method == "mp":
                K_eff, _, _ = estimate_keff(J_KS, method="mp", M_eff=M_eff)
            else:
                K_eff, _, _ = estimate_keff(J_KS, method="shuffle")

            # Disentangling (TAM) + magnetizzazioni
            xi_r, m_vec = dis_check(
                V=V,
                K=hp.K,
                L=hp.L,
                J_rec=J_rec,
                JKS_iter=J_KS,
                xi_true=xi_true,
                tam=hp.tam,
                spec=hp.spec,
                show_progress=hp.use_tqdm,
            )

            # Retrieval medio (matching ungherese) e coverage
            retr = retrieval_mean_hungarian(xi_r.astype(int), xi_true.astype(int))
            cov = compute_round_coverage(labels_t, K=hp.K)

            # FRO rispetto a J*
            fro = frobenius_relative(J_KS, J_star)

            series.append(RoundLog(retrieval=retr, fro=fro, keff=int(K_eff), coverage=float(cov)))

            # memoria per round successivo (prendi primi K candidati)
            if xi_r.shape[0] >= hp.K:
                xi_ref = xi_r[: hp.K].astype(int)
            else:
                xi_ref = xi_r.astype(int)

            # mantieni ultima J_KS
            J_server_final = J_KS

        assert J_server_final is not None and xi_ref is not None
        expo_counts = count_exposures(labels, K=hp.K)

        per_seed.append(
            SeedResult(
                seed=seed,
                series=series,
                J_server_final=J_server_final.astype(np.float32),
                xi_ref_final=xi_ref.astype(int),
                exposure_counts=expo_counts.astype(int),
            )
        )

    # --- aggregazione sui seed ---
    T = hp.n_batch
    arr_retr = np.zeros((len(per_seed), T), dtype=float)
    arr_fro = np.zeros((len(per_seed), T), dtype=float)
    arr_keff = np.zeros((len(per_seed), T), dtype=float)
    arr_cov = np.zeros((len(per_seed), T), dtype=float)
    for si, sr in enumerate(per_seed):
        for t, rl in enumerate(sr.series):
            arr_retr[si, t] = rl.retrieval
            arr_fro[si, t] = rl.fro
            arr_keff[si, t] = rl.keff
            arr_cov[si, t] = rl.coverage

    agg = {
        "retrieval_mean": arr_retr.mean(axis=0).tolist(),
        "fro_mean": arr_fro.mean(axis=0).tolist(),
        "keff_mean": arr_keff.mean(axis=0).tolist(),
        "coverage_mean": arr_cov.mean(axis=0).tolist(),
        "retrieval_se": (arr_retr.std(axis=0, ddof=1) / np.sqrt(len(per_seed))).tolist() if len(per_seed) > 1 else [0.0] * T,
        "fro_se": (arr_fro.std(axis=0, ddof=1) / np.sqrt(len(per_seed))).tolist() if len(per_seed) > 1 else [0.0] * T,
        "keff_se": (arr_keff.std(axis=0, ddof=1) / np.sqrt(len(per_seed))).tolist() if len(per_seed) > 1 else [0.0] * T,
        "coverage_se": (arr_cov.std(axis=0, ddof=1) / np.sqrt(len(per_seed))).tolist() if len(per_seed) > 1 else [0.0] * T,
    }

    # --- salvataggi opzionali ---
    if out_path is not None:
        # hyperparams
        _save_hyperparams(out_path / "hyperparams.json", hp)

        # log per seed (jsonl: una riga/seed)
        rows = []
        for sr in per_seed:
            rows.append({
                "seed": sr.seed,
                "series": [rl.__dict__ for rl in sr.series],
            })
        _save_jsonl(out_path / "log.jsonl", rows)

        # riassunto csv semplice
        try:
            import csv
            with (out_path / "results_table.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "retrieval_mean", "fro_mean", "keff_mean", "coverage_mean"])
                for t in range(T):
                    writer.writerow([
                        t,
                        agg["retrieval_mean"][t],
                        agg["fro_mean"][t],
                        agg["keff_mean"][t],
                        agg["coverage_mean"][t],
                    ])
        except Exception:
            pass

        # figura metriche
        if do_plot:
            x = np.arange(T)
            _plot_series(out_path / "fig_metrics.png", x, arr_retr.mean(0), arr_fro.mean(0), arr_keff.mean(0), arr_cov.mean(0))

    return {
        "hp": hp.__dict__,
        "per_seed": per_seed,
        "aggregate": agg,
        "final_J_list": [sr.J_server_final for sr in per_seed],
        "final_xi_list": [sr.xi_ref_final for sr in per_seed],
        "exposure_list": [sr.exposure_counts for sr in per_seed],
    }
