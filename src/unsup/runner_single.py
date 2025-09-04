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
import subprocess
import sys

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
    xi_true: Optional[np.ndarray] = None  # aggiunto per uso post-hoc (Hopfield)


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


def _plot_series(
    path_png: Path,
    x: np.ndarray,
    r_mean: np.ndarray,
    r_se: np.ndarray,
    f_mean: np.ndarray,
    f_se: np.ndarray,
    k_mean: np.ndarray,
    k_se: np.ndarray,
    c_mean: np.ndarray,
    c_se: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    # Try to use seaborn if available for nicer defaults
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        palette = sns.color_palette("tab10")
    except Exception:
        sns = None
        palette = None

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.errorbar(x, r_mean, yerr=r_se, marker='.', capsize=3, color=(palette[0] if palette is not None else None))
    ax.set_title("Retrieval (mean)", fontsize=12)

    ax = axes[0, 1]
    ax.errorbar(x, f_mean, yerr=f_se, marker='.', capsize=3, color=(palette[1] if palette is not None else None))
    ax.set_title("Frobenius (relative)", fontsize=12)

    ax = axes[1, 0]
    ax.errorbar(x, k_mean, yerr=k_se, marker='.', capsize=3, color=(palette[2] if palette is not None else None))
    # LaTeX-like label for K_eff
    ax.set_title(r"$K_{\mathrm{eff}}$", fontsize=13)

    ax = axes[1, 1]
    ax.errorbar(x, c_mean, yerr=c_se, marker='.', capsize=3, color=(palette[3] if palette is not None else None))
    ax.set_title("Coverage", fontsize=12)

    # set descriptive y-labels; show x-labels only on bottom row
    axes[0, 0].set_ylabel("retrieval")
    axes[0, 1].set_ylabel("Frobenius (relative)")
    axes[1, 0].set_ylabel(r"$K_{\mathrm{eff}}$")
    axes[1, 1].set_ylabel("coverage")
    for ax in axes[1, :]:
        ax.set_xlabel("round", fontsize=10)

    # ensure tick labels visible and set reasonable font sizes
    for ax in axes.ravel():
        ax.tick_params(axis='both', which='major', labelsize=9)

    # increase margins so y-labels are not cropped when saving
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.08, hspace=0.28, wspace=0.28)
    try:
        plt.savefig(path_png, dpi=150)
    finally:
        plt.close(fig)


def _load_existing(out_path: Path, hp: HyperParams, do_plot: bool) -> Optional[Dict[str, object]]:
    """Se presenti file salvati (hyperparams.json + log.jsonl) ricostruisce le metriche.

    Ritorna un dict nello stesso formato finale di `run_exp01_single` (eccetto che le liste
    final_J_list/final_xi_list/exposure_list saranno vuote se non disponibili). Se i file
    non esistono restituisce None.
    """
    hp_json = out_path / "hyperparams.json"
    log_jsonl = out_path / "log.jsonl"
    if not (hp_json.exists() and log_jsonl.exists()):
        return None
    try:
        saved_hp = json.loads(hp_json.read_text(encoding="utf-8"))
        # Verifica coerenza parametri chiave; se differiscono, non riusare.
        key_check = ["L", "K", "N", "n_batch", "n_seeds"]
        for k in key_check:
            if k in saved_hp and getattr(hp, k) != saved_hp[k]:
                # Incompatibile → non riuso (nuova configurazione)
                return None
        per_seed_rows = []
        with log_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                per_seed_rows.append(row)
        if not per_seed_rows:
            return None
        T = len(per_seed_rows[0]["series"])
        n_seeds = len(per_seed_rows)
        arr_retr = np.zeros((n_seeds, T)); arr_fro = np.zeros((n_seeds, T))
        arr_keff = np.zeros((n_seeds, T)); arr_cov = np.zeros((n_seeds, T))
        per_seed: List[SeedResult] = []
        for si, row in enumerate(per_seed_rows):
            series_logs: List[RoundLog] = []
            for t, s in enumerate(row["series"]):
                arr_retr[si, t] = s["retrieval"]
                arr_fro[si, t] = s["fro"]
                arr_keff[si, t] = s["keff"]
                arr_cov[si, t] = s["coverage"]
                series_logs.append(RoundLog(
                    retrieval=float(s["retrieval"]),
                    fro=float(s["fro"]),
                    keff=int(s["keff"]),
                    coverage=float(s["coverage"]),
                ))
            per_seed.append(SeedResult(
                seed=int(row.get("seed", -1)),
                series=series_logs,
                J_server_final=np.empty((0,0), dtype=np.float32),  # non salvato
                xi_ref_final=np.empty((0,0), dtype=int),            # non salvato
                exposure_counts=np.empty(0, dtype=int),             # non salvato
                xi_true=None,
            ))
        def se(a: np.ndarray) -> np.ndarray:
            return a.std(axis=0, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros(a.shape[1])
        agg = {
            "retrieval_mean": arr_retr.mean(0).tolist(),
            "fro_mean": arr_fro.mean(0).tolist(),
            "keff_mean": arr_keff.mean(0).tolist(),
            "coverage_mean": arr_cov.mean(0).tolist(),
            "retrieval_se": se(arr_retr).tolist(),
            "fro_se": se(arr_fro).tolist(),
            "keff_se": se(arr_keff).tolist(),
            "coverage_se": se(arr_cov).tolist(),
        }
        if do_plot:
            try:
                x = np.arange(T)
                img_path = out_path / "fig_metrics.png"
                _plot_series(
                    img_path,
                    x,
                    np.asarray(agg["retrieval_mean"]),
                    np.asarray(agg["retrieval_se"]),
                    np.asarray(agg["fro_mean"]),
                    np.asarray(agg["fro_se"]),
                    np.asarray(agg["keff_mean"]),
                    np.asarray(agg["keff_se"]),
                    np.asarray(agg["coverage_mean"]),
                    np.asarray(agg["coverage_se"]),
                )
                # try to open the generated image (Windows: os.startfile, else fallback)
                try:
                    if os.name == "nt":
                        os.startfile(str(img_path))
                    else:
                        opener = "open" if sys.platform == "darwin" else "xdg-open"
                        subprocess.run([opener, str(img_path)], check=False)
                except Exception:
                    pass
            except Exception:
                pass
        return {
            "hp": saved_hp,
            "per_seed": per_seed,
            "aggregate": agg,
            "final_J_list": [],
            "final_xi_list": [],
            "exposure_list": [],
        }
    except Exception:
        return None


def run_exp01_single(
    hp: HyperParams,
    seeds: Optional[List[int]] = None,
    out_dir: Optional[str] = None,
    do_plot: bool = True,
    reuse_saved: bool = True,
    force_run: bool = False,
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

    # Tentativo riuso risultati già salvati (se richiesto)
    if (out_path is not None) and reuse_saved and not force_run:
        reused = _load_existing(out_path, hp, do_plot=do_plot)
        if reused is not None:
            return reused

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
                xi_true=xi_true.astype(int),
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
            # pass mean and standard-error arrays so the plot shows error bars across seeds
            _plot_series(
                out_path / "fig_metrics.png",
                x,
                np.asarray(agg["retrieval_mean"]),
                np.asarray(agg["retrieval_se"]),
                np.asarray(agg["fro_mean"]),
                np.asarray(agg["fro_se"]),
                np.asarray(agg["keff_mean"]),
                np.asarray(agg["keff_se"]),
                np.asarray(agg["coverage_mean"]),
                np.asarray(agg["coverage_se"]),
            )

        # salva matrici finali J per seed e aggregate (npy/npz), oltre a xi_ref ed exposure se disponibili
        try:
            # salva ogni J finale come J_server_seed_<seed>.npy
            for sr in per_seed:
                try:
                    seed = int(sr.seed)
                    jpath = out_path / f"J_server_seed_{seed}.npy"
                    np.save(str(jpath), sr.J_server_final)
                except Exception:
                    # non bloccare il salvataggio globale per un errore su un seed
                    continue

            # salva array aggregato (n_seeds, N, N) in formato compresso
            try:
                allJ = np.stack([sr.J_server_final for sr in per_seed], axis=0)
                np.savez_compressed(str(out_path / "final_J_list.npz"), final_J=allJ)
            except Exception:
                pass

            # salva xi_ref_final per seed e aggregate
            for sr in per_seed:
                try:
                    np.save(str(out_path / f"xi_ref_seed_{int(sr.seed)}.npy"), sr.xi_ref_final)
                except Exception:
                    continue
            try:
                all_xi = [sr.xi_ref_final for sr in per_seed]
                np.savez_compressed(str(out_path / "final_xi_list.npz"), *all_xi)
            except Exception:
                pass

            # salva exposure counts (n_seeds, K)
            try:
                exp_arr = np.stack([sr.exposure_counts for sr in per_seed], axis=0)
                np.save(str(out_path / "exposure_list.npy"), exp_arr)
            except Exception:
                pass
        except Exception:
            # sicurezza: non interrompere l'esecuzione principale se il salvataggio fallisce
            pass

    return {
        "hp": hp.__dict__,
        "per_seed": per_seed,
        "aggregate": agg,
    "final_J_list": [sr.J_server_final for sr in per_seed],
    "final_xi_list": [sr.xi_ref_final for sr in per_seed],
    "exposure_list": [sr.exposure_counts for sr in per_seed],
    # nuova chiave per rendere disponibili gli archetipi veri
    "xi_true_list": [sr.xi_true for sr in per_seed],
    }
