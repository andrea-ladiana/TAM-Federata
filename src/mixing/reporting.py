# -*- coding: utf-8 -*-
"""
Raccolta e sintesi dei risultati per Exp-06 (single-only).

Funzioni principali:
- collect_round_metrics(run_dir)     : carica metrics.json per ciascun round
- collect_phase_metrics(pis, pi_hats): lag/ampiezza (K=3) tramite metrica complessa
- build_run_report(run_dir, ...)     : costruisce un report complessivo (JSON + CSV opzionali)
- dump_csv_*(...)                    : utilità per esportare tabelle CSV

Questo modulo NON disegna figure (grafici); per i plot è previsto un modulo dedicato.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# Utilities locali
from .io import read_json, write_json, list_round_dirs, ensure_dir  # noqa: F401
# Metriche (riuso)
from .metrics import (
    tv_distance,
    lag_and_amplitude,
    forgetting_index,
    equity_measures,
)
# Hopfield hooks per leggere la matrice M(K,T)
from .hopfield_hooks import load_magnetization_matrix_from_run  # noqa: F401


# ---------------------------------------------------------------------
# Caricamento metriche round-wise
# ---------------------------------------------------------------------
def collect_round_metrics(run_dir: str | Path) -> List[Dict[str, Any]]:
    """
    Legge i file 'metrics.json' in ciascun round_XXX e restituisce una lista ordinata per round.
    """
    rdir = Path(run_dir)
    rounds = list_round_dirs(rdir)
    items: List[Dict[str, Any]] = []
    for rd in rounds:
        mfile = rd / "metrics.json"
        if not mfile.exists():
            continue
        try:
            obj = read_json(mfile)
            obj["_round_dir"] = str(rd)
            items.append(obj)
        except Exception:
            continue
    return items


# ---------------------------------------------------------------------
# Fasi (lag/ampiezza) — solo per K=3 e sequenze complete
# ---------------------------------------------------------------------
def collect_phase_metrics_from_rounds(items: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """
    Estrae {pi_true_t} e {pi_hat_t} dagli items, allinea e calcola lag/ampiezza.
    Preferisce 'pi_hat_retrieval' se presente; altrimenti fallback su 'pi_hat' o 'pi_hat_data'.
    Ritorna None se mancano dati coerenti o K != 3.
    """
    if not items:
        return None
    pi_true_seq: List[np.ndarray] = []
    pi_hat_seq: List[np.ndarray] = []
    for it in items:
        if "pi_true" not in it:
            continue
        pt_raw = it.get("pi_true", None)
        ph_raw = it.get("pi_hat_retrieval", None)
        if ph_raw is None:
            ph_raw = it.get("pi_hat", None) or it.get("pi_hat_data", None)
        if pt_raw is None or ph_raw is None:
            continue
        pt = np.asarray(pt_raw, dtype=float)
        ph = np.asarray(ph_raw, dtype=float)
        if pt.shape != ph.shape:
            continue
        if pt.size != 3:
            continue  # supporto solo K=3
        # normalizza per robustezza
        pt = pt / (pt.sum() if pt.sum() > 0 else 1.0)
        ph = ph / (ph.sum() if ph.sum() > 0 else 1.0)
        pi_true_seq.append(pt)
        pi_hat_seq.append(ph)
    if not pi_true_seq or not pi_hat_seq:
        return None
    pi_true_matrix = np.stack(pi_true_seq, axis=0)  # (T_sel,3)
    pi_hat_matrix = np.stack(pi_hat_seq, axis=0)    # (T_sel,3)
    return lag_and_amplitude(pi_true_matrix, pi_hat_matrix)


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------
def dump_csv_round_metrics(items: List[Dict[str, Any]], path: str | Path) -> None:
    """
    Scrive un CSV con colonne standard per round: TV_pi, K_eff, pi_true, pi_hat, ecc.
    """
    if not items:
        return
    keys = ["TV_pi", "K_eff", "pi_true", "pi_hat", "M_eff", "n_eigs_sel", "coverage"]
    rows = []
    for t, it in enumerate(items):
        row = {
            "round": t,
            "TV_pi": it.get("TV_pi", np.nan),
            "K_eff": it.get("K_eff", np.nan),
            "M_eff": it.get("M_eff", np.nan),
            "n_eigs_sel": it.get("n_eigs_sel", np.nan),
            "coverage": it.get("coverage", np.nan),
            "pi_true": ";".join(map(lambda x: f"{x:.6f}", it.get("pi_true", []))) if it.get("pi_true") else "",
            "pi_hat": ";".join(map(lambda x: f"{x:.6f}", it.get("pi_hat", []))) if it.get("pi_hat") else "",
        }
        rows.append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def dump_csv_magnetization(M: np.ndarray, path: str | Path) -> None:
    """
    Salva la matrice di magnetizzazione M(K,T) in CSV, una riga per μ, colonne round.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    K, T = M.shape
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["mu"] + [f"t{t:03d}" for t in range(T)]
        writer.writerow(header)
        for mu in range(K):
            writer.writerow([mu] + [f"{float(M[mu, t]):.6f}" for t in range(T)])


# ---------------------------------------------------------------------
# Report complessivo per una run
# ---------------------------------------------------------------------
def build_run_report(
    run_dir: str | Path,
    *,
    write_json_report: bool = True,
    write_csv_round_metrics: bool = True,
    write_csv_magnetization: bool = True,
    exposure_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Costruisce un report finale a partire dagli artefatti della run:
      - round metrics (TV media/mediana, K_eff medio, ecc.)
      - lag/ampiezza (se K=3)
      - magnetizzazione Hopfield m_{μ}(t): equity e forgetting index
    Salva su disco i CSV e un JSON con il riepilogo.

    Returns
    -------
    report : dict con le statistiche principali e i path dei file scritti.
    """
    rdir = Path(run_dir)
    results_dir = ensure_dir(rdir / "results")

    # 1) Round metrics
    items = collect_round_metrics(rdir)
    tvs = [float(it.get("TV_pi", np.nan)) for it in items]
    keffs = [float(it.get("K_eff", np.nan)) for it in items]
    report: Dict[str, Any] = {
        "n_rounds_found": len(items),
        "TV_pi_mean": float(np.nanmean(tvs)) if tvs else np.nan,
        "TV_pi_median": float(np.nanmedian(tvs)) if tvs else np.nan,
        "K_eff_mean": float(np.nanmean(keffs)) if keffs else np.nan,
        "K_eff_median": float(np.nanmedian(keffs)) if keffs else np.nan,
    }

    # 1.b) w-series (se presente)
    try:
        w_path = rdir / "results" / "w_series.npy"
        if w_path.exists():
            W = np.load(w_path).astype(float).reshape(-1)
            if W.size > 0:
                report["w_mean"] = float(np.mean(W))
                report["w_median"] = float(np.median(W))
                report["w_std"] = float(np.std(W))
            else:
                report["w_mean"] = report["w_median"] = report["w_std"] = np.nan
        else:
            report["w_mean"] = report["w_median"] = report["w_std"] = np.nan
    except Exception:
        report["w_mean"] = report["w_median"] = report["w_std"] = np.nan

    # 1.c) correlazioni semplici
    try:
        if (rdir / "results" / "w_series.npy").exists() and items:
            W = np.load(rdir / "results" / "w_series.npy").astype(float).reshape(-1)
            n = min(len(items), W.size)
            if n > 1:
                Wn = W[:n]
                TVn = np.asarray([float(items[i].get("TV_pi", np.nan)) for i in range(n)], dtype=float)
                Lag = np.asarray([float(items[i].get("lag_abs_rad", np.nan)) for i in range(n)], dtype=float)
                # filtra NaN per correlazione
                def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
                    mask = np.isfinite(a) & np.isfinite(b)
                    if mask.sum() < 2:
                        return float('nan')
                    aa, bb = a[mask], b[mask]
                    if np.std(aa) <= 0 or np.std(bb) <= 0:
                        return float('nan')
                    return float(np.corrcoef(aa, bb)[0, 1])

                report["corr_w_TVpi"] = _safe_corr(Wn, TVn)
                report["corr_w_lagabs"] = _safe_corr(Wn, Lag)
            else:
                report["corr_w_TVpi"] = report["corr_w_lagabs"] = np.nan
        else:
            report["corr_w_TVpi"] = report["corr_w_lagabs"] = np.nan
    except Exception:
        report["corr_w_TVpi"] = report["corr_w_lagabs"] = np.nan

    # 2) Phase metrics (K=3)
    phase = collect_phase_metrics_from_rounds(items)
    if phase is not None:
        report.update({
            "lag_rounds": float(phase["lag_rounds"]),
            "lag_radians": float(phase["lag_radians"]),
            "amp_ratio": float(phase["amp_ratio"]),
        })

    # 3) Magnetizzazione Hopfield: M(K,T)
    M = load_magnetization_matrix_from_run(rdir)
    if M is not None:
        report["magnetization_shape"] = list(map(int, M.shape))
        # equity (var e gini medi nel tempo)
        eq = equity_measures(M)
        report["equity_variance_mean"] = float(eq["variance"])
        report["equity_gini_mean"] = float(eq["gini"])
        # forgetting index (per μ)
        FI = forgetting_index(M, exposure_mask=exposure_mask)
        report["forgetting_index"] = [float(x) for x in FI]
    else:
        report["magnetization_shape"] = None

    # Salvataggi
    if write_json_report:
        write_json(results_dir / "report.json", report)
    if write_csv_round_metrics and items:
        dump_csv_round_metrics(items, results_dir / "metrics_rounds.csv")
    if write_csv_magnetization and M is not None:
        dump_csv_magnetization(M, results_dir / "magnetization.csv")

    return report
