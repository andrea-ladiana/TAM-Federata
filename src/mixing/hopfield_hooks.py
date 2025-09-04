# -*- coding: utf-8 -*-
"""
Wrapper/utility per la valutazione di Hopfield round-wise nell'Exp-06 (single-only).

Obiettivo:
- Fornire una funzione semplice per valutare, salvare (e ricaricare) le magnetizzazioni
  corrette tramite la dinamica di Hopfield, riusando la primitiva
  `src.unsup.hopfield_eval.run_or_load_hopfield_eval(...)`.
- Aggiungere utility per lanciare/riprendere la valutazione su *tutti* i round
  di una run e per estrarre le matrici m_{μ}(t) in forma (K, T).

Nota: questo modulo NON scrive figure; si occupa solo di eseguire e raccogliere i risultati.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Tuple, List

import json
import os
import numpy as np

# Codebase (riuso)
from src.unsup.hopfield_eval import run_or_load_hopfield_eval  # noqa: F401

# Utilities locali
from .io import ensure_dir, read_json, list_round_dirs, find_files  # noqa: F401


# ---------------------------------------------------------------------
# API di base: valutazione Hopfield per un singolo round
# ---------------------------------------------------------------------
def eval_hopfield_for_round(
    round_dir: str | os.PathLike,
    *,
    J_server: np.ndarray,
    xi_true: np.ndarray,
    exposure_counts: Optional[np.ndarray] = None,
    beta: float = 3.0,
    updates: int = 30,
    reps_per_archetype: int = 32,
    start_overlap: float = 0.3,
    force_run: bool = True,
    save: bool = True,
    stochastic: bool = True,
) -> Dict[str, Any]:
    """
    Esegue (o ricarica) la valutazione Hopfield per un singolo round.

    Parametri
    ---------
    round_dir : cartella del round (es. ".../round_000/")
    J_server  : matrice sinaptica J_rec(t) del server per quel round
    xi_true   : archetipi veri binari (K,N)
    exposure_counts : (opzionale) contatori di esposizione per μ, per report
    beta, updates, reps_per_archetype, start_overlap, stochastic : parametri Hopfield
    force_run, save : pass-through al runner della codebase

    Returns
    -------
    results : dict (output del runner Hopfield) + meta (path, round_dir)
    """
    rdir = Path(round_dir)
    hop_dir = ensure_dir(rdir / "hopfield")
    results = run_or_load_hopfield_eval(
        output_dir=str(hop_dir),
        J_server=J_server,
        xi_true=xi_true,
        exposure_counts=exposure_counts,
        beta=float(beta),
        updates=int(updates),
        reps_per_archetype=int(reps_per_archetype),
        start_overlap=float(start_overlap),
        force_run=bool(force_run),
        save=bool(save),
        stochastic=bool(stochastic),
    )
    # Allego metadati minimi utili
    if isinstance(results, dict):
        results.setdefault("_meta", {})
        results["_meta"]["round_dir"] = str(rdir)
        results["_meta"]["hopfield_dir"] = str(hop_dir)
    return results


# ---------------------------------------------------------------------
# Scansione completa: lancia Hopfield su tutti i round esistenti
# ---------------------------------------------------------------------
def eval_hopfield_over_all_rounds(
    run_dir: str | os.PathLike,
    *,
    xi_true: np.ndarray,
    exposure_counts: Optional[np.ndarray] = None,
    beta: float = 3.0,
    updates: int = 30,
    reps_per_archetype: int = 32,
    start_overlap: float = 0.3,
    frequency: int = 1,
    force_run: bool = False,
    save: bool = True,
    stochastic: bool = True,
) -> List[Dict[str, Any]]:
    """
    Esegue/ripristina la valutazione Hopfield per tutti i round trovati in `run_dir`.

    Parametri
    ---------
    run_dir : cartella radice della run (contiene "round_XXX/")
    frequency : 1=ogni round, n>1=ogni n round (gli altri saltati)
    Il resto dei parametri viene passato a `eval_hopfield_for_round`.

    Returns
    -------
    results_list : lista di dict (uno per ogni round su cui è stata eseguita/ricaricata la valutazione)
    """
    rdir = Path(run_dir)
    round_dirs = list_round_dirs(rdir)
    results_list: List[Dict[str, Any]] = []

    for t, rd in enumerate(round_dirs):
        if frequency and (t % int(frequency) != 0):
            continue
        J_path = rd / "J_rec.npy"
        if not J_path.exists():
            # se il round non ha J_rec, salta
            continue
        J_rec = np.load(J_path)
        res = eval_hopfield_for_round(
            rd,
            J_server=J_rec,
            xi_true=xi_true,
            exposure_counts=exposure_counts,
            beta=beta,
            updates=updates,
            reps_per_archetype=reps_per_archetype,
            start_overlap=start_overlap,
            force_run=force_run,
            save=save,
            stochastic=stochastic,
        )
        results_list.append(res)
    return results_list


# ---------------------------------------------------------------------
# Estrazione m_{μ}(t) da file salvati
# ---------------------------------------------------------------------
def load_magnetization_matrix_from_run(
    run_dir: str | os.PathLike,
    *,
    key_candidates: Sequence[str] = ("magnetization_by_mu", "m_by_mu", "mag_by_mu"),
) -> Optional[np.ndarray]:
    """
    Legge i risultati Hopfield salvati in ciascun round e ricostruisce la matrice
    delle magnetizzazioni per archetipo nel tempo: M (K, T_sel).

    Cerca un file JSON nella cartella "round_XXX/hopfield/" che contenga almeno
    una delle chiavi in `key_candidates`.

    Returns
    -------
    M : ndarray (K, T_sel) oppure None se non trovati risultati coerenti.
    """
    rdir = Path(run_dir)
    round_dirs = list_round_dirs(rdir)
    # lista di array per round
    seq: List[np.ndarray] = []
    K_max = None

    for rd in round_dirs:
        hop_dir = rd / "hopfield"
        if not hop_dir.exists():
            continue
        jsons = find_files(hop_dir, pattern="*.json", recursive=False)
        found = None
        for jp in jsons:
            try:
                obj = read_json(jp)
            except Exception:
                continue
            for k in key_candidates:
                if k in obj:
                    arr = np.asarray(obj[k])
                    if arr.ndim == 1:
                        arr = arr[:, None]  # (K,) -> (K,1)
                    if arr.ndim == 2 and arr.shape[1] != 1:
                        # alcuni salvataggi possono avere (K, replicates) — prendi la media colonna
                        arr = arr.mean(axis=1, keepdims=True)
                    if K_max is None:
                        K_max = arr.shape[0]
                    elif arr.shape[0] != K_max:
                        # mismatch K: skip questo file
                        continue
                    seq.append(arr.astype(float))
                    found = True
                    break
            if found:
                break

        # Fallback: se non abbiamo trovato JSON con magnetizzazioni, prova il file NPZ
        if not found:
            npz_path = hop_dir / "magnetization_by_mu.npz"
            if npz_path.exists():
                try:
                    loaded = np.load(npz_path)
                    # chiavi attese: m_0, m_1, ...
                    m_keys = sorted([k for k in loaded.files if k.startswith("m_")], key=lambda s: int(s.split("_",1)[1]))
                    if m_keys:
                        cols = []
                        for mk in m_keys:
                            v = np.asarray(loaded[mk], dtype=float)
                            # v può essere (reps,) ⇒ media per ottenere valore round-wise
                            if v.ndim == 1:
                                cols.append(float(v.mean()))
                            elif v.ndim == 2:
                                # (reps, something) ⇒ media flatten
                                cols.append(float(v.reshape(-1).mean()))
                            else:
                                cols.append(float(np.mean(v)))
                        arr = np.asarray(cols, dtype=float)[:, None]  # (K,1)
                        if K_max is None:
                            K_max = arr.shape[0]
                        elif arr.shape[0] != K_max:
                            continue  # inconsistenza K => salta
                        seq.append(arr)
                except Exception:
                    pass

    if not seq:
        return None
    # concat column-wise per ottenere (K, T_sel)
    M = np.concatenate(seq, axis=1)  # (K, T_sel)
    return M


# ---------------------------------------------------------------------
# Backfill: costruisci e salva pi_hat_retrieval nei metrics.json esistenti
# ---------------------------------------------------------------------
def backfill_pi_hat_retrieval_over_run(
    run_dir: str | os.PathLike,
    *,
    ema_alpha: float = 0.0,
    reuse_previous_if_missing: bool = True,
) -> int:
    """
    Calcola ˆpi_t dal retrieval Hopfield per ogni round di una run e lo salva
    in 'round_XXX/metrics.json' come 'pi_hat_retrieval'. Restituisce il numero
    di round aggiornati. Se un round non ha risultati Hopfield, può riusare il
    valore precedente (se `reuse_previous_if_missing=True`).
    """
    rdir = Path(run_dir)
    rounds = list_round_dirs(rdir)
    updated = 0
    prev_vec = None
    for rd in rounds:
        hop_dir = rd / "hopfield"
        m_arr = None
        if hop_dir.exists():
            # 1) prova JSON
            jsons = find_files(hop_dir, pattern="*.json", recursive=False)
            for jp in jsons:
                try:
                    obj = read_json(jp)
                except Exception:
                    continue
                for k in ("magnetization_by_mu", "m_by_mu", "mag_by_mu"):
                    if k in obj:
                        try:
                            arr = np.asarray(obj[k])
                            m_arr = arr
                            break
                        except Exception:
                            m_arr = None
                if m_arr is not None:
                    break
            # 2) fallback NPZ salvato dal runner
            if m_arr is None:
                npz_path = hop_dir / "magnetization_by_mu.npz"
                if npz_path.exists():
                    try:
                        loaded = np.load(npz_path)
                        m_keys = sorted([k for k in loaded.files if k.startswith("m_")], key=lambda s: int(s.split("_",1)[1]))
                        if m_keys:
                            cols = []
                            for mk in m_keys:
                                v = np.asarray(loaded[mk], dtype=float)
                                if v.ndim == 1:
                                    cols.append(float(v.mean()))
                                else:
                                    cols.append(float(v.reshape(-1).mean()))
                            m_arr = np.asarray(cols, dtype=float)
                    except Exception:
                        m_arr = None
        pi_vec = None
        if m_arr is not None and m_arr.size > 0:
            if m_arr.ndim == 1:
                m_mu = m_arr.astype(float)
            else:
                axes = tuple(range(1, m_arr.ndim))
                m_mu = np.mean(m_arr, axis=axes).astype(float)
            eps = 1e-6
            wvec = np.maximum(m_mu, eps)
            den = float(wvec.sum()) if float(wvec.sum()) > 0 else 1.0
            pi_vec = (wvec / den)
            # EMA opzionale con il valore precedente (stabilizza)
            if prev_vec is not None and ema_alpha > 0.0:
                pi_vec = (1.0 - float(ema_alpha)) * prev_vec + float(ema_alpha) * pi_vec
                s = float(pi_vec.sum())
                if s > 0:
                    pi_vec = pi_vec / s
        elif reuse_previous_if_missing and prev_vec is not None:
            pi_vec = prev_vec

        # aggiorna metrics.json se disponibile
        mfile = rd / "metrics.json"
        if mfile.exists():
            try:
                data = read_json(mfile)
                if pi_vec is not None:
                    data["pi_hat_retrieval"] = [float(x) for x in pi_vec]
                    prev_vec = pi_vec
                else:
                    data.setdefault("pi_hat_retrieval", None)
                # mantieni alias data-driven se manca
                if "pi_hat" in data and "pi_hat_data" not in data:
                    data["pi_hat_data"] = list(data.get("pi_hat"))
                # scrivi
                (rd / "metrics.json").write_text(json.dumps(data, indent=2))
                updated += 1
            except Exception:
                continue
    return updated
