# -*- coding: utf-8 -*-
"""
Valutazione post-hoc con rete di Hopfield (single-mode).

Dato J_server (matrice hebbiana finale del server) e i veri archetipi ξ_true,
simuliamo la dinamica di Hopfield partendo da stati iniziali fortemente corrotti
di ciascun archetipo e misuriamo la magnetizzazione finale. Questo consente di
testare l'ipotesi "archetipi più esposti ⇒ retrieval migliore".

API principali
--------------
- corrupt_like_archetype(...)    : genera σ0 con overlap iniziale controllato.
- run_hopfield_test(...)         : esegue la dinamica Hopfield su più repliche.
- eval_retrieval_vs_exposure(..) : aggrega per archetipo e correla con esposizione.

Compatibilità
-------------
Fa uso di `Hopfield_Network` definita in `src.unsup.networks`. Se desideri
iniettare J_server direttamente (senza passare da prepare(η)), è sufficiente:
    net = Hopfield_Network()
    net.N = J_server.shape[0]
    net.J = J_server
e poi chiamare `net.dynamics(...)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path
import json, time
import os

import numpy as np
# Import opzionali per il solo plotting: se mancanti non devono impedire
# l'uso delle API di valutazione (run_or_load_hopfield_eval, ecc.).
try:  # pragma: no cover - import facoltativo
    import seaborn as sns  # type: ignore
except Exception:  # noqa: E722
    sns = None  # type: ignore
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Import locale
from src.unsup.networks import Hopfield_Network


__all__ = [
    "corrupt_like_archetype",
    "run_hopfield_test",
    "eval_retrieval_vs_exposure",
    # nuove API di persistenza / plotting
    "run_or_load_hopfield_eval",
    "save_hopfield_eval",
    "load_hopfield_eval",
    "plot_magnetization_distribution",
    "plot_mean_vs_exposure",
]


def corrupt_like_archetype(
    xi_true: np.ndarray,
    reps_per_archetype: int,
    start_overlap: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera stati iniziali σ0 per Hopfield come versioni corrotte degli archetipi.

    Parametri
    ---------
    xi_true : (K, N) in {±1}
    reps_per_archetype : int
        Numero di repliche per ciascun archetipo.
    start_overlap : float in [0, 1]
        Overlap atteso iniziale con l'archetipo (0 = rumore puro, 1 = identico).
        Implementazione tramite flip indipendenti con P(flip) = (1 - start_overlap)/2.
    rng : np.random.Generator opzionale

    Returns
    -------
    σ0 : (K * reps_per_archetype, N) in {±1}
        Stati iniziali per la dinamica di Hopfield.
    targets : (K * reps_per_archetype,)
        Indici degli archetipi corrispondenti a ciascuno stato iniziale.
    """
    rng = np.random.default_rng() if rng is None else rng
    K, N = xi_true.shape
    p_flip = (1.0 - float(start_overlap)) * 0.5
    total = K * int(reps_per_archetype)
    σ0 = np.empty((total, N), dtype=int)
    targets = np.empty((total,), dtype=int)

    t = 0
    for μ in range(K):
        flips = rng.random(size=(reps_per_archetype, N)) < p_flip
        # flip bit ⇒ moltiplicare per -1
        σ0[t:t + reps_per_archetype] = np.where(flips, -xi_true[μ], xi_true[μ]).astype(int)
        targets[t:t + reps_per_archetype] = μ
        t += reps_per_archetype
    return σ0, targets


@dataclass
class HopfieldEvalResult:
    """Risultati aggregati della valutazione Hopfield."""
    magnetization_by_mu: Dict[int, np.ndarray]  # μ -> array (reps,)
    mean_by_mu: Dict[int, float]
    std_by_mu: Dict[int, float]
    overall_mean: float
    overall_std: float


def run_hopfield_test(
    J_server: np.ndarray,
    xi_true: np.ndarray,
    beta: float = 3.0,
    updates: int = 30,
    reps_per_archetype: int = 32,
    start_overlap: float = 0.3,
    rng: Optional[np.random.Generator] = None,
    stochastic: bool = True,
) -> HopfieldEvalResult:
    """
    Esegue la dinamica Hopfield su J_server con iniziali corrotti degli archetipi.

    Parametri
    ---------
    J_server : (N, N)
        Matrice sinaptica da valutare (quella finale del server).
    xi_true : (K, N) in {±1}
        Archetipi di riferimento per il calcolo delle magnetizzazioni.
    beta : float
        Inverse temperature per la dinamica (controlla la spinta del segnale).
    updates : int
        Numero di aggiornamenti paralleli.
    reps_per_archetype : int
        Repliche per ogni archetipo.
    start_overlap : float
        Overlap iniziale desiderato con l'archetipo (0..1).
    rng : np.random.Generator opzionale

    Returns
    -------
    HopfieldEvalResult
        Magnetizzazioni per archetipo e statistiche aggregate.
    """
    rng = np.random.default_rng() if rng is None else rng
    K, N = xi_true.shape
    # Stati iniziali e mapping verso il "bersaglio" μ
    σ0, targets = corrupt_like_archetype(xi_true, reps_per_archetype, start_overlap, rng=rng)

    # Prepara rete di Hopfield e inietta direttamente J_server
    net = Hopfield_Network()
    net.N = int(J_server.shape[0])
    net.J = np.asarray(J_server, dtype=np.float32)

    # Dinamica parallela
    net.dynamics(σ0.astype(np.float32), β=beta, updates=updates, mode="parallel", stochastic=stochastic)
    σf = np.asarray(net.σ, dtype=int)
    if σf is None:
        raise RuntimeError("Hopfield_Network.dynamics non ha prodotto stati finali.")

    # Magnetizzazione finale verso il rispettivo target
    # m = |<σf, ξ_target>|/N
    mag_by_mu: Dict[int, list] = {μ: [] for μ in range(K)}
    for i in range(σf.shape[0]):
        μ = int(targets[i])
        m = float(np.abs(np.dot(σf[i], xi_true[μ])) / N)
        mag_by_mu[μ].append(m)

    # Aggrega
    mag_by_mu_np: Dict[int, np.ndarray] = {μ: np.asarray(mag_by_mu[μ], dtype=float) for μ in range(K)}
    mean_by_mu = {μ: float(v.mean()) if v.size else 0.0 for μ, v in mag_by_mu_np.items()}
    std_by_mu = {μ: float(v.std(ddof=1)) if v.size > 1 else 0.0 for μ, v in mag_by_mu_np.items()}
    all_vals = np.concatenate([v for v in mag_by_mu_np.values() if v.size], axis=0) if any(
        v.size for v in mag_by_mu_np.values()
    ) else np.array([0.0])

    return HopfieldEvalResult(
        magnetization_by_mu=mag_by_mu_np,
        mean_by_mu=mean_by_mu,
        std_by_mu=std_by_mu,
        overall_mean=float(all_vals.mean()),
        overall_std=float(all_vals.std(ddof=1)) if all_vals.size > 1 else 0.0,
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size == 0:
        return float("nan")
    xc = x - x.mean(); yc = y - y.mean()
    num = float(np.dot(xc, yc))
    den = float(np.linalg.norm(xc) * np.linalg.norm(yc)) + 1e-12
    return num / den


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    # rank correlation simple implementation
    def _ranks(a: np.ndarray) -> np.ndarray:
        order = a.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, a.size + 1, dtype=float)
        return ranks
    xr = _ranks(np.asarray(x, dtype=float))
    yr = _ranks(np.asarray(y, dtype=float))
    return _pearson(xr, yr)


def eval_retrieval_vs_exposure(
    J_server: np.ndarray,
    xi_true: np.ndarray,
    exposure_counts: np.ndarray,
    beta: float = 3.0,
    updates: int = 30,
    reps_per_archetype: int = 32,
    start_overlap: float = 0.3,
    rng: Optional[np.random.Generator] = None,
    stochastic: bool = True,
) -> Dict[str, object]:
    """
    Esegue `run_hopfield_test` e correla la magnetizzazione media per archetipo
    con il numero di esposizioni (quante volte l'archetipo è apparso nei round).

    Returns
    -------
    dict con chiavi:
      - 'mean_by_mu'       : dict μ -> float
      - 'std_by_mu'        : dict μ -> float
      - 'pearson'          : float
      - 'spearman'         : float
      - 'overall_mean/std' : float, float
      - 'magnetization_by_mu' : μ -> np.ndarray (tutte le repliche)
    """
    res = run_hopfield_test(
        J_server=J_server,
        xi_true=xi_true,
        beta=beta,
        updates=updates,
        reps_per_archetype=reps_per_archetype,
        start_overlap=start_overlap,
        rng=rng,
        stochastic=stochastic,
    )
    K = xi_true.shape[0]
    expo = np.asarray(exposure_counts, dtype=float).reshape(K)
    means = np.array([res.mean_by_mu.get(μ, 0.0) for μ in range(K)], dtype=float)

    return {
        "mean_by_mu": res.mean_by_mu,
        "std_by_mu": res.std_by_mu,
        "overall_mean": res.overall_mean,
        "overall_std": res.overall_std,
        "pearson": _pearson(expo, means),
        "spearman": _spearman(expo, means),
        "magnetization_by_mu": res.magnetization_by_mu,
    }


# =========================
# Persistenza & Caching
# =========================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_hopfield_eval(
    output_dir: str | os.PathLike,
    eval_dict: Dict[str, Any],
    J_server: np.ndarray,
    xi_true: np.ndarray,
    sigma0: Optional[np.ndarray],
    sigmaf: Optional[np.ndarray],
    params: Dict[str, Any],
    exposure_counts: Optional[np.ndarray] = None,
    overwrite: bool = False,
) -> str:
    """Salva su disco tutti gli artefatti (metadati + matrici).

    Struttura cartella:
      meta.json
      J_server.npy
      xi_true.npy
      (exposure_counts.npy) opzionale
      sigma0.npy, sigmaf.npy (se presenti)
      magnetization_by_mu.npz  (chiavi m_0, m_1, ...)
      stats.json  (mean/std per μ, correlazioni, overall)
    """
    out = Path(output_dir)
    if out.exists() and not overwrite:
        # non sovrascrivere: lasciamo intatto
        return str(out)
    _ensure_dir(out)

    # Meta + parametri
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
        "shapes": {
            "J_server": list(J_server.shape),
            "xi_true": list(xi_true.shape),
            "sigma0": list(sigma0.shape) if sigma0 is not None else None,
            "sigmaf": list(sigmaf.shape) if sigmaf is not None else None,
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    np.save(out / "J_server.npy", np.asarray(J_server, dtype=np.float32))
    np.save(out / "xi_true.npy", xi_true.astype(np.int8))
    if exposure_counts is not None:
        np.save(out / "exposure_counts.npy", np.asarray(exposure_counts, dtype=np.int32))
    if sigma0 is not None:
        np.save(out / "sigma0.npy", sigma0.astype(np.int8))
    if sigmaf is not None:
        np.save(out / "sigmaf.npy", sigmaf.astype(np.int8))

    # Magnetizzazioni per μ
    mag_dict: Dict[int, np.ndarray] = eval_dict.get("magnetization_by_mu", {})  # type: ignore
    if mag_dict:
        _mag_payload = {f"m_{k}": np.asarray(v, dtype=np.float32) for k, v in mag_dict.items()}
        np.savez_compressed(str(out / "magnetization_by_mu.npz"), **_mag_payload)  # type: ignore[arg-type]

    stats_payload = {
        "mean_by_mu": eval_dict.get("mean_by_mu"),
        "std_by_mu": eval_dict.get("std_by_mu"),
        "overall_mean": eval_dict.get("overall_mean"),
        "overall_std": eval_dict.get("overall_std"),
        "pearson": eval_dict.get("pearson"),
        "spearman": eval_dict.get("spearman"),
        "exposure_counts": np.asarray(exposure_counts).tolist() if exposure_counts is not None else None,
    }
    (out / "stats.json").write_text(json.dumps(stats_payload, indent=2))
    return str(out)


def load_hopfield_eval(output_dir: str | os.PathLike) -> Dict[str, Any]:
    """Carica gli artefatti salvati in precedenza.

    Restituisce un dizionario con chiavi:
      J_server, xi_true, exposure_counts (se esiste), sigma0, sigmaf,
      eval (stats + magnetization_by_mu)
      meta
    """
    out = Path(output_dir)
    if not out.exists():
        raise FileNotFoundError(f"Directory non trovata: {out}")
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    def _maybe(name: str):
        p = out / name
        return np.load(p) if p.exists() else None
    J_server = _maybe("J_server.npy")
    xi_true = _maybe("xi_true.npy")
    exposure = _maybe("exposure_counts.npy")
    sigma0 = _maybe("sigma0.npy")
    sigmaf = _maybe("sigmaf.npy")
    mag_file = out / "magnetization_by_mu.npz"
    magnetization_by_mu = {}
    if mag_file.exists():
        loaded = np.load(mag_file)
        for k in loaded.files:
            if k.startswith("m_"):
                μ = int(k.split("_", 1)[1])
                magnetization_by_mu[μ] = loaded[k]
    stats = json.loads((out / "stats.json").read_text()) if (out / "stats.json").exists() else {}
    stats["magnetization_by_mu"] = magnetization_by_mu
    return {
        "meta": meta,
        "J_server": J_server,
        "xi_true": xi_true,
        "exposure_counts": exposure,
        "sigma0": sigma0,
        "sigmaf": sigmaf,
        "eval": stats,
    }


def run_or_load_hopfield_eval(
    output_dir: str,
    J_server: Optional[np.ndarray] = None,
    xi_true: Optional[np.ndarray] = None,
    exposure_counts: Optional[np.ndarray] = None,
    *,
    beta: float = 3.0,
    updates: int = 30,
    reps_per_archetype: int = 32,
    start_overlap: float = 0.3,
    force_run: bool = False,
    save: bool = True,
    rng: Optional[np.random.Generator] = None,
    stochastic: bool = True,
) -> Dict[str, Any]:
    """Esegue (o ricarica) la valutazione Hopfield + correlazioni esposizione.

    Se la cartella esiste e force_run=False la ricarica; altrimenti richiede
    J_server & xi_true per eseguire. Restituisce dizionario come `load_hopfield_eval`.
    """
    out = Path(output_dir)
    if out.exists() and not force_run:
        return load_hopfield_eval(out)
    if J_server is None or xi_true is None:
        raise ValueError("Servono J_server e xi_true per eseguire (force_run=True o cartella assente).")

    # Run principale
    rng = np.random.default_rng() if rng is None else rng
    K, N = xi_true.shape
    sigma0, targets = corrupt_like_archetype(xi_true, reps_per_archetype, start_overlap, rng=rng)
    net = Hopfield_Network()
    net.N = int(J_server.shape[0])
    net.J = np.asarray(J_server, dtype=np.float32)
    net.dynamics(sigma0.astype(np.float32), β=beta, updates=updates, mode="parallel", stochastic=stochastic)
    sigmaf = np.asarray(net.σ, dtype=int)
    if sigmaf is None:
        raise RuntimeError("Dinamica Hopfield fallita: sigma finale None")

    # Magnetizzazione
    mag_by_mu: Dict[int, List[float]] = {μ: [] for μ in range(K)}
    for i in range(sigmaf.shape[0]):
        μ = int(targets[i])
        m = float(np.abs(np.dot(sigmaf[i], xi_true[μ])) / N)
        mag_by_mu[μ].append(m)
    mag_np = {μ: np.asarray(v, dtype=float) for μ, v in mag_by_mu.items()}
    mean_by_mu = {μ: float(v.mean()) if v.size else 0.0 for μ, v in mag_np.items()}
    std_by_mu = {μ: float(v.std(ddof=1)) if v.size > 1 else 0.0 for μ, v in mag_np.items()}
    all_vals = np.concatenate([v for v in mag_np.values() if v.size], axis=0)

    exposure_counts_local = exposure_counts if exposure_counts is not None else np.ones(K)
    expo = np.asarray(exposure_counts_local, dtype=float).reshape(K)
    means = np.array([mean_by_mu.get(μ, 0.0) for μ in range(K)], dtype=float)
    eval_dict = {
        "mean_by_mu": mean_by_mu,
        "std_by_mu": std_by_mu,
        "overall_mean": float(all_vals.mean()),
        "overall_std": float(all_vals.std(ddof=1)) if all_vals.size > 1 else 0.0,
        "pearson": _pearson(expo, means),
        "spearman": _spearman(expo, means),
        "magnetization_by_mu": mag_np,
    }
    params = dict(
        beta=beta,
        updates=updates,
        reps_per_archetype=reps_per_archetype,
        start_overlap=start_overlap,
        stochastic=stochastic,
        force_run=force_run,
    )
    if save:
        save_hopfield_eval(out, eval_dict, J_server, xi_true, sigma0, sigmaf, params, exposure_counts)
    return load_hopfield_eval(out)


# =========================
# Plotting utilities (Seaborn)
# =========================

def _to_series(mag_by_mu: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for μ, arr in sorted(mag_by_mu.items()):
        xs.extend([μ] * len(arr))
        ys.extend(arr.tolist())
    return np.array(xs), np.array(ys)


def plot_magnetization_distribution(
    eval_artifacts: Dict[str, Any] | Dict[str, np.ndarray],
    ax: Optional[Axes] = None,
    palette: str = "viridis",
    jitter: float = 0.05,
    title: Optional[str] = None,
) -> Axes:
    """Box + swarm dei magnetizzazioni per archetipo.

    Parametri
    ---------
    eval_artifacts : dizionario che include 'magnetization_by_mu' (ad es. output['eval']).
    jitter : ampiezza jitter per scatter (se 0 non viene effettuato).
    """
    mag_by_mu_any = eval_artifacts.get("magnetization_by_mu", eval_artifacts.get("eval", {}).get("magnetization_by_mu"))  # type: ignore
    if mag_by_mu_any is None:
        raise ValueError("Nessun campo 'magnetization_by_mu' trovato.")
    # enforce dict[int, np.ndarray]
    if isinstance(mag_by_mu_any, dict):
        mag_by_mu: Dict[int, np.ndarray] = {int(k): np.asarray(v) for k, v in mag_by_mu_any.items()}
    else:
        raise TypeError("'magnetization_by_mu' deve essere un dict. File salvato corrotto?")
    xs, ys = _to_series(mag_by_mu)
    df = {"archetipo": xs, "mag": ys}
    import pandas as pd
    df = pd.DataFrame(df)
    if sns is None:  # fallback minimale senza seaborn
        if ax is None:
            _, ax_local = plt.subplots(figsize=(6, 4))
            ax = ax_local
        assert ax is not None
        for μ, arr in sorted(mag_by_mu.items()):
            y = np.asarray(arr, dtype=float)
            if y.size == 0:
                continue
            ax.scatter([μ]*y.size, y, s=12, alpha=0.6)
        ax.set_xlabel("Archetipo μ"); ax.set_ylabel("Magnetizzazione finale |m|")
        if title:
            ax.set_title(title)
        return ax
    if ax is None:
        _, ax_local = plt.subplots(figsize=(6, 4))
        ax = ax_local
    assert ax is not None
    # Violin + punti jitterati
    sns.violinplot(data=df, x="archetipo", y="mag", ax=ax, palette=palette, inner="quartile", cut=0)
    if jitter > 0:
        sns.stripplot(data=df, x="archetipo", y="mag", ax=ax, color="k", alpha=0.5, size=3, jitter=jitter)
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_ylabel("Magnetizzazione finale |m|")
    ax.set_xlabel("Archetipo μ")
    if title:
        ax.set_title(title)
    sns.despine(ax=ax)
    return ax


def plot_mean_vs_exposure(
    eval_artifacts: Dict[str, Any],
    exposure_counts: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    palette: str = "mako",
    annotate: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """Scatter (mean magnetization vs exposure) con regressione lineare."""
    means_dict = eval_artifacts.get("mean_by_mu", eval_artifacts.get("eval", {}).get("mean_by_mu"))  # type: ignore
    if means_dict is None:
        raise ValueError("Mancano 'mean_by_mu'.")
    means = np.array([means_dict[k] for k in sorted(means_dict.keys())], dtype=float)
    K = means.size
    if exposure_counts is None:
        # prova a recuperare
        exposure_counts = eval_artifacts.get("exposure_counts")
        if exposure_counts is None and "eval" in eval_artifacts:
            exposure_counts = eval_artifacts.get("eval", {}).get("exposure_counts")  # type: ignore
    if exposure_counts is None:
        exposure_counts = np.ones(K)
    expo = np.asarray(exposure_counts, dtype=float).reshape(K)
    corr_p = eval_artifacts.get("pearson", eval_artifacts.get("eval", {}).get("pearson"))
    corr_s = eval_artifacts.get("spearman", eval_artifacts.get("eval", {}).get("spearman"))
    if sns is None:  # fallback scatter semplice
        if ax is None:
            _, ax_local = plt.subplots(figsize=(5.5, 4))
            ax = ax_local
        assert ax is not None
        ax.scatter(expo, means, c="C0")
        ax.set_xlabel("Exposure"); ax.set_ylabel("Magnetizzazione media")
        if title:
            ax.set_title(title)
        return ax
    if ax is None:
        _, ax_local = plt.subplots(figsize=(5.5, 4))
        ax = ax_local
    assert ax is not None
    df = {"Exposure": expo, "MeanMag": means, "μ": list(range(K))}
    import pandas as pd
    df = pd.DataFrame(df)
    sns.regplot(data=df, x="Exposure", y="MeanMag", ax=ax, scatter=False, color="black", line_kws={"lw":1, "ls":"--"})
    sns.scatterplot(data=df, x="Exposure", y="MeanMag", hue="μ", ax=ax, palette=palette, s=60)
    ax.set_ylabel("Magnetizzazione media")
    if title:
        ax.set_title(title)
    if annotate:
        txt = f"Pearson={corr_p:.3f}\nSpearman={corr_s:.3f}" if corr_p is not None else ""
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"))
    sns.despine(ax=ax)
    return ax

