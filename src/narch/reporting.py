# -*- coding: utf-8 -*-
"""
reporting.py — High-level reporting for Exp-07 (single-only).

Builds figures and a compact JSON summary from the round artefacts + the
series computed by novelty.compute_series_over_run().

Outputs (saved under outdir, default run_dir/"exp07_report"):
  • series.json  — all time series + metadata (K, K_old, t_detect, ...)
  • fig_timeseries.png — K_eff vs t, spectral gap at K_old, m_old/m_new
  • fig_pi_error.png   — TV(π, π̂) and L1(π, π̂) over t
  • fig_simplex.png    — 2D embedding of π̂(t) and (if available) π_true(t) for K=3

All plots are Matplotlib-only, no external deps. Safe to import without full project.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# project modules
try:
    from .metrics import simplex_embed_2d  # type: ignore
except Exception:  # fallback: regular 2-simplex embedding for K=3 or PCA for K!=3
    def simplex_embed_2d(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            p = p[None, :]
        T, K = p.shape
        if K == 3:
            V = np.array([[0.0, 0.0],
                          [1.0, 0.0],
                          [0.5, math.sqrt(3)/2.0]], dtype=float)
            return p @ V
        X = p - p.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        return X @ Vt[:2].T

from .novelty import compute_series_over_run, SeriesResult  # type: ignore


# ----------------------------
# I/O helpers
# ----------------------------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _save_json(p: str | Path, obj) -> None:
    Path(p).write_text(json.dumps(obj, indent=2))

# ----------------------------
# Plotting
# ----------------------------
def _style_ax(ax):
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

def plot_timeseries(series: SeriesResult, outpath: str | Path) -> None:
    T = int(series.T)
    x = np.arange(T)

    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    # (1) Keff vs t with K_old/K lines and t_detect marker
    ax = axs[0]; _style_ax(ax)
    sns.lineplot(x=x, y=series.keff, ax=ax, linewidth=2, label=r"$K_{\mathrm{eff}}(t)$")
    ax.axhline(series.K_old, color="k", lw=1.2, linestyle=":", label=r"$K_{\mathrm{old}}$")
    ax.axhline(series.K, color="k", lw=1.2, linestyle="--", alpha=0.6, label=r"$K$")
    if series.t_detect is not None:
        ax.axvline(series.t_detect, color="C3", lw=1.2, linestyle="--", label="t_detect")
    ax.set_ylabel("K_eff")
    ax.legend(loc="best")

    # (2) Relative spectral gap at the K_old boundary
    ax = axs[1]; _style_ax(ax)
    sns.lineplot(x=x, y=series.gap, ax=ax, linewidth=2, label="relative gap at K_old")
    if series.t_detect is not None:
        ax.axvline(series.t_detect, color="C3", lw=1.2, linestyle="--")
    ax.set_ylabel("(λ_Kold−λ_Kold+1)/|λ_Kold|")
    ax.legend(loc="best")

    # (3) Mean magnetisations on old vs new
    ax = axs[2]; _style_ax(ax)
    sns.lineplot(x=x, y=series.m_old, ax=ax, linewidth=2, label=r"$\overline{m}_{old}$")
    sns.lineplot(x=x, y=series.m_new, ax=ax, linewidth=2, label=r"$\overline{m}_{new}$")
    if series.t_detect is not None:
        ax.axvline(series.t_detect, color="C3", lw=1.2, linestyle="--")
    ax.set_xlabel("round t")
    ax.set_ylabel("mean Mattis overlap")
    ax.legend(loc="best")

    fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)

def plot_pi_error(series: SeriesResult, outpath: str | Path) -> None:
    x = np.arange(series.T)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    _style_ax(ax)
    ax.plot(x, series.TV, lw=2, label="TV(π, π̂)")
    ax.plot(x, series.L1, lw=1.5, linestyle="--", label="L1(π, π̂)")
    if series.t_detect is not None:
        ax.axvline(series.t_detect, color="C3", lw=1.2, linestyle="--")
    ax.set_xlabel("round t"); ax.set_ylabel("error")
    ax.legend(loc="best")
    fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)

def plot_simplex(series: SeriesResult, outpath: str | Path) -> None:
    if series.pi_hat is None or series.K != 3:
        return  # only for K=3
    P_hat = np.asarray(series.pi_hat, dtype=float)  # (T,3)
    XY_hat = simplex_embed_2d(P_hat)
    fig, ax = plt.subplots(figsize=(6, 6))
    _style_ax(ax)
    ax.plot(XY_hat[:, 0], XY_hat[:, 1], lw=2, label="π̂(t)")
    ax.scatter(XY_hat[0, 0], XY_hat[0, 1], s=40, zorder=3, label="start", marker="o")
    ax.scatter(XY_hat[-1, 0], XY_hat[-1, 1], s=40, zorder=3, label="end", marker="s")
    if series.pi_true is not None:
        XY_true = simplex_embed_2d(np.asarray(series.pi_true, dtype=float))
        ax.plot(XY_true[:, 0], XY_true[:, 1], lw=1.5, linestyle="--", label="π(t) true", alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Simplesso Δ₂ — traiettoria del mixing")
    ax.legend(loc="best")
    fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)

# ----------------------------
# Public API
# ----------------------------
def report_novelty_summary(
    run_dir: str | Path,
    *,
    K_old: int,
    outdir: Optional[str | Path] = None,
    hop_frequency: int = 1,
    hop_beta: float = 3.0,
    hop_updates: int = 30,
    hop_reps: int = 32,
    hop_start_overlap: float = 0.3,
) -> dict:
    """
    Build all Exp-07 figures + JSON series from a completed run directory.
    Magnetisations are computed from Hebb J(ξ_ref_aligned(t)).
    """
    from .novelty import HopfieldParams  # local import to avoid circulars
    run_dir = Path(run_dir)
    out = _ensure_dir(outdir or (run_dir / "exp07_report"))

    series = compute_series_over_run(
        run_dir,
        K_old=int(K_old),
        hop=HopfieldParams(
            beta=float(hop_beta),
            updates=int(hop_updates),
            reps_per_archetype=int(hop_reps),
            start_overlap=float(hop_start_overlap),
            stochastic=True,
            frequency=int(hop_frequency),
        ),
        detect_patience=2,
    )

    # Save JSON
    _save_json(out / "series.json", {
        **asdict(series),
        # convert arrays to lists for JSON
        "keff": series.keff.tolist(),
        "gap": series.gap.tolist(),
        "TV": series.TV.tolist(),
        "L1": series.L1.tolist(),
        "m_old": series.m_old.tolist(),
        "m_new": series.m_new.tolist(),
        "pi_hat": None if series.pi_hat is None else series.pi_hat.tolist(),
        "pi_true": None if series.pi_true is None else series.pi_true.tolist(),
        "eps": None if getattr(series, 'eps', None) is None else series.eps.tolist(),
        "bound_2eps": None if getattr(series, 'bound_2eps', None) is None else series.bound_2eps.tolist(),
    })

    # Figures
    plot_timeseries(series, out / "fig_timeseries.png")
    plot_pi_error(series, out / "fig_pi_error.png")
    plot_simplex(series, out / "fig_simplex.png")

    return {
        "outdir": str(out),
        "K": int(series.K),
        "K_old": int(series.K_old),
        "t_detect": None if series.t_detect is None else int(series.t_detect),
        "T": int(series.T),
        "figures": {
            "timeseries": str(out / "fig_timeseries.png"),
            "pi_error": str(out / "fig_pi_error.png"),
            "simplex": str(out / "fig_simplex.png"),
        },
        "series_json": str(out / "series.json"),
    }

# ----------------------------
# Robust variants (Seaborn-first and no-missing files)
# ----------------------------
def plot_pi_error_sns(series: SeriesResult, outpath: str | Path) -> None:
    sns.set_theme(style="whitegrid")
    x = np.arange(series.T)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    _style_ax(ax)
    sns.lineplot(x=x, y=series.TV, ax=ax, linewidth=2, label="TV(p, p^)")
    sns.lineplot(x=x, y=series.L1, ax=ax, linewidth=1.5, linestyle="--", label="L1(p, p^)")
    if getattr(series, 'bound_2eps', None) is not None:
        try:
            sns.lineplot(x=x, y=series.bound_2eps, ax=ax, linewidth=1.2, linestyle=":", label="2·ε_t bound")
        except Exception:
            pass
    if series.t_detect is not None:
        ax.axvline(series.t_detect, color="C3", lw=1.2, linestyle="--")
    ax.set_xlabel("round t"); ax.set_ylabel("error")
    ax.legend(loc="best")
    fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)

def plot_simplex_robust(series: SeriesResult, outpath: str | Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 6))
    _style_ax(ax)
    if series.pi_hat is None:
        ax.text(0.5, 0.5, "pi_hat not available", ha="center", va="center")
        ax.set_axis_off()
    else:
        P_hat = np.asarray(series.pi_hat, dtype=float)
        XY_hat = simplex_embed_2d(P_hat)
        sns.lineplot(x=XY_hat[:, 0], y=XY_hat[:, 1], ax=ax, linewidth=2, label="p^(t)")
        sns.scatterplot(x=[XY_hat[0, 0]], y=[XY_hat[0, 1]], ax=ax, s=40, label="start", marker="o")
        sns.scatterplot(x=[XY_hat[-1, 0]], y=[XY_hat[-1, 1]], ax=ax, s=40, label="end", marker="s")
        if series.pi_true is not None:
            XY_true = simplex_embed_2d(np.asarray(series.pi_true, dtype=float))
            sns.lineplot(x=XY_true[:, 0], y=XY_true[:, 1], ax=ax, linewidth=1.5, linestyle="--", label="p(t) true", alpha=0.8)
        ax.set_aspect("equal", adjustable="box")
        title = "Traiettoria sul simplesso Δ2" if series.K == 3 else "Traiettoria 2D (PCA) di π(t)"
        ax.set_title(title)
        ax.legend(loc="best")
    fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)

def report_novelty_summary_robust(
    run_dir: str | Path,
    *,
    K_old: int,
    outdir: Optional[str | Path] = None,
    hop_frequency: int = 1,
    hop_beta: float = 3.0,
    hop_updates: int = 30,
    hop_reps: int = 32,
    hop_start_overlap: float = 0.3,
) -> dict:
    from .novelty import HopfieldParams  # local import to avoid circulars
    run_dir = Path(run_dir)
    out = _ensure_dir(outdir or (run_dir / "exp07_report"))

    series = compute_series_over_run(
        run_dir,
        K_old=int(K_old),
        hop=HopfieldParams(
            beta=float(hop_beta),
            updates=int(hop_updates),
            reps_per_archetype=int(hop_reps),
            start_overlap=float(hop_start_overlap),
            stochastic=True,
            frequency=int(hop_frequency),
        ),
        detect_patience=2,
    )

    _save_json(out / "series.json", {
        "T": int(series.T),
        "K": int(series.K),
        "K_old": int(series.K_old),
        "t_detect": None if series.t_detect is None else int(series.t_detect),
        "keff": series.keff.tolist(),
        "gap": series.gap.tolist(),
        "TV": series.TV.tolist(),
        "L1": series.L1.tolist(),
        "m_old": series.m_old.tolist(),
        "m_new": series.m_new.tolist(),
        "pi_hat": None if series.pi_hat is None else series.pi_hat.tolist(),
        "pi_true": None if series.pi_true is None else series.pi_true.tolist(),
        "eps": None if getattr(series, 'eps', None) is None else series.eps.tolist(),
        "bound_2eps": None if getattr(series, 'bound_2eps', None) is None else series.bound_2eps.tolist(),
    })

    plot_timeseries(series, out / "fig_timeseries.png")
    plot_pi_error_sns(series, out / "fig_pi_error.png")
    plot_simplex_robust(series, out / "fig_simplex.png")

    return {
        "outdir": str(out),
        "K": int(series.K),
        "K_old": int(series.K_old),
        "t_detect": None if series.t_detect is None else int(series.t_detect),
        "T": int(series.T),
        "figures": {
            "timeseries": str(out / "fig_timeseries.png"),
            "pi_error": str(out / "fig_pi_error.png"),
            "simplex": str(out / "fig_simplex.png"),
        },
        "series_json": str(out / "series.json"),
    }
