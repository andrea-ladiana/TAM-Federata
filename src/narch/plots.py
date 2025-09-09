
# -*- coding: utf-8 -*-
"""
plots_exp07.py — Extended plotting utilities for Exp-07 (single-only).

This module complements `reporting.py` by offering additional figures:
  A) K_eff(t) with K_old/K lines and t_detect marker (also in reporting.py)
  B) Relative spectral gap at boundary K_old (also in reporting.py)
  C) Retrieval: m_old(t) and m_new(t) (also in reporting.py)
  D) Mixing errors: TV(π,π̂), L1(π,π̂) (also in reporting.py)
  E) Simplex Δ₂ trajectory (K=3), with time color-coding
  F) Scree plots pre/post novelty with MP / shuffle thresholds if available
  G) Heatmap of per-class magnetisations m_μ(t) from Hopfield (Hebb synapses)
  H) Ablation helpers: t_detect vs scheduler params (scatter) and gap statistics

The functions are thin wrappers around artefacts saved by the runners, and the
time series computed via `novelty.compute_series_over_run`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Sequence, Tuple, Union

import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm, colors

# Project-local helpers
try:
    from .novelty import compute_series_over_run, SeriesResult, spectral_gap_at_boundary  # type: ignore
except Exception:
    # Allow out-of-package usage if placed next to novelty.py
    from novelty import compute_series_over_run, SeriesResult, spectral_gap_at_boundary  # type: ignore

try:
    from .metrics import simplex_embed_2d  # type: ignore
except Exception:
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

# -----------------------------------------------------------------------------
# Small I/O helpers
# -----------------------------------------------------------------------------
def _read_json(p: Union[str, Path]) -> Dict[str, Any]:
    return json.loads(Path(p).read_text())

def _try_load(path: Union[str, Path]) -> Optional[np.ndarray]:
    p = Path(path)
    if p.exists():
        try:
            return np.load(p, allow_pickle=False)
        except Exception:
            return None
    return None

def _list_round_dirs(run_dir: Union[str, Path]) -> List[Path]:
    rdir = Path(run_dir)
    ds = [p for p in rdir.iterdir() if p.is_dir() and p.name.startswith("round_")]
    def _key(p: Path) -> int:
        try:
            return int(p.name.split("_")[-1])
        except Exception:
            return 10**9
    return sorted(ds, key=_key)

def _style_ax(ax):
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

def _hebb_J(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)
    N = xi.shape[1]
    return (xi.T @ xi) / float(N)

def _hopfield_dynamics(J: np.ndarray, s0: np.ndarray, steps: int = 30) -> np.ndarray:
    s = np.array(s0, dtype=float)
    for _ in range(int(steps)):
        s = np.sign(J @ s)
        s[s == 0] = 1.0
    return s

# -----------------------------------------------------------------------------
# Panel E — Simplex Δ₂ with time coloring
# -----------------------------------------------------------------------------
def plot_simplex_timecolored(series: SeriesResult, outpath: Union[str, Path]) -> None:
    """
    Plot π̂(t) (and optionally π_true(t)) in the 2-simplex with a colormap along time.
    Only valid for K=3.
    """
    if series.pi_hat is None or series.K != 3:
        return
    P_hat = np.asarray(series.pi_hat, dtype=float)
    XY_hat = simplex_embed_2d(P_hat)  # (T,2)

    T = series.T
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 6))
    _style_ax(ax)
    # Color line segments by time
    norm = colors.Normalize(vmin=0, vmax=T-1)
    cmap = cm.get_cmap("viridis")
    for i in range(T-1):
        sns.lineplot(x=XY_hat[i:i+2, 0], y=XY_hat[i:i+2, 1], ax=ax, linewidth=2, color=cmap(norm(i)))
    sns.scatterplot(x=[XY_hat[0, 0]], y=[XY_hat[0, 1]], ax=ax, s=40, marker="o", label="start")
    sns.scatterplot(x=[XY_hat[-1, 0]], y=[XY_hat[-1, 1]], ax=ax, s=40, marker="s", label="end")
    if series.pi_true is not None:
        XY_true = simplex_embed_2d(np.asarray(series.pi_true, dtype=float))
        ax.plot(XY_true[:, 0], XY_true[:, 1], lw=1.2, linestyle="--", alpha=0.8, label="π(t) true")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Simplesso Δ₂ — traiettoria colorata nel tempo")
    ax.legend(loc="best")
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8)
    cb.set_label("round t")
    fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Panel F — Scree plots pre/post novelty
# -----------------------------------------------------------------------------
def plot_scree_pre_post(run_dir: Union[str, Path],
                        K_old: int,
                        outpath: Union[str, Path],
                        t_before: Optional[int] = None,
                        t_after: Optional[int] = None,
                        max_show: int = 50) -> None:
    """
    Read eigenvalues from metrics.json or eigs_sel.npy for two rounds:
      • t_before: a round before novelty (defaults to first round)
      • t_after : a round after novelty  (defaults to last round)
    and plot both normalized spectra on the same axes.
    """
    rounds = _list_round_dirs(run_dir)
    if not rounds:
        return
    t_before = 0 if t_before is None else int(t_before)
    t_after = len(rounds)-1 if t_after is None else int(t_after)
    t_before = max(0, min(t_before, len(rounds)-1))
    t_after = max(0, min(t_after, len(rounds)-1))

    def _load_eigs(rd: Path) -> Optional[np.ndarray]:
        # try metrics.json → keff_info.eigvals → eigs_sel.npy
        mj_path = rd / "metrics.json"
        if mj_path.exists():
            try:
                mj = _read_json(mj_path)
                ev = mj.get("keff_info", {}).get("eigvals", None)
                if isinstance(ev, list) and len(ev) > 0:
                    return np.asarray(ev, dtype=float)
            except Exception:
                pass
        evp = rd / "eigs_sel.npy"
        arr = _try_load(evp)
        return arr

    ev_b = _load_eigs(rounds[t_before])
    ev_a = _load_eigs(rounds[t_after])
    if ev_b is None or ev_a is None:
        return

    ev_b = -np.sort(-np.ravel(ev_b))
    ev_a = -np.sort(-np.ravel(ev_a))
    # limit to first max_show entries for readability
    yb = ev_b[:min(max_show, ev_b.size)] / (np.max(np.abs(ev_b)) + 1e-9)
    ya = ev_a[:min(max_show, ev_a.size)] / (np.max(np.abs(ev_a)) + 1e-9)

    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, y, title in zip(axs, [yb, ya], [f"pre (t={t_before})", f"post (t={t_after})"]):
        _style_ax(ax)
        sns.lineplot(x=np.arange(y.size), y=y, ax=ax, linewidth=2)
        if K_old > 0 and K_old - 1 < y.size:
            ax.axvline(K_old - 0.5, linestyle=":", lw=1.2)
        ax.set_xlabel("eigenvalue index (desc)")
        ax.set_title(title)
    axs[0].set_ylabel("normalized λ")
    fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Panel G — Heatmap m_μ(t)
# -----------------------------------------------------------------------------
def plot_magnetization_heatmap(run_dir: Union[str, Path],
                               outpath: Union[str, Path],
                               hopfield_subdir: str = "hopfield_hebb",
                               key_order: Sequence[str] = ("magnetization_by_mu", "m_by_mu", "mag_by_mu")) -> None:
    """
    Collect m_by_mu vectors from round_i/<hopfield_subdir>/report.json (or *.npz) and plot a K×T heatmap.
    """
    rounds = _list_round_dirs(run_dir)
    if not rounds:
        return

    m_list = []
    for rd in rounds:
        rep_json = rd / hopfield_subdir / "report.json"
        m_mu = None
        if rep_json.exists():
            try:
                js = _read_json(rep_json)
                for key in key_order:
                    if key in js:
                        m_mu = np.asarray(js[key], dtype=float).ravel()
                        break
            except Exception:
                m_mu = None
        if m_mu is None:
            # fallback: try NPZ with standard key
            rep_npz = rd / hopfield_subdir / "report.npz"
            if rep_npz.exists():
                try:
                    data = np.load(rep_npz)
                    for key in key_order:
                        if key in data:
                            m_mu = np.asarray(data[key], dtype=float).ravel()
                            break
                except Exception:
                    m_mu = None
        if m_mu is None:
            return  # give up if any round missing → ensures rectangular matrix
        m_list.append(m_mu)

    M = np.stack(m_list, axis=1)  # (K,T)
    K, T = M.shape
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(1.0 + 0.4*T, 1.0 + 0.35*K))
    im = sns.heatmap(M, ax=ax, cmap="viridis", cbar=True)
    ax.set_xlabel("round t")
    ax.set_ylabel("class μ")
    ax.set_title("Hopfield magnetisations m_μ(t) — HEBB synapses")
    # seaborn heatmap already draws colorbar
    fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)

# Robust variant that falls back to overlaps when Hopfield reports are missing
def plot_magnetization_heatmap_robust(run_dir: Union[str, Path],
                                      outpath: Union[str, Path],
                                      hopfield_subdir: str = "hopfield_hebb",
                                      key_order: Sequence[str] = ("magnetization_by_mu", "m_by_mu", "mag_by_mu"),
                                      K_old: Optional[int] = None) -> None:
    run_dir = Path(run_dir)
    rounds = _list_round_dirs(run_dir)
    if not rounds:
        return
    # Try load xi_true for fallback
    xi_true = None
    xtp = run_dir / "xi_true.npy"
    if xtp.exists():
        try:
            xi_true = np.load(xtp)
        except Exception:
            xi_true = None
    m_list = []
    for rd in rounds:
        m_mu = None
        rep_json = rd / hopfield_subdir / "report.json"
        rep_npz = rd / hopfield_subdir / "report.npz"
        if rep_json.exists():
            try:
                js = _read_json(rep_json)
                for key in key_order:
                    if key in js:
                        m_mu = np.asarray(js[key], dtype=float).ravel()
                        break
            except Exception:
                m_mu = None
        if m_mu is None and rep_npz.exists():
            try:
                data = np.load(rep_npz)
                for key in key_order:
                    if key in data:
                        m_mu = np.asarray(data[key], dtype=float).ravel()
                        break
            except Exception:
                m_mu = None
        if m_mu is None and xi_true is not None:
            xa = rd / "xi_aligned.npy"
            if xa.exists():
                try:
                    xi_al = np.load(xa)
                    if xi_al.shape[0] == xi_true.shape[0]:
                        # treat zero rows (no candidate) as NaN so they don't show as 1
                        m_tmp = np.abs(np.sum(xi_al * xi_true, axis=1) / float(xi_true.shape[1]))
                        for i in range(m_tmp.size):
                            if np.all(xi_al[i] == 0):
                                m_tmp[i] = np.nan
                        m_mu = m_tmp
                except Exception:
                    m_mu = None
        if m_mu is None:
            return
        m_list.append(m_mu)
    M = np.stack(m_list, axis=1)  # (K,T)
    K, T = M.shape
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(1.0 + 0.4*T, 1.0 + 0.35*K))
    sns.heatmap(M, ax=ax, cmap="viridis")
    ax.set_xlabel("round t")
    # y labels: old vs new split if provided
    if K_old is not None and 0 < K_old < K:
        ax.hlines([K_old - 0.5], xmin=-0.5, xmax=T - 0.5, colors='w', linestyles=':', linewidth=1.2)
        ax.set_yticks(list(range(K)))
        labels = [f"old-{i}" if i < K_old else f"new-{i-K_old}" for i in range(K)]
        ax.set_yticklabels(labels)
    else:
        ax.set_ylabel("class μ")
    ax.set_title("Hopfield magnetisations m_μ(t) — HEBB synapses")
    fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Panel H — t_detect vs params (scatter)
# -----------------------------------------------------------------------------
def plot_tdetect_scatter(summaries: Iterable[Dict[str, Any]],
                         xkey: str, ykey: str,
                         outpath: Union[str, Path],
                         ckey: Optional[str] = None) -> None:
    """
    Given a list of summary dicts (e.g., one per sweep point), scatter-plot:
      x = item[xkey], y = item[ykey]  (e.g., ykey="t_detect")
      color by item[ckey] if provided.

    Example: summaries = [
        {"ramp_len": 5, "new_visibility_frac": 0.33, "t_detect": 12},
        {"ramp_len": 5, "new_visibility_frac": 0.66, "t_detect": 7}, ...
    ]
    plot_tdetect_scatter(summaries, "new_visibility_frac", "t_detect", "tdetect.png", ckey="ramp_len")
    """
    import matplotlib.pyplot as plt  # local
    xs, ys, cs = [], [], []
    for s in summaries:
        if xkey in s and ykey in s:
            xs.append(s[xkey]); ys.append(s[ykey])
            cs.append(s.get(ckey, 0.0) if ckey else 0.0)
    if not xs:
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _style_ax(ax)
    if ckey:
        sns.scatterplot(x=xs, y=ys, hue=cs, palette="viridis", ax=ax, legend=False)
        cb = plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=min(cs), vmax=max(cs)), cmap="viridis"), ax=ax, shrink=0.8); cb.set_label(ckey)
    else:
        sns.scatterplot(x=xs, y=ys, ax=ax)
    ax.set_xlabel(xkey); ax.set_ylabel(ykey)
    ax.set_title(f"{ykey} vs {xkey}")
    fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Convenience to build a full panel set for a run
# -----------------------------------------------------------------------------
def build_full_panel_set(run_dir: Union[str, Path],
                         K_old: int,
                         outdir: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """
    Produce the extended set of figures for a given run directory.
    Returns a dict of created file paths.
    """
    run_dir = Path(run_dir)
    out = Path(outdir) if outdir is not None else (run_dir / "exp07_plots")
    out.mkdir(parents=True, exist_ok=True)

    # Core series
    series = compute_series_over_run(run_dir, K_old=int(K_old))

    # Save the ones not already covered by reporting.py
    paths: Dict[str, str] = {}
    if series.K == 3:
        p = out / "simplex_timecolored.png"
        plot_simplex_timecolored(series, p)
        paths["simplex_timecolored"] = str(p)

    # Scree pre/post novelty
    p = out / "scree_pre_post.png"
    plot_scree_pre_post(run_dir, K_old=int(K_old), outpath=p)
    paths["scree_pre_post"] = str(p)

    # Magnetisation heatmap (robust)
    p = out / "magnetization_heatmap.png"
    plot_magnetization_heatmap_robust(run_dir, p, K_old=int(K_old))
    paths["magnetization_heatmap"] = str(p)

    # Final Hebb violin plot (convergence from noisy inits)
    p = out / "hebb_violin.png"
    try:
        plot_final_hebb_violin(run_dir, p, K_old=int(K_old))
        paths["hebb_violin"] = str(p)
    except Exception:
        pass

    return paths

# -----------------------------------------------------------------------------
# Final Hebb violin: convergence from noisy initial states
# -----------------------------------------------------------------------------
def plot_final_hebb_violin(run_dir: Union[str, Path],
                           outpath: Union[str, Path],
                           *,
                           K_old: Optional[int] = None,
                           noise_levels: Sequence[float] = (0.1, 0.2, 0.3),
                           reps: int = 50,
                           updates: int = 30) -> None:
    run_dir = Path(run_dir)
    xi_path = run_dir / f"round_{len(_list_round_dirs(run_dir))-1:03d}" / "xi_aligned.npy"
    if not xi_path.exists():
        return
    xi_ref = np.load(xi_path)  # (K,N)
    K, N = xi_ref.shape
    J = _hebb_J(xi_ref)

    records = []
    rng = np.random.default_rng(0)
    for mu in range(K):
        base = xi_ref[mu]
        for nl in noise_levels:
            flips = int(round(nl * N))
            for r in range(reps):
                idx = rng.choice(N, size=flips, replace=False) if flips > 0 else np.array([], dtype=int)
                s0 = base.copy().astype(float)
                s0[idx] *= -1.0
                sT = _hopfield_dynamics(J, s0, steps=updates)
                m = abs(float(np.dot(sT, base)) / float(N))
                records.append({"class": mu, "noise": nl, "m": m})

    import pandas as pd
    df = pd.DataFrame.from_records(records)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if K_old is not None and 0 < K_old < K:
        class_labels = [f"old-{i}" if i < K_old else f"new-{i-K_old}" for i in range(K)]
    else:
        class_labels = [str(i) for i in range(K)]
    df["class_label"] = df["class"].apply(lambda x: class_labels[int(x)])
    sns.violinplot(data=df, x="class_label", y="m", hue="noise", ax=ax, cut=0, inner="quartile")
    ax.set_xlabel("class (old/new)")
    ax.set_ylabel("final magnetization |m|")
    ax.set_title("Convergenza Hopfield con J_hebb(finale) da stati iniziali rumorosi")
    fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)
