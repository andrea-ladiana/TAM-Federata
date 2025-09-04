# -*- coding: utf-8 -*-
"""
Plotting per Exp-06 (single-only).

Questo modulo implementa:
1) Pannello 4× per un seed:
   (i)  Simplesso: π_t vs π̂_t con vettori di lag annotati.
   (ii) Timeseries magnetizzazioni Hopfield m_μ(t) (linee + bande s.e.m.).
   (iii) Heatmap retention m_μ(t) (K×T).
   (iv) Phase diagram locale: metriche vs w.

2) Lag–Amplitude plot:
   φ(ω) e |H(ω)| sperimentali vs w, con overlay di curve teoriche opzionali.

3) Forgetting vs Plasticity (trittico):
   per tre bucket di w (basso/intermedio/alto): (a) distanza dal baricentro nel
   simplesso, (b) lag globale, (c) media m_μ “old vs recent”.

4) Scatter “Exposure → Magnetizzazione”:
   regressione lineare + Pearson/Spearman annotati.

Tutte le funzioni accettano un oggetto Axes (quando applicabile) e restituiscono
handles/metriche utili. Niente salvataggi impliciti: usa `fig.savefig(...)`.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Dict, Any, List, Callable, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Riuso dal modulo metrics (embedding e stima lag/ampiezza)
from .metrics import simplex_embed_2d, lag_and_amplitude


# -----------------------------------------------------------------------------
# Utilità generiche
# -----------------------------------------------------------------------------
def _set_ax_equal(ax: Axes) -> None:
    ax.set_aspect("equal", adjustable="box")


def _ensure_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(-1)


def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Calcola Pearson e Spearman (senza SciPy) su vettori 1D; ignora NaN.
    """
    x = _ensure_1d(x).astype(float)
    y = _ensure_1d(y).astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return np.nan, np.nan

    # Pearson
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = np.sqrt((x_c**2).sum() * (y_c**2).sum())
    pearson = float((x_c @ y_c) / denom) if denom > 0 else np.nan

    # Spearman via rank
    def rankdata(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(v.size, dtype=float)
        # gestisce ties con media dei ranghi
        # trova blocchi con valori uguali
        same = np.flatnonzero(np.diff(v[order]) == 0.0)
        if same.size > 0:
            start = 0
            i = 0
            while i < v.size:
                j = i + 1
                while j < v.size and v[order][j] == v[order][i]:
                    j += 1
                # media dei ranghi per il blocco [i, j)
                mean_rank = (i + j - 1) / 2.0
                ranks[order][i:j] = mean_rank
                i = j
        return ranks

    rx = rankdata(x)
    ry = rankdata(y)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom_s = np.sqrt((rx_c**2).sum() * (ry_c**2).sum())
    spearman = float((rx_c @ ry_c) / denom_s) if denom_s > 0 else np.nan
    return pearson, spearman


def _triangle_vertices() -> np.ndarray:
    """Restituisce i vertici del triangolo equilatero nel sistema usato da simplex_embed_2d."""
    # usa embedding sugli e_i canonici per garantire coerenza
    I = np.eye(3, dtype=float)
    return simplex_embed_2d(I)  # (3,2)


def _draw_simplex_frame(ax: Axes, labels: Sequence[str] = ("μ0", "μ1", "μ2")) -> None:
    """Disegna i bordi del triangolo e i label ai vertici."""
    V = _triangle_vertices()  # (3,2)
    # chiude il triangolo ripetendo il primo vertice
    poly = np.vstack([V, V[:1]])
    ax.plot(poly[:, 0], poly[:, 1], lw=1.5, color="black", alpha=0.6)
    # vertici
    ax.scatter(V[:, 0], V[:, 1], s=30, color="black")
    for i in range(3):
        ax.text(V[i, 0], V[i, 1], f" {labels[i]}", va="center", ha="left", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    _set_ax_equal(ax)


# -----------------------------------------------------------------------------
# (i) Simplesso: π_t vs π̂_t con vettori di lag
# -----------------------------------------------------------------------------
def plot_simplex_trajectory(
    ax: Axes,
    pi_true_seq: np.ndarray,      # (T, 3)
    pi_hat_seq: Optional[np.ndarray] = None,  # (T, 3) o None
    *,
    labels: Sequence[str] = ("μ0", "μ1", "μ2"),
    show_arrows_every: int = 1,
    arrow_alpha: float = 0.9,
    compute_lag: bool = True,
    title: Optional[str] = None,
    simplex_style: str = "modern",  # 'modern' (embedding) oppure 'legacy'
    color_by_time: bool = True,
    cmap_name: str = "viridis",
    add_colorbar: bool = True,
    true_label: str = "mixing vero",
    hat_label: str = "mixing stimato",
    arrows_mode: str = "direct",  # 'direct' (true t -> hat t) oppure 'lag' (usa lag stimato)
) -> Dict[str, Any]:
    """Versione migliorata del plot sul simplesso ispirata alla logica richiesta.

    Novità principali:
      - Colorazione dei punti per round (colormap continua) con colorbar.
      - Frecce per ogni (o ogni n) round dal punto true → hat.
      - Arrows 'direct' (t→t) per aderire all'esempio utente; opzione 'lag'.
      - Mantiene il calcolo lag/amplitude (ritornato in info) se compute_lag=True.
    """
    pi_true_seq = np.asarray(pi_true_seq, dtype=float)
    if pi_true_seq.ndim != 2 or pi_true_seq.shape[1] != 3:
        raise ValueError("pi_true_seq deve essere (T,3).")

    if simplex_style not in {"modern", "legacy"}:
        raise ValueError("simplex_style deve essere 'modern' o 'legacy'.")

    # --- Embedding ---
    if simplex_style == "modern":
        _draw_simplex_frame(ax, labels=labels)
        xy_true = simplex_embed_2d(pi_true_seq)  # (T,2)
    else:  # legacy embedding equivalente a quello già usato nello script precedente
        a = pi_true_seq[:, 0]; b = pi_true_seq[:, 1]; c = pi_true_seq[:, 2]
        denom = a + b + c
        denom[denom == 0] = 1.0
        x = 0.5 * (2.0 * b + c) / denom
        y = (np.sqrt(3.0) / 2.0) * c / denom
        xy_true = np.stack([x, y], axis=1)
        verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0)/2.0], [0.0, 0.0]])
        ax.plot(verts[:,0], verts[:,1], color="black", lw=1.5, alpha=0.6)
        ax.scatter(verts[:3,0], verts[:3,1], s=30, color="black")
        for i,(vx,vy) in enumerate(verts[:3]):
            ax.text(vx, vy, f" {labels[i]}", va="center", ha="left", fontsize=10)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, (np.sqrt(3.0)/2.0)+0.03)
        ax.set_aspect('equal', 'box')

    T = pi_true_seq.shape[0]

    # --- Colori per round ---
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(t / max(1, T - 1)) for t in range(T)] if color_by_time else ["C0"] * T

    # --- Traiettoria True ---
    ax.plot(xy_true[:, 0], xy_true[:, 1], lw=2.0, alpha=0.95, label=true_label)

    # --- Possibly Hat ---
    info: Dict[str, Any] = {}
    if pi_hat_seq is not None:
        pi_hat_seq = np.asarray(pi_hat_seq, dtype=float)
        if pi_hat_seq.shape != pi_true_seq.shape:
            raise ValueError("pi_hat_seq deve avere shape (T,3) come pi_true_seq.")
        if simplex_style == "modern":
            xy_hat = simplex_embed_2d(pi_hat_seq)
        else:
            a = pi_hat_seq[:, 0]; b = pi_hat_seq[:, 1]; c = pi_hat_seq[:, 2]
            denom = a + b + c
            denom[denom == 0] = 1.0
            xh = 0.5 * (2.0 * b + c) / denom
            yh = (np.sqrt(3.0) / 2.0) * c / denom
            xy_hat = np.stack([xh, yh], axis=1)

        ax.plot(xy_hat[:, 0], xy_hat[:, 1], lw=2.0, alpha=0.95, ls="--", label=hat_label)

        # Calcolo lag/amplitude (info) anche se le frecce sono 'direct'
        la = lag_and_amplitude(pi_true_seq, pi_hat_seq) if compute_lag else {}
        info.update(la)
    else:
        xy_hat = None  # type: ignore

    # --- Scatter + frecce ---
    for t in range(T):
        # punti true
        ax.scatter(xy_true[t,0], xy_true[t,1], s=42, color=colors[t], edgecolor="white", linewidth=0.8, zorder=3)
        if xy_hat is not None:
            ax.scatter(xy_hat[t,0], xy_hat[t,1], s=28, color=colors[t], edgecolor="none", zorder=3)
        # frecce: true(t) -> hat(t) (direct) oppure lag-based
        if xy_hat is not None and (t % max(1, show_arrows_every) == 0):
            if arrows_mode == "lag" and compute_lag and "lag_rounds" in info:
                lag_r = int(info.get("lag_rounds", 0))
                j = (t + lag_r) % T
                x_to, y_to = xy_hat[j]
            else:
                x_to, y_to = xy_hat[t]
            ax.annotate("", xy=(x_to, y_to), xytext=(xy_true[t,0], xy_true[t,1]),
                        arrowprops=dict(arrowstyle="->", lw=1.0, color=colors[t], alpha=arrow_alpha))

    # --- Colorbar ---
    if color_by_time and add_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=T-1))
        cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("round")

    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    return info


# -----------------------------------------------------------------------------
# (ii) Timeseries magnetizzazioni con bande s.e.m.
# -----------------------------------------------------------------------------
def plot_magnetization_timeseries(
    ax: Axes,
    M_mean: np.ndarray,                # (K, T) o (T,) se singola curva aggregata
    M_sem: Optional[np.ndarray] = None,  # (K, T) o (T,)
    *,
    labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> None:
    M_mean = np.asarray(M_mean, dtype=float)
    if M_mean.ndim == 1:
        # singola curva
        T = M_mean.size
        x = np.arange(T)
        ax.plot(x, M_mean, lw=1.8, label="mean m")
        if M_sem is not None:
            M_sem = _ensure_1d(np.asarray(M_sem, dtype=float))
            lo = M_mean - M_sem
            hi = M_mean + M_sem
            ax.fill_between(x, lo, hi, alpha=0.25, linewidth=0)
    elif M_mean.ndim == 2:
        K, T = M_mean.shape
        x = np.arange(T)
        for mu in range(K):
            lab = labels[mu] if labels is not None and mu < len(labels) else f"μ{mu}"
            ax.plot(x, M_mean[mu], lw=1.8, label=lab)
            if M_sem is not None:
                sem = np.asarray(M_sem, dtype=float)
                if sem.ndim == 2 and sem.shape == (K, T):
                    lo = M_mean[mu] - sem[mu]
                    hi = M_mean[mu] + sem[mu]
                    ax.fill_between(x, lo, hi, alpha=0.25, linewidth=0)
    else:
        raise ValueError("M_mean deve essere (K,T) o (T,).")

    ax.set_xlabel("round")
    ax.set_ylabel("magnetization m")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, ncol=2)
    if title:
        ax.set_title(title, fontsize=11)


# -----------------------------------------------------------------------------
# (iii) Heatmap retention (K×T)
# -----------------------------------------------------------------------------
def plot_magnetization_heatmap(
    ax: Axes,
    M: np.ndarray,              # (K, T)
    *,
    title: Optional[str] = None,
    cbar: bool = True,
) -> None:
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError("M deve essere (K,T).")
    im = ax.imshow(M, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
    ax.set_xlabel("round")
    ax.set_ylabel("μ (archetype)")
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m")
    if title:
        ax.set_title(title, fontsize=11)


# -----------------------------------------------------------------------------
# (iv) Phase diagram locale (metriche vs w)
# -----------------------------------------------------------------------------
def plot_phase_diagram_local(
    ax: Axes,
    w_values: Sequence[float],
    metric_values: Sequence[float],
    *,
    metric_label: str = "mean m",
    title: Optional[str] = None,
) -> None:
    w = np.asarray(w_values, dtype=float)
    y = np.asarray(metric_values, dtype=float)
    ax.plot(w, y, marker="o", lw=1.8)
    ax.set_xlabel("w")
    ax.set_ylabel(metric_label)
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title, fontsize=11)


# -----------------------------------------------------------------------------
# 1) Pannello 4× (riassuntivo per un seed)
# -----------------------------------------------------------------------------
def panel4x(
    pi_true_seq: np.ndarray,              # (T,3)
    pi_hat_seq: Optional[np.ndarray],     # (T,3) o None
    M_mean: np.ndarray,                   # (K,T) o (T,)
    M_sem: Optional[np.ndarray] = None,   # (K,T) o (T,)
    *,
    w_values: Optional[Sequence[float]] = None,
    phase_metric: Optional[Sequence[float]] = None,
    phase_metric_label: str = "mean m",
    labels: Sequence[str] = ("μ0", "μ1", "μ2"),
    simplex_style: str = "modern",  # 'modern' oppure 'legacy' (passato a plot_simplex_trajectory)
    figsize: Tuple[int, int] = (14, 8),
    suptitle: Optional[str] = None,
) -> Tuple[Figure, Dict[str, Any]]:
    """
    Crea il pannello 4×:
      [0,0] simplesso; [0,1] timeseries m; [1,0] heatmap m; [1,1] phase diagram.
    Restituisce fig e dict con info (es. lag/amplitude stimati).
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    info: Dict[str, Any] = {}

    # (i) Simplesso
    info["simplex"] = plot_simplex_trajectory(
        axs[0, 0],
        pi_true_seq,
        pi_hat_seq,
        labels=labels,
        title="Simplex trajectory",
        simplex_style=simplex_style,
    )

    # (ii) Magnetizzazioni timeseries
    plot_magnetization_timeseries(axs[0, 1], M_mean, M_sem, labels=labels, title="Magnetization m_μ(t)")

    # (iii) Heatmap
    if M_mean.ndim == 2:
        plot_magnetization_heatmap(axs[1, 0], M_mean, title="Retention heatmap")
    else:
        # se M_mean è 1D, costruiamo una “heatmap” fittizia con una riga
        plot_magnetization_heatmap(axs[1, 0], M_mean[None, :], title="Retention (aggregate)")

    # (iv) Phase diagram locale
    if w_values is not None and phase_metric is not None:
        plot_phase_diagram_local(axs[1, 1], w_values, phase_metric, metric_label=phase_metric_label,
                                 title="Phase diagram (local)")
    else:
        axs[1, 1].axis("off")
        axs[1, 1].text(0.5, 0.5, "No phase metric provided", ha="center", va="center")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    return fig, info


# -----------------------------------------------------------------------------
# 2) Lag–Amplitude plot (sperimentale vs teorico)
# -----------------------------------------------------------------------------
def plot_lag_amplitude_vs_w(
    w_values: Sequence[float],
    lag_rad_values: Sequence[float],
    amp_values: Sequence[float],
    *,
    phi_theory: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    amp_theory: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    figsize: Tuple[int, int] = (12, 4),
    suptitle: Optional[str] = "Lag–Amplitude vs w",
) -> Figure:
    """
    Due pannelli affiancati:
      - sinistra: φ(w) [radianti]
      - destra: |H|(w)

    Le curve teoriche sono opzionali (callable che mappano array di w → array).
    """
    w = np.asarray(w_values, dtype=float)
    phi = np.asarray(lag_rad_values, dtype=float)
    H = np.asarray(amp_values, dtype=float)

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # φ(w)
    axs[0].plot(w, phi, marker="o", lw=1.8, label="exp")
    if phi_theory is not None:
        try:
            axs[0].plot(w, _ensure_1d(phi_theory(w)), lw=1.5, ls="--", label="theory")
        except Exception:
            pass
    axs[0].set_xlabel("w")
    axs[0].set_ylabel("lag φ (radians)")
    axs[0].grid(alpha=0.2)
    axs[0].legend(fontsize=9)

    # |H|(w)
    axs[1].plot(w, H, marker="o", lw=1.8, label="exp")
    if amp_theory is not None:
        try:
            axs[1].plot(w, _ensure_1d(amp_theory(w)), lw=1.5, ls="--", label="theory")
        except Exception:
            pass
    axs[1].set_xlabel("w")
    axs[1].set_ylabel("|H| (amplitude ratio)")
    axs[1].grid(alpha=0.2)
    axs[1].legend(fontsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    return fig


# -----------------------------------------------------------------------------
# 3) Forgetting vs Plasticity — trittico
# -----------------------------------------------------------------------------
def forgetting_vs_plasticity_triptych(
    entries: Sequence[Dict[str, Any]],
    *,
    recent_window: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 4),
    suptitle: Optional[str] = "Forgetting vs Plasticity",
) -> Figure:
    """
    entries: lista di dict, uno per bucket di w, ciascuno con:
      - 'w': float
      - 'pi_true_seq': (T,3)
      - 'pi_hat_seq':  (T,3)
      - 'M': (K,T) magnetizzazioni

    Per ogni bucket calcola:
      (a) distanza media dal baricentro (r̄) della traiettoria π_true
      (b) lag globale |φ| via lag_and_amplitude
      (c) media m_old vs m_recent, dove:
           old = primi R round, recent = ultimi R round (R=recent_window o T//3)
    Ritorna una figura 1×3 con barre aggregate per bucket.
    """
    W: List[float] = []
    dist_center: List[float] = []
    lag_abs: List[float] = []
    m_old: List[float] = []
    m_recent: List[float] = []

    for ent in entries:
        w = float(ent["w"])
        pi_true = np.asarray(ent["pi_true_seq"], dtype=float)
        pi_hat = np.asarray(ent["pi_hat_seq"], dtype=float)
        M = np.asarray(ent["M"], dtype=float)
        if pi_true.shape[-1] != 3 or pi_hat.shape != pi_true.shape:
            raise ValueError("pi_* devono essere (T,3) e coerenti.")
        if M.ndim != 2:
            raise ValueError("M deve essere (K,T).")
        T = pi_true.shape[0]
        R = recent_window if (recent_window is not None and recent_window > 0 and recent_window < T) else max(1, T // 3)

        # (a) distanza dal baricentro
        xy = simplex_embed_2d(pi_true)
        r = np.linalg.norm(xy, axis=1)
        dist_center.append(float(np.mean(r)))

        # (b) lag globale
        la = lag_and_amplitude(pi_true, pi_hat)
        lag_abs.append(float(abs(la["lag_radians"])))

        # (c) m_old vs m_recent
        m_old.append(float(np.mean(M[:, :R])))
        m_recent.append(float(np.mean(M[:, -R:])))

        W.append(w)

    # Ordina per w crescente
    order = np.argsort(W)
    W_arr = np.asarray(W, dtype=float)[order]
    dist_center_arr = np.asarray(dist_center, dtype=float)[order]
    lag_abs_arr = np.asarray(lag_abs, dtype=float)[order]
    m_old_arr = np.asarray(m_old, dtype=float)[order]
    m_recent_arr = np.asarray(m_recent, dtype=float)[order]

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # (a) distanza dal baricentro
    axs[0].bar(np.arange(W_arr.size), dist_center_arr, tick_label=[f"{w:.2f}" for w in W_arr])
    axs[0].set_xlabel("w")
    axs[0].set_ylabel("mean distance from center")
    axs[0].set_title("Simplesso: distanza media")
    axs[0].grid(axis="y", alpha=0.2)

    # (b) lag
    axs[1].bar(np.arange(W_arr.size), lag_abs_arr, tick_label=[f"{w:.2f}" for w in W_arr])
    axs[1].set_xlabel("w")
    axs[1].set_ylabel("|lag| (radians)")
    axs[1].set_title("Lag globale")
    axs[1].grid(axis="y", alpha=0.2)

    # (c) m_old vs m_recent (barre affiancate)
    idx = np.arange(W_arr.size)
    width = 0.38
    axs[2].bar(idx - width/2, m_old_arr, width=width, label="old")
    axs[2].bar(idx + width/2, m_recent_arr, width=width, label="recent")
    axs[2].set_xticks(idx)
    axs[2].set_xticklabels([f"{w:.2f}" for w in W_arr])
    axs[2].set_xlabel("w")
    axs[2].set_ylabel("mean m")
    axs[2].set_title("m: old vs recent")
    axs[2].legend()
    axs[2].grid(axis="y", alpha=0.2)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    return fig


# -----------------------------------------------------------------------------
# 4) Scatter “Exposure → Magnetizzazione” (con regressione e ρ/Spearman)
# -----------------------------------------------------------------------------
def scatter_exposure_vs_magnetization(
    ax: Axes,
    exposure: Union[np.ndarray, Sequence[float]],         # (K,) o (K,T) o (T,)
    magnetization: Union[np.ndarray, Sequence[float]],    # (K,) o (K,T) o (T,)
    *,
    title: Optional[str] = "Exposure → Magnetization",
    annotate_stats: bool = True,
) -> Dict[str, float]:
    """
    Accetta exposure e magnetization come vettori o matrici:
      - Se (K,T), appiattisce entrambi (μ,t) e correla su tutti i punti.
      - Se (K,), correla le medie per archetipo.

    Esegue regressione lineare y = a x + b, disegna la retta e annota Pearson/Spearman.
    """
    X = np.asarray(exposure, dtype=float)
    Y = np.asarray(magnetization, dtype=float)
    # appiattisci coerentemente
    if X.shape != Y.shape:
        # consenti (K,) vs (K,T) usando medie su T
        if X.ndim == 1 and Y.ndim == 2 and X.size == Y.shape[0]:
            X = np.repeat(X, Y.shape[1])
            Y = Y.reshape(-1)
        elif X.ndim == 2 and Y.ndim == 1 and Y.size == X.shape[0]:
            X = X.reshape(-1)
            Y = np.repeat(Y, X.size // Y.size)
        else:
            raise ValueError("exposure e magnetization devono avere shape compatibili.")

    x = X.reshape(-1)
    y = Y.reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        raise ValueError("Dati vuoti dopo la pulizia dei NaN.")

    # retta di regressione
    a, b = np.polyfit(x, y, deg=1)
    y_pred = a * x + b

    # correlazioni
    r_pearson, r_spearman = _pearson_spearman(x, y)

    ax.scatter(x, y, s=20, alpha=0.7)
    # retta (disegniamo nell'intervallo dei dati)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, a * xs + b, lw=1.8, ls="--", label=f"y={a:.3f}x+{b:.3f}")
    ax.set_xlabel("exposure")
    ax.set_ylabel("magnetization")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)
    if annotate_stats:
        ax.text(0.02, 0.98,
                f"Pearson r={r_pearson:.3f}\nSpearman ρ={r_spearman:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    return {"pearson_r": r_pearson, "spearman_rho": r_spearman}
