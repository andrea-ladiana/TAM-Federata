# -*- coding: utf-8 -*-
"""
Publication-quality visualization for the client-aware adaptive weight
experiment.  Style matched to  out_06/publication_panels/panel06_merged.png.

Layout:

    ┌──────────────────────┬──────────────────────┐
    │  A) Adaptive w_c(t)  │  B) Server Retrieval │
    └──────────────────────┴──────────────────────┘

Supports two modes:
  1. Single-seed  →  individual good-client lines + attacker.
  2. Multi-seed   →  mean ± SE shaded bands for good / attacker / retrieval.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .simulation import MultiSeedResult

# ── Try seaborn (graceful fallback) ──
try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    sns = None  # type: ignore
    _HAS_SNS = False


# ──────────────────────────────────────────────────────────────────
# Publication-ready defaults (matching panel06_merged.py)
# ──────────────────────────────────────────────────────────────────

def _apply_rc():
    mpl.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 1.5,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "patch.linewidth": 1.2,
    })


# ──────────────────────────────────────────────────────────────────
# Colour palette  (Okabe-Ito, colorblind-friendly)
# ──────────────────────────────────────────────────────────────────

COLOR_ATTACKER = "#D55E00"   # Vermillion
COLOR_GOOD_MEAN = "#0072B2"  # Blue
COLOR_GOOD_FILL = "#56B4E9"  # Sky blue (envelope)
COLOR_GOOD_INDIVIDUAL = [    # per-client shades
    "#A6CEE3", "#7EB8DA", "#56B4E9", "#3A8EBF",
]
COLOR_RETRIEVAL = "#009E73"  # Bluish green
COLOR_MAG = [                # Per-archetype magnetisation
    "#E69F00",               # Orange
    "#56B4E9",               # Sky blue
    "#009E73",               # Bluish green
]


# ──────────────────────────────────────────────────────────────────
# Single-seed panel  (legacy)
# ──────────────────────────────────────────────────────────────────

def plot_panel(
    w_history: np.ndarray,          # (L, T)
    retrieval_history: np.ndarray,  # (T,)
    mag_history: np.ndarray,        # (K, T)
    attacker_idx: int,
    mag_correction: Optional[np.ndarray] = None,  # (T,) additive correction
    out_path: Optional[Path] = None,
    dpi: int = 300,
) -> None:
    """Single-seed figure with individual good-client lines."""
    _apply_rc()

    L, T = w_history.shape
    K = mag_history.shape[0]
    t = np.arange(T)

    # ── Apply magnetization correction ──
    if mag_correction is None:
        mag_correction = np.zeros(T, dtype=np.float32)
    else:
        mag_correction = np.asarray(mag_correction, dtype=np.float32)
        if len(mag_correction) != T:
            raise ValueError(f"mag_correction length ({len(mag_correction)}) != T ({T})")

    # Copy and correct data
    retrieval_history = retrieval_history + mag_correction
    mag_history = mag_history + mag_correction[np.newaxis, :]  # broadcast to (K, T)

    good_mask = np.ones(L, dtype=bool)
    good_mask[attacker_idx] = False
    w_good = w_history[good_mask]
    w_att = w_history[attacker_idx]

    w_good_mean = w_good.mean(axis=0)
    w_good_min = w_good.min(axis=0)
    w_good_max = w_good.max(axis=0)

    fig, (ax_w, ax_r) = plt.subplots(1, 2, figsize=(14, 5),
                                      constrained_layout=True)

    # ── Panel A ──
    for i in range(w_good.shape[0]):
        ax_w.plot(t, w_good[i],
                  color=COLOR_GOOD_INDIVIDUAL[i % len(COLOR_GOOD_INDIVIDUAL)],
                  lw=1.0, alpha=0.5)
    ax_w.fill_between(t, w_good_min, w_good_max,
                      color=COLOR_GOOD_FILL, alpha=0.18)
    ax_w.plot(t, w_good_mean, color=COLOR_GOOD_MEAN, lw=2.5,
              label=r"Good clients ($r=0.8$, mean)")
    ax_w.plot(t, w_att, color=COLOR_ATTACKER, lw=2.5, ls="--",
              label=r"Attacker ($r\simeq0$)")
    _dress_w_axis(ax_w, T, w_good_mean)

    # ── Panel B ──
    for k in range(min(K, len(COLOR_MAG))):
        ax_r.plot(t, mag_history[k], color=COLOR_MAG[k], lw=2.0,
                  alpha=0.7, label=rf"$m_{{{k+1}}}(t)$")
    ax_r.plot(t, retrieval_history, color="k", lw=2.5,
              label=r"$\langle m \rangle$ (mean)")
    _dress_retr_axis(ax_r, T)

    for ax in (ax_w, ax_r):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _save_fig(fig, out_path, dpi)


# ──────────────────────────────────────────────────────────────────
# Multi-seed panel  (mean ± SE)
# ──────────────────────────────────────────────────────────────────

def plot_panel_multiseed(
    ms: "MultiSeedResult",
    mag_correction: Optional[np.ndarray] = None,  # (T,) additive correction
    out_path: Optional[Path] = None,
    dpi: int = 300,
    r_good_label: float = 0.9,
) -> None:
    """
    Plot mean ± standard-error bands for both panels, aggregated
    over *n_seeds* independent simulations.
    """
    _apply_rc()

    T = len(ms.w_good_mean)
    K = ms.mag_mean.shape[0]
    t = np.arange(T)

    # ── Apply magnetization correction ──
    if mag_correction is None:
        mag_correction = np.zeros(T, dtype=np.float32)
    else:
        mag_correction = np.asarray(mag_correction, dtype=np.float32)
        if len(mag_correction) != T:
            raise ValueError(f"mag_correction length ({len(mag_correction)}) != T ({T})")

    # Copy and correct data
    retr_mean_corrected = ms.retr_mean + mag_correction
    mag_mean_corrected = ms.mag_mean + mag_correction[np.newaxis, :]  # (K, T)

    fig, (ax_w, ax_r) = plt.subplots(1, 2, figsize=(14, 5),
                                      constrained_layout=True)

    # ────────── Panel A: w_c(t)  mean ± SE ──────────

    # Good clients band
    ax_w.fill_between(
        t,
        ms.w_good_mean - ms.w_good_se,
        ms.w_good_mean + ms.w_good_se,
        color=COLOR_GOOD_FILL, alpha=0.30,
    )
    ax_w.plot(
        t, ms.w_good_mean,
        color=COLOR_GOOD_MEAN, lw=2.5,
        label=rf"Good clients ($r={r_good_label}$)",
    )

    # Attacker band
    ax_w.fill_between(
        t,
        np.maximum(ms.w_att_mean - ms.w_att_se, 0),
        ms.w_att_mean + ms.w_att_se,
        color=COLOR_ATTACKER, alpha=0.20,
    )
    ax_w.plot(
        t, ms.w_att_mean,
        color=COLOR_ATTACKER, lw=2.5, ls="--",
        label=r"Attacker ($r\simeq0$)",
    )

    _dress_w_axis(ax_w, T, ms.w_good_mean)

    # Annotation: n_seeds
    ax_w.text(
        0.02, 0.02,
        f"$S = {ms.n_seeds}$ seeds",
        transform=ax_w.transAxes, fontsize=10,
        va="bottom", ha="left", color="#555555",
    )

    # ────────── Panel B: Retrieval  mean ± SE ──────────

    # Per-archetype magnetisations
    for k in range(min(K, len(COLOR_MAG))):
        ax_r.fill_between(
            t,
            np.maximum(mag_mean_corrected[k] - ms.mag_se[k], 0),
            np.minimum(mag_mean_corrected[k] + ms.mag_se[k], 1),
            color=COLOR_MAG[k], alpha=0.18,
        )
        ax_r.plot(
            t, mag_mean_corrected[k],
            color=COLOR_MAG[k], lw=2.0, alpha=0.8,
            label=rf"$m_{{{k+1}}}(t)$",
        )

    # Mean retrieval band
    ax_r.fill_between(
        t,
        np.maximum(retr_mean_corrected - ms.retr_se, 0),
        np.minimum(retr_mean_corrected + ms.retr_se, 1),
        color="k", alpha=0.10,
    )
    ax_r.plot(
        t, retr_mean_corrected,
        color="k", lw=2.5,
        label=r"$\langle m \rangle$",
    )

    _dress_retr_axis(ax_r, T)

    for ax in (ax_w, ax_r):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _save_fig(fig, out_path, dpi)


# ──────────────────────────────────────────────────────────────────
# Shared cosmetic helpers
# ──────────────────────────────────────────────────────────────────

def _dress_w_axis(ax, T: int, w_good_mean: np.ndarray):
    ax.set_title(r"A)  Adaptive Weight $w_c(t)$",
                 loc="left", fontweight="bold")
    ax.set_xlabel(r"Round $t$")
    ax.set_ylabel(r"$w_c(t)$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, T - 1)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="k", framealpha=0.9)

    # Horizontal guide at final good-client level
    w_stable = float(w_good_mean[-1])
    ax.axhline(w_stable, color=COLOR_GOOD_MEAN, lw=0.8, ls=":", alpha=0.5)
    ax.annotate(
        f"$w^*_{{\\mathrm{{good}}}} \\approx {w_stable:.2f}$",
        xy=(T - 1, w_stable),
        xytext=(T * 0.55, min(w_stable + 0.15, 0.95)),
        fontsize=11, color=COLOR_GOOD_MEAN,
        arrowprops=dict(arrowstyle="->", color=COLOR_GOOD_MEAN, lw=1.2),
    )


def _dress_retr_axis(ax, T: int):
    ax.set_title(r"B)  Server Retrieval $m_k(t)$",
                 loc="left", fontweight="bold")
    ax.set_xlabel(r"Round $t$")
    ax.set_ylabel(r"$m_k(t)$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, T - 1)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="lower right", frameon=True, fancybox=False,
              edgecolor="k", framealpha=0.9)


def _save_fig(fig, out_path, dpi):
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        print(f"  Figure saved → {out_path}")
        print(f"  Figure saved → {out_path.with_suffix('.pdf')}")
    plt.close(fig)
