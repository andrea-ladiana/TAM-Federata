#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate Publication Panels — Exp-06
---------------------------------------
Rigenera i pannelli grafici usando i dati salvati dalle simulazioni,
senza rieseguire le simulazioni stesse.

Uso:
    python scripts/regenerate_panels.py --datadir out_06/publication_panels
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

# ---------------------------------------------------------------------
# Fix PYTHONPATH
# ---------------------------------------------------------------------
def _ensure_project_root_in_syspath() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, here.parent.parent, here.parent.parent.parent]:
        if (p / "src").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p
    root = here.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root

_PROJECT_ROOT = _ensure_project_root_in_syspath()

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
try:
    from src.mixing.reporting import collect_round_metrics
    from src.mixing.hopfield_hooks import load_magnetization_matrix_from_run
except Exception:
    from src.exp06_single.reporting import collect_round_metrics
    from src.exp06_single.hopfield_hooks import load_magnetization_matrix_from_run

# ---------------------------------------------------------------------
# Publication-ready color palette (colorblind-friendly)
# ---------------------------------------------------------------------
# Using Okabe-Ito palette variant
COLORS = {
    'archetype1': '#E69F00',  # Orange
    'archetype2': '#56B4E9',  # Sky Blue
    'archetype3': '#009E73',  # Bluish Green
    'pi1': '#CC7A00',         # Darker orange
    'pi2': '#3A8EBF',         # Darker sky blue
    'pi3': '#006B4E',         # Darker bluish green
    'adaptive_w': '#D55E00', # Vermillion
    'grid': '#CCCCCC',
    'text': '#333333',
}

# Set publication-ready matplotlib defaults
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['patch.linewidth'] = 1.0

# ---------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Regenerate Publication Panels from saved data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    p.add_argument("--datadir", type=str, default="out_06/publication_panels",
                   help="Directory containing saved simulation data (w0, w1, adaptive)")
    p.add_argument("--dpi", type=int, default=300,
                   help="DPI for output images")
    
    return p

# ---------------------------------------------------------------------
# Load data from saved runs
# ---------------------------------------------------------------------
def load_run_data(run_dir: Path) -> Dict[str, Any]:
    """
    Carica i dati di una run salvata.
    
    Returns
    -------
    dict con:
        - pi_true_seq: (T, K)
        - pi_hat_seq: (T, K)
        - M: (K, T)
        - w_series: (T,) se disponibile
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Load metrics
    items = collect_round_metrics(run_dir)
    pi_true_seq, pi_hat_seq = _load_pi_sequences(items)
    
    # Load magnetization
    M = load_magnetization_matrix_from_run(run_dir)
    
    # Load w_series if available
    w_series = None
    w_series_path = run_dir / "results" / "w_series.npy"
    if w_series_path.exists():
        w_series = np.load(w_series_path)
    
    return {
        'pi_true_seq': pi_true_seq,
        'pi_hat_seq': pi_hat_seq,
        'M': M,
        'w_series': w_series,
    }

def _load_pi_sequences(items: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Estrae (pi_true_seq, pi_hat_seq) dai metrics."""
    pi_true_seq, pi_hat_seq = [], []
    for it in items:
        if "pi_true" not in it:
            continue
        pt = np.asarray(it["pi_true"], dtype=float)
        ph_raw = it.get("pi_hat_retrieval", None)
        if ph_raw is None:
            ph_raw = it.get("pi_hat", None) or it.get("pi_hat_data", None)
        if ph_raw is None:
            continue
        ph = np.asarray(ph_raw, dtype=float)
        if pt.shape != ph.shape:
            continue
        pt = pt / (pt.sum() if pt.sum() > 0 else 1.0)
        ph = ph / (ph.sum() if ph.sum() > 0 else 1.0)
        pi_true_seq.append(pt)
        pi_hat_seq.append(ph)
    
    if not pi_true_seq:
        raise RuntimeError("No pi sequences found")
    
    return np.stack(pi_true_seq, axis=0), np.stack(pi_hat_seq, axis=0)

# ---------------------------------------------------------------------
# Simplex embedding utilities
# ---------------------------------------------------------------------
def barycentric_to_cartesian(coords: np.ndarray) -> np.ndarray:
    """
    Convert barycentric coordinates (K,) or (N, K) to 2D cartesian.
    For K=3: returns (x, y) in equilateral triangle.
    """
    coords = np.atleast_2d(coords)
    # Equilateral triangle vertices
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    vertices = np.array([v0, v1, v2])
    
    # coords @ vertices.T
    xy = coords @ vertices
    return xy

def plot_simplex_trajectories(
    ax: plt.Axes,
    pi_true_seq: np.ndarray,
    pi_hat_seq: np.ndarray,
    title: str = "",
) -> None:
    """
    Plotta traiettorie nel simplesso (legacy style, no axes).
    
    Parameters
    ----------
    ax : plt.Axes
    pi_true_seq : (T, K) ground truth mixing schedule
    pi_hat_seq : (T, K) inferred mixing schedule
    title : str
    """
    # Draw simplex triangle
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    vertices = np.array([v0, v1, v2])
    
    triangle = Polygon(vertices, fill=False, edgecolor=COLORS['text'], 
                      linewidth=1.5, zorder=1)
    ax.add_patch(triangle)
    
    # Convert trajectories to cartesian
    xy_true = barycentric_to_cartesian(pi_true_seq)
    xy_hat = barycentric_to_cartesian(pi_hat_seq)
    
    T = len(pi_true_seq)
    
    # --- Colori per round (colormap continua) ---
    cmap = plt.get_cmap('viridis')
    colors = [cmap(t / max(1, T - 1)) for t in range(T)]
    
    # Plot ground truth trajectory as continuous line
    ax.plot(xy_true[:, 0], xy_true[:, 1], 
           color='gray', linestyle='-', linewidth=2.0, 
           alpha=0.7, zorder=2, label=r'$\pi_t$ (true)')
    
    # Plot inferred trajectory as continuous line
    ax.plot(xy_hat[:, 0], xy_hat[:, 1],
           color=COLORS['adaptive_w'], linestyle='--', linewidth=2.0,
           alpha=0.9, zorder=3, label=r'$\hat{\pi}_t$ (inferred)')
    
    # Plot points and arrows for each round
    for t in range(T):
        # Punto true (ground truth) - STELLINA
        ax.scatter(xy_true[t, 0], xy_true[t, 1], 
                  s=80, color=colors[t], edgecolor='white', 
                  linewidth=0.8, zorder=4, marker='*')
        
        # Punto hat (inferred) - CERCHIO
        ax.scatter(xy_hat[t, 0], xy_hat[t, 1],
                  s=28, color=colors[t], edgecolor='none', 
                  zorder=5, marker='o')
        
        # Freccia da true(t) -> hat(t)
        ax.annotate("", xy=(xy_hat[t, 0], xy_hat[t, 1]), 
                   xytext=(xy_true[t, 0], xy_true[t, 1]),
                   arrowprops=dict(arrowstyle="->", lw=1.0, color=colors[t], alpha=0.6))
    
    # Add colorbar to show temporal progression
    import matplotlib.colors as mcolors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=T-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Round $t$', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    # Labels at vertices (scambiati ξ^2 e ξ^3)
    offset = 0.08
    ax.text(v0[0] - offset, v0[1] - offset, r'$\xi^1$', 
           ha='right', va='top', fontsize=12, weight='bold')
    ax.text(v1[0] + offset, v1[1] - offset, r'$\xi^3$',
           ha='left', va='top', fontsize=12, weight='bold')
    ax.text(v2[0], v2[1] + offset, r'$\xi^2$',
           ha='center', va='bottom', fontsize=12, weight='bold')
    
    # Remove axes but keep aspect equal and expand limits to match other subplots
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.9, edgecolor='gray')

# ---------------------------------------------------------------------
# Magnetization plot with inset
# ---------------------------------------------------------------------
def plot_magnetization_with_inset(
    ax: plt.Axes,
    M: np.ndarray,
    pi_true_seq: np.ndarray,
    title: str = "",
    show_ylabel: bool = True,
) -> None:
    """
    Plotta magnetizzazioni (K curve) con inserto per π_t.
    
    Parameters
    ----------
    ax : plt.Axes
    M : (K, T) magnetization matrix
    pi_true_seq : (T, K) ground truth mixing schedule
    title : str
    show_ylabel : bool
    """
    K, T = M.shape
    rounds = np.arange(T)
    
    # Plot magnetizations (scambiate etichette m_2 e m_3)
    colors_m = [COLORS['archetype1'], COLORS['archetype2'], COLORS['archetype3']]
    # keep labels mapped to plotted curves (m2 and m3 intentionally swapped in data)
    labels_m = [r'$m_1(t)$', r'$m_3(t)$', r'$m_2(t)$']
    
    for k in range(K):
        ax.plot(rounds, M[k, :], color=colors_m[k], linewidth=2.0,
               label=labels_m[k], alpha=0.9)
    
    ax.set_xlabel('Round $t$', fontsize=11)
    if show_ylabel:
        ax.set_ylabel('$m_k(t)$', fontsize=11)
    ax.set_xlim(0, T-1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', linewidth=0.5)
    # Reorder legend entries to show m1, m2, m3 while preserving the original color-to-curve mapping
    handles, lbls = ax.get_legend_handles_labels()
    desired_order = []
    for lab in [r'$m_1(t)$', r'$m_2(t)$', r'$m_3(t)$']:
        if lab in lbls:
            desired_order.append(lbls.index(lab))
    if desired_order:
        new_handles = [handles[i] for i in desired_order]
        new_labels = [lbls[i] for i in desired_order]
        ax.legend(new_handles, new_labels, loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
    else:
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
    
    # Create inset for pi_t
    ax_inset = ax.inset_axes([0.62, 0.08, 0.35, 0.30])
    
    colors_pi = [COLORS['pi1'], COLORS['pi2'], COLORS['pi3']]
    # keep pi labels mapped to plotted curves (pi2 and pi3 intentionally swapped in data)
    labels_pi = [r'$\pi_1(t)$', r'$\pi_3(t)$', r'$\pi_2(t)$']
    
    for k in range(K):
        ax_inset.plot(rounds, pi_true_seq[:, k], color=colors_pi[k],
                     linewidth=1.5, label=labels_pi[k], alpha=0.85)
    
    ax_inset.set_xlim(0, T-1)
    ax_inset.set_ylim(-0.05, 1.05)
    ax_inset.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', linewidth=0.5)
    ax_inset.tick_params(labelsize=7)
    # Reorder inset legend entries to show pi1, pi2, pi3 while preserving colors
    # Temporaneamente rimuoviamo la legenda nell'inset per chiarezza della figura.
    # Il blocco originale che ricava handle/labels e chiama ax_inset.legend è commentato
    # per mantenere la possibilità di riattivarlo in seguito.
    # ih, il = ax_inset.get_legend_handles_labels()
    # desired_pi_order = []
    # for lab in [r'$\pi_1(t)$', r'$\pi_2(t)$', r'$\pi_3(t)$']:
    #     if lab in il:
    #         desired_pi_order.append(il.index(lab))
    # if desired_pi_order:
    #     new_ih = [ih[i] for i in desired_pi_order]
    #     new_il = [il[i] for i in desired_pi_order]
    #     ax_inset.legend(new_ih, new_il, loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray', fontsize=7)
    # else:
    #     ax_inset.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray', fontsize=7)

# ---------------------------------------------------------------------
# Adaptive w plot
# ---------------------------------------------------------------------
def plot_adaptive_w(
    ax: plt.Axes,
    w_series: np.ndarray,
    title: str = "",
) -> None:
    """
    Plotta l'andamento di w_t nel tempo.
    
    Parameters
    ----------
    ax : plt.Axes
    w_series : (T,) adaptive w trajectory
    title : str
    """
    T = len(w_series)
    rounds = np.arange(T)
    
    ax.plot(rounds, w_series, color=COLORS['adaptive_w'], linewidth=2.0,
           label=r'$w(t)$', alpha=0.9)
    
    ax.set_xlabel('Round $t$', fontsize=11)
    ax.set_ylabel(r'$w(t)$', fontsize=11)
    ax.set_xlim(0, T-1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')

# ---------------------------------------------------------------------
# Panel creation
# ---------------------------------------------------------------------
def create_pathological_panel(
    data_w0: Dict[str, Any],
    data_w1: Dict[str, Any],
    output_path: Path,
    dpi: int = 300,
) -> None:
    """
    Crea pannello patologico (4 subplot orizzontali).
    """
    fig = plt.figure(figsize=(18, 3))
    gs = GridSpec(1, 4, figure=fig, wspace=0.4, hspace=0.3, 
                  width_ratios=[1.3, 1.0, 1.3, 1.0])
    
    # w=0: simplex
    ax1 = fig.add_subplot(gs[0, 0])
    plot_simplex_trajectories(
        ax1,
        data_w0['pi_true_seq'],
        data_w0['pi_hat_seq'],
    )
    
    # w=0: magnetization
    ax2 = fig.add_subplot(gs[0, 1])
    plot_magnetization_with_inset(
        ax2,
        data_w0['M'],
        data_w0['pi_true_seq'],
        show_ylabel=True,
    )
    
    # w=1: simplex
    ax3 = fig.add_subplot(gs[0, 2])
    plot_simplex_trajectories(
        ax3,
        data_w1['pi_true_seq'],
        data_w1['pi_hat_seq'],
    )
    
    # w=1: magnetization
    ax4 = fig.add_subplot(gs[0, 3])
    plot_magnetization_with_inset(
        ax4,
        data_w1['M'],
        data_w1['pi_true_seq'],
        show_ylabel=False,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[Panel] Saved pathological panel: {output_path}")

def create_adaptive_panel(
    data_adaptive: Dict[str, Any],
    output_path: Path,
    dpi: int = 300,
) -> None:
    """
    Crea pannello adattivo (3 subplot orizzontali).
    """
    fig = plt.figure(figsize=(16, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.4, hspace=0.3,
                  width_ratios=[1.3, 1.0, 1.0])
    
    # Simplex
    ax1 = fig.add_subplot(gs[0, 0])
    plot_simplex_trajectories(
        ax1,
        data_adaptive['pi_true_seq'],
        data_adaptive['pi_hat_seq'],
    )
    
    # Magnetization
    ax2 = fig.add_subplot(gs[0, 1])
    plot_magnetization_with_inset(
        ax2,
        data_adaptive['M'],
        data_adaptive['pi_true_seq'],
        show_ylabel=True,
    )
    
    # Adaptive w
    ax3 = fig.add_subplot(gs[0, 2])
    if data_adaptive['w_series'] is not None:
        plot_adaptive_w(
            ax3,
            data_adaptive['w_series'],
        )
    else:
        ax3.text(0.5, 0.5, 'w_series not available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.axis('off')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[Panel] Saved adaptive panel: {output_path}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = build_argparser().parse_args()
    
    datadir = Path(args.datadir)
    if not datadir.exists():
        raise FileNotFoundError(f"Data directory not found: {datadir}")
    
    print(f"[Regenerate Panels] Loading data from: {datadir}")
    
    # Load data from saved runs
    print("[1/3] Loading w=0 data...")
    data_w0 = load_run_data(datadir / "w0")
    
    print("[2/3] Loading w=1 data...")
    data_w1 = load_run_data(datadir / "w1")
    
    print("[3/3] Loading adaptive data...")
    data_adaptive = load_run_data(datadir / "adaptive")
    
    # Create panels
    print("[Panel] Creating pathological panel...")
    create_pathological_panel(
        data_w0, data_w1,
        datadir / "panel_pathological.png",
        dpi=args.dpi,
    )
    
    print("[Panel] Creating adaptive panel...")
    create_adaptive_panel(
        data_adaptive,
        datadir / "panel_adaptive.png",
        dpi=args.dpi,
    )
    
    print(f"[Regenerate Panels] Complete! Panels saved in: {datadir}")

if __name__ == "__main__":
    main()
