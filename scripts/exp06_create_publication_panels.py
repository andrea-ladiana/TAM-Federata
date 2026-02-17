
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication Panels Generator — Exp-06
--------------------------------------
Genera due pannelli per l'articolo:
1. Pannello patologico: confronto w=0 (troppo solido) vs w=1 (troppo plastico)
2. Pannello buono: caso adattivo con entropy_damped

Ogni pannello è in formato orizzontale con figure quadrate e palette publication-ready.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from dataclasses import asdict

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
from src.unsup.config import HyperParams, SpectralParams, PropagationParams

try:
    from src.mixing.scheduler import make_schedule
    from src.mixing.pipeline_core import run_seed_synth
    from src.mixing.reporting import collect_round_metrics
    from src.mixing.hopfield_hooks import load_magnetization_matrix_from_run
    from src.mixing.io import ensure_dir, write_json
except Exception:
    from src.exp06_single.scheduler import make_schedule
    from src.exp06_single.pipeline_core import run_seed_synth
    from src.exp06_single.reporting import collect_round_metrics
    from src.exp06_single.hopfield_hooks import load_magnetization_matrix_from_run
    from src.exp06_single.io import ensure_dir, write_json

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
        description="Publication Panels Generator — Exp-06",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Output
    p.add_argument("--outdir", type=str, default="out_06/publication_panels",
                   help="Cartella di output per i pannelli")
    
    # Problem scale
    p.add_argument("--L", type=int, default=3, help="Numero client/layer")
    p.add_argument("--K", type=int, default=3, help="Numero archetipi")
    p.add_argument("--N", type=int, default=300, help="Dimensione dei pattern")
    p.add_argument("--rounds", type=int, default=24, help="Numero di round (T)")
    p.add_argument("--M-total", type=int, default=1200, help="Numero totale di esempi")
    p.add_argument("--r-ex", type=float, default=0.8, help="Correlazione media campione/archetipo")
    
    # Propagazione / Spettro
    p.add_argument("--prop-iters", type=int, default=30, help="Iterazioni propagate_J")
    p.add_argument("--prop-eps", type=float, default=1e-2, help="Parametro eps per propagate_J")
    p.add_argument("--tau", type=float, default=0.05, help="Cut sugli autovalori di J_KS")
    p.add_argument("--rho", type=float, default=0.1, help="Soglia allineamento spettrale")
    p.add_argument("--qthr", type=float, default=1.0, help="Pruning overlap mutuo")
    p.add_argument("--keff-method", type=str, default="shuffle", 
                   choices=("shuffle", "mp"), help="Metodo K_eff")
    p.add_argument("--ema-alpha", type=float, default=0.0, help="EMA su J_unsup")
    
    # Schedule (cyclic)
    p.add_argument("--schedule", type=str, default="cyclic",
                   choices=("cyclic", "piecewise_dirichlet", "random_walk"),
                   help="Tipo di mixing-schedule")
    p.add_argument("--period", type=int, default=24, help="[cyclic] Periodo in round")
    p.add_argument("--gamma", type=float, default=3.0, help="[cyclic] Ampiezza logits")
    p.add_argument("--temp", type=float, default=1.2, help="[cyclic] Temperatura softmax")
    p.add_argument("--center-mix", type=float, default=0.0, help="[cyclic] Blend con uniforme")
    # piecewise_dirichlet
    p.add_argument("--block", type=int, default=4, help="[piecewise_dirichlet] Lunghezza blocco")
    p.add_argument("--alpha", type=float, default=1.0, help="[piecewise_dirichlet] Parametro Dirichlet")
    # random_walk
    p.add_argument("--step-sigma", type=float, default=0.7, help="[random_walk] Deviazione passo logits")
    p.add_argument("--tv-max", type=float, default=0.35, help="[random_walk] TV step-wise massima")
    
    # Seeds
    p.add_argument("--seed-base", type=int, default=12345, help="Seed base")
    p.add_argument("--seed", type=int, default=0, help="Seed per le simulazioni")
    
    # Adaptive w parameters (for entropy_damped)
    p.add_argument("--w-init", type=float, default=0.5, help="Valore iniziale w")
    p.add_argument("--w-min", type=float, default=0.10, help="Limite inferiore per w")
    p.add_argument("--w-max", type=float, default=0.90, help="Limite superiore per w")
    p.add_argument("--alpha-w", type=float, default=0.6, help="Smoothing sul controllo di w")
    p.add_argument("--damping-mode", type=str, default="adaptive_ema",
                   choices=("ema", "rate_limit", "momentum", "adaptive_ema"),
                   help="Damping strategy per entropy_damped")
    p.add_argument("--alpha-ema", type=float, default=0.3, help="EMA smoothing factor")
    p.add_argument("--max-delta-w", type=float, default=0.15, help="Max Δw per round")
    p.add_argument("--momentum-coeff", type=float, default=0.7, help="Momentum coefficient")
    p.add_argument("--adaptive-alpha", action="store_true", default=True,
                   help="Enable adaptive alpha modulation")
    p.add_argument("--alpha-min", type=float, default=0.1, help="Minimum alpha")
    p.add_argument("--alpha-max", type=float, default=0.5, help="Maximum alpha")
    # Per-run JSON overrides (optional). If present, values in the JSON
    # will override the CLI/base arguments for that specific run.
    p.add_argument("--config-w0", type=str, default=None,
                   help="Path to JSON file with overrides for the w=0 run")
    p.add_argument("--config-w1", type=str, default=None,
                   help="Path to JSON file with overrides for the w=1 run")
    p.add_argument("--config-adaptive", type=str, default=None,
                   help="Path to JSON file with overrides for the adaptive run")
    
    return p

# ---------------------------------------------------------------------
# Hyperparameters (da argparse)
# ---------------------------------------------------------------------
def make_hyperparams_from_args(args: argparse.Namespace) -> HyperParams:
    """Crea HyperParams dagli argomenti CLI"""
    hp = HyperParams(
        L=int(args.L),
        K=int(args.K),
        N=int(args.N),
        n_batch=int(args.rounds),
        M_total=int(args.M_total),
        r_ex=float(args.r_ex),
        w=0.5,  # sarà sovrascritto per ogni run
        estimate_keff_method=str(args.keff_method),
        ema_alpha=float(args.ema_alpha),
        prop=PropagationParams(iters=int(args.prop_iters), eps=float(args.prop_eps)),
        spec=SpectralParams(tau=float(args.tau), rho=float(args.rho), qthr=float(args.qthr)),
    )
    hp.use_tqdm = False
    hp.seed_base = int(args.seed_base)
    return hp


def _load_json_overrides(path: Optional[str]) -> Optional[dict]:
    """Carica un file JSON di override e ritorna il dict, oppure None."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Override JSON not found: {path}")
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def _merge_args(base_args: argparse.Namespace, overrides: Optional[dict]) -> argparse.Namespace:
    """Merge base argparse.Namespace with overrides dict (shallow). Returns a new Namespace.

    Only keys present in overrides will replace values in the returned namespace.
    """
    if overrides is None:
        return base_args
    merged = argparse.Namespace(**vars(base_args))
    for k, v in overrides.items():
        # Only set attributes that exist or are new keys (flexible)
        setattr(merged, k, v)
    return merged

def make_schedule_from_args(hp: HyperParams, args: argparse.Namespace) -> np.ndarray:
    """Crea la schedule dagli argomenti CLI"""
    rng = np.random.default_rng(hp.seed_base)
    kind = str(args.schedule)
    
    if kind == "cyclic":
        return make_schedule(
            hp, kind="cyclic", rng=rng,
            period=int(args.period),
            gamma=float(args.gamma),
            temp=float(args.temp),
            center_mix=float(args.center_mix),
        )
    elif kind == "piecewise_dirichlet":
        return make_schedule(
            hp, kind="piecewise_dirichlet", rng=rng,
            block=int(args.block),
            alpha=float(args.alpha),
        )
    elif kind == "random_walk":
        return make_schedule(
            hp, kind="random_walk", rng=rng,
            step_sigma=float(args.step_sigma),
            tv_max=float(args.tv_max),
        )
    else:
        raise ValueError(f"Schedule '{kind}' non riconosciuta.")

# ---------------------------------------------------------------------
# Simplex embedding utilities (legacy style, no axes)
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
    
    # Plot ground truth trajectory as continuous line (con label per legenda)
    ax.plot(xy_true[:, 0], xy_true[:, 1], 
           color='gray', linestyle='-', linewidth=2.0, 
           alpha=0.7, zorder=2, label=r'$\pi_t$ (true)')
    
    # Plot inferred trajectory as continuous line (con label per legenda)
    ax.plot(xy_hat[:, 0], xy_hat[:, 1],
           color=COLORS['adaptive_w'], linestyle='--', linewidth=2.0,
           alpha=0.9, zorder=3, label=r'$\hat{\pi}_t$ (inferred)')
    
    # Plot points and arrows for each round
    for t in range(T):
        # Punto true (ground truth)
        ax.scatter(xy_true[t, 0], xy_true[t, 1], 
                  s=42, color=colors[t], edgecolor='white', 
                  linewidth=0.8, zorder=4)
        
        # Punto hat (inferred)
        ax.scatter(xy_hat[t, 0], xy_hat[t, 1],
                  s=28, color=colors[t], edgecolor='none', 
                  zorder=5)
        
        # Freccia da true(t) -> hat(t)
        ax.annotate("", xy=(xy_hat[t, 0], xy_hat[t, 1]), 
                   xytext=(xy_true[t, 0], xy_true[t, 1]),
                   arrowprops=dict(arrowstyle="->", lw=1.0, color=colors[t], alpha=0.6))
    
    # Add colorbar to show temporal progression (usando ScalarMappable)
    import matplotlib.colors as mcolors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=T-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Round $t$', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    # Labels at vertices
    offset = 0.08
    ax.text(v0[0] - offset, v0[1] - offset, r'$\xi^1$', 
           ha='right', va='top', fontsize=12, weight='bold')
    ax.text(v1[0] + offset, v1[1] - offset, r'$\xi^2$',
           ha='left', va='top', fontsize=12, weight='bold')
    ax.text(v2[0], v2[1] + offset, r'$\xi^3$',
           ha='center', va='bottom', fontsize=12, weight='bold')
    
    # Remove axes
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add legend (mostra le linee true/hat, non i punti scatter)
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
    
    # Plot magnetizations
    colors_m = [COLORS['archetype1'], COLORS['archetype2'], COLORS['archetype3']]
    labels_m = [r'$m_1(t)$', r'$m_2(t)$', r'$m_3(t)$']
    
    for k in range(K):
        ax.plot(rounds, M[k, :], color=colors_m[k], linewidth=2.0,
               label=labels_m[k], alpha=0.9)
    
    ax.set_xlabel('Round $t$', fontsize=11)
    if show_ylabel:
        ax.set_ylabel('$m_k(t)$', fontsize=11)
    ax.set_xlim(0, T-1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
    
    # Create inset for pi_t
    # Position: [x0, y0, width, height] in axes coordinates - alzato leggermente
    ax_inset = ax.inset_axes([0.62, 0.08, 0.35, 0.30])
    
    colors_pi = [COLORS['pi1'], COLORS['pi2'], COLORS['pi3']]
    labels_pi = [r'$\pi_1(t)$', r'$\pi_2(t)$', r'$\pi_3(t)$']
    
    for k in range(K):
        ax_inset.plot(rounds, pi_true_seq[:, k], color=colors_pi[k],
                     linewidth=1.5, label=labels_pi[k], alpha=0.85)
    
    ax_inset.set_xlim(0, T-1)
    ax_inset.set_ylim(-0.05, 1.05)
    ax_inset.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', linewidth=0.5)
    ax_inset.tick_params(labelsize=7)
    ax_inset.legend(loc='upper right', frameon=True, framealpha=0.9, 
                   edgecolor='gray', fontsize=7)

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
# Run simulation
# ---------------------------------------------------------------------
def run_simulation(
    hp: HyperParams,
    pis: np.ndarray,
    outdir: Path,
    w_policy: str,
    w_value: float = 0.5,
    seed: int = 0,
    args: Optional[argparse.Namespace] = None,
) -> Dict[str, Any]:
    """
    Esegue una simulazione e salva i risultati.
    
    Returns
    -------
    dict con:
        - pi_true_seq: (T, K)
        - pi_hat_seq: (T, K)
        - M: (K, T)
        - w_series: (T,) se disponibile
    """
    hp_copy = HyperParams(
        L=hp.L, K=hp.K, N=hp.N, n_batch=hp.n_batch, M_total=hp.M_total,
        r_ex=hp.r_ex, w=w_value, estimate_keff_method=hp.estimate_keff_method,
        ema_alpha=hp.ema_alpha, prop=hp.prop, spec=hp.spec,
    )
    hp_copy.use_tqdm = False
    hp_copy.seed_base = hp.seed_base
    
    run_dir = ensure_dir(outdir)
    np.save(run_dir / "pis.npy", pis.astype(np.float32))
    
    # Extract adaptive w parameters from args (if provided)
    if args is not None:
        w_init = float(args.w_init) if w_policy != "fixed" else w_value
        w_min = float(args.w_min)
        w_max = float(args.w_max)
        alpha_w = float(args.alpha_w)
        damping_mode = str(args.damping_mode)
        alpha_ema = float(args.alpha_ema)
        max_delta_w = float(args.max_delta_w)
        momentum = float(args.momentum_coeff)
        adaptive_alpha = bool(args.adaptive_alpha)
        alpha_min = float(args.alpha_min)
        alpha_max = float(args.alpha_max)
    else:
        # Default values
        w_init = w_value if w_policy == "fixed" else 0.5
        w_min = 0.10
        w_max = 0.90
        alpha_w = 0.6
        damping_mode = "adaptive_ema"
        alpha_ema = 0.3
        max_delta_w = 0.15
        momentum = 0.7
        adaptive_alpha = True
        alpha_min = 0.1
        alpha_max = 0.5
    
    # Run simulation
    _ = run_seed_synth(
        hp=hp_copy,
        seed=seed,
        outdir=str(run_dir),
        pis=pis,
        xi_true=None,
        eval_hopfield_every=1,
        w_policy=w_policy,
        w_init=w_init,
        w_min=w_min,
        w_max=w_max,
        alpha_w=alpha_w,
        a_drift=0.5, b_mismatch=1.0,
        theta_low=0.05, theta_high=0.15,
        delta_up=0.10, delta_down=0.05,
        theta_mid=0.12, beta=10.0,
        lag_target=0.3, lag_window=8,
        kp=0.8, ki=0.0, kd=0.0,
        gate_drift_theta=0.1,
        damping_mode=damping_mode,
        alpha_ema=alpha_ema,
        max_delta_w=max_delta_w,
        momentum=momentum,
        adaptive_alpha=adaptive_alpha,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    
    # Load results
    items = collect_round_metrics(run_dir)
    pi_true_seq, pi_hat_seq = _load_pi_sequences(items)
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
# Panel creation
# ---------------------------------------------------------------------
def create_pathological_panel(
    data_w0: Dict[str, Any],
    data_w1: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Crea pannello patologico (4 subplot orizzontali).
    """
    fig = plt.figure(figsize=(16, 4))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3, hspace=0.3)
    
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
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Panel] Saved pathological panel: {output_path}")

def create_adaptive_panel(
    data_adaptive: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Crea pannello adattivo (3 subplot orizzontali).
    """
    fig = plt.figure(figsize=(14, 4.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35, hspace=0.3)
    
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
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Panel] Saved adaptive panel: {output_path}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = build_argparser().parse_args()
    
    print("[Publication Panels] Starting generation...")
    print(f"[Config] Rounds={args.rounds}, Period={args.period}, Schedule={args.schedule}")
    
    # Setup
    base_out = ensure_dir(Path(args.outdir))
    
    # Save base configuration
    write_json(base_out / "config_base.json", vars(args))
    
    # Run simulations (support per-run JSON overrides)
    # Load overrides if provided and merge with base args
    w0_over = _load_json_overrides(args.config_w0)
    w1_over = _load_json_overrides(args.config_w1)
    adaptive_over = _load_json_overrides(args.config_adaptive)

    run_args_w0 = _merge_args(args, w0_over)
    run_args_w1 = _merge_args(args, w1_over)
    run_args_ad = _merge_args(args, adaptive_over)

    # Create hp and pis for each run with merged args
    print("[1/3] Running w=0 simulation...")
    hp_w0 = make_hyperparams_from_args(run_args_w0)
    print(f"  [DEBUG] w0: M_total={hp_w0.M_total}, r_ex={hp_w0.r_ex}, center_mix={getattr(run_args_w0, 'center_mix', 'N/A')}, gamma={getattr(run_args_w0, 'gamma', 'N/A')}")
    pis_w0 = make_schedule_from_args(hp_w0, run_args_w0)
    w0_dir = ensure_dir(base_out / "w0")
    np.save(w0_dir / "pis.npy", pis_w0.astype(np.float32))
    write_json(w0_dir / "config.json", vars(run_args_w0))
    
    w0_value = float(getattr(run_args_w0, 'w_value', 0.0)) if hasattr(run_args_w0, 'w_value') else 0.0
    data_w0 = run_simulation(
        hp_w0, pis_w0, base_out / "w0", w_policy="fixed", w_value=w0_value,
        seed=run_args_w0.seed, args=run_args_w0
    )

    print("[2/3] Running w=1 simulation...")
    hp_w1 = make_hyperparams_from_args(run_args_w1)
    print(f"  [DEBUG] w1: M_total={hp_w1.M_total}, r_ex={hp_w1.r_ex}, center_mix={getattr(run_args_w1, 'center_mix', 'N/A')}, gamma={getattr(run_args_w1, 'gamma', 'N/A')}")
    pis_w1 = make_schedule_from_args(hp_w1, run_args_w1)
    w1_dir = ensure_dir(base_out / "w1")
    np.save(w1_dir / "pis.npy", pis_w1.astype(np.float32))
    write_json(w1_dir / "config.json", vars(run_args_w1))
    
    w1_value = float(getattr(run_args_w1, 'w_value', 1.0)) if hasattr(run_args_w1, 'w_value') else 1.0
    data_w1 = run_simulation(
        hp_w1, pis_w1, base_out / "w1", w_policy="fixed", w_value=w1_value,
        seed=run_args_w1.seed, args=run_args_w1
    )

    print("[3/3] Running adaptive simulation...")
    hp_ad = make_hyperparams_from_args(run_args_ad)
    print(f"  [DEBUG] adaptive: M_total={hp_ad.M_total}, r_ex={hp_ad.r_ex}, center_mix={getattr(run_args_ad, 'center_mix', 'N/A')}, gamma={getattr(run_args_ad, 'gamma', 'N/A')}")
    pis_ad = make_schedule_from_args(hp_ad, run_args_ad)
    ad_dir = ensure_dir(base_out / "adaptive")
    np.save(ad_dir / "pis.npy", pis_ad.astype(np.float32))
    write_json(ad_dir / "config.json", vars(run_args_ad))
    
    # adaptive run: allow override for initial w in config under key 'w_init' or 'w_value'
    adaptive_init = None
    if hasattr(run_args_ad, 'w_init'):
        adaptive_init = float(run_args_ad.w_init)
    elif hasattr(run_args_ad, 'w_value'):
        adaptive_init = float(run_args_ad.w_value)
    else:
        adaptive_init = 0.5
    data_adaptive = run_simulation(
        hp_ad, pis_ad, base_out / "adaptive", w_policy="entropy_damped", w_value=adaptive_init,
        seed=run_args_ad.seed, args=run_args_ad
    )
    
    # Create panels
    print("[Panel] Creating pathological panel...")
    create_pathological_panel(
        data_w0, data_w1,
        base_out / "panel_pathological.png"
    )
    
    print("[Panel] Creating adaptive panel...")
    create_adaptive_panel(
        data_adaptive,
        base_out / "panel_adaptive.png"
    )
    
    print("[Publication Panels] Complete! Output in:", base_out)

if __name__ == "__main__":
    main()
