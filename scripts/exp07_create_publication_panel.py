#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp07_create_publication_panel.py — Publication-ready panels for Exp-07

Crea pannelli grafici in stile pubblicazione per l'esperimento 07,
mantenendo solo i pannelli A, B e C disposti orizzontalmente.

Uso:
    python scripts/exp07_create_publication_panel.py --datadir out_07/baseline
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
import matplotlib as mpl

# Setup project paths
_THIS = Path(__file__).resolve()
ROOT = _THIS.parent.parent  # Go to project root

# Add project root to path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------
# Publication-ready color palette (colorblind-friendly, Okabe-Ito variant)
# ---------------------------------------------------------------------
COLORS = {
    # Refined Okabe–Ito based palette (high-contrast, colorblind-friendly)
    # Strategies - visually distinct, high-contrast colors
    'baseline': '#0072B2',      # Deep Blue - primary strategy
    'ema': '#E69F00',           # Orange - warm contrast
    'rate_limit': '#009E73',    # Bluish Green - cool accent
    'momentum': '#F0E442',      # Yellow - bright highlight
    'adaptive_ema': '#D55E00',  # Vermillion - strong accent
    # Reference lines - clear and distinct from strategy colors
    'K_old': '#7570B3',         # Purple - distinct from blues
    'K_total': '#1B9E77',       # Teal - distinct but professional
    't_intro': '#E7298A',       # Magenta - highly visible for introduction time
    'grid': '#CCCCCC',          # Light gray - matches panel2
    'text': '#333333',          # Dark gray - matches panel2
}


def get_strategy_color(strategy: str) -> str:
    """Return a color for a given strategy name.

    Tries exact lookup in COLORS, then substring matches against known keys
    (so names like 'baseline_v2' still map to 'baseline'), finally falls back
    to black.
    """
    if strategy in COLORS:
        return COLORS[strategy]

    # Common strategy keys to try (ordered)
    preferred_keys = ['baseline', 'adaptive_ema', 'ema', 'rate_limit', 'momentum']
    for key in preferred_keys:
        if key in strategy:
            return COLORS.get(key, '#000000')

    # Generic fallback: any COLORS key that is substring of strategy
    for key in COLORS:
        if key in strategy:
            return COLORS[key]

    return '#000000'

# Set publication-ready matplotlib defaults (matching panel2)
mpl.rcParams['font.size'] = 13
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['grid.linewidth'] = 0.6
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['patch.linewidth'] = 1.2

# ---------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create publication-ready panels for Exp-07",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    p.add_argument("--datadir", type=str, required=True,
                   help="Directory containing exp07 results (e.g., out_07/baseline)")
    p.add_argument("--dpi", type=int, default=300,
                   help="DPI for output images")
    p.add_argument("--output", type=str, default=None,
                   help="Output filename (default: panel_publication.png in datadir)")
    
    return p

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_exp07_data(datadir: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Carica i dati dell'esperimento 07 da results_detailed.csv e hyperparams.json
    
    Returns
    -------
    df : pd.DataFrame
        Dati dettagliati per round
    hp : dict
        Hyperparameters dell'esperimento
    """
    csv_path = datadir / "results_detailed.csv"
    hp_path = datadir / "hyperparams.json"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"results_detailed.csv not found in {datadir}")
    if not hp_path.exists():
        raise FileNotFoundError(f"hyperparams.json not found in {datadir}")
    
    df = pd.read_csv(csv_path)
    
    import json
    with open(hp_path, 'r') as f:
        hp = json.load(f)
    
    return df, hp

# ---------------------------------------------------------------------
# Panel A: K_eff detection
# ---------------------------------------------------------------------
def plot_panel_A_keff(
    ax: Axes,
    df: pd.DataFrame,
    hp: Dict[str, Any],
) -> None:
    """
    Panel A: Novelty Detection - K_eff vs Round
    """
    # Aggregate by strategy and round
    agg_df = df.groupby(["strategy", "round"]).agg({
        "keff": ["mean", "std"],
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = ["_".join(col).strip() if col[1] else col[0] for col in agg_df.columns]
    
    # Get unique strategies
    strategies = df["strategy"].unique()
    
    # Plot K_eff for each strategy
    for strategy in strategies:
        data = agg_df[agg_df["strategy"] == strategy]
        color = get_strategy_color(strategy)

        # Exclude the baseline strategy from the legend (user request)
        legend_label = strategy if strategy != 'baseline' else '_nolegend_'

        ax.plot(data["round"], data["keff_mean"], 
                linewidth=2.5, label=legend_label, color=color, alpha=1.0)
        ax.fill_between(data["round"], 
                        data["keff_mean"] - data["keff_std"], 
                        data["keff_mean"] + data["keff_std"], 
                        alpha=0.2, color=color)
    
    # Reference lines
    K_old = hp.get('K_old', 3)
    K_new = hp.get('K_new', 3)
    t_intro = hp.get('t_intro', 12)
    ramp_len = hp.get('ramp_len', 4)
    
    ax.axhline(K_old, color=COLORS['K_old'], linestyle=':', 
              linewidth=2.0, label=r'$K_\mathrm{old}$', alpha=0.85)
    ax.axhline(K_old + K_new, color=COLORS['K_total'], linestyle='--', 
              linewidth=2.0, label=r'$K_\mathrm{total}$', alpha=0.85)
    ax.axvline(t_intro, color=COLORS['t_intro'], linestyle='--', 
              alpha=0.7, linewidth=2.0, label=r'$t_\mathrm{intro}$')
    ax.axvspan(t_intro, t_intro + ramp_len, 
              alpha=0.12, color=COLORS['t_intro'])
    
    # Styling
    ax.set_xlabel('Round $t$')
    ax.set_ylabel(r'$K_\mathrm{eff}$')
    ax.set_title('A) Novelty Detection', loc='left', fontweight='bold')
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-', linewidth=0.6)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95)

# ---------------------------------------------------------------------
# Panel B: Retrieval performance
# ---------------------------------------------------------------------
def plot_panel_B_retrieval(
    ax: Axes,
    df: pd.DataFrame,
    hp: Dict[str, Any],
) -> None:
    """
    Panel B: Retrieval Performance - Old vs New Archetypes
    """
    # Aggregate by strategy and round
    agg_df = df.groupby(["strategy", "round"]).agg({
        "m_old": ["mean", "std"],
        "m_new": ["mean", "std"],
    }).reset_index()

    # Flatten column names
    agg_df.columns = ["_".join(col).strip() if col[1] else col[0] for col in agg_df.columns]

    # Get unique strategies
    strategies = df["strategy"].unique()

    # Ensure legend shows only two entries: old archetypes & new archetypes
    old_plotted = False
    new_plotted = False

    # Plot retrieval for each strategy
    for strategy in strategies:
        data = agg_df[agg_df["strategy"] == strategy]
        color = get_strategy_color(strategy)

        # Labels only for the first occurrence across strategies
        label_old = "old archetypes" if not old_plotted else '_nolegend_'
        label_new = "new archetypes" if not new_plotted else '_nolegend_'

        # Old archetypes (solid line)
        ax.plot(data["round"], data["m_old_mean"], 
                linewidth=2.5, label=label_old, 
                color=color, alpha=1.0, linestyle='-')
        ax.fill_between(data["round"], 
                        data["m_old_mean"] - data["m_old_std"], 
                        data["m_old_mean"] + data["m_old_std"], 
                        alpha=0.2, color=color)
        old_plotted = True

        # New archetypes (dashed line)
        ax.plot(data["round"], data["m_new_mean"], 
                linewidth=2.5, label=label_new, 
                color=color, alpha=0.85, linestyle='--')
        ax.fill_between(data["round"], 
                        data["m_new_mean"] - data["m_new_std"], 
                        data["m_new_mean"] + data["m_new_std"], 
                        alpha=0.15, color=color)
        new_plotted = True

    # Reference lines
    t_intro = hp.get('t_intro', 12)
    ramp_len = hp.get('ramp_len', 4)

    # Make the intro time more visible: stronger vertical line and more saturated span
    ax.axvline(t_intro, color=COLORS['t_intro'], linestyle='--', 
              alpha=0.7, linewidth=2.0)
    ax.axvspan(t_intro, t_intro + ramp_len, 
              alpha=0.12, color=COLORS['t_intro'])

    # Styling
    ax.set_xlabel('Round $t$')
    ax.set_ylabel('$m_k$')
    ax.set_title('B) Retrieval Performance', loc='left', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-', linewidth=0.6)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95)

# ---------------------------------------------------------------------
# Panel C: Spectral gap
# ---------------------------------------------------------------------
def plot_panel_C_gap(
    ax: Axes,
    df: pd.DataFrame,
    hp: Dict[str, Any],
) -> None:
    """
    Panel C: Spectral Gap at K_old Boundary
    """
    # Aggregate by strategy and round
    agg_df = df.groupby(["strategy", "round"]).agg({
        "gap": ["mean", "std"],
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = ["_".join(col).strip() if col[1] else col[0] for col in agg_df.columns]
    
    # Get unique strategies
    strategies = df["strategy"].unique()
    
    # Plot gap for each strategy
    for strategy in strategies:
        data = agg_df[agg_df["strategy"] == strategy]
        color = get_strategy_color(strategy)

        # Exclude baseline from legend per user request
        legend_label = strategy if strategy != 'baseline' else '_nolegend_'

        # Filter out NaN values
        valid_mask = ~data["gap_mean"].isna()
        if valid_mask.any():
            ax.plot(data[valid_mask]["round"], data[valid_mask]["gap_mean"], 
                   linewidth=2.5, label=legend_label, color=color, alpha=1.0)
            ax.fill_between(data[valid_mask]["round"], 
                           data[valid_mask]["gap_mean"] - data[valid_mask]["gap_std"], 
                           data[valid_mask]["gap_mean"] + data[valid_mask]["gap_std"], 
                           alpha=0.2, color=color)
    
    # Reference lines
    t_intro = hp.get('t_intro', 12)
    ramp_len = hp.get('ramp_len', 4)
    
    ax.axvline(t_intro, color=COLORS['t_intro'], linestyle='--', 
              alpha=0.7, linewidth=2.0)
    ax.axvspan(t_intro, t_intro + ramp_len, 
              alpha=0.12, color=COLORS['t_intro'])
    
    # Styling
    ax.set_xlabel('Round $t$')
    ax.set_ylabel(r'$\Delta\lambda/\lambda$')
    ax.set_title('C) Spectral Gap', loc='left', fontweight='bold')
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-', linewidth=0.6)
    # Legend removed for the rightmost panel (user requested)

# ---------------------------------------------------------------------
# Main panel creation
# ---------------------------------------------------------------------
def create_publication_panel(
    df: pd.DataFrame,
    hp: Dict[str, Any],
    output_path: Path,
    dpi: int = 300,
) -> None:
    """
    Crea il pannello pubblicazione con A, B, C orizzontali
    """
    # Create figure with 3 horizontal subplots
    # Increased height to accommodate larger fonts and titles
    fig = plt.figure(figsize=(18, 5.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35, hspace=0.3)
    
    # Panel A: K_eff detection
    ax1 = fig.add_subplot(gs[0, 0])
    plot_panel_A_keff(ax1, df, hp)
    
    # Panel B: Retrieval performance
    ax2 = fig.add_subplot(gs[0, 1])
    plot_panel_B_retrieval(ax2, df, hp)
    
    # Panel C: Spectral gap
    ax3 = fig.add_subplot(gs[0, 2])
    plot_panel_C_gap(ax3, df, hp)
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[Panel] Saved publication panel: {output_path}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = build_argparser().parse_args()
    
    datadir = Path(args.datadir)
    if not datadir.exists():
        raise FileNotFoundError(f"Data directory not found: {datadir}")
    
    print(f"[Create Publication Panel] Loading data from: {datadir}")
    
    # Load data
    df, hp = load_exp07_data(datadir)
    print(f"[Data] Loaded {len(df)} records, {len(df['strategy'].unique())} strategies, {len(df['seed'].unique())} seeds")
    
    # Determine output path
    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = datadir / "panel_publication.png"
    
    # Create panel
    print("[Panel] Creating publication panel (A, B, C)...")
    create_publication_panel(df, hp, output_path, dpi=args.dpi)
    
    print(f"[Create Publication Panel] Complete! Panel saved to: {output_path}")

if __name__ == "__main__":
    main()
