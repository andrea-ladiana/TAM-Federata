#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged Panel 06 Generator
-------------------------
Generates a single figure comparing three regimes:
1. Memory-driven (w=0)
2. Data-driven (w=1)
3. Adaptive

Loads data from out_06/publication_panels/ and uses a specific layout.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import seaborn as sns

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

# Try imports
try:
    from src.mixing.hopfield_hooks import load_magnetization_matrix_from_run
except ImportError:
    try:
        from src.exp06_single.hopfield_hooks import load_magnetization_matrix_from_run
    except ImportError:
        print("Could not import load_magnetization_matrix_from_run. Please check PYTHONPATH.")
        sys.exit(1)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BASE_DIR = _PROJECT_ROOT / "out_06" / "publication_panels"
DIRS = {
    "adaptive": BASE_DIR / "adaptive",
    "w0": BASE_DIR / "w0",
    "w1": BASE_DIR / "w1"
}

# Colors from user mockup / publication palette
# Using Seaborn colorblind palette to match panel2 style
COLORS = sns.color_palette("colorblind", 3)
COLOR_W = '#D55E00' # Vermillion for w(t)

# Publication-ready matplotlib defaults (copied from graphs/panel2/plot_panel2.py)
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

def load_data():
    data = {}
    
    # 1. Load Ground Truth (from adaptive run, should be same for all if generated together)
    pis_path = DIRS["adaptive"] / "pis.npy"
    if not pis_path.exists():
        raise FileNotFoundError(f"Could not find {pis_path}")
    data["pis"] = np.load(pis_path)
    
    # 2. Load Adaptive Weight w(t)
    w_path = DIRS["adaptive"] / "results" / "w_series.npy"
    if w_path.exists():
        data["w_t"] = np.load(w_path)
    else:
        print(f"Warning: {w_path} not found. Using placeholder.")
        data["w_t"] = np.zeros(len(data["pis"])) # Placeholder
        
    # 3. Load Magnetizations for all regimes
    for key, path in DIRS.items():
        print(f"Loading magnetization for {key} from {path}...")
        M = load_magnetization_matrix_from_run(path)
        if M is None:
            print(f"Warning: Could not load magnetization for {key}")
            # Create dummy if failed
            T = len(data["pis"])
            K = data["pis"].shape[1]
            M = np.zeros((K, T))
        data[f"M_{key}"] = M
        
    return data

def plot_merged_panel(data):
    pis = data["pis"] # (T, K)
    w_t = data["w_t"] # (T,)
    M_rigid = data["M_w0"] # (K, T)
    M_plastic = data["M_w1"] # (K, T)
    M_adapt = data["M_adaptive"] # (K, T)
    
    T, K = pis.shape
    t = np.arange(T)
    
    # Ensure w_t matches T (sometimes w_series might be T-1 or T+1 depending on implementation)
    if len(w_t) != T:
        # Resize or trim
        if len(w_t) > T:
            w_t = w_t[:T]
        else:
            w_t = np.pad(w_t, (0, T - len(w_t)), 'edge')

    # Ensure M matches T
    for M in [M_rigid, M_plastic, M_adapt]:
        if M.shape[1] != T:
             print(f"Warning: M shape {M.shape} does not match T={T}. Trimming/Padding.")
             # Handle mismatch if necessary
             pass

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)

    # --- TOP ROW ---

    # AX1: Input Distribution (Ground Truth)
    ax_input = fig.add_subplot(gs[0, 0:3])
    # Assuming K=3 for colors
    for k in range(min(K, 3)):
        ax_input.plot(t, pis[:, k], color=COLORS[k], lw=2.5, label=rf'$\pi_{{{k+1}}}(t)$')
    
    ax_input.set_title('A) Input Data Distribution (Ground Truth)', loc='left', fontweight='bold')
    ax_input.set_ylabel('Cue Strength')
    ax_input.set_ylim(-0.05, 1.05)
    # Grid is handled by rcParams now, but we can enforce it
    ax_input.grid(True, linestyle='--', alpha=0.3)
    ax_input.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='k', framealpha=0.9)
    ax_input.set_xlim(0, T-1)

    # AX2: Adaptive Weight
    ax_weight = fig.add_subplot(gs[0, 3:6])
    ax_weight.plot(t, w_t, color=COLOR_W, lw=2.5)
    ax_weight.fill_between(t, 0, w_t, color=COLOR_W, alpha=0.15)
    ax_weight.set_title(r'B) Adaptive Schedule $w(t)$', loc='left', fontweight='bold')
    ax_weight.set_ylabel(r'$w(t)$')
    ax_weight.set_ylim(-0.05, 1.05)
    ax_weight.set_xlabel(r'Round $t$')
    ax_weight.grid(True, linestyle='--', alpha=0.3)
    ax_weight.set_xlim(0, T-1)
    
    # Annotations (Optional, might need adjustment based on real data)
    # Finding peaks in w_t to annotate
    peaks = np.where(w_t > 0.8)[0]
    if len(peaks) > 0:
        peak_idx = peaks[len(peaks)//2] # Middle of a peak
        ax_weight.annotate('High Plasticity\n(Novelty Detected)', xy=(peak_idx, w_t[peak_idx]), xytext=(9, 0.8),
                     arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=9)
    
    lows = np.where(w_t < 0.3)[0]
    if len(lows) > 0:
        low_idx = lows[len(lows)//2]
        ax_weight.annotate('Memory Retention\n(Stable Phase)', xy=(low_idx, w_t[low_idx]), xytext=(16, 0.3),
                     arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=9, ha='center')


    # --- BOTTOM ROW ---

    # AX3: w=0 (Pathological - Rigid)
    ax_rigid = fig.add_subplot(gs[1, 0:2])
    for k in range(min(K, 3)):
        ax_rigid.plot(t, M_rigid[k, :T], color=COLORS[k], lw=2.5)
        
    ax_rigid.set_title(r'C) Memory-Driven ($w=0$)', loc='left', fontweight='bold')
    ax_rigid.set_ylabel(r'$m_k(t)$')
    ax_rigid.set_ylim(-0.05, 1.05)
    ax_rigid.set_xlabel(r'Round $t$')
    ax_rigid.grid(True, linestyle='--', alpha=0.3)
    ax_rigid.set_xlim(0, T-1)
    # Removed FAILURE text overlay

    # AX4: w=1 (Pathological - Plastic)
    ax_plastic = fig.add_subplot(gs[1, 2:4], sharey=ax_rigid)
    for k in range(min(K, 3)):
        ax_plastic.plot(t, M_plastic[k, :T], color=COLORS[k], lw=2.5)
        
    ax_plastic.set_title(r'D) Data-Driven ($w=1$)', loc='left', fontweight='bold')
    ax_plastic.set_xlabel(r'Round $t$')
    ax_plastic.grid(True, linestyle='--', alpha=0.3)
    ax_plastic.set_xlim(0, T-1)
    plt.setp(ax_plastic.get_yticklabels(), visible=False)
    # Removed FAILURE text overlay

    # AX5: Adaptive (Success)
    ax_adapt = fig.add_subplot(gs[1, 4:6], sharey=ax_rigid)
    for k in range(min(K, 3)):
        ax_adapt.plot(t, M_adapt[k, :T], color=COLORS[k], lw=2.5, label=rf'$m_{{{k+1}}}$')
        
    ax_adapt.set_title('E) Adaptive Reconstruction', loc='left', fontweight='bold')
    ax_adapt.set_xlabel(r'Round $t$')
    ax_adapt.grid(True, linestyle='--', alpha=0.3)
    ax_adapt.set_xlim(0, T-1)
    plt.setp(ax_adapt.get_yticklabels(), visible=False)
    ax_adapt.legend(loc='center right', frameon=True, fancybox=False, edgecolor='k', framealpha=0.9)
    # Removed SUCCESS text overlay

    # Remove top and right spines for all axes for a cleaner look
    for ax in fig.get_axes():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Save
    out_path = BASE_DIR / "panel06_merged.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {out_path}")
    
    # Also save as PDF
    plt.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')

if __name__ == "__main__":
    print("Loading data...")
    try:
        data = load_data()
        print("Plotting...")
        plot_merged_panel(data)
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
