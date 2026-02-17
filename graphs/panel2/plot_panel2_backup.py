"""
PANEL 2: BBP Theorem Visualization

Generates 4 panels demonstrating:
A) Eigenspectrum vs MP bulk for different M
B) Empirical λ_out vs BBP theoretical formula
C) Eigenvector overlap vs theoretical γ(κ,q)
D) Sharpening effect on separation and alignment
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------
# Publication-ready color palette (colorblind-friendly, Okabe-Ito)
# ---------------------------------------------------------------------
COLORS = {
    'M_low': '#E69F00',      # Orange
    'M_mid': '#56B4E9',      # Sky Blue
    'M_high': '#009E73',     # Bluish Green
    'error_pre': '#F0E442',  # Yellow
    'error_post': '#D55E00', # Vermillion
    'grid': '#CCCCCC',
    'text': '#333333',
    'agreement': '#000000',  # Black for diagonal line
}

# Set publication-ready matplotlib defaults
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 14
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.linewidth'] = 1.0
matplotlib.rcParams['grid.linewidth'] = 0.5
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['patch.linewidth'] = 1.0


def load_data():
    """Carica risultati da bbp_theorem_demo.py"""
    # Try both absolute and relative paths
    script_dir = Path(__file__).parent
    data_path = script_dir / "output" / "bbp_demo_data.npz"
    
    if not data_path.exists():
        print(f"❌ File non trovato: {data_path}")
        print("   Esegui prima: python graphs/panel2/bbp_theorem_demo.py")
        return None
    
    data = np.load(data_path, allow_pickle=True)
    return data


def _array_like(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def extract_stat(section: dict, base_key: str, prefer_mean: bool = True) -> np.ndarray:
    """
    Estrae statistiche (K,) gestendo sia formato single-trial che multi-trial.
    Se prefer_mean=True e sono disponibili più prove, restituisce la media sui trial.
    """
    if prefer_mean and f"{base_key}_mean" in section:
        return _array_like(section[f"{base_key}_mean"])
    
    if base_key in section:
        arr = _array_like(section[base_key])
        if arr.ndim == 1:
            return arr
        if prefer_mean:
            return arr.mean(axis=0)
        return arr
    
    all_trials = section.get("all_trials")
    if isinstance(all_trials, dict) and base_key in all_trials:
        arr = np.asarray(all_trials[base_key], dtype=float)
        if prefer_mean:
            return arr.mean(axis=0)
        return arr
    
    raise KeyError(f"Key '{base_key}' not found in empirical section.")


def collect_across_M(results: dict, M_values, base_key: str, prefer_mean: bool = True,
                     use_sharp: bool = False) -> np.ndarray:
    """
    Restituisce matrice (len(M), K) per 'base_key' attraversando i valori di M.
    """
    rows = []
    for M in M_values:
        section = results[int(M)]['empirical_sharp' if use_sharp else 'empirical']
        vals = extract_stat(section, base_key, prefer_mean=prefer_mean)
        rows.append(vals)
    return np.vstack(rows)


def extract_rel_errors(section: dict, lambda_theory: np.ndarray) -> np.ndarray:
    """
    Restituisce matrice (n_samples, K) con errore relativo per gli autovalori.
    """
    lambda_theory = np.asarray(lambda_theory, dtype=float)
    if lambda_theory.ndim == 0:
        lambda_theory = lambda_theory.reshape(1)
    
    if 'all_trials' in section and isinstance(section['all_trials'], dict) \
            and 'lambda_emp' in section['all_trials']:
        lam = np.asarray(section['all_trials']['lambda_emp'], dtype=float)
    elif 'lambda_emp' in section:
        lam = np.asarray(section['lambda_emp'], dtype=float)
        if lam.ndim == 1:
            lam = lam[None, :]
    elif 'lambda_emp_mean' in section:
        lam = np.asarray(section['lambda_emp_mean'], dtype=float)
        if lam.ndim == 1:
            lam = lam[None, :]
    else:
        raise KeyError("lambda_emp not available for panel D")
    
    return (lam - lambda_theory) / lambda_theory


def compute_rel_error_profile(results: dict, M_values) -> dict:
    """
    Calcola statistiche sugli errori relativi pre/post sharpening.
    """
    mean_pre, mean_post = [], []
    min_pre, max_pre = [], []
    min_post, max_post = [], []
    
    for M in M_values:
        res = results[int(M)]
        lambda_theory = np.asarray(res['theory']['lambda_out'], dtype=float)
        
        rel_pre = extract_rel_errors(res['empirical'], lambda_theory)
        rel_post = extract_rel_errors(res['empirical_sharp'], lambda_theory)
        
        abs_pre_arch = np.nanmean(np.abs(rel_pre), axis=0)
        abs_post_arch = np.nanmean(np.abs(rel_post), axis=0)
        
        mean_pre.append(np.sqrt(np.nanmean(rel_pre**2)))
        mean_post.append(np.sqrt(np.nanmean(rel_post**2)))
        min_pre.append(np.nanmin(abs_pre_arch))
        max_pre.append(np.nanmax(abs_pre_arch))
        min_post.append(np.nanmin(abs_post_arch))
        max_post.append(np.nanmax(abs_post_arch))
    
    return {
        "M": np.asarray(M_values, dtype=float),
        "mean_pre": np.asarray(mean_pre, dtype=float),
        "mean_post": np.asarray(mean_post, dtype=float),
        "min_pre": np.asarray(min_pre, dtype=float),
        "max_pre": np.asarray(max_pre, dtype=float),
        "min_post": np.asarray(min_post, dtype=float),
        "max_post": np.asarray(max_post, dtype=float),
    }


def generate_summary(data) -> list[str]:
    """
    Generate quantitative summary text for the four panels.
    """
    results = data['results'].item()
    M_values = np.asarray(data['M_values'], dtype=float)
    K = int(data['K'])
    K_total = int(data.get('K_total', K))  # Backward compatibility
    summary_lines: list[str] = []
    
    M_min = int(np.min(M_values))
    M_max = int(np.max(M_values))
    
    # Panel A -----------------------------------------------------------------
    spike_counts = []
    weakest_gaps = []
    strongest_gaps = []
    for M in M_values:
        res = results[int(M)]
        lambda_plus = float(res['theory']['lambda_plus'])
        spikes = extract_stat(res['empirical'], 'lambda_emp', prefer_mean=True)
        gaps = spikes - lambda_plus
        above = gaps[gaps > 0]
        spike_counts.append(int(above.size))
        if above.size > 0:
            weakest_gaps.append(float(np.min(above)))
            strongest_gaps.append(float(np.max(above)))
    if spike_counts:
        min_spikes = min(spike_counts)
        max_spikes = max(spike_counts)
        min_gap = min(weakest_gaps) if weakest_gaps else float('nan')
        max_gap = max(strongest_gaps) if strongest_gaps else float('nan')
        line = (f"Panel A: For exposures M={M_min}-{M_max}, "
                f"{min_spikes}/{K} to {max_spikes}/{K} eigenvalues remain above the MP edge; "
                f"the weakest spike stays {min_gap:.2f} and the strongest {max_gap:.2f} units above lambda_plus.")
        summary_lines.append(line)
    
    # Panel B -----------------------------------------------------------------
    try:
        lambda_emp = collect_across_M(results, M_values, 'lambda_emp', prefer_mean=True)
    except KeyError:
        lambda_emp = collect_across_M(results, M_values, 'lambda_emp_mean', prefer_mean=True)
    lambda_theory = np.vstack([
        _array_like(results[int(M)]['theory']['lambda_out']) for M in M_values
    ])
    
    r2_list = []
    mae_list = []
    rel_mae_list = []
    for arch_idx in range(lambda_emp.shape[1]):
        theo = lambda_theory[:, arch_idx]
        emp = lambda_emp[:, arch_idx]
        valid_mask = np.isfinite(emp) & np.isfinite(theo)
        if np.count_nonzero(valid_mask) >= 2:
            theo_valid = theo[valid_mask]
            emp_valid = emp[valid_mask]
            if np.std(theo_valid) > 1e-9 and np.std(emp_valid) > 1e-9:
                corr = np.corrcoef(theo_valid, emp_valid)[0, 1]
                r2 = float(corr)**2
            else:
                r2 = float('nan')
            mae = float(np.mean(np.abs(emp_valid - theo_valid)))
            rel_mae = float(np.mean(np.abs(emp_valid - theo_valid) / np.abs(theo_valid)))
            r2_list.append(r2)
            mae_list.append(mae)
            rel_mae_list.append(rel_mae)
    if mae_list:
        finite_r2 = [r for r in r2_list if np.isfinite(r)]
        if finite_r2:
            r2_min = min(finite_r2)
            r2_max = max(finite_r2)
            r2_part = f"R^2 between {r2_min:.3f} and {r2_max:.3f}"
        else:
            r2_part = "high fidelity (R^2 not defined because values are constant)"
        mae_mean = np.mean(mae_list)
        rel_mae_mean = np.mean(rel_mae_list) * 100.0
        summary_lines.append(
            f"Panel B: Empirical lambda_out track the BBP prediction with {r2_part}; "
            f"the mean absolute error is {mae_mean:.2f} ({rel_mae_mean:.2f}% of theory)."
        )
    
    # Panel C -----------------------------------------------------------------
    overlap_emp = collect_across_M(results, M_values, 'overlap', prefer_mean=True)
    gamma_theory = np.vstack([
        _array_like(results[int(M)]['theory']['gamma']) for M in M_values
    ])
    overlap_gap = np.abs(overlap_emp - gamma_theory)
    max_gap = float(np.nanmax(overlap_gap))
    median_gap = float(np.nanmedian(overlap_gap))
    summary_lines.append(
        f"Panel C: Eigenvector overlaps stay within {max_gap:.4f} of gamma(kappa,q); "
        f"the median gap across exposures is {median_gap:.4f}."
    )
    
    # Panel D -----------------------------------------------------------------
    rel_stats = compute_rel_error_profile(results, M_values)
    rms_pre = rel_stats["mean_pre"]
    rms_post = rel_stats["mean_post"]
    ratio_vals = np.divide(rms_post, rms_pre, out=np.full_like(rms_post, np.nan),
                           where=np.abs(rms_pre) > 1e-12)
    mean_ratio = float(np.nanmean(ratio_vals))
    summary_lines.append(
        f"Panel D: Sharpening inflates the RMS eigenvalue error from {np.nanmean(rms_pre):.3f} "
        f"to {np.nanmean(rms_post):.3f} (approx {mean_ratio:.1f}x), indicating substantial compression of spike magnitudes."
    )
    
    return summary_lines


def plot_panel_A(ax, data):
    """
    Panel A: Eigenspectrum vs MP bulk - ALL top 20 eigenvalues
    
    Now shows:
    - Filled circles: eigenvalues ABOVE MP threshold (detected spikes)
    - Empty circles: eigenvalues BELOW MP threshold (undetected/weak)
    - Dashed lines: MP bulk edge λ₊
    - Dotted lines (if available): theoretical λ_out for WEAK archetypes
    
    Symlog scale to handle both large spikes and negative bulk.
    """
    results = data['results'].item()
    M_values = data['M_values']
    K = int(data['K'])
    K_total = int(data.get('K_total', K))  # Backward compatibility
    
    # Select 3 representative M values (low, mid, high)
    M_selected = [M_values[0], M_values[len(M_values)//2], M_values[-1]]
    colors = [COLORS['M_low'], COLORS['M_mid'], COLORS['M_high']]
    
    for i, M in enumerate(M_selected):
        res = results[M]
        theory = res['theory']
        emp = res['empirical']
        
        # IMPORTANT: use 'all_eigs' which contains ALL eigenvalues!
        all_eigenvalues = emp['all_eigs']  # Shape: (N,)
        
        # Top 20 eigenvalues in descending order
        sorted_eigs = np.sort(all_eigenvalues)[::-1]
        top_20 = sorted_eigs[:20]
        indices = np.arange(1, 21)
        
        # Separate above/below MP threshold
        lambda_plus = theory['lambda_plus']
        above_threshold = top_20 > lambda_plus
        
        # Plot FILLED markers for eigenvalues above threshold
        if np.any(above_threshold):
            ax.scatter(indices[above_threshold], top_20[above_threshold], 
                      s=80, alpha=0.8, color=colors[i], 
                      label=f'$M={M}$ ($q={theory["q"]:.1f}$)',
                      edgecolor='black', linewidth=0.8, marker='o', zorder=3)
        
        # Plot EMPTY markers for eigenvalues below threshold
        if np.any(~above_threshold):
            ax.scatter(indices[~above_threshold], top_20[~above_threshold], 
                      s=50, alpha=0.8,
                      edgecolor=colors[i], linewidth=2.0,
                      facecolors='none', marker='o', zorder=2+i)
        
        # MP bulk edge λ₊ - NO label in legend
        ax.axhline(lambda_plus, color=colors[i], 
                  linestyle='--', linewidth=1.5, alpha=0.6)
        
        # Optional: show theoretical λ_out for WEAK archetypes (if available)
        # This shows WHERE the weak spikes SHOULD be if they were detectable
        if 'theory_all' in res:
            theory_all = res['theory_all']
            lambda_out_all = theory_all['lambda_out']
            
            # Get weak archetypes (indices K onwards)
            lambda_out_weak = lambda_out_all[K:]
            
            # Plot thin horizontal dotted lines for weak theoretical positions
            for weak_idx, lambda_weak in enumerate(lambda_out_weak):
                if not np.isnan(lambda_weak) and lambda_weak > lambda_plus:
                    # Only plot if theoretically above threshold but empirically undetectable
                    ax.axhline(lambda_weak, color=colors[i], 
                              linestyle=':', linewidth=1.0, alpha=0.3)
    
    # Reference line at y=0
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=0)
    
    ax.set_xlabel('Eigenvalue rank', fontsize=11)
    ax.set_ylabel(r'$\lambda$ (eigenvalue)', fontsize=11)
    #ax.set_title(r'\textbf{A)} Eigenspectrum: Top 20 Eigenvalues vs MP Bulk', 
    #             fontsize=11)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', linewidth=0.5, which='both')
    
    # SYMLOG: log scale above linthresh, linear below
    # linthresh = 1.0 → everything under |λ| < 1 is linear (includes negative bulk)
    ax.set_yscale('symlog', linthresh=1.0)
    
    ax.set_xlim([0, 21])


def plot_panel_B(ax, data):
    """
    Panel B: Empirical λ_out vs BBP theoretical formula
    
    NEW: 3 subplots (one per M) to avoid vertical overlap of multiple M values.
    Each subplot shows all K archetypes horizontally distributed.
    """
    results = data['results'].item()
    M_values = data['M_values']
    K = int(data['K'])
    
    # Divide ax into 3 vertical subplots (one per M)
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=ax.get_subplotspec(), 
                                   wspace=0.3)
    
    # Symbols and colors for ARCHETYPES (not M)
    markers_base = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'h', '<', '>']
    markers = [markers_base[i % len(markers_base)] for i in range(K)]
    
    # Use tab10 colormap for archetypes
    colors_arch = matplotlib.colormaps['tab10'](np.linspace(0, 1, K))
    
    # Handles for shared legend
    legend_handles = []
    
    # For each M value (one subplot per M)
    for m_idx, M in enumerate(M_values):
        ax_sub = ax.get_figure().add_subplot(gs[m_idx])
        
        res = results[M]
        theory = res['theory']
        emp = res['empirical']
        
        lambda_emp_all = []
        lambda_emp_std_all = []
        lambda_theo_all = []
        arch_labels = []
        
        for i, M in enumerate(M_values):
            res = results[M]
            theory = res['theory']
            emp = res['empirical']
            
            # Only spikes above threshold
            if not np.isnan(theory['lambda_out'][arch_idx]):
                lambda_theo_arch.append(theory['lambda_out'][arch_idx])
                
                # Use mean/std if available, otherwise single value (backward compatibility)
                if isinstance(emp, dict) and 'lambda_emp_mean' in emp:
                    lambda_emp_arch.append(emp['lambda_emp_mean'][arch_idx])
                    lambda_emp_std_arch.append(emp['lambda_emp_std'][arch_idx])
                else:
                    # Old format (single trial)
                    lambda_emp_arch.append(emp['lambda_emp'][arch_idx])
                    lambda_emp_std_arch.append(0.0)
                
                M_labels.append(M)
                
                # Calculate horizontal offset to separate errorbars
                # Center jitter: from -(N-1)/2 to +(N-1)/2
                n_M = len(M_values)
                jitter_offset = (i - (n_M - 1) / 2.0) * jitter_scale
                x_plot = theory['lambda_out'][arch_idx] + jitter_offset
                
                # Plot point with errorbar only if variance exists
                if lambda_emp_std_arch[-1] > 1e-10:
                    eb = ax_sub.errorbar(x_plot, 
                                  lambda_emp_arch[-1],
                                  yerr=lambda_emp_std_arch[-1],
                                  marker=markers[i], markersize=8, 
                                  color=colors_M[i], alpha=0.9, 
                                  markeredgecolor='black', markeredgewidth=1.0,
                                  capsize=4, capthick=2.0,
                                  elinewidth=2.0,
                                  linestyle='none',
                                  label=f'$M={M}$')
                else:
                    # Simple scatter plot if no errorbars
                    ax_sub.scatter(x_plot, 
                                  lambda_emp_arch[-1],
                                  s=64, marker=markers[i],
                                  color=colors_M[i], alpha=0.9, 
                                  edgecolor='black', linewidth=1.0,
                                  label=f'$M={M}$', zorder=3)
                    eb = ax_sub.get_children()[-1]  # Get last artist for legend
                
                # Add errorbar handle to legend only from first subplot
                if arch_idx == 0:
                    legend_handles.append(eb)
        
        # Perfect agreement line
        if len(lambda_theo_all) > 0:
            lim = [min(min(lambda_theo_all), min(lambda_emp_all))*0.95,
                   max(max(lambda_theo_all), max(lambda_emp_all))*1.05]
            ax_sub.plot(lim, lim, '--', color=COLORS['agreement'], 
                       linewidth=2, alpha=0.5)
            ax_sub.set_xlim(lim)
            ax_sub.set_ylim(lim)
            
            # Calculate R²
            from scipy.stats import pearsonr
            lambda_emp_arr = np.array(lambda_emp_all)
            lambda_theo_arr = np.array(lambda_theo_all)
            corr_result = pearsonr(lambda_emp_arr, lambda_theo_arr)
            r_value = float(corr_result[0])
            r2 = r_value**2
            
            # Show R² in plot
            ax_sub.text(0.05, 0.95, f'$R^2={r2:.4f}$', 
                       transform=ax_sub.transAxes, 
                       fontsize=9, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='wheat', 
                                alpha=0.8, edgecolor='gray', linewidth=0.5))
        
        # Labels
        if m_idx == 0:
            ax_sub.set_ylabel(r'$\lambda_{\mathrm{out}}$ empirical', fontsize=10)
        ax_sub.set_xlabel(r'$\lambda_{\mathrm{out}}$ theory', fontsize=10)
        
        # Title with M and q
        ax_sub.set_title(f'$M={M}$ ($q={theory["q"]:.2f}$)', fontsize=10)
        
        ax_sub.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', linewidth=0.5)
        
        # Legend only in FIRST subplot
        if m_idx == 0:
            ax_sub.legend(handles=legend_handles, fontsize=7, loc='lower right', 
                         frameon=True, edgecolor='gray', framealpha=0.9)
    
    # Remove original ax
    ax.axis('off')


def plot_panel_C(ax, data):
    """
    Panel C: Eigenvector overlap vs theoretical γ(κ,q)
    
    3 subplots (one per archetype) with ONE shared legend.
    AUTOMATIC axis range to avoid excessive squashing.
    """
    results = data['results'].item()
    M_values = data['M_values']
    K = int(data['K'])
    
    # Divide ax into 3 vertical subplots
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=ax.get_subplotspec(), 
                                   wspace=0.3)
    
    # Symbols and colors for M (supports any number of M with cycling)
    markers_base = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'h', '<', '>']
    markers = [markers_base[i % len(markers_base)] for i in range(len(M_values))]
    
    # Use viridis colormap for better aesthetics (updated API)
    colors_M = matplotlib.colormaps['viridis'](np.linspace(0.2, 0.9, len(M_values)))
    
    # Handles for shared legend
    legend_handles = []
    
    # Calculate typical range for jitter (~1.5% of range to avoid overlaps)
    all_gamma_theo = []
    for M in M_values:
        res = results[M]
        theory = res['theory']
        for k in range(K):
            if not np.isnan(theory['gamma'][k]):
                all_gamma_theo.append(theory['gamma'][k])
    
    if len(all_gamma_theo) > 0:
        gamma_range = max(all_gamma_theo) - min(all_gamma_theo)
        jitter_scale = gamma_range * 0.000 if gamma_range > 1e-6 else 0.001  # 1.5% of total range
    else:
        jitter_scale = 0.001
    
    # For each archetype
    for arch_idx in range(K):
        ax_sub = ax.get_figure().add_subplot(gs[arch_idx])
        
        gamma_theo_arch = []
        overlap_emp_arch = []
        overlap_emp_std_arch = []
        q_vals = []
        
        for i, M in enumerate(M_values):
            res = results[M]
            theory = res['theory']
            emp = res['empirical']
            
            # Only spikes above threshold
            if not np.isnan(theory['gamma'][arch_idx]):
                gamma_theo_arch.append(theory['gamma'][arch_idx])
                
                # Use mean/std if available, otherwise single value (backward compatibility)
                if isinstance(emp, dict) and 'overlap_mean' in emp:
                    overlap_emp_arch.append(emp['overlap_mean'][arch_idx])
                    overlap_emp_std_arch.append(emp['overlap_std'][arch_idx])
                else:
                    # Old format (single trial)
                    overlap_emp_arch.append(emp['overlap'][arch_idx])
                    overlap_emp_std_arch.append(0.0)
                
                q_vals.append(theory['q'])
                
                # Calculate horizontal offset to separate errorbars
                # Center jitter: from -(N-1)/2 to +(N-1)/2
                n_M = len(M_values)
                jitter_offset = (i - (n_M - 1) / 2.0) * jitter_scale
                x_plot = theory['gamma'][arch_idx] + jitter_offset
                
                # Plot point with errorbar only if variance exists
                if overlap_emp_std_arch[-1] > 1e-10:
                    eb = ax_sub.errorbar(x_plot, 
                                  overlap_emp_arch[-1],
                                  yerr=overlap_emp_std_arch[-1],
                                  marker=markers[i], markersize=8, 
                                  color=colors_M[i], alpha=0.9, 
                                  markeredgecolor='black', markeredgewidth=1.0,
                                  capsize=4, capthick=2.0,
                                  elinewidth=2.0,
                                  linestyle='none',
                                  label=f'$M={M}$')
                else:
                    # Simple scatter plot if no errorbars
                    ax_sub.scatter(x_plot, 
                                  overlap_emp_arch[-1],
                                  s=64, marker=markers[i],
                                  color=colors_M[i], alpha=0.9, 
                                  edgecolor='black', linewidth=1.0,
                                  label=f'$M={M}$', zorder=3)
                    eb = ax_sub.get_children()[-1]  # Get last artist for legend
                
                # Add errorbar handle to legend only from first subplot
                if arch_idx == 0:
                    legend_handles.append(eb)
        
        # Perfect agreement line with AUTOMATIC limits
        if len(gamma_theo_arch) > 0:
            # Calculate automatic limits with 5% margin
            all_theo = np.array(gamma_theo_arch)
            all_emp = np.array(overlap_emp_arch)
            lim_min = min(np.min(all_theo), np.min(all_emp))
            lim_max = max(np.max(all_theo), np.max(all_emp))
            margin = (lim_max - lim_min) * 0.05
            lim = [lim_min - margin, lim_max + margin]
            
            ax_sub.plot(lim, lim, '--', color=COLORS['agreement'], 
                       linewidth=2, alpha=0.5)
            ax_sub.set_xlim(lim)
            ax_sub.set_ylim(lim)
            
            # Calculate mean theory-empirical gap
            gamma_theo_arr = np.array(gamma_theo_arch)
            overlap_emp_arr = np.array(overlap_emp_arch)
            mean_gap = (gamma_theo_arr - overlap_emp_arr).mean()
            
            # Show mean gap with publication-ready style
            ax_sub.text(0.05, 0.95, f'Gap$={mean_gap:.4f}$', 
                       transform=ax_sub.transAxes, 
                       fontsize=9, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='lightblue', 
                                alpha=0.8, edgecolor='gray', linewidth=0.5))
        
        # Labels
        if arch_idx == 0:
            ax_sub.set_ylabel(r'$|\langle v,u \rangle|^2$ empirical', fontsize=10)
        ax_sub.set_xlabel(r'$\gamma(\kappa,q)$ theory', fontsize=10)
        
        # Get exposure for this archetype
        alpha_val = results[M_values[0]]['info']['exposure_theory'][arch_idx]
        ax_sub.set_title(rf'$\xi_{{{arch_idx+1}}}\ (\alpha={alpha_val:.2f})$', fontsize=10)
        
        ax_sub.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', linewidth=0.5)
        
        # Legend only in FIRST subplot
        if arch_idx == 0:
            ax_sub.legend(fontsize=7, loc='lower right', frameon=True, 
                         edgecolor='gray', framealpha=0.9)
    
    # Remove original ax
    ax.axis('off')


def plot_panel_D(ax, data):
    """
    Panel D: Relative error of spiked eigenvalues w.r.t. BBP curve.
    Highlights how sharpening distorts estimation relative to theory.
    """
    results = data['results'].item()
    M_values = data['M_values']
    rel_stats = compute_rel_error_profile(results, M_values)
    
    M_plot = rel_stats["M"]
    mean_pre = rel_stats["mean_pre"]
    mean_post = rel_stats["mean_post"]
    min_pre = rel_stats["min_pre"]
    max_pre = rel_stats["max_pre"]
    min_post = rel_stats["min_post"]
    max_post = rel_stats["max_post"]
    
    # Fill between for pre/post sharpening range with publication-ready colors
    ax.fill_between(M_plot, min_pre, max_pre, 
                    color=COLORS['error_pre'], alpha=0.18,
                    label='Range pre-sharp (min-max)')
    ax.fill_between(M_plot, min_post, max_post, 
                    color=COLORS['error_post'], alpha=0.15,
                    label='Range post-sharp (min-max)')
    
    # Plot RMS lines with markers
    ax.plot(M_plot, mean_pre, color=COLORS['error_pre'], marker='o', 
            linewidth=2.5, markersize=8, markeredgecolor='black', 
            markeredgewidth=0.5, label='Rel. RMS error pre-sharp')
    ax.plot(M_plot, mean_post, color=COLORS['error_post'], marker='s', 
            linewidth=2.5, markersize=8, markeredgecolor='black', 
            markeredgewidth=0.5, label='Rel. RMS error post-sharp')
    
    ax.set_xlabel(r'$M$ (exposure)', fontsize=11)
    ax.set_ylabel(r'Relative error $|\lambda_{\mathrm{emp}}-\lambda_{\mathrm{theory}}| / \lambda_{\mathrm{theory}}$', fontsize=11)
    ax.set_xticks(M_plot)
    ax.set_xticklabels([str(int(m)) for m in M_plot])
    upper = max(np.nanmax(max_pre), np.nanmax(max_post))
    if not np.isfinite(upper) or upper <= 0:
        upper = 1e-3
    ax.set_ylim(0, upper * 1.15)
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', 
            linewidth=0.5, axis='y', zorder=1)
    #ax.set_title(r'\textbf{D)} Sharpening: relative error on eigenvalues', 
    #             fontsize=11)
    
    ratio_vals = np.divide(mean_post, mean_pre, out=np.full_like(mean_post, np.nan),
                           where=np.abs(mean_pre) > 1e-12)
    ratio = np.nanmean(ratio_vals)
    ratio_text = f'{ratio:.1f}' if np.isfinite(ratio) else 'n/a'
    ax.text(0.02, 0.92, f'RMS error $\\uparrow \\approx {ratio_text}\\times$ after sharpening',
            transform=ax.transAxes, fontsize=9, color=COLORS['error_post'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, 
                     edgecolor='gray', linewidth=0.5))
    
    ax.legend(loc='upper right', fontsize=9, frameon=True, 
             edgecolor='gray', framealpha=0.9)


def create_panel2_figure(data, output_path: Path):
    """
    Create complete figure with 4 panels in publication-ready style.
    """
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
    
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    
    plot_panel_A(ax_A, data)
    plot_panel_B(ax_B, data)
    plot_panel_C(ax_C, data)
    plot_panel_D(ax_D, data)
    
    # Suptitle with LaTeX formatting
    N = int(data['N'])
    K = int(data['K'])
    r_ex = float(data['r_ex'])
    #fig.suptitle(rf'\textbf{{Panel 2: BBP Theorem Experimental Validation}} ($N={N}$, $K={K}$, $r_{{\mathrm{{ex}}}}={r_ex}$)',
    #             fontsize=14, fontweight='bold')
    
    # Save with high DPI for publication
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Figure saved: {output_path}")
    
    # PNG version
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"[OK] PNG saved: {png_path}")
    
    plt.close()


def main():
    print("="*70)
    print("PANEL 2: BBP Theorem Visualization")
    print("="*70)
    print()
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    print("Data loaded successfully!")
    print(f"  N = {data['N']}")
    print(f"  K_strong = {data['K']}")
    if 'K_weak' in data:
        print(f"  K_weak = {data['K_weak']}")
        print(f"  K_total = {data['K_total']}")
    print(f"  r_ex = {data['r_ex']}")
    print(f"  M values = {data['M_values']}")
    print()
    
    # Create figure
    script_dir = Path(__file__).parent
    out_path = script_dir / "output" / "panel2_bbp_validation.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_panel2_figure(data, out_path)
    
    summary_lines = generate_summary(data)
    
    print()
    print("="*70)
    print("DETAILED SUMMARY")
    print("="*70)
    print()
    for line in summary_lines:
        print(f"- {line}")
    print()
    print("="*70)


if __name__ == "__main__":
    main()
