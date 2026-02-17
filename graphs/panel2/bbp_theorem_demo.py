"""
Regenerate BBP demo data with CORRECT sharpening.

Il vecchio file bbp_theorem_demo.py è vuoto (0 byte), quindi ricreo tutto da zero.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt

def sym_offdiag(J: np.ndarray) -> np.ndarray:
    """Azzera diagonale mantenendo simmetria."""
    J_new = J.copy()
    np.fill_diagonal(J_new, 0)
    return J_new


def generate_orthogonal_archetypes(N: int, K: int, seed: int = 2025) -> np.ndarray:
    """
    Genera K archetipi ortogonali in {±1}^N.
    """
    rng = np.random.RandomState(seed)
    xi = rng.choice([-1, 1], size=(K, N))
    return xi


def generate_spiked_data(
    xi: np.ndarray,
    alpha: np.ndarray,
    r_ex: float,
    M: int,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera M esempi esposti con exposure α.
    
    Returns
    -------
    eta : np.ndarray (M, N)
        Dati esposti
    J_unsup : np.ndarray (N, N)
        Matrice J = (η^T η) / M
    """
    K, N = xi.shape
    rng = np.random.RandomState(seed)
    
    # Sampling archetypes per esempio
    archetype_indices = rng.choice(K, size=M, p=alpha)
    
    # Exposure mask (Bernoulli con prob r_ex)
    chi = rng.binomial(1, r_ex, size=(M, N))
    
    # eta_i = chi_i ⊙ xi^{μ_i}
    eta = np.zeros((M, N))
    for i in range(M):
        mu = archetype_indices[i]
        eta[i] = chi[i] * xi[mu]
    
    # J = (η^T η) / M
    J_unsup = (eta.T @ eta) / M
    
    return eta, J_unsup


def compute_theoretical_predictions(
    alpha: np.ndarray,
    r_ex: float,
    N: int,
    M: int
) -> Dict:
    """BBP theoretical predictions."""
    sigma2 = 1.0 - r_ex**2
    q = N / M
    
    # Spike strength with factor N for unnormalized archetypes
    theta = (r_ex**2) * alpha * N
    kappa = theta / sigma2
    
    sqrt_q = np.sqrt(q)
    
    # λ_out(κ,q)
    lambda_out = np.zeros_like(kappa)
    for i, k in enumerate(kappa):
        if k > sqrt_q:
            lambda_out[i] = sigma2 * (1 + k) * (1 + q / k)
        else:
            lambda_out[i] = np.nan
    
    # γ(κ,q)
    gamma = np.zeros_like(kappa)
    for i, k in enumerate(kappa):
        if k > sqrt_q:
            gamma[i] = (1 - q / k**2) / (1 + q / k)
        else:
            gamma[i] = np.nan
    
    # MP bulk edges
    lambda_plus = sigma2 * (1 + sqrt_q)**2
    lambda_minus = sigma2 * (1 - sqrt_q)**2
    
    return {
        'kappa': kappa,
        'lambda_out': lambda_out,
        'gamma': gamma,
        'lambda_plus': lambda_plus,
        'lambda_minus': lambda_minus,
        'q': q,
        'sigma2': sigma2,
        'sqrt_q': sqrt_q
    }


def compute_empirical_spikes(
    J: np.ndarray,
    xi: np.ndarray,
    lambda_plus: float,
    K: int
) -> Dict:
    """Eigendecomp + overlap calculation."""
    N = J.shape[0]
    
    eigs, vecs = np.linalg.eigh(J)
    idx = np.argsort(eigs)[::-1]
    eigs = eigs[idx]
    vecs = vecs[:, idx]
    
    lambda_emp = eigs[:K]
    v_emp = vecs[:, :K]
    
    # Overlap with ORIGINAL normalized archetypes
    u_archetypes = xi / np.sqrt(N)
    
    overlap = np.zeros(K)
    for mu in range(K):
        overlaps_with_mu = np.abs(u_archetypes[mu] @ v_emp)
        best_idx = np.argmax(overlaps_with_mu)
        overlap[mu] = overlaps_with_mu[best_idx]**2
    
    return {
        'lambda_emp': lambda_emp,
        'v_emp': v_emp,
        'overlap': overlap,
        'all_eigs': eigs
    }


def logistic_sharpen(J: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Apply logistic sharpening: J_sharp = tanh(α * J)
    
    CORRECTED VERSION: element-wise sharpening that preserves structure.
    """
    return np.tanh(alpha * J)


def run_exposure_sweep(
    N: int = 400,
    K: int = 3,
    K_weak: int = 3,
    r_ex: float = 0.95,
    M_values: np.ndarray = None,
    alpha: np.ndarray = None,
    alpha_weak: np.ndarray = None,
    seed: int = 2025,
    sharp_alpha: float = 0.5,
    n_trials: int = 50
):
    """
    Main sweep over M values with multiple independent trials.
    
    Parameters
    ----------
    K : int
        Number of OVEREXPOSED archetypes (will emerge as spikes)
    K_weak : int
        Number of UNDEREXPOSED archetypes (will NOT cross BBP threshold)
    alpha : np.ndarray
        Exposure distribution for overexposed archetypes
    alpha_weak : np.ndarray
        Exposure distribution for underexposed archetypes (should be weak)
    n_trials : int
        Number of independent data generations for each M.
        Results will contain mean and std across trials.
    """
    
    if M_values is None:
        M_values = np.array([400, 1600, 6400])
    
    if alpha is None:
        alpha = np.array([0.5, 0.3, 0.2])
    
    if alpha_weak is None:
        # Default: VERY weak exposure that won't cross BBP threshold
        # We need κ = (r²/σ²) * α * N < √q
        # For q=1 (M=N=400), √q = 1
        # For r=0.95: r²/σ² = 0.9025/0.0975 ≈ 9.26
        # So we need α * N < √q / 9.26 → α < 1 / (9.26 * 400) ≈ 0.00027
        # Let's use α around 0.0001-0.0002 to be safely below threshold
        alpha_weak = np.array([0.0002, 0.00015, 0.0001])
    
    # Total number of archetypes
    K_total = K + K_weak
    
    # Combine exposures: strong first, then weak
    alpha_combined = np.concatenate([alpha, alpha_weak])
    
    # Normalize TOTAL exposure to 1
    alpha_combined = alpha_combined / alpha_combined.sum()
    
    # Split back into strong and weak after normalization
    alpha_strong_norm = alpha_combined[:K]
    alpha_weak_norm = alpha_combined[K:]
    
    # Generate ALL archetipi (FIXED across all trials)
    # First K are the strong ones, next K_weak are the weak ones
    xi = generate_orthogonal_archetypes(N, K_total, seed=seed)
    
    print(f"Generating data: N={N}, K_strong={K}, K_weak={K_weak}, K_total={K_total}, r_ex={r_ex}, n_trials={n_trials}")
    print(f"Exposure α_strong (normalized) = {alpha_strong_norm}")
    print(f"Exposure α_weak (normalized) = {alpha_weak_norm}")
    print(f"M values = {M_values}")
    print()
    
    results = {}
    
    for M in M_values:
        print(f"Processing M={M} ({n_trials} trials)...")
        
        # Accumulators for statistics across trials
        lambda_emp_trials = []
        overlap_trials = []
        lambda_emp_sharp_trials = []
        overlap_sharp_trials = []
        
        # Store one representative J and all_eigs for visualization
        J_unsup_repr = None
        J_sharp_repr = None
        all_eigs_repr = None
        
        for trial in range(n_trials):
            trial_seed = seed + M * 1000 + trial
            
            # Generate data with COMBINED exposure (strong + weak)
            eta, J_unsup = generate_spiked_data(xi, alpha_combined, r_ex, M, seed=trial_seed)
            
            # Apply sym_offdiag
            J_unsup = sym_offdiag(J_unsup)
            
            # Store first trial for visualization
            if trial == 0:
                J_unsup_repr = J_unsup.copy()
            
            # Theory for STRONG archetypes only (first K)
            theory = compute_theoretical_predictions(alpha_strong_norm, r_ex, N, M)
            
            # Theory for ALL archetypes (for diagnostic purposes)
            theory_all = compute_theoretical_predictions(alpha_combined, r_ex, N, M)
            
            # Empirical (pre-sharpening) - compute for STRONG archetypes only
            empirical = compute_empirical_spikes(J_unsup, xi[:K], theory['lambda_plus'], K)
            lambda_emp_trials.append(empirical['lambda_emp'])
            overlap_trials.append(empirical['overlap'])
            
            # Store all_eigs from trial 0 for Panel A
            # IMPORTANT: all_eigs contains spectrum for ALL K_total archetypes
            if trial == 0:
                all_eigs_repr = empirical['all_eigs']
                # Also store theory for ALL archetypes for Panel A diagnostic
                theory_all_repr = theory_all
            
            # SHARPENING
            J_sharp = logistic_sharpen(J_unsup, alpha=sharp_alpha)
            J_sharp = sym_offdiag(J_sharp)
            
            if trial == 0:
                J_sharp_repr = J_sharp.copy()
            
            # Empirical (post-sharpening) - compute for STRONG archetypes only
            empirical_sharp = compute_empirical_spikes(J_sharp, xi[:K], theory['lambda_plus'], K)
            lambda_emp_sharp_trials.append(empirical_sharp['lambda_emp'])
            overlap_sharp_trials.append(empirical_sharp['overlap'])
        
        # Compute statistics across trials
        lambda_emp_trials = np.array(lambda_emp_trials)  # (n_trials, K)
        overlap_trials = np.array(overlap_trials)  # (n_trials, K)
        lambda_emp_sharp_trials = np.array(lambda_emp_sharp_trials)
        overlap_sharp_trials = np.array(overlap_sharp_trials)
        
        results[M] = {
            'J_unsup': J_unsup_repr,  # Representative from trial 0
            'J_sharp': J_sharp_repr,
            'info': {
                'M': M,
                'exposure_theory': alpha_strong_norm,  # Only strong archetypes for plots B,C,D
                'exposure_theory_weak': alpha_weak_norm,  # For reference
                'K_strong': K,
                'K_weak': K_weak,
                'K_total': K_total,
                'r_ex': r_ex,
                'n_trials': n_trials
            },
            'theory': theory,  # Theory for STRONG archetypes only
            'theory_all': theory_all_repr,  # Theory for ALL archetypes (Panel A diagnostic)
            'empirical': {
                'lambda_emp_mean': lambda_emp_trials.mean(axis=0),
                'lambda_emp_std': lambda_emp_trials.std(axis=0),
                'overlap_mean': overlap_trials.mean(axis=0),
                'overlap_std': overlap_trials.std(axis=0),
                'all_eigs': all_eigs_repr,  # Full spectrum from trial 0 for Panel A
                'all_trials': {
                    'lambda_emp': lambda_emp_trials,
                    'overlap': overlap_trials
                }
            },
            'empirical_sharp': {
                'lambda_emp_mean': lambda_emp_sharp_trials.mean(axis=0),
                'lambda_emp_std': lambda_emp_sharp_trials.std(axis=0),
                'overlap_mean': overlap_sharp_trials.mean(axis=0),
                'overlap_std': overlap_sharp_trials.std(axis=0),
                'all_trials': {
                    'lambda_emp': lambda_emp_sharp_trials,
                    'overlap': overlap_sharp_trials
                }
            }
        }
    
    return {
        'results': results,
        'xi': xi,  # ALL archetypes (strong + weak)
        'N': N,
        'K': K,  # Number of STRONG archetypes
        'K_weak': K_weak,  # Number of WEAK archetypes
        'K_total': K_total,  # Total archetypes
        'r_ex': r_ex,
        'M_values': M_values,
        'n_trials': n_trials
    }


if __name__ == '__main__':
    print("="*70)
    print("BBP DEMO DATA REGENERATION (FIXED SHARPENING)")
    print("="*70)
    print()
    
    data = run_exposure_sweep()
    
    # Save NPZ data
    output_path = Path(__file__).parent / 'output' / 'bbp_demo_data.npz'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        results=data['results'],
        xi=data['xi'],
        N=data['N'],
        K=data['K'],
        K_weak=data['K_weak'],
        K_total=data['K_total'],
        r_ex=data['r_ex'],
        M_values=data['M_values'],
        allow_pickle=True
    )
    
    print()
    print("="*70)
    print(f"✅ Data saved to: {output_path}")
    print("="*70)
    
    # Create comprehensive JSON log with all numerical details
    print()
    print("Creating comprehensive JSON log...")
    
    json_log = {
        'metadata': {
            'N': int(data['N']),
            'K_strong': int(data['K']),
            'K_weak': int(data['K_weak']),
            'K_total': int(data['K_total']),
            'r_ex': float(data['r_ex']),
            'n_trials': int(data['n_trials']),
            'M_values': [int(m) for m in data['M_values']],
            'exposure_alpha_strong': data['results'][int(data['M_values'][0])]['info']['exposure_theory'].tolist(),
            'exposure_alpha_weak': data['results'][int(data['M_values'][0])]['info']['exposure_theory_weak'].tolist()
        },
        'results_by_M': {}
    }
    
    for M in data['M_values']:
        M_int = int(M)
        res = data['results'][M_int]
        theory = res['theory']
        emp = res['empirical']
        emp_sharp = res['empirical_sharp']
        
        json_log['results_by_M'][str(M_int)] = {
            'theory': {
                'q': float(theory['q']),
                'sigma2': float(theory['sigma2']),
                'sqrt_q': float(theory['sqrt_q']),
                'kappa': theory['kappa'].tolist(),
                'lambda_out': [float(x) if not np.isnan(x) else None for x in theory['lambda_out']],
                'gamma': [float(x) if not np.isnan(x) else None for x in theory['gamma']],
                'lambda_plus': float(theory['lambda_plus']),
                'lambda_minus': float(theory['lambda_minus'])
            },
            'empirical_pre_sharpening': {
                'lambda_emp_mean': emp['lambda_emp_mean'].tolist(),
                'lambda_emp_std': emp['lambda_emp_std'].tolist(),
                'overlap_mean': emp['overlap_mean'].tolist(),
                'overlap_std': emp['overlap_std'].tolist(),
                'all_trials': {
                    'lambda_emp': emp['all_trials']['lambda_emp'].tolist(),
                    'overlap': emp['all_trials']['overlap'].tolist()
                }
            },
            'empirical_post_sharpening': {
                'lambda_emp_mean': emp_sharp['lambda_emp_mean'].tolist(),
                'lambda_emp_std': emp_sharp['lambda_emp_std'].tolist(),
                'overlap_mean': emp_sharp['overlap_mean'].tolist(),
                'overlap_std': emp_sharp['overlap_std'].tolist(),
                'all_trials': {
                    'lambda_emp': emp_sharp['all_trials']['lambda_emp'].tolist(),
                    'overlap': emp_sharp['all_trials']['overlap'].tolist()
                }
            }
        }
    
    # Save JSON
    import json
    json_path = Path(__file__).parent / 'output' / 'bbp_demo_log.json'
    with open(json_path, 'w') as f:
        json.dump(json_log, f, indent=2)
    
    print(f"✅ JSON log saved to: {json_path}")
    print()
    
    # Quick validation (use first M actually present in the sweep)
    print()
    print("VALIDATION:")
    m0 = int(data['M_values'][0])
    res = data['results'][m0]
    print(f"M={m0}:")
    print(f"  Empirical overlap (pre):  mean={res['empirical']['overlap_mean']}, std={res['empirical']['overlap_std']}")
    print(f"  Empirical overlap (post): mean={res['empirical_sharp']['overlap_mean']}, std={res['empirical_sharp']['overlap_std']}")
    print(f"  Theory gamma:             {res['theory']['gamma']}")
    print()
    print(f"  J_sharp top 5 eigs: {np.linalg.eigvalsh(res['J_sharp'])[::-1][:5]}")
    print("="*70)
