"""
Fix empirical_sharp overlap calculation in existing data.

Il bug: compute_empirical_spikes su J_sharp calcola overlap con archetipi sbagliati.
Fix: Ricalcola overlap usando gli archetipi originali ξ.
"""
import numpy as np
from pathlib import Path
from scipy.linalg import eigh

def compute_empirical_spikes_corrected(J, xi, lambda_plus, K):
    """
    Calcola spike empirici con overlap corretto.
    
    Parameters
    ----------
    J : np.ndarray (N, N)
        Matrice di covarianza
    xi : np.ndarray (K, N)
        Archetipi ORIGINALI (non modificati da sharpening)
    lambda_plus : float
        MP bulk edge
    K : int
        Numero archetipi
    
    Returns
    -------
    dict con lambda_emp, v_emp, overlap, all_eigs
    """
    N = J.shape[0]
    
    # Eigendecomposition
    eigs, vecs = eigh(J)
    idx = np.argsort(eigs)[::-1]
    eigs = eigs[idx]
    vecs = vecs[:, idx]
    
    lambda_emp = eigs[:K]
    v_emp = vecs[:, :K]  # (N, K)
    
    # Calcola overlap con archetipi ORIGINALI normalizzati
    u_archetypes = xi / np.sqrt(N)  # (K, N)
    
    # Overlap: |⟨v_μ, u_μ⟩|²
    overlap = np.zeros(K)
    for mu in range(K):
        # Trova best match tra archetipo μ e autovettori
        overlaps_with_mu = np.abs(u_archetypes[mu] @ v_emp)  # (K,)
        best_idx = np.argmax(overlaps_with_mu)
        overlap[mu] = overlaps_with_mu[best_idx]**2
    
    return {
        'lambda_emp': lambda_emp,
        'v_emp': v_emp,
        'overlap': overlap,
        'all_eigs': eigs
    }


def fix_empirical_sharp_data(data_path: Path):
    """
    Ricarica dati e corregge empirical_sharp overlap.
    """
    print("="*70)
    print("FIX EMPIRICAL_SHARP OVERLAP BUG")
    print("="*70)
    print()
    
    # Load existing data
    data = np.load(data_path, allow_pickle=True)
    results = data['results'].item()
    xi = data['xi']  # Archetipi originali
    N = int(data['N'])
    K = int(data['K'])
    M_values = data['M_values']
    
    print(f"Loaded data: N={N}, K={K}")
    print(f"M values: {M_values}")
    print(f"Archetipi shape: {xi.shape}")
    print()
    
    # Fix overlap for each M
    fixed_count = 0
    for M in M_values:
        res = results[M]
        J_sharp = res['J_sharp']
        theory = res['theory']
        lambda_plus = theory['lambda_plus']
        
        # OLD overlap (buggy)
        old_overlap = res['empirical_sharp']['overlap']
        
        # Ricalcola con archetipi ORIGINALI
        empirical_sharp_fixed = compute_empirical_spikes_corrected(
            J_sharp, xi, lambda_plus, K
        )
        
        # Update
        results[M]['empirical_sharp'] = empirical_sharp_fixed
        
        new_overlap = empirical_sharp_fixed['overlap']
        
        print(f"M={M:4d}:")
        print(f"  OLD overlap: {old_overlap}")
        print(f"  NEW overlap: {new_overlap}")
        print(f"  Diff: {new_overlap - old_overlap}")
        print()
        
        fixed_count += 1
    
    # Save corrected data
    output_path = data_path.parent / 'bbp_demo_data_FIXED.npz'
    np.savez(
        output_path,
        results=results,
        xi=data['xi'],
        N=data['N'],
        K=data['K'],
        r_ex=data['r_ex'],
        M_values=data['M_values'],
        allow_pickle=True
    )
    
    print("="*70)
    print(f"✅ Fixed {fixed_count} entries")
    print(f"✅ Saved to: {output_path}")
    print()
    print("To use corrected data:")
    print(f"  1. Backup: mv {data_path.name} {data_path.stem}_BUGGY{data_path.suffix}")
    print(f"  2. Replace: mv {output_path.name} {data_path.name}")
    print("="*70)


if __name__ == '__main__':
    data_path = Path(__file__).parent / 'output' / 'bbp_demo_data.npz'
    fix_empirical_sharp_data(data_path)
