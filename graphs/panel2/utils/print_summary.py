"""
Print experimental parameters and theoretical predictions in tabular format.
"""

import numpy as np
from pathlib import Path
import json

def print_experiment_summary():
    """Print a comprehensive summary of the BBP experiment setup."""
    
    script_dir = Path(__file__).parent.parent  # Go up to panel2/
    json_path = script_dir / "output" / "bbp_demo_log.json"
    
    if not json_path.exists():
        print("❌ JSON log not found. Run bbp_theorem_demo.py first.")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    
    print()
    print("="*80)
    print("BBP EXPERIMENT CONFIGURATION AND THEORETICAL PREDICTIONS")
    print("="*80)
    print()
    
    # Configuration
    print("CONFIGURATION:")
    print("-" * 80)
    print(f"  Dimension N:                {metadata['N']}")
    print(f"  Strong archetypes K:        {metadata['K_strong']}")
    print(f"  Weak archetypes K_weak:     {metadata['K_weak']}")
    print(f"  Total archetypes K_total:   {metadata['K_total']}")
    print(f"  Channel quality r:          {metadata['r_ex']}")
    print(f"  Noise variance σ²:          {1 - metadata['r_ex']**2:.4f}")
    print(f"  Sample sizes M:             {metadata['M_values']}")
    print(f"  Number of trials:           {metadata['n_trials']}")
    print()
    
    # Exposure distribution
    print("EXPOSURE DISTRIBUTION:")
    print("-" * 80)
    
    alpha_strong = metadata['exposure_alpha_strong']
    alpha_weak = metadata['exposure_alpha_weak']
    
    print("  Strong archetypes (high exposure):")
    for i, a in enumerate(alpha_strong):
        print(f"    ξ{i+1}: α = {a:.6f} ({a*100:.2f}%)")
    
    print()
    print("  Weak archetypes (low exposure):")
    for i, a in enumerate(alpha_weak):
        idx = i + metadata['K_strong']
        print(f"    ξ{idx+1}: α = {a:.6f} ({a*100:.4f}%)")
    
    print()
    
    # Theoretical predictions for each M
    for M in metadata['M_values']:
        M_str = str(M)
        res = data['results_by_M'][M_str]
        theory = res['theory']
        
        print("="*80)
        print(f"THEORETICAL PREDICTIONS FOR M = {M}")
        print("="*80)
        
        q = theory['q']
        sqrt_q = theory['sqrt_q']
        lambda_plus = theory['lambda_plus']
        
        print()
        print(f"  Aspect ratio:           q = N/M = {q:.4f}")
        print(f"  BBP threshold:          √q = {sqrt_q:.4f}")
        print(f"  MP bulk edge:           λ₊ = {lambda_plus:.4f}")
        print()
        
        # Strong archetypes
        print("  STRONG ARCHETYPES (should cross BBP threshold):")
        print()
        print("    Arch.  |  Exposure α  |  Signal κ  |  λ_out theory  |  γ(κ,q) overlap  |  Detection")
        print("    " + "-" * 75)
        
        kappa = theory['kappa']
        lambda_out = theory['lambda_out']
        gamma = theory['gamma']
        
        for i in range(metadata['K_strong']):
            k = kappa[i]
            lam = lambda_out[i]
            g = gamma[i]
            a = alpha_strong[i]
            
            detectable = "✓ YES" if k > sqrt_q else "✗ NO"
            
            print(f"      ξ{i+1}   |   {a:.6f}   |  {k:8.2f}  |    {lam:8.2f}    |     {g:.6f}      |  {detectable}")
        
        print()
        
        # Weak archetypes - we need to load from NPZ for theory_all
        print("  WEAK ARCHETYPES (should NOT cross BBP threshold):")
        print()
        print("    Arch.  |  Exposure α  |  Signal κ  |  Detection")
        print("    " + "-" * 50)
        
        data_path = script_dir / "output" / "bbp_demo_data.npz"
        if data_path.exists():
            npz_data = np.load(data_path, allow_pickle=True)
            results = npz_data['results'].item()
            res_M = results[M]
            
            if 'theory_all' in res_M:
                theory_all = res_M['theory_all']
                kappa_all = theory_all['kappa']
                lambda_out_all = theory_all['lambda_out']
                
                for i in range(metadata['K_strong'], metadata['K_total']):
                    k = kappa_all[i]
                    lam = lambda_out_all[i]
                    a = alpha_weak[i - metadata['K_strong']]
                    
                    if np.isnan(lam):
                        detection = f"✓ NO (κ={k:.3f} < √q={sqrt_q:.3f})"
                    else:
                        detection = f"⚠ Marginal (κ={k:.3f}, λ={lam:.3f})"
                    
                    print(f"      ξ{i+1}   |   {a:.6f}   |  {k:8.3f}  |  {detection}")
        
        print()
    
    print("="*80)
    print()
    
    # Empirical validation summary
    print("EMPIRICAL VALIDATION SUMMARY:")
    print("-" * 80)
    
    total_detected = 0
    total_strong = 0
    
    for M in metadata['M_values']:
        M_str = str(M)
        res = data['results_by_M'][M_str]
        theory = res['theory']
        emp = res['empirical_pre_sharpening']
        
        lambda_plus = theory['lambda_plus']
        
        n_detected = 0
        for i in range(metadata['K_strong']):
            if emp['lambda_emp_mean'][i] > lambda_plus:
                n_detected += 1
                total_detected += 1
        
        total_strong += metadata['K_strong']
        
        print(f"  M={M}: {n_detected}/{metadata['K_strong']} strong archetypes detected")
    
    detection_rate = 100.0 * total_detected / total_strong
    print()
    print(f"  Overall detection rate: {total_detected}/{total_strong} ({detection_rate:.1f}%)")
    
    if detection_rate >= 99.0:
        print("  ✓ EXCELLENT: Theory and experiment match perfectly!")
    elif detection_rate >= 90.0:
        print("  ✓ GOOD: Strong agreement between theory and experiment.")
    else:
        print("  ⚠ WARNING: Some discrepancies detected.")
    
    print()
    print("="*80)


if __name__ == '__main__':
    print_experiment_summary()
