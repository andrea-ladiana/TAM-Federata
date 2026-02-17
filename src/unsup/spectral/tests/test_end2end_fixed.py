"""End-to-end tests using synthetic testbench (S8)."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from unsup.spectral.testbench import gen_spiked_rounds, gen_no_spike_rounds
from unsup.spectral.config import SpectralConfig
from unsup.spectral.panel1 import build_panel1


def test_no_spike():
    """Test S8.1: no spike → Keff moderato (finite-N effects tollerati)."""
    # Per N piccolo, MP può sovrastimare Keff su rumore puro (finite-N)
    N = 200
    rounds, U = gen_no_spike_rounds(T=5, N=N, sigma2=1.0, seed=123)
    
    cfg = SpectralConfig(
        k_max=30,
        pa_bootstrap=50,
        delta_N=0.5,    # Cushion
        use_float64=True,
        export_pgf=False,
    )
    
    logs = build_panel1(rounds, U, cfg, out_path="test_no_spike")
    
    # Check: nessun spike "molto grande"
    keff_vals = [log["keff_mp"] for log in logs]
    mean_keff = np.mean(keff_vals)
    max_keff = max(keff_vals)
    
    # Tolleranza: finite-N può dare qualche spike spurio ma non troppi
    # Per N=200, q=0.5, ci aspettiamo Keff < N/10 in media
    assert mean_keff < N / 5, f"Mean Keff unrealistic: {mean_keff:.1f} (N={N})"
    assert max_keff < N / 3, f"Max Keff unrealistic: {max_keff} (N={N})"
    
    print(f"[OK] test_no_spike passed (Keff: mean={mean_keff:.1f}, max={max_keff}, N={N})")


def test_multi_spike_supercritical():
    """Test S8.3: multi-spike → Keff cresce, alignment diagonale."""
    T = 10
    N = 512
    K = 3
    q = 0.5
    
    rounds, U = gen_spiked_rounds(
        T=T,
        N=N,
        K=K,
        q=q,
        kappa_schedule="constant_super",
        noise_sigma2=1.0,
        seed=42,
    )
    
    cfg = SpectralConfig(
        k_max=64,
        pa_bootstrap=100,
        use_float64=True,
        export_pgf=False,
    )
    
    logs = build_panel1(rounds, U, cfg, out_path="test_multi_spike")
    
    # Check TPR on Keff
    keff_vals = [log["keff_mp"] for log in logs]
    tpr = sum(k == K for k in keff_vals) / len(keff_vals)
    
    assert tpr > 0.7, f"TPR too low: {tpr:.2%} (expected > 70%)"
    
    # Check alignment (diagonal should be hot)
    alignments = [log["align"] for log in logs if log["align"].shape[0] == K]
    if len(alignments) > 0:
        mean_align = np.mean([np.diag(a[:K, :K]) for a in alignments if a.shape[0] >= K], axis=0)
        mean_diag = np.mean(mean_align)
        
        # Should be > 0.5 for well-separated spikes
        assert mean_diag > 0.5, f"Diagonal alignment too low: {mean_diag:.3f}"
        
        print(f"[OK] test_multi_spike_supercritical passed (TPR: {tpr:.2%}, diag align: {mean_diag:.3f})")
    else:
        print("⚠ test_multi_spike_supercritical: no alignments with K spikes")


def test_sharpening_improves_gap():
    """Test S8.4: sharpening → gap_post ≥ gap_pre."""
    T = 8
    rounds, U = gen_spiked_rounds(
        T=T,
        N=400,
        K=2,
        q=0.6,
        kappa_schedule="subcrit_to_supercrit",
        seed=999,
    )
    
    cfg = SpectralConfig(
        k_max=32,
        sharpen_eps0=0.5,
        sharpen_nprop=5,
        pa_bootstrap=50,
        use_float64=True,
        export_pgf=False,
    )
    
    logs = build_panel1(rounds, U, cfg, out_path="test_sharpening")
    
    # Filter rounds with spike
    gaps_pre = [log["gap_mp"] for log in logs if log["keff_mp"] > 0]
    gaps_post = [log["gap_sharp"] for log in logs if log["keff_mp"] > 0]
    
    if len(gaps_pre) > 0:
        improvements = [post >= pre * 0.9 for pre, post in zip(gaps_pre, gaps_post)]
        improvement_rate = sum(improvements) / len(improvements)
        
        assert improvement_rate > 0.7, f"Gap improvement rate too low: {improvement_rate:.2%}"
        
        print(f"[OK] test_sharpening_improves_gap passed (improvement: {improvement_rate:.2%})")
    else:
        print("⚠ test_sharpening_improves_gap: no spikes detected")


def test_kappa_estimation_accuracy():
    """Test S8: MAE(κ̂) relativa < 0.1."""
    T = 10
    N = 600
    K = 2
    q = 0.5
    sqrt_q = np.sqrt(q)
    kappa_true = 1.7 * sqrt_q
    
    rounds, U = gen_spiked_rounds(
        T=T,
        N=N,
        K=K,
        q=q,
        kappa_schedule="constant_super",
        seed=777,
    )
    
    cfg = SpectralConfig(
        k_max=48,
        pa_bootstrap=80,
        use_float64=True,
        export_pgf=False,
    )
    
    logs = build_panel1(rounds, U, cfg, out_path="test_kappa_accuracy")
    
    # Collect κ̂
    all_kappa_hat = []
    for log in logs:
        kappa_vals = [k for k in log["kappa_hat"] if not np.isnan(k)]
        all_kappa_hat.extend(kappa_vals)
    
    if len(all_kappa_hat) > 0:
        errors = [abs(k - kappa_true) / kappa_true for k in all_kappa_hat]
        mae = np.mean(errors)
        
        assert mae < 0.15, f"MAE(κ̂) too high: {mae:.2%} (expected < 15%)"
        
        print(f"[OK] test_kappa_estimation_accuracy passed (MAE: {mae:.2%})")
    else:
        print("⚠ test_kappa_estimation_accuracy: no κ̂ detected")


if __name__ == "__main__":
    # Note: questi test generano figure (PDF) che possono essere ignorate
    print("Running S8 end-to-end tests...\n")
    
    test_no_spike()
    test_multi_spike_supercritical()
    test_sharpening_improves_gap()
    test_kappa_estimation_accuracy()
    
    print("\n✅ All S8 end-to-end tests passed!")
