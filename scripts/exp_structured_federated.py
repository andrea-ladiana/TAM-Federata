"""
Federated TAM experiment with structured dataset.

This script implements the complete federated pipeline from src/unsup/:
1. Load structured archetypes (K=9, N=784)
2. Generate federated noisy examples with gen_dataset_partial_archetypes (r=0.7)
3. Run n_batch=12 federated rounds with L=3 clients
4. Each round: build_unsup_J → blend → propagate_J → eigen_cut → dis_check
5. Track metrics: retrieval, FRO, K_eff, coverage
6. Final verification: Hopfield dynamics from noisy examples

Parameters (matching exp01_synth.py baseline):
- K=9 archetypes
- N=784 (28×28)
- L=3 clients
- n_batch=12 rounds
- M_total=600 examples total across all rounds/clients
- r=0.7 (30% noise)
- w=0.4 (memory weight, default from config)
- TAM: beta_T=2.5, lam=0.2, updates=50
- Spectral: tau=0.5, rho=0.6, qthr=0.4
"""
import numpy as np
from pathlib import Path
import sys

# Add both paths to allow imports
root_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from unsup.config import HyperParams, TAMParams, SpectralParams, PropagationParams
from unsup.single_round import single_round_step
from unsup.functions import Hebb_J
from unsup.data import gen_dataset_partial_archetypes, make_client_subsets

# Load structured archetypes
from graphs.structured.prep.load_structured_dataset import load_structured_archetypes


def run_structured_federated(
    archetypi: np.ndarray,
    hp: HyperParams,
    seed: int = 42,
    out_dir: str = "out_structured"
) -> dict:
    """
    Run federated TAM experiment with structured archetypes.
    
    Parameters
    ----------
    archetypi : np.ndarray
        True archetypes, shape (K, N)
    hp : HyperParams
        Hyperparameters
    seed : int
        Random seed
    out_dir : str
        Output directory
        
    Returns
    -------
    results : dict
        Complete results including:
        - metrics per round
        - retrieved archetypes per round
        - final J matrices
        - verification data (for Hopfield test)
    """
    np.random.seed(seed)
    
    K, N = archetypi.shape
    L = hp.L
    n_batch = hp.n_batch
    
    print(f"\n{'='*60}")
    print(f"STRUCTURED FEDERATED TAM EXPERIMENT")
    print(f"{'='*60}")
    print(f"K={K} archetypes, N={N} dims")
    print(f"L={L} clients, n_batch={n_batch} rounds")
    print(f"M_total={hp.M_total} examples TOTAL, r={hp.r_ex:.2f}")
    print(f"w={hp.w:.2f} (memory weight)")
    print(f"TAM: beta_T={hp.tam.beta_T}, lam={hp.tam.lam}, updates={hp.tam.updates}")
    print(f"Spectral: tau={hp.spec.tau}, rho={hp.spec.rho}, qthr={hp.spec.qthr}")
    print(f"{'='*60}\n")
    
    # Storage
    results = {
        'archetypi': archetypi,
        'hp': hp,
        'seed': seed,
        'metrics': {
            'retrieval': [],
            'fro': [],
            'keff': [],
            'coverage': []
        },
        'xi_retrieved': [],  # Per round
        'J_unsup_history': [],
        'J_KS_history': [],
        'V_history': [],
        'examples_final': None,  # For verification
        'labels_final': None
    }
    
    # Initialize memory
    xi_prev = None
    
    # Star matrix (for metrics)
    J_star = Hebb_J(archetypi)
    
    # Create client subsets (all clients see all archetipi for now)
    rng = np.random.default_rng(seed)
    K_per_client = K  # All clients see all archetypi
    client_subsets = [list(range(K)) for _ in range(L)]
    
    # Generate ALL data upfront (for all rounds)
    print("Generating federated dataset...")
    ETA_all, labels_all = gen_dataset_partial_archetypes(
        xi_true=archetypi,
        M_total=hp.M_total,
        r_ex=hp.r_ex,
        n_batch=n_batch,
        L=L,
        client_subsets=client_subsets,
        rng=rng,
        use_tqdm=True
    )
    # ETA_all shape: (L, n_batch, M_c, N)
    # labels_all shape: (L, n_batch, M_c)
    print(f"Generated data: {ETA_all.shape}")
    
    # Run federated rounds
    for t in range(n_batch):
        print(f"\n--- Round {t+1}/{n_batch} ---")
        
        # Extract data for this round
        ETA_t = ETA_all[:, t, :, :]  # (L, M_c, N)
        labels_t = labels_all[:, t, :]  # (L, M_c)
        
        print(f"Round data: {ETA_t.shape}")
        
        # Run single round
        xi_ref_new, J_KS, log = single_round_step(
            ETA_t=ETA_t,
            labels_t=labels_t,
            xi_true=archetypi,
            J_star=J_star,
            xi_prev=xi_prev,
            hp=hp
        )
        
        # Update memory for next round
        xi_prev = xi_ref_new
        
        # Store results
        results['metrics']['retrieval'].append(log.retrieval)
        results['metrics']['fro'].append(log.fro)
        results['metrics']['keff'].append(log.keff)
        results['metrics']['coverage'].append(log.coverage)
        
        results['xi_retrieved'].append(xi_ref_new.copy())
        results['J_KS_history'].append(J_KS.copy())
        
        # Print metrics
        print(f"  Retrieval: {log.retrieval:.3f}")
        print(f"  FRO: {log.fro:.4f}")
        print(f"  K_eff: {log.keff}/{K}")
        print(f"  Coverage: {log.coverage:.2f}")
        
        # Store final round data for verification
        if t == n_batch - 1:
            results['examples_final'] = ETA_t.copy()
            results['labels_final'] = labels_t.copy()
            results['J_final'] = J_KS.copy()
            results['xi_final'] = xi_ref_new.copy()
    
    # Save results
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save xi_retrieved per round for magnetization analysis
    xi_retrieved_array = np.array([xi.copy() for xi in results['xi_retrieved']], dtype=object)
    
    np.savez(
        out_path / "results.npz",
        archetypi=archetypi,
        retrieval=np.array(results['metrics']['retrieval']),
        fro=np.array(results['metrics']['fro']),
        keff=np.array(results['metrics']['keff']),
        coverage=np.array(results['metrics']['coverage']),
        xi_retrieved_final=results['xi_final'],
        xi_retrieved_all=xi_retrieved_array,  # All rounds for magnetization
        J_final=results['J_final'],
        examples_final=results['examples_final'],
        labels_final=results['labels_final'],
        seed=seed
    )
    
    print(f"\n{'='*60}")
    print(f"Results saved to {out_path / 'results.npz'}")
    print(f"{'='*60}\n")
    
    return results


def main():
    """Run structured federated experiment."""
    
    # CONFIGURATION
    USE_K = 9  # Change to 3, 6, or 9
    USE_RANDOM = True  # True = random Rademacher, False = structured silhouettes
    
    # Load archetypi
    if USE_RANDOM:
        print(f"Loading K={USE_K} random Rademacher archetypes...")
        archetypi = np.load(f"data/structured-dataset/archetypi_random_k{USE_K}.npy")
        print(f"Loaded {archetypi.shape[0]} random patterns with shape {archetypi.shape}")
    else:
        print("Loading structured archetypes...")
        archetypi_full, filenames = load_structured_archetypes()
        
        # Select K=6 most distinct (or use all 9)
        if USE_K == 6:
            selected_idx = np.load("data/structured-dataset/selected_k6.npy")
            archetypi = archetypi_full[selected_idx]
            print(f"Using K={USE_K} selected archetypes: {selected_idx}")
        elif USE_K == 3:
            selected_idx = np.load("data/structured-dataset/selected_k3.npy")
            archetypi = archetypi_full[selected_idx]
            print(f"Using K={USE_K} selected archetypes: {selected_idx}")
        else:
            archetypi = archetypi_full
            print(f"Using all K={archetypi.shape[0]} archetypes")
    
    K, N = archetypi.shape
    
    # Setup hyperparameters
    hp = HyperParams(
        # Network structure
        L=3,  # 3 clients
        K=K,  # 9 archetypes
        N=N,  # 784 dimensions
        
        # Data generation
        n_batch=12,  # 12 federated rounds
        M_total=3000,  # Total examples across ALL rounds and clients (increased)
        r_ex=0.9,  # 5% noise (high quality, reduced from 0.9)
        
        # Memory blending (use default w=0.4)
        w=0.4,  # Default from config
        
        # Propagation
        prop=PropagationParams(
            iters=200,
            eps=1e-2
        ),
        
        # TAM dynamics
        tam=TAMParams(
            beta_T=2.5,
            lam=0.2,
            updates=50,
            anneal=True,
            schedule='linear',
            noise_scale=0.3,
            min_scale=0.02
        ),
        
        # Spectral/pruning (adaptive based on pattern type)
        spec=SpectralParams(
            tau=0.5 if USE_RANDOM else 0.15,  # Random: high threshold, Structured: low
            rho=0.6 if USE_RANDOM else 0.35,  # Random: strict alignment, Structured: relaxed
            qthr=0.4 if USE_RANDOM else 0.95  # Random: prune duplicates, Structured: keep similar
        )
    )
    
    # Run experiment
    out_suffix = "random" if USE_RANDOM else "structured"
    results = run_structured_federated(
        archetypi=archetypi,
        hp=hp,
        seed=42,
        out_dir=f"out_structured/federated_{out_suffix}_k{K}"
    )
    
    # Print final summary
    print("\nFINAL SUMMARY:")
    print(f"Final retrieval: {results['metrics']['retrieval'][-1]:.3f}")
    print(f"Final FRO: {results['metrics']['fro'][-1]:.4f}")
    print(f"Final K_eff: {results['metrics']['keff'][-1]}/{K}")
    print(f"Final coverage: {results['metrics']['coverage'][-1]:.2f}")
    
    return results


if __name__ == "__main__":
    results = main()
