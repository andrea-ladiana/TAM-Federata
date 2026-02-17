"""
Hopfield retrieval verification for structured dataset.

This module tests whether the retrieved archetypes from the federated TAM pipeline
can be used to build a Hopfield network that successfully denoises noisy examples.

Test procedure:
1. Load final retrieved archetypes from federated run
2. Build Hopfield matrix J = (1/N) ∑ξ_i ξ_i^T (Hebbian rule)
3. Initialize dynamics from noisy examples
4. Run Hopfield dynamics for multiple steps
5. Verify convergence to clean archetypes (no spurious attractors)

This answers the critical question: "Does the TAM pipeline produce usable
attractors, even if retrieval metric is low due to pattern correlation?"
"""
import numpy as np
from pathlib import Path
from typing import Tuple, List
import sys

# Add paths
root_path = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from unsup.functions import Hebb_J


def hopfield_step(s: np.ndarray, J: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Single synchronous Hopfield update.
    
    Parameters
    ----------
    s : np.ndarray
        Current state (K_test, N) or (N,)
    J : np.ndarray
        Hopfield matrix (N, N)
    beta : float
        Inverse temperature (1.0 = deterministic)
        
    Returns
    -------
    s_new : np.ndarray
        Updated state (binarized to {-1, +1})
    """
    # Compute fields
    h = s @ J  # (K_test, N) or (N,)
    
    # Deterministic update (sign)
    s_new = np.sign(h)
    
    # Handle zeros (rare with structured data, but be safe)
    s_new = np.where(s_new == 0, 1.0, s_new)
    
    return s_new


def hopfield_dynamics(
    s_init: np.ndarray,
    J: np.ndarray,
    n_steps: int = 50,
    beta: float = 1.0,
    verbose: bool = False
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Run Hopfield dynamics from initial state.
    
    Parameters
    ----------
    s_init : np.ndarray
        Initial state (K_test, N) or (N,)
    J : np.ndarray
        Hopfield matrix (N, N)
    n_steps : int
        Number of update steps
    beta : float
        Inverse temperature
    verbose : bool
        Print convergence info
        
    Returns
    -------
    s_final : np.ndarray
        Final converged state
    trajectory : List[np.ndarray]
        States at each step (for visualization)
    """
    s = s_init.copy()
    trajectory = [s.copy()]
    
    for step in range(n_steps):
        s_new = hopfield_step(s, J, beta=beta)
        trajectory.append(s_new.copy())
        
        # Check convergence
        if np.allclose(s_new, s):
            if verbose:
                print(f"  Converged at step {step+1}")
            break
        
        s = s_new
    
    return s, trajectory


def test_hopfield_retrieval(
    xi_retrieved: np.ndarray,
    xi_true: np.ndarray,
    examples: np.ndarray,
    labels: np.ndarray,
    n_steps: int = 50,
    beta: float = 1.0,
    n_test_per_archetype: int = 3
) -> dict:
    """
    Test Hopfield retrieval from noisy examples.
    
    Parameters
    ----------
    xi_retrieved : np.ndarray
        Retrieved archetypes from TAM, shape (K, N)
    xi_true : np.ndarray
        True archetypes, shape (K, N)
    examples : np.ndarray
        Noisy examples, shape (L, M_c, N)
    labels : np.ndarray
        Labels for examples, shape (L, M_c)
    n_steps : int
        Number of Hopfield update steps
    beta : float
        Inverse temperature
    n_test_per_archetype : int
        Number of examples to test per archetype
        
    Returns
    -------
    results : dict
        - success_rate: fraction of examples that converge to correct archetype
        - overlap_initial: overlaps before dynamics
        - overlap_final: overlaps after dynamics
        - trajectories: sample trajectories for visualization
        - convergence_targets: which archetype each example converged to
    """
    K, N = xi_true.shape
    
    # Build Hopfield matrix from retrieved archetypes
    J_hopfield = Hebb_J(xi_retrieved)
    
    print(f"\nHopfield Retrieval Test")
    print(f"{'='*50}")
    print(f"K={K} archetypes, N={N} dims")
    print(f"Retrieved archetypes: {xi_retrieved.shape}")
    print(f"Hopfield matrix: {J_hopfield.shape}")
    
    # Flatten examples
    L, M_c, N_check = examples.shape
    examples_flat = examples.reshape(-1, N)  # (L*M_c, N)
    labels_flat = labels.flatten()  # (L*M_c,)
    
    # Select test examples (n_test_per_archetype per true archetype)
    test_examples = []
    test_labels = []
    test_archetype_idx = []
    
    for k in range(K):
        # Find examples of this archetype
        idx_k = np.where(labels_flat == k)[0]
        
        if len(idx_k) == 0:
            print(f"  WARNING: No examples for archetype {k}")
            continue
        
        # Select random subset
        n_select = min(n_test_per_archetype, len(idx_k))
        idx_select = np.random.choice(idx_k, size=n_select, replace=False)
        
        test_examples.append(examples_flat[idx_select])
        test_labels.append(labels_flat[idx_select])
        test_archetype_idx.extend([k] * n_select)
    
    test_examples = np.vstack(test_examples)  # (K*n_test, N)
    test_labels = np.concatenate(test_labels)  # (K*n_test,)
    test_archetype_idx = np.array(test_archetype_idx)  # (K*n_test,)
    
    n_test = len(test_examples)
    print(f"Testing {n_test} examples ({n_test//K} per archetype)")
    
    # Run Hopfield dynamics for each test example
    successes = 0
    overlap_initial = np.zeros(n_test)
    overlap_final = np.zeros(n_test)
    convergence_targets = np.zeros(n_test, dtype=int)
    
    # Store sample trajectories (first example of each archetype)
    sample_trajectories = {}
    
    for i in range(n_test):
        s_init = test_examples[i]  # (N,)
        true_k = test_archetype_idx[i]
        
        # Run dynamics
        s_final, trajectory = hopfield_dynamics(
            s_init=s_init,
            J=J_hopfield,
            n_steps=n_steps,
            beta=beta,
            verbose=False
        )
        
        # Compute overlaps with all true archetypes
        overlaps_init = (1.0 / N) * (s_init @ xi_true.T)  # (K,)
        overlaps_final = (1.0 / N) * (s_final @ xi_true.T)  # (K,)
        
        # Determine convergence target (highest overlap)
        target_k = np.argmax(overlaps_final)
        convergence_targets[i] = target_k
        
        # Record overlaps with TRUE archetype
        overlap_initial[i] = overlaps_init[true_k]
        overlap_final[i] = overlaps_final[true_k]
        
        # Success if converged to correct archetype
        if target_k == true_k:
            successes += 1
        
        # Store first trajectory per archetype for visualization
        if true_k not in sample_trajectories:
            sample_trajectories[true_k] = {
                'init': s_init.copy(),
                'final': s_final.copy(),
                'trajectory': trajectory,
                'true_k': true_k,
                'target_k': target_k,
                'overlap_init': overlaps_init[true_k],
                'overlap_final': overlaps_final[true_k]
            }
    
    success_rate = successes / float(n_test)
    
    print(f"\nResults:")
    print(f"  Success rate: {success_rate:.2%} ({successes}/{n_test})")
    print(f"  Overlap (initial): {overlap_initial.mean():.3f} ± {overlap_initial.std():.3f}")
    print(f"  Overlap (final): {overlap_final.mean():.3f} ± {overlap_final.std():.3f}")
    print(f"{'='*50}\n")
    
    # Compute confusion matrix
    confusion = np.zeros((K, K), dtype=int)
    for i in range(n_test):
        true_k = test_archetype_idx[i]
        target_k = convergence_targets[i]
        confusion[true_k, target_k] += 1
    
    results = {
        'success_rate': success_rate,
        'overlap_initial': overlap_initial,
        'overlap_final': overlap_final,
        'convergence_targets': convergence_targets,
        'test_archetype_idx': test_archetype_idx,
        'confusion_matrix': confusion,
        'sample_trajectories': sample_trajectories,
        'J_hopfield': J_hopfield,
        'xi_retrieved': xi_retrieved,
        'xi_true': xi_true
    }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="out_structured/federated_run",
                        help="Directory containing results.npz")
    args = parser.parse_args()
    
    # Load results from federated run
    results_path = Path(args.run_dir) / "results.npz"
    
    if not results_path.exists():
        print(f"ERROR: Results file not found at {results_path}")
        print("Please run exp_structured_federated.py first")
        sys.exit(1)
    
    data = np.load(results_path)
    
    archetypi = data['archetypi']  # (K, N)
    xi_final = data['xi_retrieved_final']  # (K, N)
    examples_final = data['examples_final']  # (L, M_c, N)
    labels_final = data['labels_final']  # (L, M_c)
    
    print(f"Loaded results:")
    print(f"  Archetypi: {archetypi.shape}")
    print(f"  Retrieved (final): {xi_final.shape}")
    print(f"  Examples (final round): {examples_final.shape}")
    print(f"  Labels (final round): {labels_final.shape}")
    
    # Run Hopfield retrieval test
    results = test_hopfield_retrieval(
        xi_retrieved=xi_final,
        xi_true=archetypi,
        examples=examples_final,
        labels=labels_final,
        n_steps=50,
        beta=1.0,
        n_test_per_archetype=3
    )
    
    # Save results
    out_path = Path(args.run_dir)
    np.savez(
        out_path / "hopfield_verification.npz",
        **results
    )
    
    print(f"Saved verification results to {out_path / 'hopfield_verification.npz'}")
