"""
Automatic selection of K=6 and K=3 archetypes with minimal mutual similarity.

Algorithm: Greedy optimization to minimize average pairwise overlap.
"""
import numpy as np
from pathlib import Path
import sys
from itertools import combinations

root_path = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_path))

from graphs.structured.prep.load_structured_dataset import load_structured_archetypes


def compute_avg_overlap(indices, overlap_matrix):
    """Compute average absolute overlap for a subset of archetypes."""
    if len(indices) <= 1:
        return 0.0
    
    subset = overlap_matrix[np.ix_(indices, indices)]
    # Exclude diagonal (self-overlap)
    mask = ~np.eye(len(indices), dtype=bool)
    return np.abs(subset[mask]).mean()


def greedy_selection(overlap_matrix, K_target):
    """
    Greedy algorithm to select K archetypes with minimal average overlap.
    
    Strategy:
    1. Start with the pair with lowest absolute overlap
    2. Iteratively add the archetype that minimizes average overlap with selected set
    """
    K_full = overlap_matrix.shape[0]
    abs_overlaps = np.abs(overlap_matrix.copy())
    np.fill_diagonal(abs_overlaps, 1.0)  # Ignore self
    
    # Find pair with minimal overlap
    min_idx = np.unravel_index(abs_overlaps.argmin(), abs_overlaps.shape)
    selected = list(min_idx)
    
    print(f"Starting with pair: {selected[0]}, {selected[1]} (overlap: {overlap_matrix[min_idx]:.3f})")
    
    # Greedily add remaining archetypes
    while len(selected) < K_target:
        best_candidate = None
        best_avg_overlap = float('inf')
        
        for k in range(K_full):
            if k in selected:
                continue
            
            # Test adding this archetype
            test_set = selected + [k]
            avg_overlap = compute_avg_overlap(test_set, abs_overlaps)
            
            if avg_overlap < best_avg_overlap:
                best_avg_overlap = avg_overlap
                best_candidate = k
        
        selected.append(best_candidate)
        print(f"Added {best_candidate}: avg overlap = {best_avg_overlap:.3f}")
    
    return sorted(selected)


def exhaustive_search_k3(overlap_matrix, K_target=3):
    """
    Exhaustive search for K=3 (only 84 combinations, very fast).
    Finds the triplet with absolute minimum average overlap.
    """
    K_full = overlap_matrix.shape[0]
    abs_overlaps = np.abs(overlap_matrix)
    
    best_triplet = (0, 1, 2)  # Default fallback
    best_avg_overlap = float('inf')
    
    for triplet in combinations(range(K_full), K_target):
        avg_overlap = compute_avg_overlap(list(triplet), abs_overlaps)
        
        if avg_overlap < best_avg_overlap:
            best_avg_overlap = avg_overlap
            best_triplet = triplet
    
    return sorted(list(best_triplet)), best_avg_overlap


def main():
    # Load archetypes
    archetypi, filenames = load_structured_archetypes()
    K_full, N = archetypi.shape
    
    print("\n" + "="*70)
    print("AUTOMATIC ARCHETYPE SELECTION - MINIMAL MUTUAL SIMILARITY")
    print("="*70)
    print(f"\nTotal archetypes: {K_full}")
    print("\nArchetypes:")
    for i, fname in enumerate(filenames):
        print(f"  [{i}] {fname.replace('.png', '')}")
    
    # Compute overlap matrix
    overlaps = (archetypi @ archetypi.T) / float(N)
    
    print("\nPairwise overlaps:")
    for i in range(K_full):
        for j in range(i+1, K_full):
            print(f"  [{i}] ↔ [{j}]: {overlaps[i,j]:+.3f}")
    
    # Select K=6 (greedy)
    print("\n" + "="*70)
    print("SELECTING K=6 ARCHETYPES (Greedy Algorithm)")
    print("="*70)
    selected_k6 = greedy_selection(overlaps, K_target=6)
    avg_overlap_k6 = compute_avg_overlap(selected_k6, np.abs(overlaps))
    
    print(f"\nBest K=6 selection: {selected_k6}")
    print(f"Files: {[filenames[i].replace('.png', '') for i in selected_k6]}")
    print(f"Average absolute overlap: {avg_overlap_k6:.3f}")
    
    # Select K=3 (exhaustive search for global optimum)
    print("\n" + "="*70)
    print("SELECTING K=3 ARCHETYPES (Exhaustive Search)")
    print("="*70)
    selected_k3, avg_overlap_k3 = exhaustive_search_k3(overlaps, K_target=3)
    
    print(f"\nOptimal K=3 selection: {selected_k3}")
    print(f"Files: {[filenames[i].replace('.png', '') for i in selected_k3]}")
    print(f"Average absolute overlap: {avg_overlap_k3:.3f}")
    
    # Verify K=3 is subset of K=6 (if not, show both)
    k3_in_k6 = all(k in selected_k6 for k in selected_k3)
    
    if not k3_in_k6:
        print("\nNOTE: Optimal K=3 is NOT a subset of K=6.")
        print("Searching for best K=3 subset within K=6...")
        
        # Find best K=3 within K=6
        best_k3_subset = (selected_k6[0], selected_k6[1], selected_k6[2])  # Default
        best_overlap_subset = float('inf')
        
        for triplet in combinations(selected_k6, 3):
            avg_overlap = compute_avg_overlap(list(triplet), np.abs(overlaps))
            if avg_overlap < best_overlap_subset:
                best_overlap_subset = avg_overlap
                best_k3_subset = triplet
        
        best_k3_subset = sorted(list(best_k3_subset))
        print(f"\nBest K=3 within K=6: {best_k3_subset}")
        print(f"Files: {[filenames[i].replace('.png', '') for i in best_k3_subset]}")
        print(f"Average absolute overlap: {best_overlap_subset:.3f}")
        
        print("\nWhich K=3 to use?")
        print(f"  A) Global optimal: {selected_k3} (overlap: {avg_overlap_k3:.3f})")
        print(f"  B) Subset of K=6: {best_k3_subset} (overlap: {best_overlap_subset:.3f})")
        
        choice = input("\nChoice (A/B, default=A): ").strip().upper()
        if choice == 'B':
            selected_k3 = best_k3_subset
            avg_overlap_k3 = best_overlap_subset
            print("Using K=3 subset of K=6")
        else:
            print("Using global optimal K=3")
    
    # Show overlap matrices for selected subsets
    print("\n" + "="*70)
    print("OVERLAP MATRICES FOR SELECTED SUBSETS")
    print("="*70)
    
    print("\nK=6 overlap matrix:")
    subset_k6 = overlaps[np.ix_(selected_k6, selected_k6)]
    print("     ", end="")
    for k in selected_k6:
        print(f"{k:5d}", end="")
    print()
    for i, k_i in enumerate(selected_k6):
        print(f"[{k_i}] ", end="")
        for j, k_j in enumerate(selected_k6):
            print(f"{subset_k6[i,j]:5.2f}", end="")
        print(f"  {filenames[k_i].replace('.png', '')[:15]}")
    
    print("\nK=3 overlap matrix:")
    subset_k3 = overlaps[np.ix_(selected_k3, selected_k3)]
    print("     ", end="")
    for k in selected_k3:
        print(f"{k:5d}", end="")
    print()
    for i, k_i in enumerate(selected_k3):
        print(f"[{k_i}] ", end="")
        for j, k_j in enumerate(selected_k3):
            print(f"{subset_k3[i,j]:5.2f}", end="")
        print(f"  {filenames[k_i].replace('.png', '')[:15]}")
    
    # Save selections
    np.save("data/structured-dataset/selected_k6.npy", np.array(selected_k6))
    np.save("data/structured-dataset/selected_k3.npy", np.array(selected_k3))
    
    print("\n" + "="*70)
    print("SAVED!")
    print("="*70)
    print(f"K=6: {selected_k6} → data/structured-dataset/selected_k6.npy")
    print(f"K=3: {selected_k3} → data/structured-dataset/selected_k3.npy")
    print("\nSelection complete! Run federated experiments with these optimized subsets.")
    print("="*70)


if __name__ == "__main__":
    main()
