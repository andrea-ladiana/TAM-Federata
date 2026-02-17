"""
Select K=6 most distinct archetypes from the structured dataset.

Strategy: compute pairwise overlaps and select subset with minimal correlation.
"""
import numpy as np
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_path))

from graphs.structured.prep.load_structured_dataset import load_structured_archetypes

# Load all 9 archetypes
archetypi, filenames = load_structured_archetypes()
K_full, N = archetypi.shape

print(f"Loaded {K_full} archetypes")
print(f"Filenames: {filenames}")

# Compute pairwise overlaps
overlaps = (archetypi @ archetypi.T) / float(N)
print(f"\nPairwise overlaps:")
print(overlaps)

# Find most distinct subset (greedy approach)
# Start with pair with lowest overlap
abs_overlaps = np.abs(overlaps)
np.fill_diagonal(abs_overlaps, 1.0)  # Ignore self-overlap

# Find most distinct pair
min_overlap_idx = np.unravel_index(abs_overlaps.argmin(), abs_overlaps.shape)
selected = list(min_overlap_idx)

print(f"\nStarting with most distinct pair: {selected[0]}, {selected[1]}")
print(f"  {filenames[selected[0]]} <-> {filenames[selected[1]]}")
print(f"  Overlap: {overlaps[selected[0], selected[1]]:.3f}")

# Greedily add archetypes with minimal average overlap to already selected
while len(selected) < 6:
    best_candidate = None
    best_avg_overlap = 1.0
    
    for k in range(K_full):
        if k in selected:
            continue
        
        # Compute average overlap with already selected
        avg_overlap = np.mean([abs(overlaps[k, s]) for s in selected])
        
        if avg_overlap < best_avg_overlap:
            best_avg_overlap = avg_overlap
            best_candidate = k
    
    selected.append(best_candidate)
    print(f"Added {best_candidate}: {filenames[best_candidate]} (avg overlap: {best_avg_overlap:.3f})")

selected = sorted(selected)
print(f"\nFinal selection (K=6): {selected}")
print(f"Files: {[filenames[i] for i in selected]}")

# Compute average overlap in selected subset
selected_overlaps = overlaps[np.ix_(selected, selected)]
np.fill_diagonal(selected_overlaps, 0.0)
avg_overlap = np.abs(selected_overlaps).mean()
print(f"\nAverage absolute overlap in selected subset: {avg_overlap:.3f}")

# Save selected indices
np.save("data/structured-dataset/selected_k6.npy", np.array(selected))
print(f"Saved selected indices to data/structured-dataset/selected_k6.npy")

# Similarly for K=3
selected_k3 = selected[:3]
print(f"\nFor K=3, use: {selected_k3}")
print(f"Files: {[filenames[i] for i in selected_k3]}")
np.save("data/structured-dataset/selected_k3.npy", np.array(selected_k3))
print(f"Saved to data/structured-dataset/selected_k3.npy")
