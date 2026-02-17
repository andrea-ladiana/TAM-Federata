"""
Generate 3 random Rademacher archetypes for comparison test.

This creates unstructured random patterns to verify the pipeline works
with truly uncorrelated data.
"""
import numpy as np
from pathlib import Path

# Generate 3 random archetypes (N=784, like 28x28 images)
N = 784
K = 3

np.random.seed(42)  # For reproducibility

# Random Rademacher: each entry ±1 with equal probability
archetypi_random = np.random.choice([-1, 1], size=(K, N))

print(f"Generated {K} random Rademacher archetypes")
print(f"Shape: {archetypi_random.shape}")

# Check they are different
overlaps = (archetypi_random @ archetypi_random.T) / float(N)
print(f"\nPairwise overlaps:")
for i in range(K):
    for j in range(i+1, K):
        print(f"  [{i}] ↔ [{j}]: {overlaps[i,j]:+.3f}")

avg_overlap = 0
count = 0
for i in range(K):
    for j in range(i+1, K):
        avg_overlap += abs(overlaps[i,j])
        count += 1
avg_overlap /= count
print(f"\nAverage absolute overlap: {avg_overlap:.3f}")
print("(Expected ~0 for random patterns)")

# Save in data/structured-dataset as backup
out_path = Path("data/structured-dataset")
out_path.mkdir(parents=True, exist_ok=True)

# Save full random set (K=3)
np.save(out_path / "archetypi_random_k3.npy", archetypi_random)
print(f"\nSaved to {out_path / 'archetypi_random_k3.npy'}")

# Also create K=6 and K=9 versions for completeness
archetypi_random_k6 = np.random.choice([-1, 1], size=(6, N))
archetypi_random_k9 = np.random.choice([-1, 1], size=(9, N))

np.save(out_path / "archetypi_random_k6.npy", archetypi_random_k6)
np.save(out_path / "archetypi_random_k9.npy", archetypi_random_k9)

print(f"Also saved K=6 and K=9 random versions")

# Compute stats for K=9
overlaps_k9 = (archetypi_random_k9 @ archetypi_random_k9.T) / float(N)
avg_overlap_k9 = 0
count = 0
for i in range(9):
    for j in range(i+1, 9):
        avg_overlap_k9 += abs(overlaps_k9[i,j])
        count += 1
avg_overlap_k9 /= count
print(f"\nK=9 average absolute overlap: {avg_overlap_k9:.3f}")

print("\nTo use random archetypes, modify exp_structured_federated.py:")
print("  Change: USE_RANDOM = True")
