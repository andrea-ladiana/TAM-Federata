"""Debug script to understand why dis_check returns empty candidates."""
import numpy as np
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from unsup.config import HyperParams, TAMParams, SpectralParams, PropagationParams
from unsup.estimators import build_unsup_J_single, blend_with_memory
from unsup.functions import propagate_J, Hebb_J
from unsup.spectrum import eigen_cut as spectral_cut
from unsup.dynamics import init_candidates_from_eigs, TAM_Network
from unsup.data import gen_dataset_partial_archetypes
from graphs.structured.prep.load_structured_dataset import load_structured_archetypes

# Load data
archetypi, _ = load_structured_archetypes()
K, N = archetypi.shape

print(f"Archetypi: K={K}, N={N}")
print(f"Archetype means: {[f'{a.mean():.2f}' for a in archetypi]}")
print(f"Archetype norms: {[f'{np.linalg.norm(a):.1f}' for a in archetypi]}")

# Setup HP
hp = HyperParams(
    L=3, K=K, N=N,
    n_batch=12,
    M_total=1200,
    r_ex=0.9,
    w=0.4,
    prop=PropagationParams(iters=200, eps=1e-2),
    tam=TAMParams(beta_T=2.5, lam=0.2, updates=50),
    spec=SpectralParams(tau=0.15, rho=0.6, qthr=0.4)
)

# Generate one round of data
rng = np.random.default_rng(42)
client_subsets = [list(range(K)) for _ in range(hp.L)]

ETA_all, labels_all = gen_dataset_partial_archetypes(
    xi_true=archetypi,
    M_total=hp.M_total,
    r_ex=hp.r_ex,
    n_batch=1,  # Just one round
    L=hp.L,
    client_subsets=client_subsets,
    rng=rng,
    use_tqdm=False
)

ETA_t = ETA_all[:, 0, :, :]  # (L, M_c, N)
print(f"\nData shape: {ETA_t.shape}")

# Build J
J_unsup, M_eff = build_unsup_J_single(ETA_t, K=hp.K)
J_rec = blend_with_memory(J_unsup, xi_prev=None, w=hp.w)
J_KS = np.asarray(propagate_J(J_rec, J_real=-1, verbose=False, iters=hp.prop.iters), dtype=np.float32)

print(f"J_KS norm: {np.linalg.norm(J_KS):.3f}")

# Spectral cut
_spec_out = spectral_cut(J_KS, tau=hp.spec.tau, return_info=True)
V, k_eff_cut, info_spec = _spec_out

print(f"\nSpectral cut: k_eff={k_eff_cut}, tau={hp.spec.tau}")
evals = info_spec.get("evals")
print(f"Top 10 eigenvalues: {evals[:10]}")

if k_eff_cut == 0:
    print("\nERROR: No eigenvectors passed tau threshold!")
    sys.exit(1)

# Initialize candidates
sigma0 = init_candidates_from_eigs(V, L=hp.L)
print(f"\nInitialized candidates: {sigma0.shape}")
print(f"Candidate means: {[f'{s.mean():.2f}' for s in sigma0[:min(5, len(sigma0))]]}")

# Run TAM dynamics
sigma = np.repeat(sigma0[:, None, :], hp.L, axis=1)
net = TAM_Network()
net.prepare(J_rec, hp.L)
net.dynamics(
    sigma,
    hp.tam.beta_T,
    hp.tam.lam,
    hp.tam.h_in,
    updates=hp.tam.updates,
    noise_scale=hp.tam.noise_scale,
    min_scale=hp.tam.min_scale,
    anneal=hp.tam.anneal,
    schedule=hp.tam.schedule,
    show_progress=False,
    desc="TAM debug"
)

xi_tam = np.reshape(np.asarray(net.σ), (sigma0.shape[0] * hp.L, N)).astype(int)
print(f"\nAfter TAM: {xi_tam.shape}")
print(f"TAM output means: {[f'{x.mean():.2f}' for x in xi_tam[:min(5, len(xi_tam))]]}")

# Check spectral alignment
align = (xi_tam @ J_KS @ xi_tam.T).diagonal() / float(N)
print(f"\nSpectral alignments (before pruning):")
print(f"  Min: {align.min():.3f}, Max: {align.max():.3f}, Mean: {align.mean():.3f}")
print(f"  Threshold rho={hp.spec.rho}")
print(f"  Passing: {(align >= hp.spec.rho).sum()}/{len(align)}")

# Check overlap with true archetypi
overlaps = (xi_tam @ archetypi.T) / float(N)  # (n_cand, K)
max_overlaps = np.abs(overlaps).max(axis=1)
print(f"\nOverlaps with true archetypi:")
print(f"  Max absolute overlap per candidate:")
print(f"  Min: {max_overlaps.min():.3f}, Max: {max_overlaps.max():.3f}, Mean: {max_overlaps.mean():.3f}")

# Show which candidates pass rho
passing = align >= hp.spec.rho
print(f"\nCandidates passing rho threshold:")
for i, (passed, al, mo) in enumerate(zip(passing, align, max_overlaps)):
    status = "PASS" if passed else "FAIL"
    print(f"  Candidate {i}: align={al:.3f} max_overlap={mo:.3f} [{status}]")

if passing.sum() == 0:
    print("\n*** PROBLEM: No candidates pass spectral alignment threshold! ***")
    print(f"*** Consider reducing rho from {hp.spec.rho} to ~{align.max():.2f} ***")
