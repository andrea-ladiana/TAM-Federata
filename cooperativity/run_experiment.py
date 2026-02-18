
import sys
import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Sequence
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to import src.unsup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.unsup.config import HyperParams
from src.unsup.functions import gen_patterns, propagate_J, JK_real
from src.unsup.data import make_client_subsets, new_round_single
from src.unsup.estimators import build_unsup_J_single, blend_with_memory
from src.unsup.spectrum import eigen_cut as spectral_cut
from src.unsup.dynamics import dis_check
from src.unsup.metrics import retrieval_mean_hungarian

# --- Custom Dataset Generation for Heterogeneous r ---

def gen_dataset_heterogeneous(
    xi_true: np.ndarray,
    M_c: int,
    r_list: List[float],
    n_batch: int,
    L: int,
    client_subsets: Sequence[Sequence[int]],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates unsupervised dataset with unique r per client.
    
    Args:
        xi_true: (K, N)
        M_c: Number of examples per client per round.
        r_list: List of length L with r values for each client.
        n_batch: Number of rounds.
        L: Number of clients.
        client_subsets: specific archetypes visible to each client.
        rng: Random generator.
        
    Returns:
        ETA: (L, n_batch, M_c, N)
        labels: (L, n_batch, M_c)
    """
    K, N = xi_true.shape
    if len(r_list) != L:
        raise ValueError(f"r_list length {len(r_list)} must match L={L}")
    
    ETA = np.zeros((L, n_batch, M_c, N), dtype=np.float32)
    labels = np.zeros((L, n_batch, M_c), dtype=np.int32)
    
    # Precompute p_keep for each client
    p_keep_list = [0.5 * (1.0 + float(r_val)) for r_val in r_list]
    
    for l in range(L):
        allowed = list(client_subsets[l])
        p_keep = p_keep_list[l]
        
        for t in range(n_batch):
            # Sample archetypes from allowed subset
            mus = rng.choice(allowed, size=M_c, replace=True).astype(int)
            
            # Generate noise
            probs = rng.uniform(size=(M_c, N))
            chi = np.where(probs <= p_keep, 1.0, -1.0).astype(np.float32)
            
            # Select true patterns
            xi_sel = xi_true[mus].astype(np.float32)
            
            # Combine
            ETA[l, t] = (chi * xi_sel).astype(np.float32)
            labels[l, t] = mus.astype(np.int32)
            
    return ETA, labels

# --- Simulation Logic ---

@dataclass
class RegimeConfig:
    name: str
    r_list: List[float]

def run_single_seed(
    seed: int,
    hp: HyperParams,
    r_list: List[float],
    M_c: int
) -> List[float]:
    """Runs one full simulation (all rounds) for a given seed."""
    rng = np.random.default_rng(seed)
    
    # 1. Generate archetypes
    xi_true = np.asarray(gen_patterns(hp.N, hp.K), dtype=np.int32)
    
    # 2. Assign subsets (ensure full coverage if possible)
    # Using defaults from src/unsup logic but we want to ensure basic coverage for this small example
    # With K=3, L=3, K_per_client=None -> ceil(3/3)=1. So 1 archetype per client.
    # But usually we want some overlap or at least full coverage. 
    # Let's trust make_client_subsets to do round robin.
    K_per_client = 1  # Minimal overlap to match "partial coverage" idea, or maybe more?
    # Text says: "pipeline consistently reconstructs all archetypes". 
    # Let's stick to standard src logic.
    subsets = make_client_subsets(K=hp.K, L=hp.L, K_per_client=K_per_client, rng=rng)
    
    # 3. Generate Data
    ETA, _ = gen_dataset_heterogeneous(
        xi_true=xi_true,
        M_c=M_c,
        r_list=r_list,
        n_batch=hp.n_batch,
        L=hp.L,
        client_subsets=subsets,
        rng=rng
    )
    
    # 4. Simulation Loop
    xi_ref: Optional[np.ndarray] = None
    retrievals = []
    
    for t in range(hp.n_batch):
        # Current round data
        ETA_t = new_round_single(ETA, t) # (L, M_c, N)
        
        # Build Unsupervised J
        J_unsup, _ = build_unsup_J_single(ETA_t, K=hp.K)
        
        # Blend with memory
        J_rec = blend_with_memory(J_unsup, xi_prev=xi_ref, w=hp.w)
        
        # Propagation
        J_KS = np.asarray(propagate_J(J_rec, J_real=-1, verbose=False, iters=hp.prop.iters), dtype=np.float32)
        
        # DEBUG: Check eigenvalues
        if t == 0:
            e_rec = np.linalg.eigvalsh(J_rec)
            e_ks = np.linalg.eigvalsh(J_KS)
            print(f"DEBUG: seed={seed} t={t} max_eig(J_rec)={e_rec.max():.3f} max_eig(J_KS)={e_ks.max():.3f}")

        # Spectral Cut and Disentangling
        V, _ = spectral_cut(J_KS, tau=hp.spec.tau)
        
        xi_r, m_vec = dis_check(
            V=V,
            K=hp.K,
            L=hp.L,
            J_rec=J_rec,
            JKS_iter=J_KS,
            xi_true=xi_true,
            tam=hp.tam,
            spec=hp.spec,
            show_progress=False
        )
        
        # Calculate Max Overlap (Magnetization)
        # The text says "Maximum overlap (magnetization)". 
        # Usually we report mean retrieval, but the figure caption says "Maximum overlap".
        # Let's compute the mean max overlap across archetypes, 
        # or just standard hungarian matching metric which matches archetypes 1-to-1 maximimizing sum of overlaps.
        # "retrieval_mean_hungarian" gives avg overlap of best matching.
        retr = retrieval_mean_hungarian(xi_r.astype(int), xi_true.astype(int))
        retrievals.append(retr)
        
        # Update memory
        if xi_r.shape[0] >= hp.K:
            xi_ref = xi_r[: hp.K].astype(int)
        else:
            xi_ref = xi_r.astype(int)
            
    return retrievals

def main():
    # Parameters from text
    N = 400
    T = 12
    M_c = 400
    L = 3
    K = 3
    w = 0.6
    
    # Seeds
    n_seeds = 20
    base_seed = 42
    
    # Output dir
    out_dir = Path("cooperativity/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Hyperparams container
    hp = HyperParams(
        L=L,
        K=K,
        N=N,
        n_batch=T,
        M_total=M_c * L * T, # Not really used since we force M_c
        w=w,
        mode="single",
        K_per_client=3, # Full coverage
        spec=HyperParams().spec, # default
        prop=HyperParams().prop, # default
    )
    # Override defaults for this experiment to handle small eigenvalues from K-way mixture
    hp.spec = hp.spec.__class__(tau=0.03, rho=0.6, qthr=0.4)
    hp.prop = hp.prop.__class__(iters=1000, eps=1e-2)
    
    # Regimes
    regimes = [
        RegimeConfig(name="High", r_list=[0.8, 0.8, 0.8]),
        RegimeConfig(name="Hetero", r_list=[0.8, 0.2, 0.2]),
        RegimeConfig(name="Low", r_list=[0.2, 0.2, 0.2]),
    ]
    
    results = {}
    
    for reg in regimes:
        print(f"Running Regime: {reg.name} with r={reg.r_list}")
        
        seeds = [base_seed + i for i in range(n_seeds)]
        
        # Parallel execution
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(run_single_seed, s, hp, reg.r_list, M_c) 
                for s in seeds
            ]
            
            regime_retrievals = [f.result() for f in futures]
            
        results[reg.name] = np.array(regime_retrievals) # (n_seeds, T)
        
    # Save results
    np.savez(
        out_dir / "results.npz",
        High=results["High"],
        Hetero=results["Hetero"],
        Low=results["Low"],
        t_steps=np.arange(T)
    )
    print("Simulation complete. Results saved.")

if __name__ == "__main__":
    main()
