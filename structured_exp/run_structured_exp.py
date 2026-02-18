"""
Structured Dataset Federated TAM Experiment
============================================

Loads 3 structured image archetypes (anchor, motorbike, bass) from
data/structured-dataset/archetypi.npy[selected_k3], binarizes them,
applies Gram-Schmidt decorrelation, and runs the federated TAM pipeline
over n_runs seeds.

Output: structured_exp/results/results.npz

Usage:
    cd c:\\Users\\ladia\\Desktop\\TAMFed\\UNSUP
    python structured_exp/run_structured_exp.py
"""
import sys
import argparse
import numpy as np
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from unsup.config import HyperParams, TAMParams, SpectralParams, PropagationParams
from unsup.estimators import build_unsup_J_single, blend_with_memory
from unsup.functions import propagate_J, Hebb_J
from unsup.spectrum import eigen_cut as spectral_cut, estimate_keff
from unsup.dynamics import dis_check
from unsup.metrics import retrieval_mean_hungarian
from unsup.data import gen_dataset_partial_archetypes, compute_round_coverage

# ── configuration ────────────────────────────────────────────────────────────
N_RUNS   = 10
N_BATCH  = 12
L        = 3       # clients
R_EX     = 0.6     # noise quality
W        = 0.6     # memory blend weight

TAM_CFG = TAMParams(
    beta_T      = 3.0,
    lam         = 0.15,
    h_in        = 0.1,
    updates     = 60,
    noise_scale = 0.25,
    min_scale   = 0.02,
    anneal      = True,
    schedule    = "linear",
)
SPEC_CFG = SpectralParams(
    tau  = 0.15,   # relaxed for correlated patterns
    rho  = 0.25,   # relaxed spectral alignment threshold
    qthr = 0.95,   # keep near-duplicate candidates (structured data)
)
PROP_CFG = PropagationParams(iters=300, eps=1e-2)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_archetypes() -> np.ndarray:
    """Load and binarize the 3 selected structured archetypes."""
    data_dir = ROOT / "data" / "structured-dataset"
    archetypi_all = np.load(data_dir / "archetypi.npy")   # (9, 784)
    selected_idx  = np.load(data_dir / "selected_k3.npy") # [2, 3, 5]
    arch = archetypi_all[selected_idx].astype(np.float64)  # (3, 784)

    # Images are in [0, 1]; binarize to {-1, +1}
    # Threshold at 0.5: bright pixels → +1, dark pixels → -1
    arch_bin = np.where(arch >= 0.5, 1.0, -1.0)
    return arch_bin.astype(np.float32)


def gram_schmidt_binarize(xi: np.ndarray) -> np.ndarray:
    """
    Apply Gram-Schmidt orthogonalization to reduce inter-pattern correlation,
    then re-binarize each resulting vector to {-1, +1}.

    Parameters
    ----------
    xi : (K, N) float array of binarized archetypes

    Returns
    -------
    xi_gs : (K, N) binarized orthogonalized archetypes
    """
    K, N = xi.shape
    xi_gs = np.zeros_like(xi, dtype=np.float64)
    xi_f  = xi.astype(np.float64)

    for k in range(K):
        v = xi_f[k].copy()
        for j in range(k):
            proj = np.dot(xi_gs[j], xi_f[k]) / (np.dot(xi_gs[j], xi_gs[j]) + 1e-12)
            v -= proj * xi_gs[j]
        # Re-binarize: sign of the residual
        v_bin = np.where(v >= 0.0, 1.0, -1.0)
        xi_gs[k] = v_bin

    return xi_gs.astype(np.float32)


def compute_per_archetype_magnetization(xi_r: np.ndarray, xi_true: np.ndarray) -> np.ndarray:
    """
    For each true archetype mu, compute max overlap with any reconstructed pattern.

    Returns
    -------
    m : (K,) array of magnetizations in [0, 1]
    """
    K = xi_true.shape[0]
    N = xi_true.shape[1]
    if xi_r.shape[0] == 0:
        return np.zeros(K, dtype=np.float32)
    overlaps = np.abs(xi_r @ xi_true.T) / N   # (K_r, K)
    m = overlaps.max(axis=0)                   # (K,)
    return m.astype(np.float32)


def run_one_seed(archetypi: np.ndarray, hp: HyperParams, seed: int) -> dict:
    """Run the full federated pipeline for one seed."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    K, N = archetypi.shape
    n_batch = hp.n_batch

    # All clients see all archetypes
    client_subsets = [list(range(K)) for _ in range(hp.L)]

    # Generate all data upfront
    ETA_all, labels_all = gen_dataset_partial_archetypes(
        xi_true       = archetypi,
        M_total       = hp.M_total,
        r_ex          = hp.r_ex,
        n_batch       = n_batch,
        L             = hp.L,
        client_subsets= client_subsets,
        rng           = rng,
        use_tqdm      = False,
    )
    # ETA_all: (L, n_batch, M_c, N)

    J_star  = Hebb_J(archetypi)
    xi_prev = None

    # Per-round magnetizations: shape (n_batch, K)
    mag_history = np.zeros((n_batch, K), dtype=np.float32)

    # Store final reconstructions and one noisy example per archetype
    xi_final = None
    noisy_examples = None  # (K, N) — one noisy example per archetype

    for t in range(n_batch):
        ETA_t   = ETA_all[:, t, :, :]   # (L, M_c, N)
        labels_t = labels_all[:, t, :]  # (L, M_c)

        # 1) Build unsup J
        J_unsup, M_eff = build_unsup_J_single(ETA_t, K=hp.K)

        # 2) Blend with memory
        J_rec = blend_with_memory(J_unsup, xi_prev=xi_prev, w=hp.w)

        # 3) Propagate to pseudo-inverse
        J_KS = np.asarray(
            propagate_J(J_rec, J_real=-1, verbose=False, iters=hp.prop.iters),
            dtype=np.float32,
        )

        # 4) Spectral cut
        _spec_out = spectral_cut(J_KS, tau=hp.spec.tau, return_info=True)
        if len(_spec_out) == 3:
            V, k_eff_cut, _ = _spec_out
        else:
            V, k_eff_cut = _spec_out

        # 5) Disentangling
        xi_r, _m_vec = dis_check(
            V          = V,
            K          = hp.K,
            L          = hp.L,
            J_rec      = J_rec,
            JKS_iter   = J_KS,
            xi_true    = archetypi,
            tam        = hp.tam,
            spec       = hp.spec,
            show_progress = False,
        )

        # 6) Binarize reconstructions
        xi_r_bin = np.where(xi_r >= 0, 1, -1).astype(np.float32)

        # 7) Per-archetype magnetizations
        m_t = compute_per_archetype_magnetization(xi_r_bin, archetypi)
        mag_history[t] = m_t

        # 8) Update memory
        if xi_r_bin.shape[0] >= K:
            xi_prev = xi_r_bin[:K]
        else:
            xi_prev = xi_r_bin

        xi_final = xi_r_bin

        print(f"  seed={seed} round={t+1:2d}/{n_batch}  m={m_t.round(3)}")

    # Collect one noisy example per archetype (from last round, first client)
    # Find the example closest to each archetype
    ETA_last = ETA_all[0, -1, :, :]  # (M_c, N)
    labels_last = labels_all[0, -1, :]  # (M_c,)
    noisy_examples = np.zeros((K, N), dtype=np.float32)
    for mu in range(K):
        idxs = np.where(labels_last == mu)[0]
        if len(idxs) > 0:
            noisy_examples[mu] = ETA_last[idxs[0]]
        else:
            # fallback: random noisy version
            noisy_examples[mu] = archetypi[mu] * np.where(
                np.random.rand(N) < 0.5 * (1 + R_EX), 1.0, -1.0
            )

    return {
        "mag_history":    mag_history,    # (n_batch, K)
        "xi_final":       xi_final,       # (K_r, N)
        "noisy_examples": noisy_examples, # (K, N)
    }


def main():
    parser = argparse.ArgumentParser(description="Structured Dataset Federated TAM Experiment")
    parser.add_argument("--run-name", type=str, default="exp_1", help="Name of the experiment run (folder name)")
    args = parser.parse_args()

    RUN_NAME = args.run_name

    print("=" * 60)
    print(f"STRUCTURED DATASET FEDERATED TAM EXPERIMENT: {RUN_NAME}")
    print("=" * 60)

    # Load and prepare archetypes
    print("\n[1] Loading archetypes...")
    archetypi_raw = load_archetypes()
    K, N = archetypi_raw.shape
    print(f"    Raw archetypes: {K} patterns, N={N}")

    # Check correlations before GS
    C_raw = (archetypi_raw @ archetypi_raw.T) / N
    print(f"    Correlation matrix (raw):\n{C_raw.round(3)}")

    # Apply Gram-Schmidt decorrelation
    print("\n[2] Applying Gram-Schmidt decorrelation...")
    archetypi = gram_schmidt_binarize(archetypi_raw)
    C_gs = (archetypi @ archetypi.T) / N
    print(f"    Correlation matrix (after GS):\n{C_gs.round(3)}")

    # Build hyperparams
    hp = HyperParams(
        L       = L,
        K       = K,
        N       = N,
        n_batch = N_BATCH,
        M_total = 4000,
        r_ex    = R_EX,
        w       = W,
        tam     = TAM_CFG,
        prop    = PROP_CFG,
        spec    = SPEC_CFG,
        estimate_keff_method = "shuffle",
        use_tqdm = False,
    )

    print(f"\n[3] Running {N_RUNS} seeds...")
    print(f"    K={K}, N={N}, L={L}, n_batch={N_BATCH}, r={R_EX}, w={W}")
    print(f"    TAM: beta_T={TAM_CFG.beta_T}, lam={TAM_CFG.lam}, updates={TAM_CFG.updates}")
    print(f"    Spectral: tau={SPEC_CFG.tau}, rho={SPEC_CFG.rho}, qthr={SPEC_CFG.qthr}")

    all_mag_histories = []   # (N_RUNS, n_batch, K)
    all_xi_finals     = []   # list of (K_r, N)
    all_noisy         = []   # (N_RUNS, K, N)

    for run_idx in range(N_RUNS):
        seed = 1000 + run_idx * 7
        print(f"\n--- Run {run_idx+1}/{N_RUNS} (seed={seed}) ---")
        res = run_one_seed(archetypi, hp, seed)
        all_mag_histories.append(res["mag_history"])
        all_xi_finals.append(res["xi_final"])
        all_noisy.append(res["noisy_examples"])

    # Stack results
    mag_array = np.stack(all_mag_histories, axis=0)  # (N_RUNS, n_batch, K)
    noisy_array = np.stack(all_noisy, axis=0)         # (N_RUNS, K, N)

    # Summary
    mag_mean = mag_array.mean(axis=0)   # (n_batch, K)
    mag_std  = mag_array.std(axis=0)    # (n_batch, K)
    print("\n[4] Final magnetizations (mean ± std over last round):")
    for mu in range(K):
        print(f"    m_{mu+1}: {mag_mean[-1, mu]:.3f} ± {mag_std[-1, mu]:.3f}")

    # Save
    out_dir = ROOT / "structured_exp" / "results" / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.npz"

    # Use seed=0 run's noisy examples and final reconstruction for display
    display_noisy = all_noisy[0]                    # (K, N)
    display_xi    = all_xi_finals[0]                # (K_r, N)

    np.savez(
        out_path,
        archetypi       = archetypi,          # (K, N) GS-binarized
        archetypi_raw   = archetypi_raw,       # (K, N) original binarized
        mag_mean        = mag_mean,            # (n_batch, K)
        mag_std         = mag_std,             # (n_batch, K)
        mag_all         = mag_array,           # (N_RUNS, n_batch, K)
        display_noisy   = display_noisy,       # (K, N)
        display_xi      = display_xi,          # (K_r, N)
        r_ex            = np.float32(R_EX),
        w               = np.float32(W),
        n_runs          = np.int32(N_RUNS),
        n_batch         = np.int32(N_BATCH),
    )
    print(f"\n[5] Results saved to: {out_path}")
    print("=" * 60)
    print(f"Done. Run: python structured_exp/plot_structured_exp.py --run-name {RUN_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()
