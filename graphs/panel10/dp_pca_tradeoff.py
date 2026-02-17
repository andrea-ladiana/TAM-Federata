"""
Panel 10 - DP PCA / Wishart trade-off (data generation)

Simulates a federated covariance aggregation pipeline with per-client clipping
and Gaussian mechanism noise to study how the privacy budget ε impacts novelty
detection, K_eff bias, and eigenvector overlap with true archetypes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsup.spectral import marchenko_pastur_edge  # noqa: E402


@dataclass(frozen=True)
class Panel10Config:
    N: int = 256
    q: float = 0.5
    L: int = 100
    T: int = 30
    clients_per_round: int = 100
    clip_C: float = 1.0
    delta: float = 1e-5
    k_base: int = 2
    kappa_base: float = 1.5
    kappa_novel: float = 1.2
    novelty_prob: float = 0.35
    cushion: float = 0.02
    persistence_rounds: int = 3  # for novelty definition if needed

    @property
    def n_samples(self) -> int:
        return int(round(self.N / self.q))

    @property
    def samples_per_client(self) -> int:
        return max(8, self.n_samples // self.L)

    @property
    def tau(self) -> float:
        return (1.0 + self.cushion) * marchenko_pastur_edge(self.q, variance=1.0)


def random_orthonormal(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.standard_normal((dim, k))
    q, _ = np.linalg.qr(mat)
    return q[:, :k]


def clip_operator_norm(M: np.ndarray, C: float) -> np.ndarray:
    evals = np.linalg.eigvalsh(M)
    lmax = float(evals.max())
    if lmax <= C:
        return M
    return (C / lmax) * M


def build_sigma_sqrt(
    vecs: np.ndarray,
    kappas: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    # vecs shape (N, K_total), kappas same length
    N = vecs.shape[0]
    Sigma_sqrt = np.eye(N, dtype=np.float64)
    for idx, active in enumerate(active_mask):
        if not active:
            continue
        gain = np.sqrt(1.0 + kappas[idx]) - 1.0
        Sigma_sqrt += gain * np.outer(vecs[:, idx], vecs[:, idx])
    return Sigma_sqrt


def gaussian_symmetric_noise(N: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return np.zeros((N, N), dtype=np.float64)
    tri = rng.normal(0.0, sigma, size=(N, N))
    noise = np.tril(tri) + np.tril(tri, -1).T
    np.fill_diagonal(noise, rng.normal(0.0, sigma, size=N))
    return noise


def compute_auc(labels: List[int], scores: List[float]) -> float:
    labels_arr = np.asarray(labels, dtype=np.int32)
    scores_arr = np.asarray(scores, dtype=np.float64)
    pos = labels_arr == 1
    neg = labels_arr == 0
    n_pos = np.count_nonzero(pos)
    n_neg = np.count_nonzero(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores_arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores_arr) + 1)
    sum_pos = float(np.sum(ranks[pos]))
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def overlaps_with_spikes(
    eigenvectors: np.ndarray,
    spike_vecs: np.ndarray,
) -> np.ndarray:
    """
    Parameters
    ----------
    eigenvectors : (N, k)
    spike_vecs : (k_spike, N)
    """
    if eigenvectors.size == 0 or spike_vecs.size == 0:
        return np.zeros(spike_vecs.shape[0], dtype=np.float64)
    proj = np.abs(spike_vecs @ eigenvectors)
    proj_sq = proj ** 2
    return proj_sq.max(axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Panel 10 DP-PCA trade-off data.")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[4.0, 6.0, 8.0, 12.0, 16.0, np.inf])
    parser.add_argument("--output", type=Path,
                        default=Path("graphs/panel10/output/panel10_dp_pca_data.npz"))
    parser.add_argument("--log", type=Path,
                        default=Path("graphs/panel10/output/panel10_dp_pca_log.json"))
    parser.add_argument("--novelty_prob", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Panel10Config(novelty_prob=args.novelty_prob)
    eps_list = [float(eps) for eps in args.epsilons]
    rng = np.random.default_rng(args.seed)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    n_eps = len(eps_list)
    metrics = {
        "auc": np.full((n_eps, args.replicates), np.nan, dtype=np.float64),
        "delta_keff": np.zeros((n_eps, args.replicates), dtype=np.float64),
        "overlap": np.zeros((n_eps, args.replicates), dtype=np.float64),
    }

    # Precompute DP noise scales for each epsilon
    delta_F = 2 * cfg.clip_C / cfg.L
    sigma_map: Dict[float, float] = {}
    for eps in eps_list:
        if np.isinf(eps) or eps <= 0:
            sigma_map[eps] = 0.0
        else:
            sigma_map[eps] = (2 * delta_F * np.sqrt(cfg.T * np.log(1.0 / cfg.delta))) / eps

    example_spectra = {
        "epsilons": [4.0, 8.0, np.inf],
        "eigs": {},
        "tau": None,
    }
    example_captured = False

    for rep in range(args.replicates):
        rep_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        # Spike directions: base + novel
        total_spikes = cfg.k_base + 1
        vecs = random_orthonormal(cfg.N, total_spikes, rep_rng)
        kappas = np.zeros(total_spikes, dtype=np.float64)
        kappas[:cfg.k_base] = cfg.kappa_base
        kappas[-1] = cfg.kappa_novel

        labels: List[int] = []
        per_eps_scores = {eps: [] for eps in eps_list}
        per_eps_delta = {eps: [] for eps in eps_list}
        per_eps_overlap = {eps: [] for eps in eps_list}

        for t in range(cfg.T):
            novel = int(rep_rng.random() < cfg.novelty_prob)
            labels.append(novel)
            active_mask = np.zeros(total_spikes, dtype=bool)
            active_mask[:cfg.k_base] = True
            active_mask[-1] = bool(novel)

            Sigma_sqrt = build_sigma_sqrt(vecs, kappas, active_mask)

            # Aggregate clipped covariances
            M_accum = np.zeros((cfg.N, cfg.N), dtype=np.float64)
            for _ in range(cfg.clients_per_round):
                Z = rep_rng.standard_normal((cfg.samples_per_client, cfg.N))
                X = Z @ Sigma_sqrt
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                clip_norm = np.sqrt(cfg.N)
                scale_norm = np.maximum(1.0, norms / clip_norm)
                X = X / scale_norm
                M_l = (X.T @ X) / cfg.samples_per_client
                M_l = clip_operator_norm(M_l, cfg.clip_C)
                M_accum += M_l
            M_bar = M_accum / cfg.clients_per_round

            v_t = float(np.trace(M_bar)) / max(cfg.N, 1e-9)
            tau_t = (1.0 + cfg.cushion) * marchenko_pastur_edge(cfg.q, variance=v_t)

            evals_clean, evecs_clean = np.linalg.eigh(M_bar)
            evals_clean = evals_clean[::-1]
            evecs_clean = evecs_clean[:, ::-1]
            keff_clean = int(np.sum(evals_clean > tau_t))

            # store example spectrum (first replicate, first positive round)
            if (not example_captured) and novel == 1:
                example_spectra["round_index"] = t
                example_spectra["tau"] = tau_t

            # Precompute spike vectors active this round
            active_vecs = vecs[:, active_mask]

            for eps in eps_list:
                base_sigma = sigma_map[eps]
                sigma_round = base_sigma * max(v_t, 1e-9)
                noise = gaussian_symmetric_noise(cfg.N, sigma_round, rep_rng)
                M_dp = M_bar + noise
                evals_dp, evecs_dp = np.linalg.eigh(M_dp)
                evals_dp = evals_dp[::-1]
                evecs_dp = evecs_dp[:, ::-1]

                mask_outlier = evals_dp > tau_t
                keff_dp = int(np.sum(mask_outlier))
                per_eps_delta[eps].append(abs(keff_dp - keff_clean))

                top_k = min(total_spikes, evecs_dp.shape[1])
                top_vecs = evecs_dp[:, :top_k]
                overlaps = overlaps_with_spikes(top_vecs, active_vecs.T)
                if overlaps.size == 0:
                    per_eps_overlap[eps].append(0.0)
                else:
                    per_eps_overlap[eps].append(float(np.mean(overlaps)))

                score_t = float(np.sum(np.maximum(evals_dp - tau_t, 0.0)))
                per_eps_scores[eps].append(score_t)

                if (not example_captured) and novel == 1 and eps in example_spectra["epsilons"]:
                    example_spectra["eigs"][str(eps)] = evals_dp.copy()

            if (not example_captured) and novel == 1:
                if all(str(eps) in example_spectra["eigs"] for eps in example_spectra["epsilons"]):
                    example_captured = True

        for idx, eps in enumerate(eps_list):
            metrics["auc"][idx, rep] = compute_auc(labels, per_eps_scores[eps])
            metrics["delta_keff"][idx, rep] = float(np.mean(per_eps_delta[eps]))
            metrics["overlap"][idx, rep] = float(np.mean(per_eps_overlap[eps]))

    np.savez_compressed(
        output_path,
        epsilons=np.array(eps_list, dtype=np.float64),
        auc=metrics["auc"],
        delta_keff=metrics["delta_keff"],
        overlap=metrics["overlap"],
        config=np.array(cfg.__dict__, dtype=object),
        spectra_example=np.array(example_spectra, dtype=object),
    )

    summary = {
        "config": cfg.__dict__,
        "epsilons": eps_list,
        "sigma_map": sigma_map,
        "replicates": args.replicates,
        "metrics_mean": {
            "auc": metrics["auc"].mean(axis=1).tolist(),
            "delta_keff": metrics["delta_keff"].mean(axis=1).tolist(),
            "overlap": metrics["overlap"].mean(axis=1).tolist(),
        },
    }
    with open(args.log, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[Panel10] Saved data to {output_path}")
    print(f"[Panel10] Log written to {args.log}")


if __name__ == "__main__":
    main()
