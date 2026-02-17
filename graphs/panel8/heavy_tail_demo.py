"""
Panel 8 - Heavy-tail stress test (data generation)
==================================================

This script generates the synthetic datasets and summary statistics required to
plot Panel 8.  It compares classical sample covariance (SCM) against the Tyler
shape estimator under heavy-tailed elliptical noise with planted spikes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsup.spectral import (  # noqa: E402
    tyler_shape_matrix,
    normalize_diagonal,
    marchenko_pastur_edge,
)


PRESETS = {
    # Quick sanity-check configuration suggested in the protocol footnote
    "quick": {
        "N_list": [256],
        "q_list": [0.5],
        "nu_list": [3, 5, np.inf],
        "K_list": [2],
        "kappa_list": [1.5],
    },
    # Full sweep used for publication-ready statistics
    "full": {
        "N_list": [256, 512],
        "q_list": [0.3, 0.6],
        "nu_list": [3, 5, 8, np.inf],
        "K_list": [2, 4],
        "kappa_list": [1.2, 1.5],
    },
}


@dataclass(frozen=True)
class Panel8Condition:
    N: int
    q: float
    nu: float
    K: int
    kappa: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "N": int(self.N),
            "q": float(self.q),
            "nu": float(self.nu),
            "K": int(self.K),
            "kappa": float(self.kappa),
            "n_samples": int(self.n_samples),
        }

    @property
    def n_samples(self) -> int:
        return max(1, int(round(self.N / float(self.q))))

    @property
    def q_eff(self) -> float:
        return self.N / float(self.n_samples)


def generate_spike_basis(N: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthonormal basis for the planted spikes."""
    mat = rng.standard_normal((N, K))
    Q, _ = np.linalg.qr(mat)
    return Q[:, :K]


def build_population_cov(N: int, K: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """
    Population scatter = I + sum_{mu=1}^K kappa * v_mu v_mu^T.
    """
    V = generate_spike_basis(N, K, rng)
    Sigma = np.eye(N)
    for mu in range(K):
        Sigma += float(kappa) * np.outer(V[:, mu], V[:, mu])
    return 0.5 * (Sigma + Sigma.T)


def sample_t_student_radii(df: float, n: int, rng: np.random.Generator) -> np.ndarray:
    if np.isinf(df):
        return np.ones(n, dtype=np.float64)
    chi2 = rng.chisquare(df, size=n)
    chi2 = np.clip(chi2, 1e-9, None)
    return np.sqrt(df / chi2)


def simulate_condition(
    cond: Panel8Condition,
    n_rep: int,
    rng: np.random.Generator,
    cushion: float,
    tyler_tol: float,
    tyler_max_iter: int,
) -> Dict:
    Sigma = build_population_cov(cond.N, cond.K, cond.kappa, rng)
    evals, vecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, 1e-9, None)
    Sigma_sqrt = (vecs * np.sqrt(evals)) @ vecs.T
    Sigma_norm = normalize_diagonal(Sigma)

    q_eff = cond.q_eff
    tau_mp = marchenko_pastur_edge(q_eff, variance=1.0)
    tau = tau_mp * (1.0 + cushion)

    metrics = {
        "sample": {"keff": [], "false_outliers": [], "fro": [], "bulk_edge": [], "lambda_max": []},
        "tyler": {"keff": [], "false_outliers": [], "fro": [], "bulk_edge": [], "lambda_max": [], "iters": []},
    }

    fro_norm = np.linalg.norm(Sigma_norm, ord="fro")

    for rep in range(n_rep):
        rep_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        Z = rep_rng.standard_normal((cond.N, cond.n_samples))
        radii = sample_t_student_radii(cond.nu, cond.n_samples, rep_rng)
        X = Sigma_sqrt @ (Z * radii)

        S_sample = (X @ X.T) / float(cond.n_samples)
        S_sample = normalize_diagonal(S_sample)
        evals_sample = np.linalg.eigvalsh(S_sample)[::-1]

        res_tyler = tyler_shape_matrix(X, tol=tyler_tol, max_iter=tyler_max_iter)
        S_tyler = normalize_diagonal(res_tyler.scatter)
        evals_tyler = np.linalg.eigvalsh(S_tyler)[::-1]

        for name, evals, matrix, extra in [
            ("sample", evals_sample, S_sample, {}),
            ("tyler", evals_tyler, S_tyler, {"iters": res_tyler.iters}),
        ]:
            keff = int(np.sum(evals > tau))
            false_out = max(0, keff - cond.K)
            bulk_edge = evals[cond.K] if cond.K < len(evals) else evals[-1]
            fro_err = np.linalg.norm(matrix - Sigma_norm, ord="fro") / fro_norm

            metrics[name]["keff"].append(keff)
            metrics[name]["false_outliers"].append(false_out)
            metrics[name]["fro"].append(fro_err)
            metrics[name]["bulk_edge"].append(bulk_edge)
            metrics[name]["lambda_max"].append(evals[0])
            if "iters" in extra:
                metrics[name]["iters"].append(extra["iters"])

    for method in metrics:
        for k, values in metrics[method].items():
            metrics[method][k] = np.asarray(values, dtype=np.float32)

    return {
        "condition": cond.as_dict(),
        "thresholds": {"mp_edge": tau_mp, "mp_edge_cushioned": tau, "cushion": cushion},
        "population": {
            "fro_norm": fro_norm,
            "trace": float(np.trace(Sigma)),
            "spike_strength": float(1.0 + cond.kappa),
        },
        "metrics": metrics,
        "tyler": {
            "tol": tyler_tol,
            "max_iter": tyler_max_iter,
            "convergence_rate": float(np.mean(metrics["tyler"]["iters"])) if len(metrics["tyler"]["iters"]) else 0.0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Panel 8 heavy-tail stress-test data.")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="quick",
                        help="Parameter grid to use (default: quick).")
    parser.add_argument("--N", type=int, nargs="+", help="Override list of N values.")
    parser.add_argument("--q", type=float, nargs="+", help="Override list of q values.")
    parser.add_argument("--nu", type=float, nargs="+", help="Override nu grid.")
    parser.add_argument("--K", type=int, nargs="+", help="Override K grid.")
    parser.add_argument("--kappa", type=float, nargs="+", help="Override kappa grid.")
    parser.add_argument("--replicates", type=int, default=100, help="Number of Monte-Carlo replicates per condition.")
    parser.add_argument("--seed", type=int, default=2025, help="Base RNG seed.")
    parser.add_argument("--cushion", type=float, default=0.02, help="Relative cushion added to MP edge.")
    parser.add_argument("--tyler_tol", type=float, default=1e-6, help="Tyler relative Frobenius tolerance.")
    parser.add_argument("--tyler_max_iter", type=int, default=200, help="Tyler maximum iterations.")
    parser.add_argument("--output", type=Path, default=Path("graphs/panel8/output/panel8_heavy_tail_data.npz"),
                        help="Path to the compressed NPZ file to write.")
    parser.add_argument("--log", type=Path, default=Path("graphs/panel8/output/panel8_heavy_tail_log.json"),
                        help="Path to the JSON log file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = PRESETS[args.preset].copy()
    if args.N:
        base_cfg["N_list"] = args.N
    if args.q:
        base_cfg["q_list"] = args.q
    if args.nu:
        base_cfg["nu_list"] = args.nu
    if args.K:
        base_cfg["K_list"] = args.K
    if args.kappa:
        base_cfg["kappa_list"] = args.kappa

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    conditions: List[Dict] = []
    summary = []

    grid = [
        Panel8Condition(N=N, q=q, nu=nu, K=K, kappa=kappa)
        for N in base_cfg["N_list"]
        for q in base_cfg["q_list"]
        for nu in base_cfg["nu_list"]
        for K in base_cfg["K_list"]
        for kappa in base_cfg["kappa_list"]
    ]

    print(f"[Panel8] Running {len(grid)} conditions × {args.replicates} replicates each (preset={args.preset}).")
    for idx, cond in enumerate(grid, 1):
        print(f"  - Condition {idx}/{len(grid)} | N={cond.N}, q={cond.q}, nu={cond.nu}, "
              f"K={cond.K}, kappa={cond.kappa}")
        cond_result = simulate_condition(
            cond,
            n_rep=args.replicates,
            rng=rng,
            cushion=args.cushion,
            tyler_tol=args.tyler_tol,
            tyler_max_iter=args.tyler_max_iter,
        )
        conditions.append(cond_result)

        sample_keff = cond_result["metrics"]["sample"]["keff"].mean()
        tyler_keff = cond_result["metrics"]["tyler"]["keff"].mean()
        summary.append({
            **cond.as_dict(),
            "sample_keff_mean": float(sample_keff),
            "tyler_keff_mean": float(tyler_keff),
            "sample_false_rate": float(np.mean(cond_result["metrics"]["sample"]["false_outliers"])),
            "tyler_false_rate": float(np.mean(cond_result["metrics"]["tyler"]["false_outliers"])),
        })

    np.savez_compressed(
        output_path,
        results=np.array(conditions, dtype=object),
        meta=np.array({"preset": args.preset, "replicates": args.replicates}, dtype=object),
    )
    print(f"[Panel8] Saved data to {output_path}")

    with open(args.log, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "preset": args.preset,
                "replicates": args.replicates,
                "cushion": args.cushion,
                "tyler": {"tol": args.tyler_tol, "max_iter": args.tyler_max_iter},
                "summary": summary,
            },
            fh,
            indent=2,
        )
    print(f"[Panel8] Wrote log to {args.log}")


if __name__ == "__main__":
    main()
