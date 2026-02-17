"""
Panel 9 - Separable noise / deformed-MP (data generation)
=========================================================

Generates correlated-noise datasets with planted spikes and records the metrics
needed to demonstrate how classical MP thresholds over-estimate K_eff whereas a
deformed edge (or whitening) recovers the true number of spikes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsup.spectral import (  # noqa: E402
    normalize_diagonal,
    marchenko_pastur_edge,
    approximate_deformed_edge,
)


PRESETS = {
    "quick": {
        "N_list": [256],
        "q_list": [0.5],
        "models": [
            {"kind": "ar", "rho": 0.3},
            {"kind": "ar", "rho": 0.6},
        ],
        "K_list": [2],
        "kappa_list": [1.5],
    },
    "full": {
        "N_list": [256, 512],
        "q_list": [0.3, 0.6],
        "models": [
            {"kind": "ar", "rho": 0.3},
            {"kind": "ar", "rho": 0.6},
            {"kind": "block", "variances": [1.0, 1.5, 2.0, 0.7]},
        ],
        "K_list": [2, 4],
        "kappa_list": [1.2, 1.5],
    },
}


@dataclass(frozen=True)
class NoiseModel:
    kind: str
    rho: float | None = None
    variances: List[float] | None = None

    def describe(self) -> Dict:
        base = {"kind": self.kind}
        if self.rho is not None:
            base["rho"] = self.rho
        if self.variances is not None:
            base["variances"] = self.variances
        return base


@dataclass(frozen=True)
class Panel9Condition:
    N: int
    q: float
    model: NoiseModel
    K: int
    kappa: float

    @property
    def n_samples(self) -> int:
        return max(1, int(round(self.N / float(self.q))))

    @property
    def q_eff(self) -> float:
        return self.N / float(self.n_samples)

    def describe(self) -> Dict:
        return {
            "N": self.N,
            "q": self.q_eff,
            "n_samples": self.n_samples,
            "K": self.K,
            "kappa": self.kappa,
            "noise": self.model.describe(),
        }


def build_noise_cov(cond: Panel9Condition) -> np.ndarray:
    if cond.model.kind == "ar":
        rho = cond.model.rho or 0.3
        idx = np.arange(cond.N)
        Sigma = rho ** np.abs(np.subtract.outer(idx, idx))
    elif cond.model.kind == "block":
        variances = cond.model.variances or [1.0, 1.5, 2.0, 0.7]
        n_blocks = len(variances)
        block_size = cond.N // n_blocks
        diag = np.concatenate([np.full(block_size, v) for v in variances])
        if diag.size < cond.N:
            diag = np.pad(diag, (0, cond.N - diag.size), constant_values=variances[-1])
        Sigma = np.diag(diag)
    else:
        raise ValueError(f"Unsupported noise model: {cond.model.kind}")
    return 0.5 * (Sigma + Sigma.T)


def add_spikes(Sigma: np.ndarray, K: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    N = Sigma.shape[0]
    mat = rng.standard_normal((N, K))
    V, _ = np.linalg.qr(mat)
    scatter = Sigma.copy()
    for mu in range(K):
        scatter += kappa * np.outer(V[:, mu], V[:, mu])
    return 0.5 * (scatter + scatter.T)


def whiten_matrix(Sigma: np.ndarray) -> np.ndarray:
    evals, vecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, 1e-9, None)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(evals)) @ vecs.T
    return inv_sqrt


def simulate_condition(
    cond: Panel9Condition,
    n_rep: int,
    rng: np.random.Generator,
    cushion: float,
    n_mc_edge: int,
) -> Dict:
    Sigma_noise = normalize_diagonal(build_noise_cov(cond))
    Sigma_total = add_spikes(Sigma_noise, cond.K, cond.kappa, rng)

    evals_tot, vecs_tot = np.linalg.eigh(Sigma_total)
    evals_tot = np.clip(evals_tot, 1e-9, None)
    Sigma_total_sqrt = (vecs_tot * np.sqrt(evals_tot)) @ vecs_tot.T

    q_eff = cond.q_eff
    mp_edge = marchenko_pastur_edge(q_eff, variance=1.0)
    def_edge = approximate_deformed_edge(Sigma_noise, q_eff, n_mc=n_mc_edge)
    mp_edge_cushioned = mp_edge * (1.0 + cushion)
    def_edge_cushioned = def_edge * (1.0 + cushion)

    Sigma_inv_sqrt = whiten_matrix(Sigma_noise)

    metrics = {
        "mp": {"keff": [], "false_outliers": [], "bias": [], "bulk_edge": [], "lambda_max": []},
        "def": {"keff": [], "false_outliers": [], "bias": [], "bulk_edge": [], "lambda_max": []},
        "white": {"keff": [], "false_outliers": [], "bias": [], "bulk_edge": [], "lambda_max": []},
    }

    for rep in range(n_rep):
        rep_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        Z = rep_rng.standard_normal((cond.N, cond.n_samples))
        X = Sigma_total_sqrt @ Z
        S = (X @ X.T) / float(cond.n_samples)
        S_norm = normalize_diagonal(S)
        evals = np.linalg.eigvalsh(S_norm)[::-1]

        # Naive MP
        bulk = evals[cond.K] if cond.K < len(evals) else evals[-1]
        keff = int(np.sum(evals > mp_edge_cushioned))
        false_out = max(0, keff - cond.K)
        metrics["mp"]["keff"].append(keff)
        metrics["mp"]["false_outliers"].append(false_out)
        metrics["mp"]["bias"].append(bulk - mp_edge_cushioned)
        metrics["mp"]["bulk_edge"].append(bulk)
        metrics["mp"]["lambda_max"].append(evals[0])

        # Deformed MP
        keff_def = int(np.sum(evals > def_edge_cushioned))
        bulk_def = bulk  # same bulk candidate
        false_def = max(0, keff_def - cond.K)
        metrics["def"]["keff"].append(keff_def)
        metrics["def"]["false_outliers"].append(false_def)
        metrics["def"]["bias"].append(bulk_def - def_edge_cushioned)
        metrics["def"]["bulk_edge"].append(bulk_def)
        metrics["def"]["lambda_max"].append(evals[0])

        # Whitening with true Sigma
        X_white = Sigma_inv_sqrt @ X
        S_white = (X_white @ X_white.T) / float(cond.n_samples)
        S_white = normalize_diagonal(S_white)
        evals_white = np.linalg.eigvalsh(S_white)[::-1]
        bulk_white = evals_white[cond.K] if cond.K < len(evals_white) else evals_white[-1]
        keff_white = int(np.sum(evals_white > mp_edge_cushioned))
        false_white = max(0, keff_white - cond.K)
        metrics["white"]["keff"].append(keff_white)
        metrics["white"]["false_outliers"].append(false_white)
        metrics["white"]["bias"].append(bulk_white - mp_edge_cushioned)
        metrics["white"]["bulk_edge"].append(bulk_white)
        metrics["white"]["lambda_max"].append(evals_white[0])

    for method in metrics:
        for key in metrics[method]:
            metrics[method][key] = np.asarray(metrics[method][key], dtype=np.float32)

    return {
        "condition": cond.describe(),
        "thresholds": {
            "mp_edge": mp_edge,
            "mp_edge_cushioned": mp_edge_cushioned,
            "def_edge": def_edge,
            "def_edge_cushioned": def_edge_cushioned,
            "cushion": cushion,
        },
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Panel 9 separable-noise data.")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="quick")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--cushion", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--edge_mc", type=int, default=16, help="Monte-Carlo reps for deformed edge.")
    parser.add_argument("--output", type=Path, default=Path("graphs/panel9/output/panel9_separable_data.npz"))
    parser.add_argument("--log", type=Path, default=Path("graphs/panel9/output/panel9_separable_log.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PRESETS[args.preset]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    models = []
    for model_dict in cfg["models"]:
        models.append(
            NoiseModel(
                kind=model_dict["kind"],
                rho=model_dict.get("rho"),
                variances=model_dict.get("variances"),
            )
        )

    grid = [
        Panel9Condition(N=N, q=q, model=model, K=K, kappa=kappa)
        for N in cfg["N_list"]
        for q in cfg["q_list"]
        for model in models
        for K in cfg["K_list"]
        for kappa in cfg["kappa_list"]
    ]

    print(f"[Panel9] Running {len(grid)} conditions × {args.replicates} replicates (preset={args.preset}).")
    results = []
    summary = []
    for idx, cond in enumerate(grid, 1):
        noise_desc = cond.model.describe()
        print(f"  - Condition {idx}/{len(grid)} | N={cond.N}, q={cond.q_eff:.2f}, "
              f"model={noise_desc}, K={cond.K}, kappa={cond.kappa}")
        cond_result = simulate_condition(
            cond,
            n_rep=args.replicates,
            rng=rng,
            cushion=args.cushion,
            n_mc_edge=args.edge_mc,
        )
        results.append(cond_result)

        summary.append({
            **cond.describe(),
            "mp_false": float(np.mean(cond_result["metrics"]["mp"]["false_outliers"])),
            "def_false": float(np.mean(cond_result["metrics"]["def"]["false_outliers"])),
            "white_false": float(np.mean(cond_result["metrics"]["white"]["false_outliers"])),
            "mp_keff": float(np.mean(cond_result["metrics"]["mp"]["keff"])),
            "def_keff": float(np.mean(cond_result["metrics"]["def"]["keff"])),
        })

    np.savez_compressed(
        output_path,
        results=np.array(results, dtype=object),
        meta=np.array({"preset": args.preset, "replicates": args.replicates}, dtype=object),
    )
    print(f"[Panel9] Saved data to {output_path}")

    with open(args.log, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "preset": args.preset,
                "replicates": args.replicates,
                "cushion": args.cushion,
                "summary": summary,
            },
            fh,
            indent=2,
        )
    print(f"[Panel9] Wrote log to {args.log}")


if __name__ == "__main__":
    main()
