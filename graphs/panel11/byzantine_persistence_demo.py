"""
Panel 11 - Fake spikes / Byzantine persistence (data generation)

Simulates adversarial clients injecting rank-1 components and evaluates
baseline, persistence, and persistence+cross-shard defenses in terms of ROC,
false-positive rate, and ΔK_eff.
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
class Panel11Config:
    N: int = 256
    q: float = 0.5
    L: int = 50
    s_shards: int = 5
    T: int = 80
    k_base: int = 2
    kappa_base: float = 1.5
    kappa_decay: float = 1.0
    alpha_values: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    attacker_counts: Tuple[int, ...] = (0, 1, 3, 5)
    attack_threshold: float = 0.2
    cushion: float = 0.02
    m_persist: int = 3
    eta_align: float = 0.9
    eta_shard: float = 0.8
    theta_vote: float = 0.4

    @property
    def n_samples(self) -> int:
        return int(round(self.N / self.q))

    @property
    def shard_samples(self) -> int:
        return max(100, self.n_samples // self.s_shards)

    @property
    def clients_per_shard(self) -> int:
        return max(1, self.L // self.s_shards)

    @property
    def tau(self) -> float:
        return (1.0 + self.cushion) * marchenko_pastur_edge(self.q, variance=1.0)


def random_orthonormal(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.standard_normal((dim, k))
    q, _ = np.linalg.qr(mat)
    return q[:, :k]


def build_sigma_sqrt(vecs: np.ndarray, kappas: np.ndarray) -> np.ndarray:
    N, K = vecs.shape
    Sigma_sqrt = np.eye(N, dtype=np.float64)
    for idx in range(K):
        gain = np.sqrt(1.0 + kappas[idx]) - 1.0
        Sigma_sqrt += gain * np.outer(vecs[:, idx], vecs[:, idx])
    return Sigma_sqrt


def sample_shard_cov(cfg: Panel11Config, Sigma_sqrt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Z = rng.standard_normal((cfg.shard_samples, cfg.N))
    X = Z @ Sigma_sqrt
    M = (X.T @ X) / cfg.shard_samples
    return M


def rotate_vector(prev: np.ndarray, rng: np.random.Generator, angle: float = np.deg2rad(30)) -> np.ndarray:
    noise = rng.standard_normal(prev.shape[0])
    noise -= noise.dot(prev) * prev
    norm = np.linalg.norm(noise)
    if norm < 1e-8:
        noise = rng.standard_normal(prev.shape[0])
        noise -= noise.dot(prev) * prev
        norm = np.linalg.norm(noise)
    noise /= norm
    vec = np.cos(angle) * prev + np.sin(angle) * noise
    vec /= np.linalg.norm(vec)
    return vec


def roc_curve(scores: List[float], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)
    pos = np.count_nonzero(labels_arr)
    neg = len(labels_arr) - pos
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(scores_arr)[::-1]
    sorted_labels = labels_arr[order]
    sorted_scores = scores_arr[order]
    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    prev_score = np.inf
    for score, label in zip(sorted_scores, sorted_labels):
        if score != prev_score:
            tpr.append(tp / pos)
            fpr.append(fp / neg)
            prev_score = score
        if label:
            tp += 1
        else:
            fp += 1
    tpr.append(tp / pos)
    fpr.append(fp / neg)
    return np.array(fpr), np.array(tpr)


def fpr_at_tpr(scores: List[float], labels: List[int], target: float = 0.9) -> float:
    fpr, tpr = roc_curve(scores, labels)
    if np.all(np.isnan(tpr)) or len(tpr) == 0:
        return float("nan")
    if np.max(tpr) < target:
        return float("nan")
    return float(np.interp(target, tpr, fpr))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Panel 11 Byzantine persistence data.")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("graphs/panel11/output/panel11_byzantine_data.npz"))
    parser.add_argument("--log", type=Path,
                        default=Path("graphs/panel11/output/panel11_byzantine_log.json"))
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Panel11Config()
    rng = np.random.default_rng(args.seed)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    defenses = ["baseline", "persistence", "combo"]
    all_scores = {d: [] for d in defenses}
    all_labels: List[int] = []
    scenario_scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        d: {} for d in defenses
    }
    delta_keff = {d: np.zeros(args.replicates, dtype=np.float64) for d in defenses}
    heatmap = {"scores": [], "labels": []}
    heatmap_recorded = False

    attack_scenarios = [(c, alpha) for c in cfg.attacker_counts if c > 0 for alpha in cfg.alpha_values]

    for rep in range(args.replicates):
        rep_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        vecs = random_orthonormal(cfg.N, cfg.k_base, rep_rng)
        kappas = np.array(
            [max(0.5, cfg.kappa_base - cfg.kappa_decay * i) for i in range(cfg.k_base)],
            dtype=np.float64,
        )
        Sigma_sqrt = build_sigma_sqrt(vecs, kappas)

        u_stationary = rep_rng.standard_normal(cfg.N)
        u_stationary /= np.linalg.norm(u_stationary)
        u_wander = rep_rng.standard_normal(cfg.N)
        u_wander /= np.linalg.norm(u_wander)

        round_plan = []
        repeats = max(1, cfg.T // max(1, len(attack_scenarios)))
        for _ in range(repeats):
            round_plan.extend(attack_scenarios)
        round_plan = round_plan[: cfg.T // 2]
        clean_rounds = [(0, 0.0)] * (cfg.T - len(round_plan))
        combined = round_plan + clean_rounds
        rep_rng.shuffle(combined)

        labels_round: List[int] = []
        scores_round = {d: [] for d in defenses}
        dels_round = {d: [] for d in defenses}
        S_series: List[float] = []

        align_history: List[float] = []
        prev_vec = None

        for t, (c_attack, alpha) in enumerate(combined):
            label = int((c_attack > 0) and (alpha >= cfg.attack_threshold))
            labels_round.append(label)

            # decide attack direction
            if c_attack > 0:
                stationary = rep_rng.random() < 0.5
                if stationary:
                    u_vec = u_stationary
                else:
                    u_wander = rotate_vector(u_wander, rep_rng)
                    u_vec = u_wander
            else:
                u_vec = None

            shard_attacks = np.zeros(cfg.s_shards, dtype=int)
            if c_attack > 0:
                shard_attacks = rep_rng.multinomial(c_attack, [1 / cfg.s_shards] * cfg.s_shards)

            shard_mats = []
            shard_eval_cache = []
            for j in range(cfg.s_shards):
                M_j = sample_shard_cov(cfg, Sigma_sqrt, rep_rng)
                if c_attack > 0 and shard_attacks[j] > 0 and u_vec is not None:
                    coeff = alpha * shard_attacks[j]
                    M_j += coeff * np.outer(u_vec, u_vec)
                shard_mats.append(M_j)
                shard_eval_cache.append(np.linalg.eigh(M_j))
            M_full = np.mean(shard_mats, axis=0)

            evals_full, evecs_full = np.linalg.eigh(M_full)
            evals_full = evals_full[::-1]
            evecs_full = evecs_full[:, ::-1]
            keff = int(np.sum(evals_full > cfg.tau))
            lambda_top = float(evals_full[0])
            vec_top = evecs_full[:, 0]
            baseline_score = max(0.0, lambda_top - cfg.tau)

            # persistence state via rolling alignment window
            if lambda_top > cfg.tau:
                align = 1.0 if prev_vec is None else abs(prev_vec.dot(vec_top))
                align_history.append(align)
                if len(align_history) > cfg.m_persist:
                    align_history.pop(0)
                prev_vec = vec_top
            else:
                align_history.clear()
                prev_vec = None

            persist_strength = float(np.mean(align_history)) if align_history else 0.0
            persist_ok = len(align_history) >= cfg.m_persist and persist_strength >= cfg.eta_align
            combo_ready = persist_strength >= 0.3

            votes = 0
            shard_scores = []
            for evals_j, evecs_j in shard_eval_cache:
                v_j = evecs_j[:, -1]
                score_j = max(0.0, float(evals_j[-1]) - cfg.tau)
                shard_scores.append(score_j)
                if abs(v_j.dot(vec_top)) >= cfg.eta_shard:
                    votes += 1
            S_t = votes / cfg.s_shards
            shard_scores = np.array(shard_scores, dtype=np.float64)
            combo_ok = combo_ready and (S_t >= cfg.theta_vote)

            combo_strength = S_t

            scores_round["baseline"].append(baseline_score + shard_scores.mean())
            scores_round["persistence"].append((baseline_score + shard_scores.mean()) * persist_strength)
            scores_round["combo"].append((baseline_score + shard_scores.mean()) * combo_strength)

            reject_persist = 0
            if t < cfg.m_persist:
                reject_combo = 0
            else:
                reject_combo = int(
                    lambda_top > cfg.tau and combo_ready and (S_t < cfg.theta_vote) and baseline_score > 0
                )

            dels_round["baseline"].append(abs(keff - cfg.k_base))
            dels_round["persistence"].append(abs(max(0, keff - reject_persist) - cfg.k_base))
            dels_round["combo"].append(abs(max(0, keff - reject_combo) - cfg.k_base))

            scenario_key = f"c={c_attack},alpha={alpha:.1f}"
            for def_name in defenses:
                scenario_scores.setdefault(def_name, {})
                scenario_scores[def_name].setdefault(scenario_key, {"scores": [], "labels": []})
                scenario_scores[def_name][scenario_key]["scores"].append(scores_round[def_name][-1])
                scenario_scores[def_name][scenario_key]["labels"].append(label)

            if not heatmap_recorded:
                S_series.append(S_t)

        if not heatmap_recorded:
            heatmap["scores"] = S_series
            heatmap["labels"] = labels_round
            heatmap_recorded = True

        for def_name in defenses:
            all_scores[def_name].extend(scores_round[def_name])
            delta_keff[def_name][rep] = float(np.mean(dels_round[def_name]))
        all_labels.extend(labels_round)

    roc_curves = {}
    for def_name in defenses:
        fpr, tpr = roc_curve(all_scores[def_name], all_labels)
        roc_curves[def_name] = {"fpr": fpr, "tpr": tpr}

    fpr_table = {def_name: {} for def_name in defenses}
    for def_name in defenses:
        for scenario, payload in scenario_scores[def_name].items():
            fpr_val = fpr_at_tpr(payload["scores"], payload["labels"], target=0.9)
            fpr_table[def_name][scenario] = fpr_val

    np.savez_compressed(
        output_path,
        defenses=np.array(defenses),
        roc=np.array(roc_curves, dtype=object),
        fpr_table=np.array(fpr_table, dtype=object),
        delta_keff=np.array(delta_keff, dtype=object),
        heatmap=np.array(heatmap, dtype=object),
        config=np.array(cfg.__dict__, dtype=object),
    )

    log = {
        "config": cfg.__dict__,
        "replicates": args.replicates,
        "delta_keff_mean": {d: float(delta_keff[d].mean()) for d in defenses},
        "roc_summary": {
            d: {
                "auc": float(np.trapz(roc_curves[d]["tpr"], roc_curves[d]["fpr"]))
            }
            for d in defenses
        },
    }
    with open(args.log, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)

    print(f"[Panel11] Saved data to {output_path}")
    print(f"[Panel11] Log written to {args.log}")


if __name__ == "__main__":
    main()
