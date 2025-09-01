from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Tuple

# Reuse core primitives from existing modules (kept intact)
# Import lazily to avoid hard failure if dynamics has encoding issues in some environments
try:
    from .dynamics import (
        new_round,
        dis_check,
    )
except Exception:  # pragma: no cover
    def new_round(*args, **kwargs):  # type: ignore
        raise ImportError("new_round unavailable: failed to import from unsup.dynamics")
    def dis_check(*args, **kwargs):  # type: ignore
        raise ImportError("dis_check unavailable: failed to import from unsup.dynamics")

# Local helpers mirroring Dynamics.py impl (imported lazily where needed)
def _match_and_overlap(estimated: np.ndarray, true: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    from scipy.optimize import linear_sum_assignment  # local import
    K_hat, N = estimated.shape
    K, N2 = true.shape
    assert N == N2
    M = np.abs(estimated @ true.T / N)
    cost = 1.0 - M
    row_ind, col_ind = linear_sum_assignment(cost)
    overlaps = M[row_ind, col_ind]
    if K_hat < K:
        mean_overlap = overlaps.sum() / K
    else:
        mean_overlap = overlaps.mean()
    return float(mean_overlap), overlaps, col_ind

def _fro_norm_rel(J_est: np.ndarray, J_true: np.ndarray) -> float:
    return float(np.linalg.norm(J_est - J_true, ord='fro') / (np.linalg.norm(J_true, ord='fro') + 1e-9))

def robust_z(values: List[float]) -> List[float]:
    vals = np.array(values, dtype=float)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-9
    return list((vals - med) / (1.4826 * mad))

def _stability_across_seeds(pattern_sets: List[np.ndarray], true: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment  # noqa: F401  # only to ensure scipy present
    if len(pattern_sets) < 2:
        return 0.0
    overlaps_collected: List[float] = []
    for i in range(len(pattern_sets)):
        for j in range(i + 1, len(pattern_sets)):
            A = pattern_sets[i]
            B = pattern_sets[j]
            Ka, N = A.shape; Kb, _ = B.shape
            M = np.abs(A @ B.T / N)
            cost = 1 - M
            rI, cI = np.array(np.nonzero(M == M.max()))[:, 0] if (Ka == 0 or Kb == 0) else (None, None)
            try:
                from scipy.optimize import linear_sum_assignment as _lsa
                rI, cI = _lsa(cost)
            except Exception:
                if rI is None or cI is None:
                    continue
            overlaps_collected.append(M[rI, cI].mean())
    if not overlaps_collected:
        return 0.0
    arr = np.array(overlaps_collected)
    return float(max(0.0, 1 - arr.var()))


def run_experiment_single_mode(base_params: Dict[str, Any], w_value: float, seed: int) -> Dict[str, Any]:
    """
    Single-mode variant: at round b compute J from ONLY round-b examples (no extend).
    Returns metrics and series analogous to Dynamics.run_single_experiment but in 'single' mode.
    """
    # Lazy imports to avoid hard dependency when unused
    from .functions import (
        gen_patterns, JK_real, gen_dataset_unsup, unsupervised_J, propagate_J, estimate_K_eff_from_J
    )

    rng = np.random.default_rng(seed)
    L = base_params['L']; K = base_params['K']; N = base_params['N']
    M_unsup = base_params['M_unsup']; r_ex = base_params['r_ex']; n_batch = base_params['n_batch']
    up = base_params['updates']
    use_mp = base_params.get('use_mp_keff', True)
    shuffle_alpha = base_params.get('shuffle_alpha', 0.01)
    shuffle_random = base_params.get('shuffle_random', 32)

    xi_true = gen_patterns(N, K)
    J_star = JK_real(xi_true)
    ETA_unsup = gen_dataset_unsup(xi_true, M_unsup, r_ex, n_batch, L)

    def _mean_unsup_J_per_layer_local(tensor_L_M_N, Kx):
        L_loc, M_eff_actual, N_loc = tensor_L_M_N.shape
        M_eff_param = max(1, M_eff_actual // Kx)
        Js = [unsupervised_J(tensor_L_M_N[l], M_eff_param) for l in range(L_loc)]
        return np.sum(Js, axis=0) / L_loc, M_eff_param

    xi_ref = None
    fro_single_rounds: List[float] = []
    magn_single_rounds: List[float] = []
    keff_single_rounds: List[int] = []
    patterns_final = None
    K_eff_final = None

    for b in range(n_batch):
        ETA_round = new_round(ETA_unsup, round=b, collection=False)
        J_unsup_single, M_eff_round = _mean_unsup_J_per_layer_local(ETA_round, K)
        if b == 0:
            J_rec = J_unsup_single.copy()
        else:
            # Hebbian memory from previously disentangled patterns
            J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
            J_rec = w_value * J_unsup_single + (1 - w_value) * J_hebb_prev

        # One-step propagation and disentangling
        JKS_iter = propagate_J(J_rec, iters=1, verbose=False)
        vals, vecs = np.linalg.eig(JKS_iter)
        mask = (np.real(vals) > 0.5)
        autov = np.real(vecs[:, mask]).T
        xi_hat, Magn = dis_check(autov, K, L, J_rec, JKS_iter, xi=xi_true, updates=up, show_bar=False)
        xi_ref = xi_hat

        fro_rel = _fro_norm_rel(JKS_iter, J_star)
        fro_single_rounds.append(fro_rel)
        magn_single_rounds.append(float(np.mean(Magn)))
        if use_mp:
            try:
                K_eff_mp, _, _ = estimate_K_eff_from_J(JKS_iter, method='shuffle', M_eff=M_eff_round,
                                                       alpha=shuffle_alpha, random=shuffle_random)
            except Exception:
                K_eff_mp = autov.shape[0]
        else:
            K_eff_mp = autov.shape[0]
        keff_single_rounds.append(int(K_eff_mp))
        if b == n_batch - 1:
            patterns_final = xi_hat
            K_eff_final = K_eff_mp

    # Baseline first round (single)
    ETA_first = new_round(ETA_unsup, round=0, collection=False)
    J_unsup_first, _ = _mean_unsup_J_per_layer_local(ETA_first, K)
    JKS_first = propagate_J(J_unsup_first, iters=1, verbose=False)
    vals0, vecs0 = np.linalg.eig(JKS_first)
    mask0 = (np.real(vals0) > 0.5)
    autov0 = np.real(vecs0[:, mask0]).T
    xi_first, Magn_first = dis_check(autov0, K, L, J_unsup_first, JKS_first, xi=xi_true, updates=up, show_bar=False)

    m_first, _, _ = _match_and_overlap(xi_first, xi_true)
    m_final, _, _ = _match_and_overlap(patterns_final, xi_true)
    G_single = m_final - m_first
    deltaK = abs(int(K_eff_final) - int(K))

    result = {
        'mode': 'single',
        'w': w_value,
        'seed': seed,
        'm_retr_first': m_first,
        'm_retr_final': m_final,
        'G_single': G_single,
        'fro_final': fro_single_rounds[-1],
        'fro_AUC': float(np.trapz(fro_single_rounds, dx=1) / max(1, len(fro_single_rounds) - 1)),
        'm_AUC': float(np.trapz(magn_single_rounds, dx=1) / max(1, len(magn_single_rounds) - 1)),
        'deltaK': deltaK,
        'K_eff_final': int(K_eff_final),
        'patterns_final': patterns_final,
        'magn_final_mean': float(np.mean(magn_single_rounds)),
        'magn_first_mean': float(magn_single_rounds[0]),
        'fro_series': fro_single_rounds,
        'm_series': magn_single_rounds,
    }
    return result


def grid_search_w_single(base_params: Dict[str, Any], w_grid: List[float], seeds: List[int], weights=None, verbose: bool = True) -> Dict[str, Any]:
    """Grid search for w using single-mode metrics."""
    import math
    if weights is None:
        weights = dict(alpha=0.5, beta=0.3, gamma=0.1, delta=0.05, epsilon=0.05)
    all_results = []
    per_w_patterns = {}
    for wv in w_grid:
        patterns_for_stability = []
        if verbose:
            print(f"[grid-single] w={wv}")
        for sd in seeds:
            r = run_experiment_single_mode(base_params, wv, sd)
            all_results.append(r)
            patterns_for_stability.append(r['patterns_final'])
        per_w_patterns[wv] = patterns_for_stability
    summary = {}
    for wv in w_grid:
        subset = [r for r in all_results if r['w'] == wv]
        def agg(key):
            vals = [r[key] for r in subset]
            return float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
        m_final_mean, m_final_se = agg('m_retr_final')
        fro_mean, fro_se = agg('fro_final')
        deltaK_mean, deltaK_se = agg('deltaK')
        G_single_mean, G_single_se = agg('G_single')
        # For stability we reference one example patterns_final as true for matching
        S_arch = _stability_across_seeds(per_w_patterns[wv], subset[0]['patterns_final']) if subset else 0.0
        summary[wv] = dict(m_retr=m_final_mean, m_retr_se=m_final_se,
                           fro=fro_mean, fro_se=fro_se,
                           deltaK=deltaK_mean, deltaK_se=deltaK_se,
                           G_single=G_single_mean, G_single_se=G_single_se,
                           S_arch=S_arch)
    m_vec = [summary[w]['m_retr'] for w in w_grid]
    fro_vec = [summary[w]['fro'] for w in w_grid]
    deltaK_vec = [summary[w]['deltaK'] for w in w_grid]
    S_vec = [summary[w]['S_arch'] for w in w_grid]
    G_vec = [summary[w]['G_single'] for w in w_grid]
    zm = robust_z(m_vec); zfro = robust_z(fro_vec); zdeltaK = robust_z(deltaK_vec); zS = robust_z(S_vec); zG = robust_z(G_vec)
    scores: List[float] = []
    for i, wv in enumerate(w_grid):
        sc = (weights['alpha'] * zm[i]
              - weights['beta'] * zfro[i]
              - weights['gamma'] * zdeltaK[i]
              + weights['delta'] * zS[i]
              + weights['epsilon'] * zG[i])
        scores.append(sc)
        summary[wv]['Score'] = sc
    best_idx = int(np.argmax(scores))
    best_w = w_grid[best_idx]
    best_score = scores[best_idx]
    score_se_approx = [weights['alpha'] * summary[w]['m_retr_se'] for w in w_grid]
    best_se = score_se_approx[best_idx]
    candidates = [w_grid[i] for i, sc in enumerate(scores) if sc >= best_score - best_se]
    w_one_se = min(candidates)
    report_rows = []
    for i, wv in enumerate(w_grid):
        report_rows.append({
            'w': wv,
            **summary[wv],
            'Score_se_approx': score_se_approx[i]
        })
    report = {
        'mode': 'single',
        'grid': w_grid,
        'summary': report_rows,
        'best_w': best_w,
        'best_w_score': float(best_score),
        'one_se_w': w_one_se,
        'weights': weights
    }
    return report


def refine_grid_search_single(initial_report: dict,
                              base_params: dict,
                              seeds: list,
                              fine_step: float = 0.02,
                              span: float = 0.12,
                              center_mode: str = 'one_se',
                              improve_threshold: float = 0.01,
                              max_refinements: int = 3,
                              weights=None,
                              results_dir: str = 'results_w_search_single',
                              fixed_range: tuple | None = None,
                              force_recompute_existing: bool = False) -> dict:
    """Iteratively refine w-grid using single-mode objective. Also writes quick plots and CSVs."""
    import json, csv, os
    from pathlib import Path
    import matplotlib.pyplot as plt

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    history = []
    prev_best_score = initial_report['best_w_score']
    current_report = initial_report

    for level in range(1, max_refinements + 1):
        if fixed_range is not None:
            lo, hi = fixed_range
            lo = max(0.0, min(1.0, lo))
            hi = max(lo, min(1.0, hi))
            candidate_grid = list(np.unique([round(x, 10) for x in np.arange(lo, hi + fine_step / 2, fine_step)]))
        else:
            center = current_report['one_se_w'] if center_mode == 'one_se' else current_report['best_w']
            lo = max(0.0, center - span)
            hi = min(1.0, center + span)
            candidate_grid = list(np.unique([round(x, 10) for x in np.arange(lo, hi + fine_step / 2, fine_step)]))
        tested = set(row['w'] for row in current_report['summary'])
        new_w = candidate_grid if force_recompute_existing else [w for w in candidate_grid if w not in tested]
        if not new_w:
            break
        new_report = grid_search_w_single(base_params, new_w, seeds, weights=weights, verbose=False)
        merged_rows = ( [r for r in current_report['summary'] if r['w'] not in new_w] + new_report['summary'] ) if force_recompute_existing else ( current_report['summary'] + new_report['summary'] )
        union_grid = sorted({row['w'] for row in merged_rows})
        by_w = {row['w']: row for row in merged_rows}
        m_vec = [by_w[w]['m_retr'] for w in union_grid]
        fro_vec = [by_w[w]['fro'] for w in union_grid]
        dK_vec = [by_w[w]['deltaK'] for w in union_grid]
        S_vec = [by_w[w]['S_arch'] for w in union_grid]
        G_vec = [by_w[w]['G_single'] for w in union_grid]
        zm = robust_z(m_vec); zfro = robust_z(fro_vec); zdeltaK = robust_z(dK_vec); zS = robust_z(S_vec); zG = robust_z(G_vec)
        scores = []
        for i, wv in enumerate(union_grid):
            sc = (weights['alpha'] * zm[i]
                  - weights['beta'] * zfro[i]
                  - weights['gamma'] * zdeltaK[i]
                  + weights['delta'] * zS[i]
                  + weights['epsilon'] * zG[i])
            scores.append(sc)
            by_w[wv]['Score'] = sc
        best_idx = int(np.argmax(scores))
        best_w = union_grid[best_idx]
        best_score = scores[best_idx]
        best_se = max([row.get('m_retr_se', 0.0) for row in merged_rows])
        candidates = [union_grid[i] for i, sc in enumerate(scores) if sc >= best_score - best_se]
        w_one_se = min(candidates) if candidates else best_w
        current_report = {
            'mode': 'single',
            'grid': union_grid,
            'summary': [by_w[w] for w in union_grid],
            'best_w': best_w,
            'best_w_score': float(best_score),
            'one_se_w': w_one_se,
            'weights': weights
        }
        # persist step
        level_dir = Path(results_dir) / f"refine_level{level}"
        level_dir.mkdir(parents=True, exist_ok=True)
        with open(level_dir / 'refine_level1.json', 'w') as fjson:
            json.dump(current_report, fjson, indent=2)
        with open(level_dir / 'refine_level1.csv', 'w', newline='') as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=list(current_report['summary'][0].keys()))
            writer.writeheader(); writer.writerows(current_report['summary'])
        # plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(union_grid, [by_w[w]['m_retr'] for w in union_grid], label='m_retr')
        ax.plot(union_grid, [by_w[w]['fro'] for w in union_grid], label='fro')
        ax.plot(union_grid, [by_w[w]['deltaK'] for w in union_grid], label='deltaK')
        ax.plot(union_grid, [by_w[w]['G_single'] for w in union_grid], label='G_single')
        ax.set_xlabel('w'); ax.set_title(f'grid refine lvl {level}')
        ax.legend(); fig.tight_layout(); fig.savefig(level_dir / 'grid_search_fine.png', dpi=150)
        plt.close(fig)

        history.append(dict(level=level, best_w=best_w, best_score=float(best_score)))
        if (best_score - prev_best_score) < improve_threshold:
            break
        prev_best_score = best_score

    return current_report





