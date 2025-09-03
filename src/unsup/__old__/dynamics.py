# -*- coding: utf-8 -*-
"""
Dynamics.py

Raccolta di TUTTE le definizioni di funzione presenti nel notebook `split_unsup.ipynb`.
Il codice è copiato (con minime aggiunte di docstring / import difensivi) per poter
riutilizzare facilmente le routine senza sporcare ulteriormente il notebook.

NOTA:
- Alcune funzioni dipendono da simboli definiti in `Functions.py` (es: unsupervised_J,
  propagate_J, estimate_K_eff_from_J, gen_patterns, Hebb_J, gen_dataset_unsup, JK_real)
  e dalle classi di rete in `Networks.py` (TAM_Network). Assicurarsi che il path del
  progetto sia nel PYTHONPATH prima di importare questo modulo.
- Le variabili con lettere greche (ξ) sono state lasciate inalterate come nel notebook.
- Manteniamo anche funzioni "private" con underscore perché comparivano nel notebook.

Se alcune import falliscono, verificare che i file originali siano presenti.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Tuple

# TensorFlow (shim come nel notebook)
try:
    import tensorflow as tf  # type: ignore
    _USING_REAL_TF = True
except Exception:  # pragma: no cover - fallback
    _USING_REAL_TF = False
    class _TFShim:
        float32 = np.float32
        def convert_to_tensor(self, x, dtype=None):
            return np.array(x, dtype=dtype if dtype is not None else None)
        def transpose(self, x, perm=None):
            return np.transpose(x, axes=perm)
        def sign(self, x):
            return np.sign(x)
        def einsum(self, subscripts, *operands, **kwargs):
            return np.einsum(subscripts, *operands, **kwargs)
        def expand_dims(self, a, axis):
            return np.expand_dims(a, axis=axis)
        def reshape(self, a, shape):
            return np.reshape(a, shape)
    tf = _TFShim()  # type: ignore

# Import delle funzioni esterne usate da alcune routine (se disponibili)
try:  # import selettivo per evitare side effects indesiderati
    from src.unsup.functions import (
        unsupervised_J,
        propagate_J,
        estimate_K_eff_from_J,
        gen_patterns,
        Hebb_J,
        gen_dataset_unsup,
        JK_real,
    )
except Exception:  # noqa: E722 - vogliamo solo continuare; funzioni verranno lazy-import nei punti necessari
    pass

try:
    from src.unsup.networks import TAM_Network  # Hopfield_Network non usato direttamente qui
except Exception:  # noqa
    TAM_Network = None  # verrà controllato a runtime quando necessario

__all__ = [
    'diag_generale', 'r_empirico', 'new_round', 'calcolo_autovett', 'generazione_input',
    'comb_esempi', 'disentangling', 'dis_check', '_eig_cut', '_match_and_overlap',
    '_stability_across_seeds', '_delta_K', '_fro_norm_rel', 'robust_z',
    'run_single_experiment', 'grid_search_w', 'refine_grid_search', '_get_data_for_seed',
    'timed_partial_run', '_mean_unsup_J_per_layer'
]

# ---------------------------------------------------------------------------
# Sezione: Funzioni iniziali (magnetizzazioni / autovettori / batch)
# ---------------------------------------------------------------------------

def diag_generale(result: np.ndarray) -> np.ndarray:
    """Applica una maschera diagonale (1 sulla diagonale, 0 altrove) al tensore 3D.
    result: shape (k, a, b).
    Ritorna lo stesso tensore con zeri fuori diagonale in ciascuna matrice (a,b).
    """
    k, a, b = result.shape
    mask = np.eye(a)[None, :, :]  # shape (1, a, b)
    return result * mask

def r_empirico(x: np.ndarray) -> float:
    """Stima della correlazione media fuori diagonale.
    x: shape (M, K, N)
    """
    M, K, N = x.shape
    temp = np.einsum('aki,bki->kab', x, x) / N
    return float(np.mean(np.sqrt(np.sum(temp - diag_generale(temp), axis=(1, 2)) / (M * (M - 1)))))

def new_round(data, round: int = 0, collection: bool = False):
    """Concatena esempi dei batch fino al round indicato.
    data shape attesa: (L, n_batch, M_per_batch, N) oppure simile (come nel notebook originale).
    Nel notebook l'asse è (L, n_batch, M_batch, N) ma la funzione era usata su ETA_unsup.
    Ritorna tensore float32 (tf o np.array) shape (L, M_eff, N).
    """
    if not collection:
        new_data = data[:, round, :, :]
    else:
        new_data = data[:, 0, :, :]
        new_data = tf.transpose(new_data, (1, 0, 2))
        for r in range(round):
            new_data = np.vstack((new_data, tf.transpose(data[:, r + 1, :, :], (1, 0, 2))))
        new_data = tf.transpose(new_data, (1, 0, 2))
    return tf.convert_to_tensor(new_data, dtype=np.float32)

def calcolo_autovett(JKS: np.ndarray) -> Tuple[np.ndarray, int]:
    """Calcola autovettori con autovalori reali > 0.5 e restituisce (autovettori, K_eff)."""
    vals, vecs = np.linalg.eig(JKS)
    index = np.real(vals) > 0.5
    autovett = np.real(vecs[:, index]).T  # shape (k_eff, N)
    K_eff = autovett.shape[0]
    return autovett, K_eff

def generazione_input(K_eff: int, autovett: np.ndarray, L: int):
    """Crea combinazioni lineari in segno degli autovettori (inizializzazione stati TAM).
    K_eff: numero effettivo (target K nel notebook originale quando chiamata da disentangling)
    autovett: shape (k_eff, N)
    L: numero di layer / client
    """
    M = 10 * int(K_eff / L * np.log(K_eff / 0.01))
    weigth = np.random.normal(0, 1, size=(M, autovett.shape[0]))
    new = tf.sign(tf.einsum('ij,mi->mj', autovett, weigth))
    return new

def comb_esempi(ETA_batch, M_expamples: int):
    """Crea combinazioni random (in segno) degli esempi usati per Jij per ogni layer.
    ETA_batch shape attesa: (L, M_data, N)
    Ritorna: vv shape (M_expamples, L, N) dopo riarrangiamenti come in notebook.
    """
    L, M_data, N = ETA_batch.shape
    weigth = np.random.normal(0, 1, size=(M_expamples, M_data, L))
    vv = tf.sign(np.sum(tf.einsum('mal, lai ->  lami', weigth, ETA_batch), axis=1))
    return vv

# ---------------------------------------------------------------------------
# Sezione: Disentangling TAM
# ---------------------------------------------------------------------------

def disentangling(x: np.ndarray, K: int, L: int, J_rec: np.ndarray, JKS_iter: np.ndarray,
                  ξ: np.ndarray, updates: int, ξr_old: Any = -10, βT: float = 2.5, h: float = 0.1,
                  λ: float = 0.2, show_bar: bool = True):
    """Esegue la dinamica TAM su stati iniziali generati da autovettori / archetipi.

    Parametri principali identici alla cella notebook.
    Ritorna: (ξr, ms) con pattern ricostruiti filtrati e magnetizzazioni massime.
    """
    if TAM_Network is None:
        raise RuntimeError("TAM_Network non disponibile: importa prima Networks.TAM_Network")
    K_eff_tot, N = x.shape
    x = generazione_input(K, x, L)
    σr = tf.convert_to_tensor([x] * L)
    σr = np.swapaxes(σr, 0, 1)
    Net = TAM_Network()
    Net.prepare(J_rec, L)
    Net.dynamics(σr, βT, λ, h, updates=updates, show_progress=show_bar, desc="TAM round")
    if Net.σ is None:
        return np.empty((0, N)), np.array([])
    ξr_new = np.reshape(np.asarray(Net.σ), (L * x.shape[0], N))
    if np.mean(ξr_old) != -10:
        ξr = np.vstack((ξr_new, ξr_old))
    else:
        ξr = ξr_new
    ξr_pre_prune = np.copy(ξr)  # backup per fallback
    # Check eigenvectors of KS
    temp = tf.einsum('ij,Aj->Ai', JKS_iter, ξr)
    check2 = tf.einsum('Ai,Ai->A', ξr, temp) / N
    toremove2 = (check2 < 0.6)
    if np.any(toremove2):
        ξr = np.delete(ξr, toremove2, axis=0)
    # mutual overlap pruning
    qξr = np.abs(np.array(tf.einsum('ai,bi->ab', ξr, ξr) / N))
    qξr = np.triu(qξr, 1)
    check = (qξr > 0.4)
    toremove = []
    for _ in range(ξr.shape[0]):
        wh = np.where(check[_])[0]
        if len(wh) > 0:
            for a in range(len(wh)):
                toremove.append(wh[a])
    if toremove:
        ξr = np.delete(ξr, toremove, axis=0)
    # Fallback se tutto rimosso
    if ξr.shape[0] == 0:
        ξr = ξr_pre_prune[: max(1, min(K, ξr_pre_prune.shape[0]))]
    # Calcolo magnetizzazioni (gestione edge-case singolo pattern)
    if ξr.shape[0] > 0:
        ms = np.max(np.abs(tf.einsum('ai,bi->ab', ξr, ξ) / N), axis=1)
    else:  # ulteriore guardia (non dovrebbe succedere dopo fallback)
        ms = np.array([])
    return ξr, ms

def dis_check(autovett: np.ndarray, K: int, L: int, J_rec: np.ndarray, JKS_iter: np.ndarray,
              ξ: np.ndarray, updates: int = 80, show_bar: bool = True):
    """Wrapper iterativo che continua a chiamare disentangling finché non ottiene >= K magnetizzazioni."""
    K_eff, N = autovett.shape
    ξr, ms = disentangling(autovett, K, L, J_rec, JKS_iter, ξ, updates, ξr_old=-10, show_bar=show_bar)
    attempts = 0
    max_attempts = 10
    while len(ms) < K and attempts < max_attempts:
        attempts += 1
        ξr, ms = disentangling(autovett, K, L, J_rec, JKS_iter, ξ, updates, ξr_old=ξr, show_bar=show_bar)
    # Se dopo i tentativi rimane insufficiente, restituiamo quel che abbiamo (evita loop infinito)
    return ξr, ms

def _eig_cut(JK: np.ndarray) -> Tuple[np.ndarray, int]:
    """Funzione legacy dal notebook: autovettori con autovalori > 0.5."""
    vals, vecs = np.linalg.eig(JK)
    mask = (np.real(vals) > 0.5)
    autov = np.real(vecs[:, mask]).T
    return autov, autov.shape[0]

# ---------------------------------------------------------------------------
# Sezione: Metriche e supporto per grid-search / stabilità
# ---------------------------------------------------------------------------

def _match_and_overlap(estimated: np.ndarray, true: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Hungarian matching su overlap assoluto. Ritorna (mean_overlap, overlaps, mapping)."""
    from scipy.optimize import linear_sum_assignment  # import locale
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

def _stability_across_seeds(pattern_sets: List[np.ndarray], true: np.ndarray) -> float:
    """Stima stabilità archetipi: media overlap pairwise dopo matching (0..1)."""
    from scipy.optimize import linear_sum_assignment
    if len(pattern_sets) < 2:
        return 0.0
    overlaps_collected = []
    for i in range(len(pattern_sets)):
        for j in range(i + 1, len(pattern_sets)):
            A = pattern_sets[i]
            B = pattern_sets[j]
            Ka, N = A.shape; Kb, _ = B.shape
            M = np.abs(A @ B.T / N)
            cost = 1 - M
            rI, cI = linear_sum_assignment(cost)
            overlaps_collected.append(M[rI, cI].mean())
    if not overlaps_collected:
        return 0.0
    arr = np.array(overlaps_collected)
    return float(max(0.0, 1 - arr.var()))

def _delta_K(K_eff: int, K_true: int) -> int:
    return abs(int(K_eff) - int(K_true))

def _fro_norm_rel(J_est: np.ndarray, J_true: np.ndarray) -> float:
    return float(np.linalg.norm(J_est - J_true, ord='fro') / (np.linalg.norm(J_true, ord='fro') + 1e-9))

def robust_z(values: List[float]) -> List[float]:
    vals = np.array(values, dtype=float)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-9
    return list((vals - med) / (1.4826 * mad))

# ---------------------------------------------------------------------------
# Sezione: Esperimenti (run_single_experiment, grid_search_w)
# ---------------------------------------------------------------------------

def run_single_experiment(base_params: Dict[str, Any], w_value: float, seed: int) -> Dict[str, Any]:
    """Esegue loop principale ridotto per dato w, seed (versione notebook)."""
    # Lazy import per sicurezza
    from src.unsup.functions import (
        gen_patterns, JK_real, gen_dataset_unsup, unsupervised_J, propagate_J, estimate_K_eff_from_J
    )
    rng = np.random.default_rng(seed)
    L = base_params['L']; K = base_params['K']; N = base_params['N']
    M_unsup = base_params['M_unsup']; r_ex = base_params['r_ex']; n_batch = base_params['n_batch']
    up = base_params['updates']
    use_mp = base_params.get('use_mp_keff', True)
    shuffle_alpha = base_params.get('shuffle_alpha', 0.01)
    shuffle_random = base_params.get('shuffle_random', 32)

    ξ_true = gen_patterns(N, K)
    J_star = JK_real(ξ_true)
    ETA_unsup = gen_dataset_unsup(ξ_true, M_unsup, r_ex, n_batch, L)

    def _mean_unsup_J_per_layer_local(tensor_L_M_N, Kx):
        L_loc, M_eff_actual, N_loc = tensor_L_M_N.shape
        M_eff_param = max(1, M_eff_actual // Kx)
        Js = [unsupervised_J(tensor_L_M_N[l], M_eff_param) for l in range(L_loc)]
        return np.sum(Js, axis=0) / L_loc, M_eff_param

    ξr_ref = None
    fro_extend_rounds = []
    magn_extend_rounds = []
    keff_mp_extend_rounds = []
    patterns_extend_final = None
    K_eff_final = None

    for b in range(n_batch):
        ETA_round = new_round(ETA_unsup, round=b, collection=False)
        ETA_extend = new_round(ETA_unsup, round=b, collection=True)
        J_unsup_single, M_eff_round = _mean_unsup_J_per_layer_local(ETA_round, K)
        J_unsup_extend, M_eff_extend = _mean_unsup_J_per_layer_local(ETA_extend, K)
        if b == 0:
            J_rec = J_unsup_single.copy()
            J_rec_extend = J_unsup_extend.copy()
        else:
            J_hebb_prev = unsupervised_J(tf.convert_to_tensor(ξr_ref, dtype=tf.float32), 1)
            J_rec = w_value * J_unsup_single + (1 - w_value) * J_hebb_prev
            J_rec_extend = w_value * J_unsup_extend + (1 - w_value) * J_hebb_prev
        JKS_iter_extend = propagate_J(J_rec_extend, iters=1, verbose=False)
        vals_ext, vecs_ext = np.linalg.eig(JKS_iter_extend)
        mask_ext = (np.real(vals_ext) > 0.5)
        autov_ext = np.real(vecs_ext[:, mask_ext]).T
        ξr_extend, Magn_extend = dis_check(autov_ext, K, L, J_rec_extend, JKS_iter_extend, ξ=ξ_true, updates=up, show_bar=False)
        ξr_ref = ξr_extend
        fro_rel = _fro_norm_rel(JKS_iter_extend, J_star)
        fro_extend_rounds.append(fro_rel)
        magn_extend_rounds.append(float(np.mean(Magn_extend)))
        if use_mp:
            try:
                K_eff_mp_ext, _, _ = estimate_K_eff_from_J(JKS_iter_extend, method='shuffle', M_eff=M_eff_extend)
            except Exception:
                K_eff_mp_ext = autov_ext.shape[0]
        else:
            K_eff_mp_ext = autov_ext.shape[0]
        keff_mp_extend_rounds.append(int(K_eff_mp_ext))
        if b == n_batch - 1:
            patterns_extend_final = ξr_extend
            K_eff_final = K_eff_mp_ext

    # Iniziale (round 0) per confronto
    ETA_first = new_round(ETA_unsup, round=0, collection=True)
    J_unsup_first, _ = _mean_unsup_J_per_layer_local(ETA_first, K)
    JKS_first = propagate_J(J_unsup_first, iters=1, verbose=False)
    vals0, vecs0 = np.linalg.eig(JKS_first)
    mask0 = (np.real(vals0) > 0.5)
    autov0 = np.real(vecs0[:, mask0]).T
    ξr_first, Magn_first = dis_check(autov0, K, L, J_unsup_first, JKS_first, ξ=ξ_true, updates=up, show_bar=False)

    m_first, _, _ = _match_and_overlap(ξr_first, ξ_true)
    if patterns_extend_final is not None:
        m_final, overlaps_final, _ = _match_and_overlap(patterns_extend_final, ξ_true)
    else:
        m_final, overlaps_final = 0.0, np.array([])
    G_ext = m_final - m_first
    deltaK = _delta_K(K_eff_final if K_eff_final is not None else 0, K)

    result = {
        'w': w_value,
        'seed': seed,
        'm_retr_first': m_first,
        'm_retr_final': m_final,
        'G_ext': G_ext,
        'fro_final': fro_extend_rounds[-1],
        'fro_AUC': float(float(np.trapz(fro_extend_rounds, dx=1)) / max(1, len(fro_extend_rounds) - 1)),
        'm_AUC': float(float(np.trapz(magn_extend_rounds, dx=1)) / max(1, len(magn_extend_rounds) - 1)),
        'deltaK': deltaK,
        'K_eff_final': int(K_eff_final if K_eff_final is not None else 0),
        'patterns_final': patterns_extend_final,
        'magn_final_mean': float(np.mean(magn_extend_rounds)),
        'magn_first_mean': float(magn_extend_rounds[0]),
        'fro_series': fro_extend_rounds,
        'm_series': magn_extend_rounds
    }
    return result

def grid_search_w(base_params: Dict[str, Any], w_grid: List[float], seeds: List[int], weights=None, verbose: bool = True) -> Dict[str, Any]:
    """Grid search dei pesi w (versione notebook)."""
    import math
    if weights is None:
        weights = dict(alpha=0.5, beta=0.3, gamma=0.1, delta=0.05, epsilon=0.05)
    all_results = []
    per_w_patterns = {}
    for wv in w_grid:
        patterns_for_stability = []
        if verbose:
            print(f"[grid] w={wv}")
        for sd in seeds:
            r = run_single_experiment(base_params, wv, sd)
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
        G_ext_mean, G_ext_se = agg('G_ext')
        S_arch = _stability_across_seeds(per_w_patterns[wv], subset[0]['patterns_final'])
        summary[wv] = dict(m_retr=m_final_mean, m_retr_se=m_final_se,
                           fro=fro_mean, fro_se=fro_se,
                           deltaK=deltaK_mean, deltaK_se=deltaK_se,
                           G_ext=G_ext_mean, G_ext_se=G_ext_se,
                           S_arch=S_arch)
    m_vec = [summary[w]['m_retr'] for w in w_grid]
    fro_vec = [summary[w]['fro'] for w in w_grid]
    deltaK_vec = [summary[w]['deltaK'] for w in w_grid]
    S_vec = [summary[w]['S_arch'] for w in w_grid]
    G_vec = [summary[w]['G_ext'] for w in w_grid]
    zm = robust_z(m_vec); zfro = robust_z(fro_vec); zdeltaK = robust_z(deltaK_vec); zS = robust_z(S_vec); zG = robust_z(G_vec)
    scores = []
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
        'grid': w_grid,
        'summary': report_rows,
        'best_w': best_w,
        'best_w_score': float(best_score),
        'one_se_w': w_one_se,
        'weights': weights
    }
    return report

# ---------------------------------------------------------------------------
# Sezione: Raffinamento griglia (refine_grid_search)
# ---------------------------------------------------------------------------

def refine_grid_search(initial_report: dict,
                        base_params: dict,
                        seeds: list,
                        fine_step: float = 0.02,
                        span: float = 0.12,
                        center_mode: str = 'one_se',
                        improve_threshold: float = 0.01,
                        max_refinements: int = 3,
                        weights=None,
                        results_dir: str = 'results_w_search',
                        fixed_range: tuple | None = None,
                        force_recompute_existing: bool = False) -> dict:
    """Raffinamento iterativo dei pesi w (copiato dal notebook)."""
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
        if force_recompute_existing:
            new_w = candidate_grid
        else:
            new_w = [w for w in candidate_grid if w not in tested]
        if not new_w:
            break
        new_report = grid_search_w(base_params, new_w, seeds, weights=weights, verbose=False)
        if force_recompute_existing:
            filtered_old = [r for r in current_report['summary'] if r['w'] not in new_w]
            merged_rows = filtered_old + new_report['summary']
        else:
            merged_rows = current_report['summary'] + new_report['summary']
        union_grid = sorted({row['w'] for row in merged_rows})
        by_w = {row['w']: row for row in merged_rows}
        m_vec = [by_w[w]['m_retr'] for w in union_grid]
        fro_vec = [by_w[w]['fro'] for w in union_grid]
        dK_vec = [by_w[w]['deltaK'] for w in union_grid]
        S_vec = [by_w[w]['S_arch'] for w in union_grid]
        G_vec = [by_w[w]['G_ext'] for w in union_grid]
        zm = robust_z(m_vec); zfro = robust_z(fro_vec); zdK = robust_z(dK_vec); zS = robust_z(S_vec); zG = robust_z(G_vec)
        if weights is None:
            weights_loc = dict(alpha=0.5, beta=0.3, gamma=0.1, delta=0.05, epsilon=0.05)
        else:
            weights_loc = weights
        scores = []
        for i, wval in enumerate(union_grid):
            sc = (weights_loc['alpha'] * zm[i] - weights_loc['beta'] * zfro[i] -
                  weights_loc['gamma'] * zdK[i] + weights_loc['delta'] * zS[i] +
                  weights_loc['epsilon'] * zG[i])
            by_w[wval]['Score'] = sc
            scores.append(sc)
        best_idx = int(np.argmax(scores))
        best_w = union_grid[best_idx]
        best_score = scores[best_idx]
        score_se_approx = {wval: weights_loc['alpha'] * by_w[wval]['m_retr_se'] for wval in union_grid}
        best_se = score_se_approx[best_w]
        candidates = [wval for wval, sc in zip(union_grid, scores) if sc >= best_score - best_se]
        one_se_w = min(candidates)
        current_report = {
            'grid': union_grid,
            'summary': [by_w[wval] for wval in union_grid],
            'best_w': best_w,
            'best_w_score': best_score,
            'one_se_w': one_se_w,
            'weights': weights_loc
        }
        out_json = Path(results_dir) / f'refine_level{level}.json'
        with open(out_json, 'w') as f:
            json.dump(current_report, f, indent=2)
        out_csv = Path(results_dir) / f'refine_level{level}.csv'
        with open(out_csv, 'w', newline='') as f:
            import csv as _csv
            writer = _csv.writer(f)
            header = ['w', 'Score', 'm_retr', 'fro', 'deltaK', 'S_arch', 'G_ext', 'Score_se_approx']
            writer.writerow(header)
            for wval in union_grid:
                row = by_w[wval]
                writer.writerow([wval, row['Score'], row['m_retr'], row['fro'], row['deltaK'], row['S_arch'], row['G_ext'], score_se_approx[wval]])
        history.append({
            'level': level,
            'best_w': best_w,
            'one_se_w': one_se_w,
            'best_score': best_score,
            'range_used': (lo, hi),
            'mode': 'fixed_range' if fixed_range is not None else f'center_{center_mode}',
            'new_w_tested': new_w
        })
        rel_improve = (best_score - prev_best_score) / (abs(prev_best_score) + 1e-9)
        if rel_improve < improve_threshold:
            break
        prev_best_score = best_score
    # (Plot opzionale rimosso per mantenere il modulo pulito; replicare nel notebook se serve)
    current_report['history'] = history
    return current_report

# ---------------------------------------------------------------------------
# Sezione: Benchmark (funzioni dal blocco benchmark)
# ---------------------------------------------------------------------------

_dataset_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

def _get_data_for_seed(seed: int, N_b: int, K_b: int, M_unsup_b: int, r_ex_b: float, n_batch_b: int, L_b: int):
    from src.unsup.functions import gen_patterns, gen_dataset_unsup, JK_real
    if seed in _dataset_cache:
        return _dataset_cache[seed]
    xi_true = gen_patterns(N_b, K_b)
    ETA_unsup_loc = gen_dataset_unsup(xi_true, M_unsup_b, r_ex_b, n_batch_b, L_b)
    J_star_loc = JK_real(xi_true)
    _dataset_cache[seed] = (xi_true, ETA_unsup_loc, J_star_loc)
    return _dataset_cache[seed]

def _mean_unsup_J_per_layer(tensor_L_M_N: np.ndarray, K: int):
    from src.unsup.functions import unsupervised_J
    L_loc, M_eff_actual, N_loc = tensor_L_M_N.shape
    M_eff_param = max(1, M_eff_actual // K)
    Js = [unsupervised_J(tensor_L_M_N[l], M_eff_param) for l in range(L_loc)]
    return np.sum(Js, axis=0) / L_loc, M_eff_param

def timed_partial_run(w_value: float, seed: int, rounds: int, *,
                      L_b: int, K_b: int, N_b: int,
                      M_unsup_b: int, r_ex_b: float, n_batch_b: int,
                      updates_b: int, show_detail: bool = False) -> Dict[str, Any]:
    """Esegue un sotto-run temporizzato (dal notebook benchmark)."""
    import time
    from src.unsup.functions import unsupervised_J, propagate_J
    xi_true, ETA_unsup_loc, J_star_loc = _get_data_for_seed(seed, N_b, K_b, M_unsup_b, r_ex_b, n_batch_b, L_b)
    timings = { 'round': [], 't_unsup_single': [], 't_unsup_extend': [], 't_hebb_prev': [], 't_blend': [],
                't_propagate': [], 't_disentangle': [], 't_total_round': [] }
    ξr_ref = None
    for b in range(rounds):
        t_round_start = time.perf_counter()
        ETA_round = new_round(ETA_unsup_loc, round=b, collection=False)
        ETA_extend = new_round(ETA_unsup_loc, round=b, collection=True)
        t0 = time.perf_counter(); J_unsup_single, M_eff_round = _mean_unsup_J_per_layer(ETA_round, K_b); t1 = time.perf_counter()
        J_unsup_extend, M_eff_extend = _mean_unsup_J_per_layer(ETA_extend, K_b); t2 = time.perf_counter()
        if b == 0:
            t_hebb = 0.0
            J_rec = J_unsup_single.copy(); J_rec_extend = J_unsup_extend.copy()
            t_blend = 0.0
        else:
            if (w_value < 1.0) and (ξr_ref is not None):
                tH0 = time.perf_counter(); J_hebb_prev = unsupervised_J(tf.convert_to_tensor(ξr_ref, dtype=tf.float32), 1); tH1 = time.perf_counter(); t_hebb = tH1 - tH0
                tB0 = time.perf_counter(); J_rec = w_value * J_unsup_single + (1 - w_value) * J_hebb_prev; J_rec_extend = w_value * J_unsup_extend + (1 - w_value) * J_hebb_prev; tB1 = time.perf_counter(); t_blend = tB1 - tB0
            else:
                t_hebb = 0.0; t_blend = 0.0; J_rec = J_unsup_single.copy(); J_rec_extend = J_unsup_extend.copy()
        tP0 = time.perf_counter(); JKS_iter_extend = propagate_J(J_rec_extend, iters=1, verbose=False); tP1 = time.perf_counter()
        tD0 = time.perf_counter(); vals_ext, vecs_ext = np.linalg.eig(JKS_iter_extend); mask_ext = (np.real(vals_ext) > 0.5); autov_ext = np.real(vecs_ext[:, mask_ext]).T; ξr_extend, _ = dis_check(autov_ext, K_b, L_b, J_rec_extend, JKS_iter_extend, ξ=xi_true, updates=updates_b, show_bar=False); tD1 = time.perf_counter(); ξr_ref = ξr_extend
        timings['round'].append(b)
        timings['t_unsup_single'].append(t1 - t0)
        timings['t_unsup_extend'].append(t2 - t1)
        timings['t_hebb_prev'].append(t_hebb)
        timings['t_blend'].append(0.0 if b == 0 else (t_blend if w_value < 1.0 else 0.0))
        timings['t_propagate'].append(tP1 - tP0)
        timings['t_disentangle'].append(tD1 - tD0)
        timings['t_total_round'].append(time.perf_counter() - t_round_start)
        if show_detail:
            print(f"[seed {seed}] w={w_value:.3f} round {b} total={timings['t_total_round'][-1]:.3f}s")
    return timings

# Fine del file
