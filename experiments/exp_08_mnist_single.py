#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 08 — MNIST (single-mode)
===================================
Primo scaffold per valutazione su dataset reale (MNIST) in modalità SINGLE.

Strategia:
- Costruisce 10 prototipi "veri" xi_true come segno della media per classe (±1 su N=784).
- Genera uno stream round-wise per L client, con etichette bilanciate (uniformi) o con drift opzionale.
- A ciascun round calcola J solo sul round corrente, propaga e stima i prototipi con TAM.
- Metriche: K_eff (MP), retrieval vs xi_true, Frobenius relativa vs JK_real(xi_true).

Nota: il download di MNIST ora avviene senza tensorflow (file .npz ufficiale, ~11MB, cached). Rimuovere quindi eventuale dipendenza da tensorflow.
"""
from __future__ import annotations

import os
import sys
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_THIS = Path(__file__).resolve()
ROOT = _THIS
while ROOT != ROOT.parent and not (ROOT / "Functions.py").exists():
    ROOT = ROOT.parent

SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from unsup.functions import JK_real, unsupervised_J, propagate_J, estimate_K_eff_from_J  # type: ignore
from unsup.dynamics import dis_check  # type: ignore


@dataclass
class HyperParams:
    L: int = 5
    n_batch: int = 20
    M_total: int = 10000   # esempi totali
    w: float = 0.9
    updates: int = 60
    n_seeds: int = 3
    seed_base: int = 42
    pb_seeds: bool = True
    use_mp_keff: bool = True


def load_mnist_pm1(cache_dir: Path | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Scarica (se necessario) e carica MNIST restituendo X in {-1,+1} e y.

    Evita la dipendenza da tensorflow: usa il file pubblico mnist.npz
    (lo stesso usato internamente da keras) scaricandolo da GCS.
    """
    import urllib.request
    import hashlib
    if cache_dir is None:
        cache_dir = ROOT / 'data' / 'mnist'
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / 'mnist.npz'
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    # Facoltativo: checksum noto (sha256) per integrità (valore aggiornato se cambia upstream)
    expected_sha256 = '731c5ac602752760c8e48fbffcf8c3b850d6c52b7bb0a7f4f3f8a4f74a5a5a33'
    if not npz_path.exists():
        print(f"[mnist] Download {url} -> {npz_path}")
        try:
            urllib.request.urlretrieve(url, npz_path)  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Download MNIST fallito: {e}")
    # Verifica (best-effort)
    try:
        h = hashlib.sha256()
        with open(npz_path, 'rb') as f: h.update(f.read())
        digest = h.hexdigest()
        if digest != expected_sha256:
            print('[mnist] Warning: checksum diverso, continuo comunque.')
    except Exception:
        pass
    with np.load(npz_path) as data:  # type: ignore
        x_train = data['x_train']  # (60000,28,28)
        y_train = data['y_train']
    X = x_train.astype(np.float32) / 255.0
    X = (X.reshape(-1, 28 * 28) >= 0.5).astype(np.float32) * 2.0 - 1.0
    y = y_train.astype(np.int64)
    return X, y


def make_xi_true_from_means(X: np.ndarray, y: np.ndarray, K: int = 10) -> np.ndarray:
    N = X.shape[1]
    xi = np.zeros((K, N), dtype=np.float32)
    for k in range(K):
        m = X[y == k].mean(axis=0)
        xi[k] = np.where(m >= 0.0, 1.0, -1.0)
    return xi


def gen_stream_roundwise(
    X: np.ndarray, y: np.ndarray, *, L: int, n_batch: int, M_total: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ritorna ETA (L, n_batch, M_c, N) e labels (L, n_batch, M_c). Distribuzione uniforme sulle classi.
    """
    rng = np.random.default_rng(0)
    N = X.shape[1]; K = 10
    M_c = math.ceil(M_total / (L * n_batch))
    ETA = np.zeros((L, n_batch, M_c, N), dtype=np.float32)
    labels = np.zeros((L, n_batch, M_c), dtype=np.int32)
    idx_by_k = [np.where(y == k)[0] for k in range(K)]
    for l in range(L):
        for t in range(n_batch):
            # uniforme tra le classi
            ks = rng.choice(K, size=M_c, replace=True)
            sel = np.array([rng.choice(idx_by_k[k]) for k in ks], dtype=np.int64)
            ETA[l, t] = X[sel]
            labels[l, t] = ks
    return ETA, labels


def run_one_seed(hp: HyperParams, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    X, y = load_mnist_pm1()
    K = 10; N = X.shape[1]; L = hp.L
    xi_true = make_xi_true_from_means(X, y, K=K)
    J_star = JK_real(xi_true)
    ETA, labels = gen_stream_roundwise(X, y, L=L, n_batch=hp.n_batch, M_total=hp.M_total)

    fro_series: List[float] = []
    m_series: List[float] = []
    keff_series: List[int] = []
    xi_ref = None
    K_eff_final = None

    for t in range(hp.n_batch):
        ETA_round = ETA[:, t, :, :]  # (L, M_c, N)
        L_loc, M_eff, N_loc = ETA_round.shape
        M_eff_param = max(1, M_eff // K)
        Js = [unsupervised_J(ETA_round[l], M_eff_param) for l in range(L_loc)]
        J_unsup = np.sum(Js, axis=0) / L_loc
        if t == 0:
            J_rec = J_unsup.copy()
        else:
            J_hebb_prev = unsupervised_J(np.asarray(xi_ref, dtype=np.float32), 1)
            J_rec = hp.w * J_unsup + (1.0 - hp.w) * J_hebb_prev

        JKS_iter = propagate_J(J_rec, iters=1, verbose=False)
        vals, vecs = np.linalg.eig(JKS_iter)
        mask = (np.real(vals) > 0.5)
        autov = np.real(vecs[:, mask]).T
        xi_hat, Magn = dis_check(autov, K, L, J_rec, JKS_iter, xi=xi_true, updates=hp.updates, show_bar=False)
        xi_ref = xi_hat

        fro_series.append(float(np.linalg.norm(JKS_iter - J_star, ord='fro') / (np.linalg.norm(J_star, ord='fro') + 1e-9)))
        m_series.append(float(np.mean(Magn)))
        if hp.use_mp_keff:
            try:
                K_eff_mp, _, _ = estimate_K_eff_from_J(JKS_iter, method='shuffle', M_eff=M_eff)
            except Exception:
                K_eff_mp = autov.shape[0]
        else:
            K_eff_mp = autov.shape[0]
        keff_series.append(int(K_eff_mp))
        K_eff_final = K_eff_mp

    # first/final retrieval
    ETA_first = ETA[:, 0, :, :]
    L_loc, M_eff_f, _ = ETA_first.shape
    M_eff_param = max(1, M_eff_f // K)
    Js_f = [unsupervised_J(ETA_first[l], M_eff_param) for l in range(L_loc)]
    J_unsup_first = np.sum(Js_f, axis=0) / L_loc
    JKS_first = propagate_J(J_unsup_first, iters=1, verbose=False)
    vals0, vecs0 = np.linalg.eig(JKS_first)
    mask0 = (np.real(vals0) > 0.5)
    autov0 = np.real(vecs0[:, mask0]).T
    xi_first, Magn_first = dis_check(autov0, K, L, J_unsup_first, JKS_first, xi=xi_true, updates=hp.updates, show_bar=False)

    def _match_and_overlap(estimated: np.ndarray, true: np.ndarray) -> float:
        from scipy.optimize import linear_sum_assignment
        K_hat, N = estimated.shape
        K_t, N2 = true.shape
        assert N == N2
        M = np.abs(estimated @ true.T / N)
        cost = 1.0 - M
        rI, cI = linear_sum_assignment(cost)
        overlaps = M[rI, cI]
        if K_hat < K_t:
            return float(overlaps.sum() / K_t)
        return float(overlaps.mean())

    m_first = _match_and_overlap(xi_first, xi_true)
    m_final = _match_and_overlap(xi_ref, xi_true) if xi_ref is not None else 0.0
    deltaK = abs(int(K_eff_final) - int(K)) if K_eff_final is not None else int(K)

    return dict(
        seed=seed,
        m_first=m_first,
        m_final=m_final,
        G_single=m_final - m_first,
        fro_final=fro_series[-1],
        deltaK=deltaK,
        series=dict(
            rounds=list(range(hp.n_batch)),
            m_single_mean=m_series,
            fro_single=fro_series,
            keff_single=keff_series,
        ),
    )


def aggregate_and_plot(hp: HyperParams, results: List[Dict[str, Any]], exp_dir: Path) -> None:
    import math
    rounds = np.arange(hp.n_batch)
    arr_m = np.array([r['series']['m_single_mean'] for r in results])
    arr_f = np.array([r['series']['fro_single'] for r in results])
    arr_k = np.array([r['series']['keff_single'] for r in results])

    def m_se(a):
        return a.mean(axis=0), a.std(axis=0, ddof=1) / max(1, math.sqrt(a.shape[0]))

    m_mean, m_sev = m_se(arr_m)
    f_mean, f_sev = m_se(arr_f)
    k_mean, k_sev = m_se(arr_k)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)
    ax = axes[0]
    ax.plot(rounds, m_mean, lw=2.0)
    ax.fill_between(rounds, m_mean - m_sev, m_mean + m_sev, alpha=0.2)
    ax.set_title("MNIST â€” Retrieval vs round (single)")
    ax.set_xlabel("round"); ax.set_ylabel("Mattis overlap")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(rounds, f_mean, lw=2.0)
    ax.fill_between(rounds, f_mean - f_sev, f_mean + f_sev, alpha=0.2)
    ax.set_title("Frobenius rel. vs round (single)")
    ax.set_xlabel("round"); ax.set_ylabel("||J-J*||_F / ||J*||_F")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(rounds, k_mean, lw=2.0)
    ax.fill_between(rounds, k_mean - k_sev, k_mean + k_sev, alpha=0.2)
    ax.set_title("K_eff (MP) vs round")
    ax.set_xlabel("round"); ax.set_ylabel("K_eff")
    ax.grid(True, alpha=0.3)

    fig_path = exp_dir / "fig_mnist_single.png"
    fig.savefig(fig_path, dpi=150)
    print(f"[plot] Figure salvata in: {fig_path}")
    plt.show()


def main():
    hp = HyperParams()
    base_dir = ROOT / "stress_tests" / "exp08_mnist_single"
    tag = f"L{hp.L}_T{hp.n_batch}_M{hp.M_total}_w{hp.w}"
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    results: List[Dict[str, Any]] = []
    log_path = exp_dir / "log.jsonl"
    with open(log_path, "w") as flog:
        seed_iter = range(hp.n_seeds)
        if hp.pb_seeds:
            seed_iter = tqdm(seed_iter, desc="seeds")
        for s in seed_iter:
            seed = hp.seed_base + s
            out = run_one_seed(hp, seed)
            results.append(out)
            flog.write(json.dumps({
                'mode': 'single',
                **out,
            }) + "\n")
    aggregate_and_plot(hp, results, exp_dir)


if __name__ == '__main__':
    main()

