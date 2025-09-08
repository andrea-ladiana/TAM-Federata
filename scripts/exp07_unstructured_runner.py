
# -*- coding: utf-8 -*-
"""
exp07_unstructured_runner.py — Driver per Exp-07 (single-only) su dati NON strutturati.

Obiettivo:
  • Generare round single-mode con emersione di nuovi archetipi (novità) secondo uno scheduler a rampa.
  • Per ciascun round: stimare J_unsup, fare blend con memoria ebraica, propagare, tagliare lo spettro,
    TAM + disentangling, stimare K_eff, stimare π̂_t dai soli esempi del round, e salvare artefatti.
  • Loggare metriche per round in `round_XXX/metrics.json` e artefatti (.npy) riusando le primitive esistenti.
  • Salvare in run_dir: `xi_true.npy`, `pis.npy`, (opzionale) `exposure_counts.npy`.

Note:
  • Setting **single** bloccato (nessuna logica "extend").
  • Le magnetizzazioni Hopfield NON sono calcolate qui: verranno calcolate dopo con novelty/reporting
    usando sinapsi Hebb da `xi_aligned.npy` generati per round.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple

import json
import math
import numpy as np

# -----------------------------
# Import dalla codebase esistente
# -----------------------------
# Config + iperparametri
try:
    from src.unsup.config import HyperParams, TAMParams, SpectralParams, PropagationParams  # type: ignore
except Exception:  # fallback leggero per test/scripting isolato
    @dataclass
    class TAMParams:
        beta_T: float = 2.5
        lam: float = 0.2
        h_in: float = 0.1
        updates: int = 50
        noise_scale: float = 0.3

    @dataclass
    class SpectralParams:
        tau: float = 0.5
        rho: float = 0.6
        qthr: float = 0.4

    @dataclass
    class PropagationParams:
        iters: int = 20

    @dataclass
    class HyperParams:
        L: int = 3
        K: int = 3
        N: int = 300
        n_batch: int = 24
        M_total: int = 2400
        r_ex: float = 0.8
        K_per_client: Optional[int] = None
        w: float = 0.0
        tam: TAMParams = TAMParams()
        prop: PropagationParams = PropagationParams()
        spec: SpectralParams = SpectralParams()
        estimate_keff_method: str = "shuffle"
        ema_alpha: float = 0.0
        use_tqdm: bool = False
        mode: str = "single"

# Funzioni core (riuso dalla codebase)
try:
    from src.unsup.functions import gen_patterns  # type: ignore
except Exception:
    # fallback minimale (solo per test isolato): genera ±1
    def gen_patterns(N: int, P: int) -> np.ndarray:
        return (2 * np.random.randint(0, 2, size=(P, N)) - 1).astype(np.int8)

try:
    from src.unsup.single_mode import (  # type: ignore
        build_unsup_J_single,
        blend_with_memory,
        propagate_J,
        eigen_cut,           # potremmo preferire spectral_cut se presente
        estimate_keff as estimate_keff_wrapped,
        dis_check,
        overlap_matrix,
    )
except Exception:  # pragma: no cover
    # Percorsi alternativi (se l'utente organizza diversamente i moduli)
    from src.unsup.functions import build_unsup_J_single, blend_with_memory, propagate_J, eigen_cut, dis_check, overlap_matrix  # type: ignore
    # Stima K_eff wrapper (MP/shuffle)
    def estimate_keff_wrapped(J, method="shuffle", **kwargs):
        from src.unsup.functions import estimate_K_eff_from_J  # type: ignore
        Keff, evals, info = estimate_K_eff_from_J(J, method=method, **kwargs)
        return Keff, evals, info

# Metriche mixing
try:
    from src.mixing.metrics import tv_distance, estimate_pi_hat_from_examples  # type: ignore
except Exception:
    def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
        p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
        return 0.5 * float(np.sum(np.abs(p - q)))
    def estimate_pi_hat_from_examples(xi_ref: np.ndarray, E_t: np.ndarray) -> np.ndarray:
        L, M_c, N = E_t.shape
        K = xi_ref.shape[0]
        X = E_t.reshape(L * M_c, N)
        Ov = X @ xi_ref[:K].T
        mu_hat = np.argmax(Ov, axis=1)
        counts = np.bincount(mu_hat, minlength=K).astype(float)
        return counts / (counts.sum() + 1e-9)

# Scheduler della novità (dal modulo novelty.py creato prima)
from .novelty import novelty_schedule  # type: ignore

# -----------------------------
# Helper di filesystem
# -----------------------------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _save_json(p: str | Path, obj: Dict[str, Any]) -> None:
    Path(p).write_text(json.dumps(obj, indent=2))

def _round_dir(run_dir: Path, t: int) -> Path:
    rd = run_dir / f"round_{t:03d}"
    rd.mkdir(parents=True, exist_ok=True)
    return rd

# -----------------------------
# Novelty spec (parametri esperimento 07)
# -----------------------------
@dataclass
class NoveltySpec:
    K_old: int
    K_new: int
    t_intro: int
    ramp_len: int
    alpha_max: float = 1.0
    new_visibility_frac: float = 1.0  # frazione di client che “vedono” i nuovi archetipi per round

# -----------------------------
# Sampling SINGLE per round (non strutturato)
# -----------------------------
def _sample_round_unstructured(
    xi_true: np.ndarray,
    pi_t: np.ndarray,
    L: int,
    M_c: int,
    r_ex: float,
    K_old: int,
    new_visibility_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera (ETA_t, labels_t) per un round in SINGLE-mode rispettando la visibilità parziale
    dei nuovi archetipi: solo una frazione di client (scelta random ad ogni round) può ricevere
    campioni dalle classi nuove (μ >= K_old).
    """
    K, N = xi_true.shape
    # Quanti client vedono i nuovi archetipi in questo round
    vis_count = int(np.floor(L * max(0.0, min(1.0, new_visibility_frac))))
    new_enabled = np.zeros(L, dtype=bool)
    if vis_count > 0:
        idx = rng.choice(np.arange(L), size=vis_count, replace=False)
        new_enabled[idx] = True

    # Numero totale esempi round
    total = L * M_c
    pi_t = np.asarray(pi_t, dtype=float)
    pi_t = pi_t / (pi_t.sum() + 1e-9)
    counts = rng.multinomial(total, pi_t)  # (K,)

    # Allocazione per client: i "nuovi" solo su client abilitati
    per_client_slots = [[None] * M_c for _ in range(L)]  # ogni entry → (class_idx)
    # round-robin per classe k sui client ammessi
    for k in range(K):
        n_k = counts[k]
        if n_k <= 0:
            continue
        if k >= K_old:
            allowed_clients = np.where(new_enabled)[0]
        else:
            allowed_clients = np.arange(L)
        if allowed_clients.size == 0:
            # se nessun client può ricevere classi nuove, riassegna agli "old" più vicini: fallback conservativo
            allowed_clients = np.arange(L)
        # costruisci lista di (l, slot) disponibili fra i client ammessi
        slots = []
        for l in allowed_clients:
            for m in range(M_c):
                if per_client_slots[l][m] is None:
                    slots.append((l, m))
        # assegna n_k elementi (se slots insufficienti, tronca → si riempirà con classi "old" di default)
        if not slots:
            continue
        pick = min(n_k, len(slots))
        chosen_idx = rng.choice(np.arange(len(slots)), size=pick, replace=False)
        for idx in chosen_idx:
            l, m = slots[int(idx)]
            per_client_slots[l][m] = k

    # Riempimento dei buchi (None) con classi "old" a proporzione residua
    resid = np.array([max(0, counts[k] - sum((np.array(per_client_slots[l]) == k).sum() for l in range(L))) for k in range(K)], dtype=int)
    resid[:K_old] += resid[K_old:]  # convoglia il residuo dei nuovi nelle classi old
    resid[K_old:] = 0
    # lista di classi old disponibili secondo resid
    old_pool = []
    for k in range(K_old):
        old_pool += [k] * int(resid[k])
    if not old_pool:
        # default: distribuzione uniforme tra i vecchi
        old_pool = [int(k) for k in range(K_old)] * max(1, (L * M_c) // max(1, K_old))

    optr = 0
    for l in range(L):
        for m in range(M_c):
            if per_client_slots[l][m] is None:
                per_client_slots[l][m] = old_pool[optr % len(old_pool)]
                optr += 1

    # Costruisci esempi con r_ex via moltiplicazione di maschera χ ∈ {±1}^N
    p_keep = 0.5 * (1.0 + float(r_ex))
    ETA_t = np.empty((L, M_c, N), dtype=np.float32)
    labels_t = np.empty((L, M_c), dtype=np.int32)
    for l in range(L):
        for m in range(M_c):
            k = int(per_client_slots[l][m])
            chi = (rng.uniform(size=N) <= p_keep).astype(np.float32) * 2.0 - 1.0  # ±1
            ETA_t[l, m] = chi * xi_true[k].astype(np.float32)
            labels_t[l, m] = k

    return ETA_t, labels_t

# -----------------------------
# Allineamento candidati → true (Hungarian + flip di segno)
# -----------------------------
def _align_candidates_to_true(xi_cand: np.ndarray, xi_true: np.ndarray, K: int) -> np.ndarray:
    """
    Restituisce una matrice (K, N) con i primi K candidati allineati a xi_true secondo matching
    Ungherese (max overlap) e orientati con segno coerente.
    Se i candidati sono < K, pad con i migliori disponibili.
    """
    from scipy.optimize import linear_sum_assignment
    if xi_cand.size == 0:
        return xi_true.copy().astype(np.int8)
    Ka, N = xi_cand.shape
    Kt, Nt = xi_true.shape
    if Nt != N:
        raise ValueError("Dimensioni incompatibili tra xi_cand e xi_true.")
    # Overlap assoluto
    M = np.abs(xi_cand @ xi_true.T) / float(N)  # (Ka,Kt)
    cost = 1.0 - M
    rI, cI = linear_sum_assignment(cost)
    # Costruisci array allineato (primi K match)
    order = np.argsort(M[rI, cI])[::-1]  # opzionale: ordina per overlap decrescente
    rI = rI[order]; cI = cI[order]
    take = min(K, min(Ka, Kt))
    xi_aligned = np.zeros((K, N), dtype=np.int8)
    used = 0
    for rr, cc in zip(rI[:take], cI[:take]):
        v = xi_cand[int(rr)]
        # binarizza e allinea il segno
        sgn = 1.0 if float(v @ xi_true[int(cc)]) >= 0.0 else -1.0
        xi_aligned[used] = np.where(v >= 0, 1, -1).astype(np.int8) * int(sgn)
        used += 1
    # padding se necessario
    for j in range(used, K):
        xi_aligned[j] = xi_true[j]
    return xi_aligned

# -----------------------------
# Funzione principale
# -----------------------------
def run_exp07_unstructured(
    run_dir: str | Path,
    hp: HyperParams,
    nov: NoveltySpec,
    *,
    seed: int = 0,
    save_mats: bool = True,
    save_spectrum: bool = True,
) -> Dict[str, Any]:
    """
    Esegue Exp-07 (single) su dati sintetici non strutturati.

    Parametri
    ---------
    run_dir : path in cui salvare i risultati
    hp : HyperParams (single-mode) dalla tua codebase
    nov : NoveltySpec (parametri di emersione)
    seed : random seed
    save_mats : se True salva J_unsup/J_rec/J_KS per round
    save_spectrum : se True salva V_sel/eigs per round

    Output
    ------
    dict con riferimenti principali (file generati, K_old/K, ecc.).
    """
    run_dir = _ensure_dir(run_dir)
    rng = np.random.default_rng(seed)

    # K totale e archetipi veri
    K = int(nov.K_old + nov.K_new)
    if hp.K != K:
        hp = HyperParams(**{**asdict(hp), "K": K})  # copia hp con K aggiornato
    xi_true = gen_patterns(hp.N, K).astype(np.int8)
    np.save(run_dir / "xi_true.npy", xi_true)

    # Schedule della novità (T x K) + salvataggio
    pis = novelty_schedule(
        T=int(hp.n_batch),
        K_old=int(nov.K_old),
        K_new=int(nov.K_new),
        t_intro=int(nov.t_intro),
        ramp_len=int(nov.ramp_len),
        alpha_max=float(nov.alpha_max),
        new_visibility_frac=float(nov.new_visibility_frac),
    )
    np.save(run_dir / "pis.npy", pis.astype(np.float32))

    # Deriva M_c dal budget
    M_c = int(math.ceil(hp.M_total / float(hp.L * hp.n_batch)))

    # Per memoria ebraica
    xi_ref: Optional[np.ndarray] = None

    # Exposure counts (per classe) aggregato su tutta la run (utile per Hopfield)
    exposure_counts = np.zeros(K, dtype=np.int64)

    # Loop dei round
    for t in range(hp.n_batch):
        rd = _round_dir(run_dir, t)

        # Dati round (single) con visibilità parziale dei nuovi
        ETA_t, labels_t = _sample_round_unstructured(
            xi_true=xi_true, pi_t=pis[t], L=hp.L, M_c=M_c, r_ex=hp.r_ex,
            K_old=nov.K_old, new_visibility_frac=nov.new_visibility_frac, rng=rng
        )
        # Aggiorna exposure
        vals, cnts = np.unique(labels_t, return_counts=True)
        exposure_counts[vals] += cnts

        # 1) stima J_unsup per round t
        J_unsup, M_eff = build_unsup_J_single(ETA_t, K=hp.K)

        # 2) blending con memoria ebraica precedente
        J_rec = blend_with_memory(J_unsup, xi_prev=xi_ref, w=hp.w)

        # 3) propagazione pseudo-inversa
        J_KS = np.asarray(propagate_J(J_rec, J_real=-1, verbose=False, iters=hp.prop.iters), dtype=np.float32)

        # 4) cut spettrale + spettro
        try:
            # preferisci interfaccia che restituisce info con evals
            from src.unsup.single_mode import spectral_cut  # type: ignore
            spec_out = spectral_cut(J_KS, tau=hp.spec.tau, return_info=True)
            if len(spec_out) == 3:
                V_sel, k_from_cut, info_spec = spec_out
            else:
                V_sel, k_from_cut = spec_out
                info_spec = {"evals": None}
        except Exception:
            # fallback su eigen_cut
            ec = eigen_cut(J_KS, tau=hp.spec.tau, return_info=True)
            if len(ec) == 3:
                V_sel, k_from_cut, info_spec = ec
            else:
                V_sel, k_from_cut = ec
                info_spec = {"evals": None}

        # 5) stima K_eff (shuffle|mp, MP usa M_eff del round)
        if hp.estimate_keff_method == "mp":
            K_eff, evals, info_keff = estimate_keff_wrapped(J_KS, method="mp", M_eff=M_eff)
        else:
            K_eff, evals, info_keff = estimate_keff_wrapped(J_KS, method="shuffle")
        # usa evals disponibili
        if info_spec.get("evals", None) is None and evals is not None:
            info_spec["evals"] = evals

        # 6) Disentangling + TAM (restituisce candidati xi_r in ordine informativo)
        xi_r, _m = dis_check(
            V=V_sel, K=hp.K, L=hp.L,
            J_rec=J_rec, JKS_iter=J_KS,
            xi_true=xi_true, tam=hp.tam, spec=hp.spec,
            show_progress=False, max_attempts=10,
        )
        xi_r = np.asarray(xi_r, dtype=float)
        # 6b) Allineamento candidati → true (binarizza + flip di segno)
        xi_aligned = _align_candidates_to_true(xi_r, xi_true, K=hp.K)

        # 7) π̂_t dai soli esempi del round e TV(π̂,π)
        pi_hat = estimate_pi_hat_from_examples(xi_true, ETA_t)  # usa i prototipi veri come riferimenti
        pi_true_t = pis[t] / (pis[t].sum() + 1e-9)
        TV_t = tv_distance(pi_hat, pi_true_t)

        # 8) Salvataggi per round
        metrics = {
            "round": int(t),
            "K_eff": int(K_eff),
            "TV_pi": float(TV_t),
            "pi_hat": pi_hat.tolist(),
            "pi_true": pi_true_t.tolist(),
            "keff_info": {
                "method": hp.estimate_keff_method,
                "eigvals": None if info_spec.get("evals", None) is None else np.asarray(info_spec["evals"], dtype=float).tolist(),
            },
            "shapes": {
                "V_sel": list(np.shape(V_sel)),
                "J_unsup": list(np.shape(J_unsup)),
                "J_rec": list(np.shape(J_rec)),
                "J_KS": list(np.shape(J_KS)),
                "xi_aligned": list(np.shape(xi_aligned)),
            },
        }
        _save_json(rd / "metrics.json", metrics)
        if save_mats:
            np.save(rd / "J_unsup.npy", np.asarray(J_unsup, dtype=np.float32))
            np.save(rd / "J_rec.npy", np.asarray(J_rec, dtype=np.float32))
            np.save(rd / "J_KS.npy", np.asarray(J_KS, dtype=np.float32))
        if save_spectrum:
            np.save(rd / "V_sel.npy", np.asarray(V_sel, dtype=np.float32))
            if info_spec.get("evals", None) is not None:
                np.save(rd / "eigs_sel.npy", np.asarray(info_spec["evals"], dtype=np.float32))
        np.save(rd / "xi_aligned.npy", xi_aligned.astype(np.int8))

        # 9) Memoria per round successivo (primi K candidati allineati)
        xi_ref = xi_aligned.copy()

    # Salvataggi top-level utili a novelty/reporting
    np.save(run_dir / "exposure_counts.npy", exposure_counts.astype(np.int64))

    # Meta run
    meta = {
        "K": int(K),
        "K_old": int(nov.K_old),
        "K_new": int(nov.K_new),
        "t_intro": int(nov.t_intro),
        "ramp_len": int(nov.ramp_len),
        "alpha_max": float(nov.alpha_max),
        "new_visibility_frac": float(nov.new_visibility_frac),
        "hp": {
            "L": int(hp.L), "N": int(hp.N), "n_batch": int(hp.n_batch),
            "M_total": int(hp.M_total), "r_ex": float(hp.r_ex),
            "w": float(hp.w),
            "prop.iters": int(hp.prop.iters),
            "spec.tau": float(hp.spec.tau),
            "estimate_keff_method": str(hp.estimate_keff_method),
        },
    }
    _save_json(run_dir / "meta_exp07.json", meta)

    return {
        "run_dir": str(run_dir),
        "K": int(K),
        "K_old": int(nov.K_old),
        "K_new": int(nov.K_new),
        "n_rounds": int(hp.n_batch),
        "paths": {
            "xi_true": str(run_dir / "xi_true.npy"),
            "pis": str(run_dir / "pis.npy"),
            "exposure_counts": str(run_dir / "exposure_counts.npy"),
            "meta": str(run_dir / "meta_exp07.json"),
        },
    }

# -----------------------------------------------------------------------------
# Esempio d'uso (script)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Directory di output della run")
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--K_old", type=int, default=2)
    ap.add_argument("--K_new", type=int, default=1)
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--M_total", type=int, default=2400)
    ap.add_argument("--r_ex", type=float, default=0.8)
    ap.add_argument("--w", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--keff", type=str, default="shuffle", choices=["shuffle", "mp"])
    ap.add_argument("--t_intro", type=int, default=8)
    ap.add_argument("--ramp_len", type=int, default=6)
    ap.add_argument("--alpha_max", type=float, default=1.0)
    ap.add_argument("--vis_frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hp = HyperParams(
        L=args.L, K=args.K_old + args.K_new, N=args.N, n_batch=args.T, M_total=args.M_total,
        r_ex=args.r_ex, w=args.w, estimate_keff_method=args.keff,
        prop=PropagationParams(iters=args.iters),
        spec=SpectralParams(tau=args.tau),
    )
    nov = NoveltySpec(
        K_old=args.K_old, K_new=args.K_new,
        t_intro=args.t_intro, ramp_len=args.ramp_len,
        alpha_max=args.alpha_max, new_visibility_frac=args.vis_frac,
    )
    out = run_exp07_unstructured(args.out, hp, nov, seed=args.seed)
    print(json.dumps(out, indent=2))
