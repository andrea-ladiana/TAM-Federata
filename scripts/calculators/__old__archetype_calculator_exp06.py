#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calcolatore CLI / interattivo per Experiment 06 (mixing drift K=3).

Obiettivo: dato l'insieme di iperparametri (K, L, n_batch, M_total, r_ex, drift_type,
period, drift_strength, seed) mostrare quanti esempi (attesi e simulati) vengono
assegnati per (round, client, archetipo) e i totali aggregati.

Logica dataset (come in exp_06_mixing_drift_k_3_single):
  M_c = ceil(M_total / (L * n_batch))
  Per ogni client l, round t: si estraggono M_c archetipi i.i.d. da distribuzione π_t (uguale per tutti i client) => conteggi ~ Multinomiale(M_c, π_t)
  Opzione simulazione: ripete S volte la generazione e riporta media ± std.

Output principali:
  - Schedule π_t (T x K)
  - TV(π_t, π_{t-1}) per t
  - Conteggi attesi per client/round/archetipo: M_c * π_t[k]
  - Expected per round aggregato sui client: L * M_c * π_t
  - Totali su tutti i round (expected): L * M_c * sum_t π_t
  - (Simulazione) media e std per le stesse grandezze
  - Esportazione JSON opzionale

Uso:
  python scripts/mixing_drift_calculator_exp06.py --n-batch 24 --M-total 2400 --L 3 --K 3 --drift-type cyclic --period 12 --drift-strength 1.0 --simulate 100
  python scripts/mixing_drift_calculator_exp06.py --interactive

"""
from __future__ import annotations
import argparse, math, json
from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any
import numpy as np

# ---------------- Iperparametri ----------------
@dataclass
class HParams:
    K: int = 3
    L: int = 3
    N: int = 300  # non entra nei conteggi ma lo manteniamo per completezza
    n_batch: int = 24
    M_total: int = 2400
    r_ex: float = 0.8  # non influenza conteggi archetipi
    drift_type: Literal["cyclic", "piecewise_dirichlet", "random_walk"] = "cyclic"
    drift_strength: float = 1.0
    period: int = 12
    seed: int = 120001

    @property
    def M_c(self) -> int:
        return math.ceil(self.M_total / (self.L * self.n_batch))

    @property
    def T(self) -> int:
        return self.n_batch

# -------------- Schedule mixing (copiata/adattata) --------------

def schedule_mixing(hp: HParams, rng: np.random.Generator) -> np.ndarray:
    T = hp.n_batch
    if hp.drift_type == "cyclic":
        A = float(hp.drift_strength)
        t = np.arange(T)
        phi = 2 * np.pi * t / max(1, hp.period)
        x = 1.0/3 + (A/3) * np.cos(phi)
        y = 1.0/3 + (A/3) * np.cos(phi + 2*np.pi/3)
        z = 1.0 - x - y
        pis = np.stack([x, y, z], axis=1)
    elif hp.drift_type == "piecewise_dirichlet":
        pis = []
        seg = max(1, hp.period)
        for t in range(T):
            if t % seg == 0:
                base = rng.dirichlet(alpha=[1.0]*hp.K)
            pis.append(base)
        pis = np.array(pis)
    else:  # random_walk
        p = np.full(hp.K, 1.0/hp.K, dtype=float)
        pis = []
        for _ in range(T):
            p = p + 0.05 * rng.normal(size=hp.K)
            p = np.maximum(1e-4, p)
            p /= p.sum()
            pis.append(p)
        pis = np.array(pis)
    # Clamp ed normalize (numerica)
    pis = np.maximum(1e-9, pis)
    pis /= pis.sum(axis=1, keepdims=True)
    return pis

# -------------- TV distance --------------

def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q)))

# -------------- Expected counts --------------

def expected_counts(hp: HParams, pis: np.ndarray) -> Dict[str, Any]:
    # shape pis: (T, K)
    M_c = hp.M_c
    L = hp.L
    # Expected per (t, l, k) = M_c * pis[t,k]
    exp_t_l_k = np.repeat(pis[:, None, :], L, axis=1) * M_c  # (T, L, K)
    # Aggregated per round across clients
    exp_round_tot_k = exp_t_l_k.sum(axis=1)  # (T, K) = L*M_c*pis[t]
    # Global totals over all rounds
    exp_total_k = exp_round_tot_k.sum(axis=0)  # (K,)
    # TV series
    tv_series = [0.0]
    for t in range(1, hp.T):
        tv_series.append(tv_distance(pis[t], pis[t-1]))
    return dict(
        M_c=M_c,
        expected_per_t_l_k=exp_t_l_k.tolist(),
        expected_per_round_total=exp_round_tot_k.tolist(),
        expected_total_per_archetype=exp_total_k.tolist(),
        tv_series=tv_series,
    )

# -------------- Simulation --------------

def simulate_counts(hp: HParams, pis: np.ndarray, sims: int, rng: np.random.Generator) -> Dict[str, Any]:
    T, K = pis.shape; L = hp.L; M_c = hp.M_c
    # counts: sims x T x L x K
    counts = np.zeros((sims, T, L, K), dtype=int)
    for s in range(sims):
        for t in range(T):
            p = pis[t]
            for l in range(L):
                mus = rng.choice(K, size=M_c, p=p)
                for mu in mus:
                    counts[s, t, l, mu] += 1
    mean_t_l_k = counts.mean(axis=0)
    std_t_l_k = counts.std(axis=0, ddof=1) if sims > 1 else np.zeros_like(mean_t_l_k)
    mean_round_tot = mean_t_l_k.sum(axis=1)  # T x K
    std_round_tot = np.sqrt((std_t_l_k**2).sum(axis=1))  # approx (var somma) T x K
    mean_total_k = mean_round_tot.sum(axis=0)  # K
    # var totale: somma var sui round + 2*cov ~ assumiamo indip per semplicità
    std_total_k = np.sqrt((std_round_tot**2).sum(axis=0))
    return dict(
        mean_per_t_l_k=mean_t_l_k.tolist(),
        std_per_t_l_k=std_t_l_k.tolist(),
        mean_round_total=mean_round_tot.tolist(),
        std_round_total=std_round_tot.tolist(),
        mean_total_per_archetype=mean_total_k.tolist(),
        std_total_per_archetype=std_total_k.tolist(),
        sims=sims,
    )

# -------------- Formatting --------------

def fmt_table(headers, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    def fmt(r): return " | ".join(str(x).rjust(w) for x, w in zip(r, widths))
    sep = "-+-".join('-'*w for w in widths)
    return '\n'.join([fmt(headers), sep] + [fmt(r) for r in rows])

# -------------- Core run --------------

def run(args):
    hp = HParams(K=args.K, L=args.L, N=args.N, n_batch=args.n_batch, M_total=args.M_total,
                 r_ex=args.r_ex, drift_type=args.drift_type, drift_strength=args.drift_strength,
                 period=args.period, seed=args.seed)
    rng = np.random.default_rng(hp.seed)
    pis = schedule_mixing(hp, rng)
    exp_data = expected_counts(hp, pis)

    sim_data = None
    if args.simulate > 0:
        sim_rng = np.random.default_rng(hp.seed + 777)
        sim_data = simulate_counts(hp, pis, args.simulate, sim_rng)

    if args.json:
        out = dict(hyperparams=asdict(hp), pis=pis.tolist(), expected=exp_data, simulation=sim_data)
        print(json.dumps(out, indent=2))
        return

    # Human readable
    print("== Hyperparams ==")
    for k, v in asdict(hp).items():
        print(f"{k}: {v}")
    print(f"M_c (per client per round) = {hp.M_c}\n")

    print("== Schedule π_t == (t -> probs)")
    for t, p in enumerate(pis):
        print(f"t={t:02d}: " + ", ".join(f"π{k}={p[k]:.4f}" for k in range(hp.K)))

    print("\nTV consecutivo (0 per t=0):")
    print(", ".join(f"{x:.3f}" for x in exp_data['tv_series']))

    # Expected per round aggregato
    print("\n== Expected per round (aggregato su client) ==")
    headers = ["round"] + [f"k{j}" for j in range(hp.K)] + ["tot"]
    rows = []
    exp_round = exp_data['expected_per_round_total']  # T x K
    for t, vec in enumerate(exp_round):
        tot = sum(vec)
        rows.append([t] + [f"{v:.2f}" for v in vec] + [f"{tot:.0f}"])
    print(fmt_table(headers, rows))

    # Expected total
    print("\n== Expected totale per archetipo (tutti i round, tutti i client) ==")
    tot_k = exp_data['expected_total_per_archetype']
    rows2 = [[f"k{j}", f"{tot_k[j]:.2f}"] for j in range(hp.K)]
    print(fmt_table(["archetipo", "expected_total"], rows2))
    print(f"Somma totale attesa: {sum(tot_k):.0f}\n")

    if sim_data:
        print("== Simulazione: media ± std totale per archetipo ==")
        m = sim_data['mean_total_per_archetype']
        s = sim_data['std_total_per_archetype']
        rows3 = [[f"k{j}", f"{m[j]:.2f} ± {s[j]:.2f}"] for j in range(hp.K)]
        print(fmt_table(["archetipo", "mean_total ± std"], rows3))
        print("(Nota: std teorica Multinomiale ~ sqrt(M_c * p * (1-p)) per round, poi propagata)\n")

    print("Fatto.")

# -------------- Argparse & interactive --------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Calcolatore conteggi Experiment 06 mixing drift")
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--L', type=int, default=3)
    p.add_argument('--N', type=int, default=300)
    p.add_argument('--n-batch', dest='n_batch', type=int, default=24)
    p.add_argument('--M-total', dest='M_total', type=int, default=2400)
    p.add_argument('--r-ex', dest='r_ex', type=float, default=0.8)
    p.add_argument('--drift-type', choices=['cyclic','piecewise_dirichlet','random_walk'], default='cyclic')
    p.add_argument('--drift-strength', type=float, default=1.0)
    p.add_argument('--period', type=int, default=12)
    p.add_argument('--seed', type=int, default=120001)
    p.add_argument('--simulate', type=int, default=0)
    p.add_argument('--json', action='store_true')
    p.add_argument('--interactive', action='store_true')
    return p.parse_args(argv)


def interactive_loop():
    print("Modalità interattiva. Invio = default. Ctrl+C per uscire.")
    hp = HParams()
    while True:
        try:
            def ask(name, cur, typ):
                val = input(f"{name} [{cur}]: ").strip()
                return typ(val) if val else cur
            hp.K = ask('K', hp.K, int)
            hp.L = ask('L', hp.L, int)
            hp.n_batch = ask('n_batch', hp.n_batch, int)
            hp.M_total = ask('M_total', hp.M_total, int)
            hp.r_ex = ask('r_ex', hp.r_ex, float)
            hp.drift_type = ask('drift_type (cyclic|piecewise_dirichlet|random_walk)', hp.drift_type, str)
            hp.drift_strength = ask('drift_strength', hp.drift_strength, float)
            hp.period = ask('period', hp.period, int)
            hp.seed = ask('seed', hp.seed, int)
            sims = ask('simulate', 0, int)
            args = argparse.Namespace(**asdict(hp), simulate=sims, json=False, interactive=True)
            run(args)
            print("\n--- Nuovo calcolo ---\n")
        except KeyboardInterrupt:
            print("\nUscita.")
            break


def main(argv=None):
    args = parse_args(argv)
    if args.interactive:
        interactive_loop()
    else:
        run(args)

if __name__ == '__main__':
    main()
