#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calcolatore CLI / interattivo per Experiment 07 (nuovi archetipi).

Replica la logica di generazione di `gen_dataset_new_archetypes` nello script
`exp_07_new_archetypes_single.py` per analizzare il numero di esempi assegnati
ad ogni archetipo per round / client / totale, distinguendo archetipi vecchi e
nuovi introdotti con rampa.

Assunzioni dataset:
  - K = K_old + new_k (i new archetypes hanno indici K_old .. K-1)
  - Prima di t_intro solo i K_old sono visibili.
  - Dopo t_intro parte una rampa lineare di lunghezza ramp_len (>=1) che aumenta
    il numero K_add(t) = round(frac * new_k) di nuovi archetipi visibili (stesso
    set per tutti i client abilitati) con frac = (t - t_intro + 1)/ramp_len.
  - Solo una frazione new_visibility_frac dei client (round(L*frac)) può vedere
    i nuovi archetipi (client_has_new = 1). Gli altri restano sempre confinati
    ai K_old.
  - Ogni client genera M_c = ceil(M_total / (L * n_batch)) esempi per round
    campionando uniformemente dagli archetipi nel proprio pool visibile.

Output:
  - M_c, K_old, mapping archetipi old/new
  - Flag dei client che possono vedere nuovi archetipi
  - Per ogni round: K_true_round (numero effettivo di archetipi massimi visibili
    dai client abilitati), K_add(t)
  - Conteggi attesi per (round, client, archetipo)
  - Conteggi attesi aggregati per round, e totali su tutti i round
  - (Opzionale) Simulazione Monte Carlo: media ± std dei conteggi
  - JSON opzionale per elaborazioni successive

Uso rapido:
  python scripts/new_archetypes_calculator_exp07.py --K 6 --new-k 3 --t-intro 12 --ramp-len 4 --new-vis-frac 0.5 --n-batch 24 --M-total 1200 --L 3 --simulate 200
  python scripts/new_archetypes_calculator_exp07.py --interactive
"""
from __future__ import annotations
import argparse, math, json
from dataclasses import dataclass, asdict
from typing import Dict, Any
import numpy as np

# ------------------------ Hyperparams minimal ------------------------
@dataclass
class HParams:
    K: int = 6          # totale finale (old + new)
    new_k: int = 3      # nuovi archetipi introdotti
    L: int = 3          # numero client
    n_batch: int = 24   # round
    M_total: int = 1200
    t_intro: int = 12   # primo round (0-based) in cui parte introduzione
    ramp_len: int = 4   # durata rampa lineare
    new_visibility_frac: float = 0.5  # frazione client che possono vedere i nuovi
    seed: int = 111

    @property
    def K_old(self) -> int:
        return self.K - self.new_k

    @property
    def M_c(self) -> int:
        return math.ceil(self.M_total / (self.L * self.n_batch))

    @property
    def T(self) -> int:
        return self.n_batch

# ------------------------ Client flags ------------------------

def assign_client_flags(hp: HParams, rng: np.random.Generator) -> np.ndarray:
    L_new = int(round(hp.new_visibility_frac * hp.L))
    flags = np.array([1]*L_new + [0]*(hp.L - L_new), dtype=int)
    rng.shuffle(flags)
    return flags

# ------------------------ Helper: ramp K_add ------------------------

def k_add_at_round(hp: HParams, t: int) -> int:
    if t < hp.t_intro:
        return 0
    frac = (t - hp.t_intro + 1) / max(1, hp.ramp_len)
    frac = min(1.0, frac)
    return int(round(frac * hp.new_k))

# ------------------------ Expected counts ------------------------

def expected_counts(hp: HParams, client_flags: np.ndarray) -> Dict[str, Any]:
    T, L = hp.T, hp.L
    K_old, K_tot = hp.K_old, hp.K
    M_c = hp.M_c

    # expected_per_t_l_k: (T,L,K)
    exp_t_l_k = np.zeros((T, L, K_tot), dtype=float)
    K_true_round = np.zeros(T, dtype=int)
    K_add_round = np.zeros(T, dtype=int)

    for t in range(T):
        K_add = k_add_at_round(hp, t)
        K_vis = K_old + K_add  # visibili ai client abilitati
        K_true_round[t] = K_vis
        K_add_round[t] = K_add
        for l in range(L):
            if client_flags[l] == 0:  # client non abilitatI => sempre solo old
                probs = np.zeros(K_tot, dtype=float)
                probs[:K_old] = 1.0 / K_old
            else:
                if K_add == 0:  # ancora nessun nuovo
                    probs = np.zeros(K_tot, dtype=float)
                    probs[:K_old] = 1.0 / K_old
                else:
                    probs = np.zeros(K_tot, dtype=float)
                    probs[:K_vis] = 1.0 / K_vis  # old + primi K_add new
            exp_t_l_k[t, l] = M_c * probs

    exp_round_total = exp_t_l_k.sum(axis=1)  # (T,K)
    exp_total_per_k = exp_round_total.sum(axis=0)  # (K,)

    return dict(
        expected_per_t_l_k=exp_t_l_k.tolist(),
        expected_per_round_total=exp_round_total.tolist(),
        expected_total_per_archetype=exp_total_per_k.tolist(),
        K_true_round=K_true_round.tolist(),
        K_add_round=K_add_round.tolist(),
    )

# ------------------------ Simulation ------------------------

def simulate_counts(hp: HParams, client_flags: np.ndarray, sims: int, rng: np.random.Generator) -> Dict[str, Any]:
    if sims <= 0:
        return {}
    T, L, K_old, K_tot = hp.T, hp.L, hp.K_old, hp.K
    M_c = hp.M_c
    counts = np.zeros((sims, T, L, K_tot), dtype=int)
    for s in range(sims):
        for t in range(T):
            K_add = k_add_at_round(hp, t)
            K_vis = K_old + K_add
            for l in range(L):
                if client_flags[l] == 0:
                    pool = np.arange(K_old)
                else:
                    pool = np.arange(K_old) if K_add == 0 else np.arange(K_vis)
                mus = rng.choice(pool, size=M_c, replace=True)
                for mu in mus:
                    counts[s, t, l, mu] += 1
    mean_t_l_k = counts.mean(axis=0)
    std_t_l_k = counts.std(axis=0, ddof=1) if sims > 1 else np.zeros_like(mean_t_l_k)
    mean_round_total = mean_t_l_k.sum(axis=1)
    std_round_total = np.sqrt((std_t_l_k**2).sum(axis=1))
    mean_total_k = mean_round_total.sum(axis=0)
    std_total_k = np.sqrt((std_round_total**2).sum(axis=0))
    return dict(
        mean_per_t_l_k=mean_t_l_k.tolist(),
        std_per_t_l_k=std_t_l_k.tolist(),
        mean_round_total=mean_round_total.tolist(),
        std_round_total=std_round_total.tolist(),
        mean_total_per_archetype=mean_total_k.tolist(),
        std_total_per_archetype=std_total_k.tolist(),
        sims=sims,
    )

# ------------------------ Formatting ------------------------

def fmt_table(headers, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    def fmt(r): return " | ".join(str(x).rjust(w) for x, w in zip(r, widths))
    sep = "-+-".join('-'*w for w in widths)
    return '\n'.join([fmt(headers), sep] + [fmt(r) for r in rows])

# ------------------------ Core run ------------------------

def run(args):
    hp = HParams(K=args.K, new_k=args.new_k, L=args.L, n_batch=args.n_batch, M_total=args.M_total,
                 t_intro=args.t_intro, ramp_len=args.ramp_len, new_visibility_frac=args.new_visibility_frac,
                 seed=args.seed)
    rng = np.random.default_rng(hp.seed)
    client_flags = assign_client_flags(hp, rng)

    exp_data = expected_counts(hp, client_flags)
    sim_data = None
    if args.simulate > 0:
        sim_rng = np.random.default_rng(hp.seed + 999)
        sim_data = simulate_counts(hp, client_flags, args.simulate, sim_rng)

    if args.json:
        out = dict(hyperparams=asdict(hp), client_flags=client_flags.tolist(), expected=exp_data, simulation=sim_data)
        print(json.dumps(out, indent=2))
        return

    print("== Hyperparams ==")
    for k, v in asdict(hp).items():
        print(f"{k}: {v}")
    print(f"K_old={hp.K_old}, new_k={hp.new_k}, M_c={hp.M_c}\n")

    print("Client flags (1=vede nuovi):" )
    print(" ".join(str(f) for f in client_flags))

    print("\nRound -> K_add(t) / K_true_round(t) (per client abilitati)")
    K_add = exp_data['K_add_round']; K_true = exp_data['K_true_round']
    line = ", ".join(f"t{t}:{K_add[t]}/{K_true[t]}" for t in range(hp.T))
    print(line)

    # Expected per round aggregato
    print("\n== Expected per round (aggregato su client) ==")
    headers = ["round"] + [f"k{j}" for j in range(hp.K)] + ["tot"]
    rows = []
    exp_round = exp_data['expected_per_round_total']
    for t, vec in enumerate(exp_round):
        tot = sum(vec)
        rows.append([t] + [f"{v:.2f}" for v in vec] + [f"{tot:.0f}"])
    print(fmt_table(headers, rows))

    # Totali
    print("\n== Expected totale per archetipo (tutti i round, tutti i client) ==")
    tot_k = exp_data['expected_total_per_archetype']
    rows2 = [[f"k{j}", f"{tot_k[j]:.2f}"] for j in range(hp.K)]
    print(fmt_table(["archetipo","expected_total"], rows2))
    print(f"Somma totale attesa: {sum(tot_k):.0f}\n")

    if sim_data:
        print("== Simulazione: media ± std totale per archetipo ==")
        m = sim_data['mean_total_per_archetype']; s = sim_data['std_total_per_archetype']
        rows3 = [[f"k{j}", f"{m[j]:.2f} ± {s[j]:.2f}"] for j in range(hp.K)]
        print(fmt_table(["archetipo","mean_total ± std"], rows3))
        print("(Nota: std aggregata approssimata assumendo indipendenza round nella propagazione)\n")

    print("Fatto.")

# ------------------------ Argparse & interactive ------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Calcolatore nuovi archetipi (Exp07)")
    p.add_argument('--K', type=int, default=6, help='K totale finale (old+new)')
    p.add_argument('--new-k', dest='new_k', type=int, default=3, help='Numero nuovi archetipi')
    p.add_argument('--L', type=int, default=3)
    p.add_argument('--n-batch', dest='n_batch', type=int, default=24)
    p.add_argument('--M-total', dest='M_total', type=int, default=1200)
    p.add_argument('--t-intro', dest='t_intro', type=int, default=12)
    p.add_argument('--ramp-len', dest='ramp_len', type=int, default=4)
    p.add_argument('--new-vis-frac', dest='new_visibility_frac', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=111)
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
            hp.new_k = ask('new_k', hp.new_k, int)
            hp.L = ask('L', hp.L, int)
            hp.n_batch = ask('n_batch', hp.n_batch, int)
            hp.M_total = ask('M_total', hp.M_total, int)
            hp.t_intro = ask('t_intro', hp.t_intro, int)
            hp.ramp_len = ask('ramp_len', hp.ramp_len, int)
            hp.new_visibility_frac = ask('new_visibility_frac', hp.new_visibility_frac, float)
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
