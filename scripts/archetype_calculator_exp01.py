#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive / CLI calculator for archetype example distribution.

Dato l'insieme di iperparametri (K, L, n_batch, M_total, K_per_client, seed ...)
calcola:
- M_c per round per client
- Numero totale di esempi per client (attesi e simulati)
- Sottoinsiemi di archetipi assegnati a ciascun client (algoritmo come exp_01_partial_archetypes_single)
- Conteggi attesi per archetipo (per client, per round, totali) assumendo campionamento uniforme sugli archetipi consentiti
- (Opzionale) Simulazione stocastica del dataset come nello script originale per stimare media e deviazione standard dei conteggi

Uso rapido (PowerShell):
  python scripts/archetype_calculator.py --K 6 --L 3 --n-batch 10 --M-total 200 --K-per-client 3 --seed 123
  python scripts/archetype_calculator.py --interactive

Argomenti chiave:
  --simulate N    -> esegue N simulazioni (default 0 = solo valori attesi)
  --deterministic -> mostra solo valori attesi (uguale a simulate=0)
  --json          -> output in JSON (utile per piping / automazione)

"""
from __future__ import annotations

import argparse
import math
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np

# ---------------------------------------------------------------------------
# Dataclass iperparametri minimi necessari
# ---------------------------------------------------------------------------
@dataclass
class HParams:
    K: int = 6
    L: int = 3
    n_batch: int = 10  # numero round
    M_total: int = 200  # numero esempi totali target (approssimato)
    K_per_client: int = 3
    seed: int = 123

    @property
    def M_c(self) -> int:
        """Numero di esempi per client per round (stessa definizione esperimenti)."""
        return math.ceil(self.M_total / (self.L * self.n_batch))

    @property
    def M_client_tot(self) -> int:
        return self.n_batch * self.M_c

# ---------------------------------------------------------------------------
# Replica di make_client_subsets dallo script esperimento
# ---------------------------------------------------------------------------

def make_client_subsets(K: int, L: int, K_per_client: int, rng: np.random.Generator) -> List[List[int]]:
    if K_per_client <= 0:
        raise ValueError("K_per_client deve essere > 0")
    if K_per_client > K:
        raise ValueError("K_per_client non può superare K")
    subsets: List[List[int]] = [[] for _ in range(L)]
    for i, mu in enumerate(rng.permutation(K)):
        subsets[i % L].append(int(mu))
    for l in range(L):
        need = K_per_client - len(subsets[l])
        pool = [mu for mu in range(K) if mu not in subsets[l]]
        if need > 0:
            choices = rng.choice(pool, size=need, replace=False)
            subsets[l].extend(int(x) for x in choices)
    return [sorted(set(s)) for s in subsets]

# ---------------------------------------------------------------------------
# Calcoli attesi
# ---------------------------------------------------------------------------

def expected_counts(hp: HParams, subsets: List[List[int]]) -> Dict[str, Any]:
    """Restituisce dizionario con conteggi attesi.

    Ogni client l genera M_c esempi per round scegliendo uniformemente tra i K_per_client archetipi consentiti.
    Quindi atteso per archetipo (se presente nel subset) per round: M_c / K_per_client.
    """
    M_c = hp.M_c
    per_round_per_client = M_c
    exp_per_archetype_per_round_if_present = M_c / hp.K_per_client

    # Matrice attesa per client (L x K)
    mat = np.zeros((hp.L, hp.K), dtype=float)
    for l, s in enumerate(subsets):
        for mu in s:
            mat[l, mu] = exp_per_archetype_per_round_if_present

    exp_round_total_per_archetype = mat.sum(axis=0)  # somma sui client
    exp_all_rounds_per_client = mat * hp.n_batch  # L x K (tutti i round)
    exp_all_rounds_total_per_archetype = exp_round_total_per_archetype * hp.n_batch

    return dict(
        M_c=M_c,
        M_client_tot=hp.M_client_tot,
        expected_client_matrix_round=mat.tolist(),
        expected_round_total_per_archetype=exp_round_total_per_archetype.tolist(),
        expected_all_rounds_per_client=exp_all_rounds_per_client.tolist(),
        expected_all_rounds_total_per_archetype=exp_all_rounds_total_per_archetype.tolist(),
        exp_per_archetype_per_round_if_present=exp_per_archetype_per_round_if_present,
    )

# ---------------------------------------------------------------------------
# Simulazione dataset come nello script originale (solo per i label)
# ---------------------------------------------------------------------------

def simulate_counts(hp: HParams, subsets: List[List[int]], sims: int, rng: np.random.Generator) -> Dict[str, Any]:
    K, L = hp.K, hp.L
    M_c = hp.M_c
    nB = hp.n_batch

    # Accumulatori
    # sims x n_batch x L x K
    counts = np.zeros((sims, nB, L, K), dtype=int)
    for s in range(sims):
        for t in range(nB):
            for l in range(L):
                allowed = subsets[l]
                mus = rng.choice(allowed, size=M_c, replace=True)
                # conteggi
                for mu in mus:
                    counts[s, t, l, mu] += 1
    # Statistiche
    mean_per_round = counts.mean(axis=0)          # n_batch x L x K
    std_per_round = counts.std(axis=0, ddof=1) if sims > 1 else np.zeros_like(mean_per_round)

    total_all_rounds = counts.sum(axis=1)  # sims x L x K (sommato sui round)
    mean_total = total_all_rounds.mean(axis=0)
    std_total = total_all_rounds.std(axis=0, ddof=1) if sims > 1 else np.zeros_like(mean_total)

    agg_round_total = counts.sum(axis=2)  # sims x n_batch x K (somma sui client)
    mean_round_total = agg_round_total.mean(axis=0)
    std_round_total = agg_round_total.std(axis=0, ddof=1) if sims > 1 else np.zeros_like(mean_round_total)

    agg_all_total = total_all_rounds.sum(axis=1)  # sims x K (somma sui client)
    mean_all_total = agg_all_total.mean(axis=0)
    std_all_total = agg_all_total.std(axis=0, ddof=1) if sims > 1 else np.zeros_like(mean_all_total)

    return dict(
        mean_per_round=mean_per_round.tolist(),
        std_per_round=std_per_round.tolist(),
        mean_total_per_client=mean_total.tolist(),
        std_total_per_client=std_total.tolist(),
        mean_round_total_per_archetype=mean_round_total.tolist(),
        std_round_total_per_archetype=std_round_total.tolist(),
        mean_all_rounds_total_per_archetype=mean_all_total.tolist(),
        std_all_rounds_total_per_archetype=std_all_total.tolist(),
        sims=sims,
    )

# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

def format_table(header: List[str], rows: List[List[Any]]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    def fmt_row(r):
        return " | ".join(str(x).rjust(w) for x, w in zip(r, widths))
    sep = "-+-".join('-'*w for w in widths)
    lines = [fmt_row(header), sep]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main CLI / interactive
# ---------------------------------------------------------------------------

def run_calc(args: argparse.Namespace) -> None:
    hp = HParams(K=args.K, L=args.L, n_batch=args.n_batch, M_total=args.M_total, K_per_client=args.K_per_client, seed=args.seed)
    rng = np.random.default_rng(hp.seed)
    subsets = make_client_subsets(hp.K, hp.L, hp.K_per_client, rng)

    exp_data = expected_counts(hp, subsets)

    sim_data = None
    if args.simulate > 0:
        sim_rng = np.random.default_rng(hp.seed + 10_000)
        sim_data = simulate_counts(hp, subsets, args.simulate, sim_rng)

    if args.json:
        out = dict(hyperparams=asdict(hp), subsets=subsets, expected=exp_data, simulation=sim_data)
        print(json.dumps(out, indent=2))
        return

    # Output umano leggibile
    print("== Hyperparams ==")
    for k, v in asdict(hp).items():
        print(f"{k}: {v}")
    print(f"M_c (per client per round) = {exp_data['M_c']}")
    print(f"M_client_tot (per client su tutti i round) = {exp_data['M_client_tot']}")

    print("\n== Sottoinsiemi archetipi per client ==")
    for l, s in enumerate(subsets):
        print(f"Client {l}: {s}")

    print("\n== Conteggi attesi per archetipo per round (aggregati su tutti i client) ==")
    header = ["archetype", "exp_per_round_total", "exp_all_rounds_total"]
    rows = []
    exp_round = exp_data['expected_round_total_per_archetype']
    exp_all = exp_data['expected_all_rounds_total_per_archetype']
    for mu in range(hp.K):
        rows.append([mu, f"{exp_round[mu]:.2f}", f"{exp_all[mu]:.2f}"])
    print(format_table(header, rows))

    if sim_data:
        print("\n== Simulazione (media ± std) conteggi totali per archetipo (tutti i client, tutti i round) ==")
        mean_all = sim_data['mean_all_rounds_total_per_archetype']
        std_all = sim_data['std_all_rounds_total_per_archetype']
        header2 = ["archetype", "mean_all_rounds_total ± std"]
        rows2 = []
        for mu in range(hp.K):
            rows2.append([mu, f"{mean_all[mu]:.2f} ± {std_all[mu]:.2f}"])
        print(format_table(header2, rows2))

        print("\n(NB: le deviazioni standard calano ~ sqrt(sims); aumentare --simulate per stime più stabili)")

    print("\nFatto.")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calcolatore distribuzione esempi per archetipo nei dataset parziali")
    p.add_argument('--K', type=int, default=6, help='Numero archetipi')
    p.add_argument('--L', type=int, default=3, help='Numero client')
    p.add_argument('--n-batch', dest='n_batch', type=int, default=10, help='Numero round (n_batch)')
    p.add_argument('--M-total', dest='M_total', type=int, default=200, help='Totale target esempi M_total')
    p.add_argument('--K-per-client', dest='K_per_client', type=int, default=3, help='Archetipi per client (subset)')
    p.add_argument('--seed', type=int, default=123, help='Seed per subset')
    p.add_argument('--simulate', type=int, default=0, help='Numero simulazioni dataset (0 = none)')
    p.add_argument('--json', action='store_true', help='Output JSON')
    p.add_argument('--interactive', action='store_true', help='Modalità interattiva (loop)')
    return p.parse_args(argv)


def interactive_loop():
    print("Modalità interattiva. Premi invio per usare il default tra [] oppure digita un valore. Ctrl+C per uscire.")
    hp = HParams()
    while True:
        try:
            def ask(name, cur):
                val = input(f"{name} [{cur}]: ").strip()
                return type(cur)(val) if val else cur
            hp.K = ask('K', hp.K)
            hp.L = ask('L', hp.L)
            hp.n_batch = ask('n_batch', hp.n_batch)
            hp.M_total = ask('M_total', hp.M_total)
            hp.K_per_client = ask('K_per_client', hp.K_per_client)
            hp.seed = ask('seed', hp.seed)
            sims_in = input('simulate (0=none) [0]: ').strip()
            sims = int(sims_in) if sims_in else 0
            args = argparse.Namespace(**asdict(hp), simulate=sims, json=False, interactive=True)
            run_calc(args)
            print("\n--- Nuovo calcolo ---\n")
        except KeyboardInterrupt:
            print("\nUscita.")
            break


def main(argv=None):
    args = parse_args(argv)
    if args.interactive:
        interactive_loop()
    else:
        run_calc(args)


if __name__ == '__main__':
    main()
