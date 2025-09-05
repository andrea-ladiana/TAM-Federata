#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calcolatore CLI / interattivo per Exp-06 (sintetico, schedule π_t condivisa).

Obiettivo
---------
Dato l'insieme di iperparametri (stessi nomi / semantica di `scripts/exp06_synth.py`)
fornisce una panoramica completa di QUANTI ESEMPI ottiene ciascun client per
archetipo in ogni round, sia in termini ATTESI (analitici) che (opzionale)
SIMULATI via Monte Carlo (Multinomiali indipendenti per client & round).

Assunzioni dataset (coerenti con `run_seed_synth`):
  - Tutti i client condividono la stessa mixing schedule π_t ∈ Δ^{K-1}.
  - In ogni round t ciascun client genera M_c = ceil(M_total / (L * rounds)) esempi.
  - Ogni esempio del client l al round t sceglie un archetipo k ~ Categorical(π_t).
  - Non c'è differenza fra client: differenze emergono solo dal rumore multinomiale.

Output (human readable):
  - Riepilogo iperparametri e M_c (con overshoot vs M_total)
  - Tabella schedule π_t (t, π0..π{K-1}) e TV(π_t, π_{t-1})
  - Expected counts per (round, client, archetipo) (solo aggregati in forma utile)
  - Expected per round (aggregato su client)
  - Expected totale per archetipo (tutti i round, tutti i client)
  - Metriche riepilogo: imbalance ratio, min/max per archetipo, entropia media
  - (Se --simulate S) media ± std per le stesse grandezze + coefficiente di variazione
  - Opzione --json per avere un JSON completo (schedule, expected, simulation, meta)

Uso rapido
----------
  python scripts/calculators/archetype_calculator_exp06_synth.py --K 3 --L 3 --rounds 12 --M-total 4800 --schedule cyclic --period 12 --gamma 1.5 --temp 1.2 --center-mix 0.15
  python scripts/calculators/archetype_calculator_exp06_synth.py --schedule random_walk --K 3 --L 5 --rounds 24 --M-total 12000 --simulate 200
  python scripts/calculators/archetype_calculator_exp06_synth.py --interactive
  python scripts/calculators/archetype_calculator_exp06_synth.py --json --simulate 50 > out.json

Argomenti principali (subset di exp06_synth):
  --K, --L, --rounds, --M-total, --schedule {cyclic|piecewise_dirichlet|random_walk}
  Parametri schedule specifici (period,gamma,temp,center-mix,block,alpha,step-sigma,tv-max)
  --seed-base (seed per RNG)  --simulate S (Monte Carlo)  --json  --interactive

Note
----
I parametri non coinvolti nei conteggi (N, r_ex, w, ecc.) sono omessi; si possono
aggiungere facilmente se servisse tracciarli nel JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import numpy as np

# ---------------------------------------------------------------------------
# Inserimento automatico della root di progetto (contenente 'src') in sys.path
# ---------------------------------------------------------------------------
def _ensure_project_root_in_syspath() -> Path:
    here = Path(__file__).resolve()
    # prova alcune risalite comuni
    for p in [here.parent, here.parent.parent, here.parent.parent.parent]:
        if (p / "src").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p
    # fallback: cartella superiore
    fallback = here.parent.parent
    if (fallback / "src").exists() and str(fallback) not in sys.path:
        sys.path.insert(0, str(fallback))
    return fallback

_PROJECT_ROOT = _ensure_project_root_in_syspath()

try:
    from src.mixing.scheduler import make_schedule, total_variation  # type: ignore
except Exception:  # pragma: no cover - fallback se struttura alternativa
    try:
        from src.exp06_single.scheduler import make_schedule, total_variation  # type: ignore
    except Exception as e:  # ultima spiaggia: messaggio chiaro
        raise ImportError(
            "Impossibile importare 'src.mixing.scheduler' né 'src.exp06_single.scheduler'. "
            "Assicurati di eseguire il comando dalla root del progetto o che la cartella 'src' esista."
        ) from e


# ---------------------------------------------------------------------------
# Hyperparams minimi per i conteggi
# ---------------------------------------------------------------------------
@dataclass
class HParams:
    K: int = 3
    L: int = 3
    rounds: int = 12  # = n_batch / T
    M_total: int = 4800
    seed_base: int = 12345
    schedule: str = "cyclic"  # kind
    # --- schedule specific (default identici a exp06_synth.py) ---
    period: int = 12
    gamma: float = 0.5
    temp: float = 1.2
    center_mix: float = 0.60
    block: int = 4
    alpha: float = 1.0
    step_sigma: float = 0.7
    tv_max: float = 0.35

    @property
    def T(self) -> int:
        return int(self.rounds)

    # Compatibilità con scheduler originale che accede a hp.n_batch
    @property
    def n_batch(self) -> int:  # pragma: no cover - semplice alias
        return self.T

    @property
    def M_c(self) -> int:
        return math.ceil(self.M_total / (self.L * self.T))

    def schedule_kwargs(self) -> Dict[str, Any]:
        if self.schedule == "cyclic":
            return dict(period=self.period, gamma=self.gamma, temp=self.temp, center_mix=self.center_mix)
        if self.schedule == "piecewise_dirichlet":
            return dict(block=self.block, alpha=self.alpha)
        if self.schedule == "random_walk":
            return dict(step_sigma=self.step_sigma, tv_max=self.tv_max)
        raise ValueError(f"Schedule '{self.schedule}' non riconosciuta")

# ---------------------------------------------------------------------------
# Expected counts
# ---------------------------------------------------------------------------

def expected_counts(hp: HParams, pis: np.ndarray) -> Dict[str, Any]:
    """Calcola conteggi attesi.

    pis: (T,K)
    Expected per (t,l,k) = M_c * π_t[k].
    """
    T, K = pis.shape
    M_c = hp.M_c
    L = hp.L
    exp_t_l_k = np.repeat(pis[:, None, :], L, axis=1) * M_c  # T x L x K
    exp_round_total = exp_t_l_k.sum(axis=1)  # T x K
    exp_total_k = exp_round_total.sum(axis=0)  # K
    exp_total_per_client_k = exp_t_l_k.sum(axis=0)  # L x K
    tv_series = [0.0]
    for t in range(1, T):
        tv_series.append(total_variation(pis[t], pis[t-1]))

    # Metriche riepilogo
    imbalance_ratio = float(exp_total_k.max() / max(1e-12, exp_total_k.min()))
    entropies = (-pis * np.log(np.clip(pis, 1e-12, 1.0))).sum(axis=1)

    return dict(
        M_c=M_c,
        expected_per_t_l_k=exp_t_l_k.tolist(),
        expected_per_round_total=exp_round_total.tolist(),
        expected_total_per_archetype=exp_total_k.tolist(),
        expected_total_per_client_per_archetype=exp_total_per_client_k.tolist(),
        tv_series=tv_series,
        imbalance_ratio=imbalance_ratio,
        entropy_mean=float(entropies.mean()),
        entropy_min=float(entropies.min()),
        entropy_max=float(entropies.max()),
    )

# ---------------------------------------------------------------------------
# Simulation (Monte Carlo)
# ---------------------------------------------------------------------------

def simulate_counts(hp: HParams, pis: np.ndarray, sims: int, rng: np.random.Generator) -> Dict[str, Any]:
    if sims <= 0:
        return {}
    T, K = pis.shape
    L, M_c = hp.L, hp.M_c
    counts = np.zeros((sims, T, L, K), dtype=int)
    for s in range(sims):
        for t in range(T):
            p = pis[t]
            for l in range(L):
                # Multinomial generazione efficiente
                draws = rng.multinomial(M_c, pvals=p)
                counts[s, t, l] = draws
    mean_t_l_k = counts.mean(axis=0)
    std_t_l_k = counts.std(axis=0, ddof=1) if sims > 1 else np.zeros_like(mean_t_l_k)

    mean_round_total = mean_t_l_k.sum(axis=1)  # T x K
    std_round_total = np.sqrt((std_t_l_k**2).sum(axis=1))  # T x K (approx independence across clients)

    mean_total_k = mean_round_total.sum(axis=0)  # K
    std_total_k = np.sqrt((std_round_total**2).sum(axis=0))

    cv_total_k = (std_total_k / np.maximum(1e-12, mean_total_k)).tolist()

    return dict(
        mean_per_t_l_k=mean_t_l_k.tolist(),
        std_per_t_l_k=std_t_l_k.tolist(),
        mean_round_total=mean_round_total.tolist(),
        std_round_total=std_round_total.tolist(),
        mean_total_per_archetype=mean_total_k.tolist(),
        std_total_per_archetype=std_total_k.tolist(),
        cv_total_per_archetype=cv_total_k,
        sims=sims,
    )

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_table(headers, rows) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)] if rows else [len(h) for h in headers]
    def fmt(r):
        return " | ".join(str(x).rjust(w) for x, w in zip(r, widths))
    sep = "-+-".join('-'*w for w in widths)
    return "\n".join([fmt(headers), sep] + [fmt(r) for r in rows])

# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    hp = HParams(
        K=args.K, L=args.L, rounds=args.rounds, M_total=args.M_total,
        seed_base=args.seed_base, schedule=args.schedule,
        period=args.period, gamma=args.gamma, temp=args.temp, center_mix=args.center_mix,
        block=args.block, alpha=args.alpha, step_sigma=args.step_sigma, tv_max=args.tv_max,
    )
    rng = np.random.default_rng(hp.seed_base)
    pis = make_schedule(hp, kind=hp.schedule, rng=rng, **hp.schedule_kwargs())  # (T,K)

    exp_data = expected_counts(hp, pis)
    sim_data = simulate_counts(hp, pis, args.simulate, rng=np.random.default_rng(hp.seed_base + 999)) if args.simulate > 0 else None

    overshoot = hp.L * hp.T * hp.M_c - hp.M_total

    if args.json:
        out = dict(hyperparams=asdict(hp), schedule=pis.tolist(), expected=exp_data, simulation=sim_data,
                   derived=dict(total_effective=int(hp.L*hp.T*hp.M_c), overshoot=int(overshoot)))
        print(json.dumps(out, indent=2))
        return

    # Human readable --------------------------------------------------
    print("== Hyperparams ==")
    for k, v in asdict(hp).items():
        print(f"{k}: {v}")
    print(f"M_c (esempi per client per round) = {hp.M_c}  |  Totale effettivo = {hp.L*hp.T*hp.M_c}  (overshoot={overshoot})")

    # Schedule
    print("\n== Schedule π_t (t, componenti) ==")
    for t, p in enumerate(pis):
        probs = ", ".join(f"π{k}={p[k]:.4f}" for k in range(hp.K))
        tv = exp_data['tv_series'][t]
        print(f"t={t:02d}: {probs}  | TV(prev)={tv:.4f}")

    # Expected per round (aggregato)
    print("\n== Expected per round (aggregato su client) ==")
    headers = ["round"] + [f"k{j}" for j in range(hp.K)] + ["tot"]
    rows = []
    exp_round = exp_data['expected_per_round_total']
    for t, vec in enumerate(exp_round):
        tot = sum(vec)
        rows.append([t] + [f"{v:.2f}" for v in vec] + [f"{tot:.0f}"])
    print(_fmt_table(headers, rows))

    # Expected totale per archetipo
    print("\n== Expected totale per archetipo (tutti i round & client) ==")
    tot_k = exp_data['expected_total_per_archetype']
    rows2 = [[f"k{j}", f"{tot_k[j]:.2f}"] for j in range(hp.K)]
    print(_fmt_table(["archetipo", "expected_total"], rows2))
    print(f"Somma totale attesa (usando M_c): {sum(tot_k):.0f}")

    # Expected per client totale
    print("\n== Expected totale per client per archetipo ==")
    per_client = exp_data['expected_total_per_client_per_archetype']  # L x K
    headers_pc = ["client"] + [f"k{j}" for j in range(hp.K)] + ["tot_client"]
    rows_pc = []
    for l, row in enumerate(per_client):
        totc = sum(row)
        rows_pc.append([l] + [f"{v:.2f}" for v in row] + [f"{totc:.0f}"])
    print(_fmt_table(headers_pc, rows_pc))

    # Riepilogo metriche
    print("\n== Metriche riepilogo schedule ==")
    print(f"Imbalance ratio (max/min totale) = {exp_data['imbalance_ratio']:.4f}")
    print(f"Entropia media = {exp_data['entropy_mean']:.4f}  (min={exp_data['entropy_min']:.4f}, max={exp_data['entropy_max']:.4f})")

    if sim_data:
        print("\n== Simulazione (media ± std) totale per archetipo ==")
        m = sim_data['mean_total_per_archetype']; s = sim_data['std_total_per_archetype']; cv = sim_data['cv_total_per_archetype']
        rows3 = [[f"k{j}", f"{m[j]:.2f} ± {s[j]:.2f}", f"CV={float(cv[j]):.4f}"] for j in range(hp.K)]
        print(_fmt_table(["archetipo","mean_total ± std","note"], rows3))
        print("(CV = std/mean; la varianza singola Multinomiale per un client-round è M_c*p*(1-p))")

    print("\nFatto.")

# ---------------------------------------------------------------------------
# Argparse & interactive
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calcolatore counts Exp06 (synth)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--L', type=int, default=3)
    p.add_argument('--rounds', type=int, default=12, help='Numero di round (T)')
    p.add_argument('--M-total', dest='M_total', type=int, default=4800)
    p.add_argument('--seed-base', dest='seed_base', type=int, default=12345)
    p.add_argument('--schedule', type=str, default='cyclic', choices=('cyclic','piecewise_dirichlet','random_walk'))
    # cyclic
    p.add_argument('--period', type=int, default=12)
    p.add_argument('--gamma', type=float, default=1.5)
    p.add_argument('--temp', type=float, default=1.2)
    p.add_argument('--center-mix', dest='center_mix', type=float, default=0.15)
    # piecewise_dirichlet
    p.add_argument('--block', type=int, default=4)
    p.add_argument('--alpha', type=float, default=1.0)
    # random_walk
    p.add_argument('--step-sigma', dest='step_sigma', type=float, default=0.7)
    p.add_argument('--tv-max', dest='tv_max', type=float, default=0.35)

    p.add_argument('--simulate', type=int, default=0, help='Numero simulazioni Monte Carlo (0=off)')
    p.add_argument('--json', action='store_true', help='Output JSON')
    p.add_argument('--interactive', action='store_true', help='Modalità interattiva (loop)')
    return p


def interactive_loop():
    print("Modalità interattiva. Invio = default. Ctrl+C per uscire.")
    hp = HParams()
    while True:
        try:
            def ask(name, cur, typ):
                v = input(f"{name} [{cur}]: ").strip()
                return typ(v) if v else cur
            hp.K = ask('K', hp.K, int)
            hp.L = ask('L', hp.L, int)
            hp.rounds = ask('rounds', hp.rounds, int)
            hp.M_total = ask('M_total', hp.M_total, int)
            hp.seed_base = ask('seed_base', hp.seed_base, int)
            hp.schedule = ask('schedule (cyclic|piecewise_dirichlet|random_walk)', hp.schedule, str)
            if hp.schedule == 'cyclic':
                hp.period = ask('period', hp.period, int)
                hp.gamma = ask('gamma', hp.gamma, float)
                hp.temp = ask('temp', hp.temp, float)
                hp.center_mix = ask('center_mix', hp.center_mix, float)
            elif hp.schedule == 'piecewise_dirichlet':
                hp.block = ask('block', hp.block, int)
                hp.alpha = ask('alpha', hp.alpha, float)
            else:  # random_walk
                hp.step_sigma = ask('step_sigma', hp.step_sigma, float)
                hp.tv_max = ask('tv_max', hp.tv_max, float)
            sims = ask('simulate', 0, int)
            args = argparse.Namespace(**asdict(hp), simulate=sims, json=False, interactive=True)
            run(args)
            print("\n--- Nuovo calcolo ---\n")
        except KeyboardInterrupt:
            print("\nUscita.")
            break


def main(argv: Optional[list[str]] = None) -> None:
    p = build_argparser()
    args = p.parse_args(argv)
    if args.interactive:
        interactive_loop()
    else:
        run(args)


if __name__ == '__main__':
    main()
