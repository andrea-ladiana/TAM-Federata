#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-res wrapper for exp_01_partial_archetypes_single
-----------------------------------------------------
Esegue la versione single con iperparametri piÃ¹ alti per ottenere curve piÃ¹ smooth.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import sys
from pathlib import Path as _P

_THIS = _P(__file__).resolve()
ROOT = _THIS
while ROOT != ROOT.parent and not (ROOT / "Functions.py").exists():
    ROOT = ROOT.parent

SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import numpy as np
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

import experiments.exp_01_partial_archetypes_single as mod
from dataclasses import asdict


def main():
    hp = mod.HyperParams(
        L=5,
        K=12,
        N=400,
        n_batch=24,
        M_total=2400,
        r_ex=0.7,
        K_per_client=4,
        updates=80,
        w=0.85,
        n_seeds=8,
        seed_base=123,
        pb_seeds=True,
        use_mp_keff=True,
    )

    base_dir = ROOT / "stress_tests" / "exp01_partial_archetypes_single_highres"
    tag = f"K{hp.K}_N{hp.N}_L{hp.L}_nb{hp.n_batch}_M{hp.M_total}_r{hp.r_ex}_c{hp.K_per_client}_w{hp.w}"
    exp_dir = base_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "hyperparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    results = []
    log_path = exp_dir / "log.jsonl"
    with open(log_path, "w") as flog:
        seed_iter = range(hp.n_seeds)
        if hp.pb_seeds:
            seed_iter = tqdm(seed_iter, desc="seeds")
        for s in seed_iter:
            seed = hp.seed_base + s
            t0 = time.perf_counter()
            res = mod.run_one_seed_single(hp, seed)
            t1 = time.perf_counter()
            results.append(res)
            row = {
                "mode": "single",
                "seed": res['seed'],
                "m_first": res['m_first'],
                "m_final": res['m_final'],
                "G_single": res['G_single'],
                "fro_final": res['fro_final'],
                "deltaK": res['deltaK'],
                "rounds": res['series']['rounds'],
                "m_series": res['series']['m_single_mean'],
                "fro_series": res['series']['fro_single'],
                "keff_series": res['series']['keff_single'],
                "coverage_series": res['series']['coverage_single'],
                "elapsed_s": t1 - t0,
            }
            flog.write(json.dumps(row) + "\n")
    mod.aggregate_and_plot(hp, results, exp_dir)


if __name__ == '__main__':
    main()
