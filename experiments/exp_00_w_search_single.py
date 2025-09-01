#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 00 â€” Grid search di w in modalitÃ  SINGLE
===================================================
Esegue una grid-search di w usando solo i dati del round corrente (no extend),
seguendo lo stesso schema di Dynamics.py ma con le API in Dynamics_single_mode.py.

Output tipico (cartella `risultati/results_w_search_single/`):
- JSON/CSV dei risultati aggregati
- PNG con curve aggregate della griglia (refine level)

Nota: non modifica i file originali; questa Ã¨ una variante dedicata alla modalitÃ  single.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

# Ensure src/ on sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from unsup.dynamics_single_mode import (
    grid_search_w_single,
    refine_grid_search_single,
)


def main():
    # Base params (allineati a Dynamics.py)
    base_params = dict(
        L=3,
        K=6,
        N=200,
        M_unsup=600,
        r_ex=0.7,
        n_batch=12,
        updates=60,
        use_mp_keff=True,
        shuffle_alpha=0.01,
        shuffle_random=32,
    )
    seeds = [123, 124, 125]

    # Griglia coarse e save
    w_grid_coarse = [round(x, 2) for x in np.linspace(0.0, 1.0, 11)]
    report = grid_search_w_single(base_params, w_grid_coarse, seeds, verbose=True)

    out_dir = Path("risultati") / "results_w_search_single"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'grid_search_coarse.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Refinement nel range alto (tipicamente w in [0.8,1.0])
    refined = refine_grid_search_single(
        report,
        base_params,
        seeds,
        fine_step=0.02,
        span=0.10,
        center_mode='one_se',
        max_refinements=1,
        results_dir=str(out_dir / 'refine_level1'),
        fixed_range=(0.8, 1.0),
    )
    with open(out_dir / 'grid_search_fine.json', 'w') as f:
        json.dump(refined, f, indent=2)


if __name__ == '__main__':
    main()
