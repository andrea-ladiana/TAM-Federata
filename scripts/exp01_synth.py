import os
import sys

# Make project root importable so top-level `src` package can be imported
# when running this script directly (e.g. `python scripts/exp01_synth.py`).
# Assumption: this file is located at <project_root>/scripts/...
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.unsup.config import HyperParams
from src.unsup.runner_single import run_exp01_single

# Baseline SINGLE
hp = HyperParams(
    mode="single",
    L=3, K=9, N=300,
    n_batch=12,
    M_total=600,       # ~M_c = ceil(M_total / (L*n_batch)) = 34
    r_ex=0.80,          # qualità esempi
    K_per_client=3,     # partizione “disgiunta” (3+3+3) → coverage piena per round
    w=0.8,              # blending unsup vs memoria Hebb
    n_seeds=1, seed_base=2025,
    use_tqdm=True,
)

res = run_exp01_single(hp, out_dir="out_01/synth_baseline", do_plot=True, force_run=True)

# res["aggregate"] contiene le serie mediate per round (retrieval, FRO, K_eff, coverage)
# out/synth_baseline/ avrà: hyperparams.json, log.jsonl, results_table.csv, fig_m etrics.png

# python scripts/run_exp01_synth.py
 