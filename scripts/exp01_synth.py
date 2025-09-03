from src.unsup.config import HyperParams
from src.unsup.runner_single import run_exp01_single

# Baseline SINGLE
hp = HyperParams(
    mode="single",
    L=3, K=9, N=300,
    n_batch=24,
    M_total=2400,       # ~M_c = ceil(M_total / (L*n_batch)) = 34
    r_ex=0.85,          # qualità esempi
    K_per_client=3,     # partizione “disgiunta” (3+3+3) → coverage piena per round
    w=0.8,              # blending unsup vs memoria Hebb
    n_seeds=5, seed_base=1234,
    use_tqdm=True,
)

res = run_exp01_single(hp, out_dir="out_01/synth_baseline", do_plot=True)

# res["aggregate"] contiene le serie mediate per round (retrieval, FRO, K_eff, coverage)
# out/synth_baseline/ avrà: hyperparams.json, log.jsonl, results_table.csv, fig_metrics.png

# python scripts/run_exp01_synth.py
