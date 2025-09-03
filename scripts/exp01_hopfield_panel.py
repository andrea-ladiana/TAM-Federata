import numpy as np
import matplotlib.pyplot as plt
from src.unsup.hopfield_eval import eval_retrieval_vs_exposure
from typing import Dict, Any

# Supponi di avere 'res' dal run precedente (o carica un seed a scelta)
from src.unsup.config import HyperParams
from src.unsup.runner_single import run_exp01_single

hp = HyperParams(K=9, L=3, N=300, n_batch=24, M_total=2400, r_ex=0.85, K_per_client=3, w=0.4)

res: Dict[str, Any] = run_exp01_single(hp, out_dir=None, do_plot=False)
seed_idx = 0
J_server = res["final_J_list"][seed_idx]
xi_ref   = res["final_xi_list"][seed_idx]      # disentangled final (puoi anche usare xi_true se li vuoi come ground truth)
expo     = res["exposure_list"][seed_idx]      # esposizioni per archetipo
K        = xi_ref.shape[0]

# Se vuoi usare i veri archetipi, sostituisci xi_ref con i veri ξ_true (dipende dalla tua pipeline).
# Qui usiamo i disentangled finali come reference per la magnetizzazione.
out: Dict[str, Any] = eval_retrieval_vs_exposure(J_server, xi_ref, exposure_counts=expo,
                                                 beta=3.0, updates=30, reps_per_archetype=32, start_overlap=0.3)

# Scatter: esposizioni vs magnetizzazione media
means = np.array([out["mean_by_mu"][mu] for mu in range(K)])
expo  = np.asarray(expo[:K], float)

plt.figure(figsize=(5,4))
plt.scatter(expo, means)
plt.xlabel("# esposizioni per archetipo")
plt.ylabel("magnetizzazione media (Hopfield)")
plt.title(f"corr: pearson={out['pearson']:.3f} spearman={out['spearman']:.3f}")
plt.tight_layout()
plt.savefig("out/synth_baseline/hopfield_exposure_scatter.png", dpi=150)
plt.close()
