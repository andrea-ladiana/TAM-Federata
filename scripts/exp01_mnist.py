import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict

from src.unsup.config import HyperParams
from src.unsup.mnist_hfl import load_mnist, binarize_images, class_prototypes_sign_mean
from src.unsup.mnist_hfl import build_class_mapping, make_mnist_hfl_subsets, gen_dataset_from_mnist_single
from src.unsup.single_round import single_round_step, RoundLog
from src.unsup.metrics import retrieval_mean_hungarian, frobenius_relative
from src.unsup.functions import JK_real

OUT = Path("out_01/mnist_single"); OUT.mkdir(parents=True, exist_ok=True)

# --- setup MNIST HFL: client1 {1,2,3}, client2 {4,5,6}, client3 {7,8,9} (zero escluso) ---
groups = [[1,2,3],[4,5,6],[7,8,9]]
classes = [c for g in groups for c in g]  # 1..9
class_to_arch, arch_to_class = build_class_mapping(classes)

# --- hyperparams (SINGLE) ---
hp = HyperParams(mode="single",
                 L=3, K=len(classes), N=28*28,     # K=9, N=784
                 n_batch=12,
                 M_total=3*12*200,                 # ~200 esempi/client/round
                 r_ex=1.0,                         # non usato qui: immagini reali
                 K_per_client=3,
                 w=0.8,
                 n_seeds=1, seed_base=2025,
                 use_tqdm=True)

# --- data ---
Xtr, ytr = load_mnist("./data", train=True)
ETA, labels = gen_dataset_from_mnist_single(
    X=Xtr, y=ytr,
    client_classes=groups,
    n_batch=hp.n_batch, L=hp.L, M_total=hp.M_total,
    class_to_arch=class_to_arch,
    rng=np.random.default_rng(hp.seed_base),
    binarize_threshold=0.5,
    use_tqdm=True,
)

# Prototipi come reference “ξ_true” (uno per classe in 'classes' nell'ordine 1..9)
Xtr_bin = binarize_images(Xtr, 0.5)
xi_true = class_prototypes_sign_mean(Xtr_bin, ytr, classes=classes).astype(int)   # (9, 784)
J_star  = JK_real(xi_true).astype(np.float32)

# --- loop per-round con single_round_step ---
xi_ref = None
series = []
J_server_last = None

for t in range(hp.n_batch):
    ETA_t    = ETA[:, t, :, :]      # (L, M_c, N)
    labels_t = labels[:, t, :]      # (L, M_c)
    xi_ref, JKS, log = single_round_step(
        ETA_t=ETA_t, labels_t=labels_t,
        xi_true=xi_true, J_star=J_star,
        xi_prev=xi_ref, hp=hp,
    )
    series.append(log)
    J_server_last = JKS

# --- grafici metriche ---
x = np.arange(hp.n_batch)
retr = np.array([s.retrieval for s in series]); fro = np.array([s.fro for s in series])
keff = np.array([s.keff for s in series]);     cov = np.array([s.coverage for s in series])

fig = plt.figure(figsize=(10,6))
plt.subplot(2,2,1); plt.plot(x, retr, marker="o"); plt.title("Retrieval (mean)")
plt.subplot(2,2,2); plt.plot(x, fro,  marker="o"); plt.title("Frobenius (relative)")
plt.subplot(2,2,3); plt.plot(x, keff, marker="o"); plt.title("K_eff")
plt.subplot(2,2,4); plt.plot(x, cov,  marker="o"); plt.title("Coverage")
plt.tight_layout(); plt.savefig(OUT/"fig_metrics.png", dpi=150); plt.close()

# --- Hopfield post-hoc: magnetizzazione vs esposizione (con i prototipi come ξ_true) ---
from src.unsup.data import count_exposures
from src.unsup.hopfield_eval import eval_retrieval_vs_exposure
expo = count_exposures(labels, K=hp.K)    # quante volte appare ogni classe 1..9
assert J_server_last is not None
out: Dict[str, Any] = eval_retrieval_vs_exposure(J_server_last, xi_true, exposure_counts=expo,
                                 beta=3.0, updates=30, reps_per_archetype=32, start_overlap=0.3)

means = np.array([out["mean_by_mu"][mu] for mu in range(hp.K)])
means = np.array([out["mean_by_mu"][mu] for mu in range(hp.K)])
expo9 = np.asarray(expo[:hp.K], float)

plt.figure(figsize=(5,4))
plt.scatter(expo9, means)
plt.xlabel("# esposizioni per classe")
plt.ylabel("magnetizzazione media (Hopfield)")
plt.title(f"MNIST HFL — pearson={out['pearson']:.3f}")
plt.tight_layout(); plt.savefig(OUT/"hopfield_exposure_scatter.png", dpi=150); plt.close()
