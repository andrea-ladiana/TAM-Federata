import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # estetica
from pathlib import Path
from typing import Any, Dict
import os, sys, json

# Assicura che la root del progetto (contenente 'src/') sia importabile anche eseguendo lo script da 'scripts/'.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.unsup.config import HyperParams  # type: ignore  # noqa: E402
from src.unsup.mnist_hfl import load_mnist, binarize_images, class_prototypes_sign_mean  # type: ignore  # noqa: E402
from src.unsup.mnist_hfl import build_class_mapping, make_mnist_hfl_subsets, gen_dataset_from_mnist_single  # type: ignore  # noqa: E402
from src.unsup.single_round import single_round_step, RoundLog  # type: ignore  # noqa: E402
from src.unsup.metrics import retrieval_mean_hungarian, frobenius_relative  # type: ignore  # noqa: E402
from src.unsup.functions import JK_real  # type: ignore  # noqa: E402

# =============================
# Config caching / rerun
# =============================
FORCE_RERUN = False          # True per rigenerare tutto (dati + Hopfield)
FORCE_ONLY_HOPFIELD = False  # True per rifare solo la valutazione Hopfield (non rigenera round federati)

# Se entrambi True, prevale FORCE_RERUN
if FORCE_RERUN and FORCE_ONLY_HOPFIELD:
    print("[WARN] FORCE_RERUN e FORCE_ONLY_HOPFIELD entrambi True → uso solo FORCE_RERUN.")
    FORCE_ONLY_HOPFIELD = False
HOPFIELD_PARAMS = dict(beta=5.0, updates=30, reps_per_archetype=32, start_overlap=1.0)

OUT = Path("out_01/mnist_single"); OUT.mkdir(parents=True, exist_ok=True)

# --- setup MNIST HFL con overlap tra client ---
# ATTENZIONE: per costruire i prototipi (archetipi) e J* serve l'insieme delle classi UNIVOCO.
# Se usassimo direttamente la lista concatenata con duplicati otterremmo archetipi ripetuti →
# matrice di correlazione singolare in JK_real (inversione fallisce).
groups = [[1,2,3,4,5], [1,2,4,5,6], [1,3,7,8,9]]  # ciascun client vede 5 cifre, con overlap
classes_all = [c for g in groups for c in g]
classes = sorted(set(classes_all))                 # => [1..9] senza duplicati
class_to_arch, arch_to_class = build_class_mapping(classes)

# --- hyperparams (SINGLE) ---
hp = HyperParams(mode="single",
                 L=3, K=len(classes), N=28*28,     # K=9, N=784 (len(classes) usa set unico)
                 n_batch=12,
                 M_total=3*12*200,                 # ~200 esempi/client/round
                 r_ex=1.0,                         # non usato qui: immagini reali
                 K_per_client=5,
                 w=0.9,
                 n_seeds=1, seed_base=2025,
                 use_tqdm=True)

## =============================
## Caching artefatti
## =============================
CACHE_DIR = OUT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RUN_CACHE = CACHE_DIR / "run_metrics.npz"          # metriche round + hopfield
META_JSON = CACHE_DIR / "meta.json"                # parametri usati per invalidare cache

def _load_cache() -> Dict[str, Any] | None:
    if not RUN_CACHE.exists():
        return None
    try:
        data = np.load(RUN_CACHE, allow_pickle=True)
        out = {k: data[k] for k in data.files}
        return out
    except Exception as e:
        print(f"[WARN] Lettura cache fallita ({e}).")
        return None

def _meta_changed() -> bool:
    if not META_JSON.exists():
        return True
    try:
        meta = json.loads(META_JSON.read_text())
        # Confronta subset minimo di parametri che impattano i dati
        keys = ["n_batch", "M_total", "seed_base", "K", "L", "N", "hopfield"]
        ref = dict(n_batch=hp.n_batch, M_total=hp.M_total, seed_base=hp.seed_base,
                   K=hp.K, L=hp.L, N=hp.N, hopfield=HOPFIELD_PARAMS)
        return any(meta.get(k) != ref.get(k) for k in keys)
    except Exception:
        return True

cache_data = None
if not FORCE_RERUN and not FORCE_ONLY_HOPFIELD:
    if not _meta_changed():
        cache_data = _load_cache()
    else:
        print("[INFO] Parametri cambiati → invalido cache round.")
elif FORCE_ONLY_HOPFIELD:
    cache_data = _load_cache()
    if cache_data is None:
        print("[WARN] Cache assente: non posso fare solo Hopfield → forzo recompute completo.")
        FORCE_ONLY_HOPFIELD = False
        FORCE_RERUN = True

if FORCE_RERUN:
    recompute_rounds = True
else:
    recompute_rounds = (cache_data is None) and (not FORCE_ONLY_HOPFIELD)

recompute_hopfield = FORCE_RERUN or FORCE_ONLY_HOPFIELD

if recompute_rounds:
    print("[RUN] Eseguo pipeline federata MNIST (rounds)...")
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
    Xtr_bin = binarize_images(Xtr, 0.5)
    xi_true = class_prototypes_sign_mean(Xtr_bin, ytr, classes=classes).astype(int)
    J_star  = JK_real(xi_true).astype(np.float32)
    xi_ref = None
    series: list[RoundLog] = []
    J_server_last = None
    for t in range(hp.n_batch):
        ETA_t    = ETA[:, t, :, :]
        labels_t = labels[:, t, :]
        xi_ref, JKS, log = single_round_step(
            ETA_t=ETA_t, labels_t=labels_t,
            xi_true=xi_true, J_star=J_star,
            xi_prev=xi_ref, hp=hp,
        )
        series.append(log)
        J_server_last = JKS
    # Metriche per-round
    retr = np.array([s.retrieval for s in series])
    fro = np.array([s.fro for s in series])
    keff = np.array([s.keff for s in series])
    cov = np.array([s.coverage for s in series])
elif cache_data is not None:
    # Carica da cache
    retr = cache_data.get('retr')
    fro = cache_data.get('fro')
    keff = cache_data.get('keff')
    cov = cache_data.get('cov')
    xi_true = cache_data.get('xi_true')
    J_server_last = cache_data.get('J_server')
    xi_ref = cache_data.get('xi_final') if 'xi_final' in cache_data else None
    labels = cache_data.get('labels') if 'labels' in cache_data else None
    exposure = cache_data.get('exposure') if 'exposure' in cache_data else None
    means = cache_data.get('means') if 'means' in cache_data else None
    pearson_cached = cache_data.get('pearson') if 'pearson' in cache_data else None
    print("[CACHE] Round metrics caricati.")
else:
    print("[WARN] Nessuna cache disponibile e nessun recompute previsto: forzo recompute rounds.")
    recompute_rounds = True
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
    Xtr_bin = binarize_images(Xtr, 0.5)
    xi_true = class_prototypes_sign_mean(Xtr_bin, ytr, classes=classes).astype(int)
    J_star  = JK_real(xi_true).astype(np.float32)
    xi_ref = None
    series: list[RoundLog] = []
    J_server_last = None
    for t in range(hp.n_batch):
        ETA_t    = ETA[:, t, :, :]
        labels_t = labels[:, t, :]
        xi_ref, JKS, log = single_round_step(
            ETA_t=ETA_t, labels_t=labels_t,
            xi_true=xi_true, J_star=J_star,
            xi_prev=xi_ref, hp=hp,
        )
        series.append(log)
        J_server_last = JKS
    retr = np.array([s.retrieval for s in series])
    fro = np.array([s.fro for s in series])
    keff = np.array([s.keff for s in series])
    cov = np.array([s.coverage for s in series])

# Hopfield eval (può essere ricalcolata separatamente)
from src.unsup.data import count_exposures  # noqa: E402
from src.unsup.hopfield_eval import eval_retrieval_vs_exposure  # noqa: E402

if recompute_rounds or recompute_hopfield:
    if J_server_last is None or xi_true is None:
        raise SystemExit("J_server_last/xi_true mancanti: impossibile valutare Hopfield.")
    if recompute_rounds:
        if 'labels' not in locals() or labels is None:
            raise SystemExit("Labels mancanti dopo recompute_rounds.")
        exposure = count_exposures(np.asarray(labels), K=hp.K)
    elif 'exposure' not in locals() or exposure is None:
        # Se vogliamo rifare solo Hopfield ma manca exposure abortiamo
        raise SystemExit("Exposure assente: rieseguire con FORCE_RERUN=True.")
    # Parametri espliciti per evitare mismatch **kwargs (typing)
    out_h: Dict[str, Any] = eval_retrieval_vs_exposure(
        J_server_last,
        xi_true,
        exposure_counts=np.asarray(exposure),
        beta=int(HOPFIELD_PARAMS['beta']),
        updates=int(HOPFIELD_PARAMS['updates']),
        reps_per_archetype=int(HOPFIELD_PARAMS['reps_per_archetype']),
        start_overlap=float(HOPFIELD_PARAMS['start_overlap'])
    )
    mean_by_mu = out_h.get("mean_by_mu", {})
    if isinstance(mean_by_mu, dict):
        means = np.array([mean_by_mu.get(mu, np.nan) for mu in range(hp.K)])
    else:
        means = np.zeros(hp.K)
    # Matrice magnetizzazioni per violin (K, reps)
    mag_dict = out_h.get("magnetization_by_mu", {})
    if isinstance(mag_dict, dict) and len(mag_dict):
        try:
            # ordina per indice μ
            mag_mat = np.vstack([np.asarray(mag_dict[mu], dtype=float) for mu in range(len(mag_dict))])
        except Exception:
            mag_mat = None
    else:
        mag_mat = None
    pearson_raw = out_h.get('pearson', np.nan)
    try:
        pearson_val = float(pearson_raw)
    except Exception:
        pearson_val = float('nan')
else:
    pearson_val = float(pearson_cached) if 'pearson_cached' in locals() and pearson_cached is not None else np.nan
    mag_mat = cache_data.get('magnetizations') if cache_data and 'magnetizations' in cache_data else None

# Salva cache (se abbiamo appena ricomputato rounds o hopfield)
if recompute_rounds or recompute_hopfield:
    try:
        # Assicura ndarray
        def _ensure(a, fallback_shape=()):
            if a is None:
                return np.zeros(fallback_shape)
            return np.asarray(a)
        np.savez_compressed(
            RUN_CACHE,
            retr=_ensure(retr), fro=_ensure(fro), keff=_ensure(keff), cov=_ensure(cov),
            xi_true=_ensure(xi_true, (0,)), J_server=_ensure(J_server_last, (0, 0)),
            xi_final=_ensure(xi_ref, (0,)),
            exposure=_ensure(exposure, (0,)), means=_ensure(means, (0,)), pearson=np.asarray([pearson_val]),
            magnetizations=_ensure(mag_mat, (0, 0)),
            labels=_ensure(labels, (0,)), hopfield_beta=np.asarray([HOPFIELD_PARAMS['beta']])
        )
        META_JSON.write_text(json.dumps(dict(n_batch=hp.n_batch, M_total=hp.M_total, seed_base=hp.seed_base,
                                             K=hp.K, L=hp.L, N=hp.N, hopfield=HOPFIELD_PARAMS), indent=2))
        print(f"[CACHE] Aggiornata cache → {RUN_CACHE}")
    except Exception as e:
        print(f"[WARN] Salvataggio cache fallito: {e}")

expo = exposure if 'exposure' in locals() and exposure is not None else (cache_data.get('exposure') if cache_data else np.array([]))
expo9 = np.asarray(expo[:hp.K], float) if isinstance(expo, np.ndarray) and expo.size else np.asarray([])
means = means if 'means' in locals() and means is not None else (cache_data.get('means') if cache_data else np.array([]))
pearson_for_plot = pearson_val if 'pearson_val' in locals() else (pearson_cached if 'pearson_cached' in locals() else np.nan)

## =============================
## Plotting migliorato (seaborn)
## =============================

# Parametri estetica (facili da modificare se serve):
STYLE = "whitegrid"
PALETTE = "viridis"
MARKER = "o"
DPI = 150
SAVE_PANEL = OUT / "panel_metrics.png"
SAVE_LONGFORM = None  # plot combinato rimosso

sns.set_style(STYLE)

# Costruzione dei dati per plotting (già presenti se da cache)
retr = np.asarray(retr) if retr is not None else np.zeros(0)
fro = np.asarray(fro) if fro is not None else np.zeros(0)
keff = np.asarray(keff) if keff is not None else np.zeros(0)
cov = np.asarray(cov) if cov is not None else np.zeros(0)
x = np.arange(retr.shape[0])

# 1) Pannello 2x2 con seaborn.lineplot
fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
axes = axes.ravel()

def _line(ax, y, title, color):
    sns.lineplot(x=x, y=y, ax=ax, marker=MARKER, color=color)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Round")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Griglia leggera solo sull'asse y
    ax.grid(axis='y', alpha=0.35)

palette_colors = sns.color_palette(PALETTE, 4)
_line(axes[0], retr, "Retrieval (mean)", palette_colors[0])
_line(axes[1], fro,  "Frobenius (rel)", palette_colors[1])
_line(axes[2], keff, "K_eff", palette_colors[2])
_line(axes[3], cov,  "Coverage", palette_colors[3])
for ax in axes:
    ax.set_xlim(x[0], x[-1])
fig.suptitle("MNIST Federated – Metriche per Round", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(SAVE_PANEL, dpi=DPI)
plt.close(fig)

# (Plot long-form rimosso)

if expo9.size and isinstance(means, np.ndarray) and means.size:
    # --- Singolo scatter (retrocompatibilità) ---
    plt.figure(figsize=(5.8, 4.2))
    sns.regplot(x=expo9, y=means, scatter_kws=dict(s=55, edgecolor="black", linewidths=0.6, alpha=0.85),
                line_kws=dict(color="crimson", alpha=0.7), color=sns.color_palette("mako", 6)[2])
    for mu in range(len(expo9)):
        plt.text(expo9[mu]+0.5, means[mu], str(mu+1), fontsize=8, ha='left', va='center')
    plt.xlabel("# esposizioni per classe", fontsize=10)
    plt.ylabel("Magnetizzazione media (Hopfield)", fontsize=10)
    plt.title(f"Hopfield – Exposure vs Magnetizzazione (pearson={pearson_for_plot:.3f})", fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT/"hopfield_exposure_scatter.png", dpi=DPI); plt.close()

    # --- Pannello violin + scatter con legenda ---
    PANEL_PATH = OUT / "hopfield_panel.png"
    try:
        import pandas as pd
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
        # Violin plot
        if mag_mat is not None and isinstance(mag_mat, np.ndarray) and mag_mat.size:
            cols = [f"{i+1}" for i in range(mag_mat.shape[0])]
            df_mag = pd.DataFrame(mag_mat.T, columns=cols)
            dfm = df_mag.melt(var_name="classe", value_name="mag")
            sns.violinplot(data=dfm, x="classe", y="mag", palette="viridis", inner="quartile", cut=0, ax=axL)
            axL.set_xlabel("Classe")
            axL.set_ylabel("Magnetizzazione")
            axL.set_title("Distribuzione magnetizzazioni")
        else:
            axL.text(0.5, 0.5, "(mag assenti)", ha='center', va='center')
            axL.set_axis_off()

        # Scatter + regressione con legenda
        palette_scatter = sns.color_palette("mako", len(means))
        for mu in range(len(means)):
            axR.scatter(expo9[mu], means[mu], s=65, color=palette_scatter[mu], edgecolor='black', linewidths=0.6, label=f"{mu+1}")
        # regression line (global)
        sns.regplot(x=expo9, y=means, scatter=False, ax=axR, line_kws=dict(color='crimson', alpha=0.7, linewidth=1.5))
        axR.set_xlabel("# esposizioni per classe")
        axR.set_ylabel("Magnetizzazione media")
        axR.set_title(f"Exposure vs Magnetizzazione (r={pearson_for_plot:.2f})")
        axR.grid(axis='y', alpha=0.3)
        # legenda fuori
        axR.legend(title="Classe", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, ncol=1, fontsize=8)
        fig.tight_layout()
        fig.savefig(PANEL_PATH, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Impossibile generare pannello Hopfield: {e}")
else:
    print("[INFO] Scatter Hopfield non generato (dati assenti).")

extra = []
if (OUT/"hopfield_panel.png").exists():
    extra.append("hopfield_panel.png")
print(f"[PLOT] Salvati: {SAVE_PANEL.name}, hopfield_exposure_scatter.png" + (", " + ', '.join(extra) if extra else "") + f" in {OUT}")

## =============================
## REPORTING COMPLETO
## =============================
REPORT_DIR = OUT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def _write_text(path: Path, txt: str):
    try:
        path.write_text(txt, encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Impossibile scrivere {path.name}: {e}")

def _safe_json(path: Path, obj: Dict[str, Any]):
    try:
        import json as _json
        path.write_text(_json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Impossibile salvare JSON {path.name}: {e}")

# 1. Report per-round (se disponibili)
if retr.size:
    import csv
    rounds_path_csv = REPORT_DIR / "metrics_rounds.csv"
    rounds_path_json = REPORT_DIR / "metrics_rounds.json"
    try:
        with rounds_path_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["round","retrieval","frobenius_rel","K_eff","coverage"])
            for i in range(retr.shape[0]):
                w.writerow([i, float(retr[i]), float(fro[i]) if i < fro.size else np.nan,
                            float(keff[i]) if i < keff.size else np.nan,
                            float(cov[i]) if i < cov.size else np.nan])
    except Exception as e:
        print(f"[WARN] CSV per-round fallito: {e}")
    # JSON structure
    _safe_json(rounds_path_json, dict(
        metrics=[dict(round=int(i), retrieval=float(retr[i]), frobenius_rel=float(fro[i]) if i < fro.size else None,
                       K_eff=float(keff[i]) if i < keff.size else None,
                       coverage=float(cov[i]) if i < cov.size else None)
                 for i in range(retr.shape[0])]
    ))

# 2. Summary metriche rounds
summary_rounds = {
    "final": {
        "retrieval": float(retr[-1]) if retr.size else None,
        "frobenius_rel": float(fro[-1]) if fro.size else None,
        "K_eff": float(keff[-1]) if keff.size else None,
        "coverage": float(cov[-1]) if cov.size else None,
    },
    "mean_over_rounds": {
        "retrieval": float(retr.mean()) if retr.size else None,
        "frobenius_rel": float(fro.mean()) if fro.size else None,
        "K_eff": float(keff.mean()) if keff.size else None,
        "coverage": float(cov.mean()) if cov.size else None,
    },
    "best": {
        "retrieval_max": float(retr.max()) if retr.size else None,
        "frobenius_rel_min": float(fro.min()) if fro.size else None,
        "K_eff_max": float(keff.max()) if keff.size else None,
        "coverage_max": float(cov.max()) if cov.size else None,
    },
    "n_rounds": int(retr.shape[0]) if retr.size else 0,
}
_safe_json(REPORT_DIR / "metrics_summary.json", summary_rounds)

# 3. Hopfield summary & distributions
def _as_clean_float(v: Any):
    try:
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return None
            v = v.reshape(-1)[0]
        vf = float(v)
        if np.isnan(vf):
            return None
        return vf
    except Exception:
        return None

hopfield_summary = {
    "pearson": _as_clean_float(pearson_for_plot),
    "exposure_per_class": expo9.tolist() if expo9.size else [],
    "magnetization_mean_per_class": means.tolist() if isinstance(means, np.ndarray) and means.size else [],
}

# Add distribution stats if available
if 'mag_mat' in locals() and isinstance(mag_mat, np.ndarray) and mag_mat.size:
    per_class_stats = []
    for mu in range(mag_mat.shape[0]):
        vals = mag_mat[mu]
        if vals.size == 0:
            continue
        per_class_stats.append(dict(
            class_index=mu+1,
            reps=int(vals.size),
            mean=float(np.mean(vals)),
            std=float(np.std(vals, ddof=1)) if vals.size>1 else 0.0,
            p25=float(np.percentile(vals,25)),
            p50=float(np.percentile(vals,50)),
            p75=float(np.percentile(vals,75)),
            min=float(vals.min()),
            max=float(vals.max()),
        ))
    hopfield_summary["distribution_stats"] = per_class_stats

_safe_json(REPORT_DIR / "hopfield_summary.json", hopfield_summary)

# 4. Hopfield exposure vs magnetization CSV
if expo9.size and isinstance(means, np.ndarray) and means.size:
    import csv as _csv
    evm_csv = REPORT_DIR / "hopfield_exposure_vs_magnetization.csv"
    try:
        with evm_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["class","exposure","magnetization_mean"])
            for mu in range(len(expo9)):
                w.writerow([mu+1, float(expo9[mu]), float(means[mu])])
    except Exception as e:
        print(f"[WARN] CSV exposure_vs_magnetization fallito: {e}")

# 5. Magnetizations long-form CSV
if 'mag_mat' in locals() and isinstance(mag_mat, np.ndarray) and mag_mat.size:
    import csv as _csv
    mag_csv = REPORT_DIR / "hopfield_magnetizations_long.csv"
    try:
        with mag_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["class","rep","magnetization"])
            for mu in range(mag_mat.shape[0]):
                for r_idx, val in enumerate(mag_mat[mu]):
                    w.writerow([mu+1, r_idx, float(val)])
    except Exception as e:
        print(f"[WARN] CSV magnetizations fallito: {e}")

# 6. Config & digests
def _digest_arr(a: Any) -> str:
    try:
        import hashlib
        arr = np.asarray(a)
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    except Exception:
        return "NA"

config_report = {
    "hyperparams": {
        "L": hp.L, "K": hp.K, "N": hp.N, "n_batch": hp.n_batch, "M_total": hp.M_total,
        "K_per_client": hp.K_per_client, "w": hp.w, "seed_base": hp.seed_base
    },
    "hopfield_params": HOPFIELD_PARAMS,
    "digests": {
        "J_server_sha": _digest_arr(J_server_last) if 'J_server_last' in locals() else None,
        "xi_true_sha": _digest_arr(xi_true) if 'xi_true' in locals() else None,
        "xi_final_sha": _digest_arr(xi_ref) if 'xi_ref' in locals() else None,
    }
}
_safe_json(REPORT_DIR / "config_and_digests.json", config_report)

# 7. README markdown sintetico
readme_md = f"""# Report MNIST Federated

Questa cartella contiene i report generati automaticamente.

## File
* metrics_rounds.csv / .json: metriche per round (retrieval, frobenius_rel, K_eff, coverage)
* metrics_summary.json: valori finali, medie e best.
* hopfield_summary.json: correlazioni e statistiche distribuzioni magnetizzazione.
* hopfield_exposure_vs_magnetization.csv: relazione esposizione vs magnetizzazione media.
* hopfield_magnetizations_long.csv: valori grezzi delle magnetizzazioni (long-form).
* config_and_digests.json: iperparametri e digest SHA delle matrici principali.

## Note
- I digest consentono di verificare se le matrici cambiano tra run.
- Per rigenerare completamente i dati: impostare FORCE_RERUN=True nello script.

Generato da `exp01_mnist.py`.
"""
_write_text(REPORT_DIR / "README.md", readme_md)

print(f"[REPORT] Report completi salvati in {REPORT_DIR}")
print(f"[INFO] Usa FORCE_RERUN=True per rigenerare completamente oppure modifica solo stile e rilancia per aggiornare figure.")

## =============================
## Hopfield: evoluzione di un esempio molto corrotto
## =============================
"""
Come prova del nove: prendiamo un'immagine reale di una delle classi usate (1..9),
la binarizziamo e la corrompiamo pesantemente (overlap iniziale basso), quindi
applichiamo la dinamica di Hopfield (con J del server finale) per un certo numero
di step e mostriamo visivamente come evolve.

Output: out_01/mnist_single/hopfield_evolution_example.png
"""
try:
    from src.unsup.networks import Hopfield_Network  # type: ignore
    from src.unsup.mnist_hfl import binarize_images  # type: ignore
    import numpy as _np
    import matplotlib.pyplot as _plt
    from matplotlib import gridspec as _gridspec

    if 'J_server_last' in locals() and J_server_last is not None and 'xi_true' in locals() and xi_true is not None:
        # Scegli una classe presente (prendiamo la prima del gruppo per riproducibilità)
        target_class = classes[0] if 'classes' in locals() and len(classes) else 1
        # Se non abbiamo le immagini in RAM, ricarica il train set per selezionare un esempio
        if 'Xtr' not in locals() or 'ytr' not in locals():
            from src.unsup.mnist_hfl import load_mnist  # lazy import
            Xtr, ytr = load_mnist("./data", train=True)
        # Filtra immagini della classe target
        idxs = _np.where(_np.asarray(ytr) == int(target_class))[0]
        if idxs.size:
            _rng = _np.random.default_rng(hp.seed_base + 7)
            i_sel = int(_rng.integers(0, idxs.size))
            x0 = _np.asarray(binarize_images(_np.asarray(Xtr[idxs[i_sel]][None, ...]), 0.5)[0], dtype=int)  # (N,)
            # Corrompi pesantemente per avere overlap iniziale molto basso
            start_overlap = 1.0  # «ancora più corrotto»
            p_flip = (1.0 - float(start_overlap)) * 0.5
            flips = _rng.random(size=x0.shape) < p_flip
            s0 = _np.where(flips, -x0, x0).astype(_np.int8)

            # Dinamica Hopfield step-by-step per tracciare l'evoluzione
            net = Hopfield_Network()
            net.N = int(J_server_last.shape[0])
            net.J = _np.asarray(J_server_last, dtype=_np.float32)
            beta = float(HOPFIELD_PARAMS.get('beta', 100.0))
            T_total = int(HOPFIELD_PARAMS.get('updates', 50))
            # istanti da mostrare
            snap_ts = [0, 1, 2, 5, 10, 20, T_total]
            snaps = []
            mags = []
            # trova indice archetipo corrispondente alla classe target
            mu = class_to_arch[int(target_class)] if 'class_to_arch' in locals() else 0
            xi_mu = _np.asarray(xi_true[mu], dtype=int)
            N = xi_mu.size
            s = s0.copy()[None, :]  # (1, N)
            for t in range(T_total + 1):
                if t in snap_ts:
                    snaps.append(s[0].copy())
                    m = float(_np.abs(_np.dot(s[0], xi_mu)) / N)
                    mags.append(m)
                if t == T_total:
                    break
                # un singolo aggiornamento (parallel Glauber come in Hopfield_Network)
                net.dynamics(s.astype(_np.float32), beta, updates=1, mode="parallel", stochastic=True, rng=_rng)
                # Recupera lo stato aggiornato dal network (attributo `σ`). In caso di ambienti che non
                # supportano il carattere unicode, prova anche un alias ascii 'sigma'.
                try:
                    s_next = getattr(net, 'σ')
                except AttributeError:
                    s_next = getattr(net, 'sigma', None)
                if s_next is None:
                    raise RuntimeError("Hopfield_Network: stato finale assente dopo dynamics (attributo 'σ').")
                s = _np.asarray(s_next, dtype=int)

            # Plot: griglia con snapshot + curva magnetizzazione
            ncols = len(snap_ts)
            _plt.figure(figsize=(1.8 * ncols + 3.5, 3.8))
            gridspec = _gridspec.GridSpec(2, ncols, height_ratios=[1, 0.45])
            for j, (t, sj, mj) in enumerate(zip(snap_ts, snaps, mags)):
                ax = _plt.subplot(gridspec[0, j])
                img = sj.reshape(28, 28)
                ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
                ax.set_title(f"t={t}\n|m|={mj:.2f}", fontsize=9)
                ax.axis('off')
            # curva magnetizzazione
            ax_curve = _plt.subplot(gridspec[1, :])
            ax_curve.plot(snap_ts, mags, marker='o', color='tab:blue')
            ax_curve.set_xlabel("Aggiornamenti Hopfield (t)")
            ax_curve.set_ylabel("|m(t)| vs archetipo")
            ax_curve.grid(axis='y', alpha=0.3)
            _plt.tight_layout()
            _plt.savefig(OUT / "hopfield_evolution_example.png", dpi=DPI)
            _plt.close()
            print("[PLOT] Salvato hopfield_evolution_example.png")
        else:
            print("[INFO] Nessuna immagine trovata per la classe target: salto grafico evoluzione Hopfield.")
    else:
        print("[INFO] J_server_last/xi_true assenti: salto grafico evoluzione Hopfield.")
except Exception as e:
    print(f"[WARN] Impossibile generare grafico evoluzione Hopfield: {e}")
