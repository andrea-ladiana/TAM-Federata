"""Pannello di valutazione Hopfield (exp01) SENZA CLI.

Stile semplificato come `exp01_synth.py`: modifica le costanti qui sotto e lancia:
        python scripts/exp01_hopfield_panel.py

Pipeline:
 1. Esegue (se necessario) l'esperimento federato exp01 per ottenere J_server, archetipi finali e conteggi esposizioni.
 2. Esegue/ricarica la valutazione Hopfield post-hoc (caching in sotto-cartella `hopfield_eval/`).
 3. Produce figure seaborn (distribuzione magnetizzazioni, mean vs exposure, pannello combinato) in `figs/`.
 4. Salva un riassunto JSON (`summary_hopfield.json`).

Logica caching:
    - Se esiste `OUTPUT_DIR/hopfield_eval` e `FORCE_RUN = False` allora ricarica direttamente la valutazione Hopfield
        (senza rifare il run federato) e riproduce le figure.
    - Se `FORCE_RUN = True` oppure la cartella non esiste: riesegue federato + valutazione Hopfield.

Per cambiare parametri basta modificare le costanti e rilanciare. Nessun argomento da riga di comando.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional
import shutil

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.unsup.hopfield_eval import (  # noqa: E402
    run_or_load_hopfield_eval,
    load_hopfield_eval,
    plot_magnetization_distribution,
    plot_mean_vs_exposure,
)
from src.unsup.config import HyperParams  # noqa: E402
from src.unsup.runner_single import run_exp01_single  # noqa: E402

# =============================
# Config federato (exp01)
# =============================
HP = HyperParams(
    mode="single",
    L=3, K=9, N=300,
    n_batch=12,
    M_total=900,
    r_ex=0.80,
    K_per_client=5,
    w=0.8,
    n_seeds=1, seed_base=2025,
    use_tqdm=True,
)
SEED_IDX = 0  # indice seed per estrarre J / archetipi finali

# =============================
# Config Hopfield post-hoc
# =============================
HOPFIELD_BETA = 4
HOPFIELD_UPDATES = 100
HOPFIELD_REPS = 100
HOPFIELD_START_OVERLAP = 0.7
HOPFIELD_STOCHASTIC = True   # True: Glauber probabilistico; False: deterministico
USE_XI_TRUE = True      # se disponibili archetipi veri
# Controlli separati:
# - FORCE_FEDERATED: forza riesecuzione dell'apprendimento federato (run_exp01_single)
# - FORCE_HOPFIELD: forza riesecuzione della valutazione Hopfield anche se esiste una cartella di valutazione
FORCE_FEDERATED = True
FORCE_HOPFIELD = True   # forza SOLO la parte Hopfield (ignora cache hopfield_eval)
# backward compat: se FORCE_RUN era usato, manteniamo comportamento (entrambi True)
FORCE_RUN = False        # Deprecated: usa FORCE_FEDERATED / FORCE_HOPFIELD

# =============================
# Output & plotting
# =============================
OUTPUT_DIR = "out_01/hop_eval/exp01_panel"
STYLE = "whitegrid"
PALETTE_DIST = "viridis"
PALETTE_SCATTER = "mako"
DPI = 150
SAVE_FORMATS = "png"   # separa con virgole per più formati es: "png,pdf"
NO_SHOW = False         # True per non aprire le figure (es: run headless)

def _maybe_get_true_xi(res: Dict[str, Any], seed_idx: int) -> Optional[np.ndarray]:
    # Prova alcuni nomi plausibili
    keys = [
        "xi_true_list",  # lista di archetipi veri
        "true_xi_list",
        "xi_true"  # fallback singolo
    ]
    for k in keys:
        if k in res:
            val = res[k]
            if isinstance(val, list):
                if seed_idx < len(val):
                    return val[seed_idx]
            elif isinstance(val, np.ndarray):
                return val
    return None


def run_federated_and_extract(hp: HyperParams, seed_idx: int) -> Dict[str, Any]:
    res: Dict[str, Any] = run_exp01_single(hp, out_dir=None, do_plot=False)
    J_server = res["final_J_list"][seed_idx]
    xi_final = res["final_xi_list"][seed_idx]
    exposure = res["exposure_list"][seed_idx]
    xi_true = _maybe_get_true_xi(res, seed_idx)
    return dict(J_server=J_server, xi_final=xi_final, exposure=exposure, xi_true=xi_true)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def run():
    sns.set_style(STYLE)

    out_dir = ensure_dir(OUTPUT_DIR)
    hop_eval_dir = out_dir / "hopfield_eval"

    # Parametri Hopfield correnti (usati anche per invalidare cache se cambiano)
    current_hopfield_params = dict(
        beta=HOPFIELD_BETA,
        updates=HOPFIELD_UPDATES,
        reps_per_archetype=HOPFIELD_REPS,
        start_overlap=HOPFIELD_START_OVERLAP,
        stochastic=HOPFIELD_STOCHASTIC,
    )

    # Determina se la cache hopfield andrebbe forzata
    force_hopfield_effective = FORCE_HOPFIELD or FORCE_RUN
    # If user explicitly requested to force Hopfield-only re-run, remove existing cache dir
    if FORCE_HOPFIELD:
        try:
            if hop_eval_dir.exists():
                shutil.rmtree(hop_eval_dir)
                print(f"[INFO] FORCE_HOPFIELD=True → rimossa cache hopfield_eval: {hop_eval_dir}")
        except Exception as e:
            print(f"[WARN] Impossibile rimuovere {hop_eval_dir}: {e}")
    if not force_hopfield_effective and hop_eval_dir.exists():
        # Prova a leggere meta.json e confrontare i parametri salvati
        meta_path = hop_eval_dir / "meta.json"
        try:
            if meta_path.exists():
                import json as _json
                meta = _json.loads(meta_path.read_text())
                saved_params = (meta.get("params") or {})
                # Confronta solo le chiavi di interesse
                diffs = []
                for k, v in current_hopfield_params.items():
                    if k not in saved_params or saved_params.get(k) != v:
                        diffs.append(k)
                if diffs:
                    print(f"[INFO] Parametri Hopfield cambiati ({', '.join(diffs)}). Rigenero valutazione.")
                    force_hopfield_effective = True
        except Exception as e:
            print(f"[WARN] Impossibile leggere/parsare meta.json per invalidare cache: {e}")

    # load_only = True => ricarica pura senza eseguire dinamica
    load_only = hop_eval_dir.exists() and not force_hopfield_effective

    # 1) Ottieni J_server / xi_ref / exposure / xi_true
    J_server = xi_ref = exposure = xi_true = None

    # Preferisci caricare artefatti direttamente da hop_eval_dir se presenti
    if hop_eval_dir.exists() and load_only:
        try:
            hj = hop_eval_dir / "J_server.npy"
            hx = hop_eval_dir / "xi_true.npy"
            he = hop_eval_dir / "exposure_counts.npy"
            loaded = False
            if hj.exists():
                try:
                    J_server = np.load(hj)
                    print(f"[INFO] Caricato J_server da hop_eval: {hj}")
                    loaded = True
                except Exception:
                    J_server = None
            if hx.exists():
                try:
                    xi_true = np.load(hx)
                    print(f"[INFO] Caricato xi_true da hop_eval: {hx}")
                    loaded = True
                except Exception:
                    xi_true = None
            if he.exists():
                try:
                    exposure = np.load(he)
                    print(f"[INFO] Caricato exposure da hop_eval: {he}")
                    loaded = True
                except Exception:
                    exposure = None
            if loaded:
                print(f"[INFO] Utilizzo artefatti presenti in {hop_eval_dir} per evitare run federato.")
        except Exception:
            pass

    # If not forcing a fresh federated run, try to load saved artifacts from OUTPUT_DIR
    def _try_load_saved_run(out_dir: Path, seed_idx: int):
        # Look for saved matrices from previous run_exp01_single
        # Candidates: J_server_seed_{seed}.npy, final_J_list.npz, xi_ref_seed_{seed}.npy, final_xi_list.npz, exposure_list.npy
        J = None; xi_r = None; expo = None; xi_t = None
        # J by seed
        p_j_seed = out_dir / f"J_server_seed_{seed_idx}.npy"
        if p_j_seed.exists():
            try:
                J = np.load(p_j_seed)
                print(f"[INFO] Caricato J_server da: {p_j_seed}")
            except Exception:
                J = None
        # fallback to final_J_list.npz
        if J is None and (out_dir / "final_J_list.npz").exists():
            try:
                arr = np.load(str(out_dir / "final_J_list.npz"))
                # file saved as final_J (n_seeds, N, N) earlier
                if 'final_J' in arr.files:
                    allJ = arr['final_J']
                    if seed_idx < allJ.shape[0]:
                        J = allJ[seed_idx]
                        print(f"[INFO] Caricato J_server dal file aggregato: {out_dir / 'final_J_list.npz'} (indice {seed_idx})")
            except Exception:
                J = None
        # xi_ref by seed
        p_xi_seed = out_dir / f"xi_ref_seed_{seed_idx}.npy"
        if p_xi_seed.exists():
            try:
                xi_r = np.load(p_xi_seed)
                print(f"[INFO] Caricato xi_ref da: {p_xi_seed}")
            except Exception:
                xi_r = None
        # fallback final_xi_list.npz
        if xi_r is None and (out_dir / "final_xi_list.npz").exists():
            try:
                arr = np.load(str(out_dir / "final_xi_list.npz"))
                # this archive stored arrays without a named key earlier; take index
                if isinstance(arr, np.lib.npyio.NpzFile):
                    keys = arr.files
                    # try to pick seed_idx-th entry if ordering preserved
                    if seed_idx < len(keys):
                        xi_r = arr[keys[seed_idx]]
                        print(f"[INFO] Caricato xi_ref dal file aggregato: {out_dir / 'final_xi_list.npz'} (chiave {keys[seed_idx]})")
            except Exception:
                xi_r = None
        # xi_true by seed (nuovo: salviamo xi_true separatamente se disponibile)
        p_xi_true_seed = out_dir / f"xi_true_seed_{seed_idx}.npy"
        if p_xi_true_seed.exists():
            try:
                xi_t = np.load(p_xi_true_seed)
                print(f"[INFO] Caricato xi_true da: {p_xi_true_seed}")
            except Exception:
                xi_t = None
        # exposure list
        if (out_dir / "exposure_list.npy").exists():
            try:
                exp_arr = np.load(out_dir / "exposure_list.npy")
                if seed_idx < exp_arr.shape[0]:
                    expo = exp_arr[seed_idx]
                    print(f"[INFO] Caricato exposure da: {out_dir / 'exposure_list.npy'} (indice {seed_idx})")
            except Exception:
                expo = None
        # xi_true list (if saved by run_exp01_single)
        if (out_dir / "final_xi_list.npz").exists():
            try:
                arr = np.load(str(out_dir / "final_xi_list.npz"))
                # try to find xi_true saved previously as xi_true_list via our runner
                # but more likely xi_true not saved; fallback to xi_ref
            except Exception:
                pass
        # If nothing found in out_dir, try a lightweight project-wide search for likely files
        if J is None and xi_r is None and expo is None:
            try:
                print(f"[DEBUG] Provo una ricerca rapida nel progetto per file 'final_J_list.npz' o 'J_server_seed_{seed_idx}.npy'...")
                proj_root = Path(_PROJECT_ROOT)
                # search for exact filenames first
                candidates = list(proj_root.rglob(f"J_server_seed_{seed_idx}.npy"))
                if not candidates:
                    candidates = list(proj_root.rglob("final_J_list.npz"))
                if not candidates:
                    candidates = list(proj_root.rglob("exposure_list.npy"))
                if candidates:
                    found = candidates[0]
                    print(f"[INFO] Trovato file utile in progetto: {found} → proverò a caricarlo")
                    # attempt to load according to name
                    name = found.name
                    if name.startswith("J_server_seed_") and name.endswith('.npy'):
                        try:
                            J = np.load(found)
                            print(f"[INFO] Caricato J_server da: {found}")
                        except Exception:
                            J = None
                    elif name == 'final_J_list.npz':
                        try:
                            arr = np.load(found)
                            if 'final_J' in arr.files:
                                allJ = arr['final_J']
                                if seed_idx < allJ.shape[0]:
                                    J = allJ[seed_idx]
                                    print(f"[INFO] Caricato J_server dal file aggregato: {found} (indice {seed_idx})")
                        except Exception:
                            J = None
                    elif name == 'exposure_list.npy':
                        try:
                            exp_arr = np.load(found)
                            if seed_idx < exp_arr.shape[0]:
                                expo = exp_arr[seed_idx]
                                print(f"[INFO] Caricato exposure da: {found} (indice {seed_idx})")
                        except Exception:
                            expo = None
            except Exception:
                pass
        return J, xi_r, expo, xi_t

    # Try load saved if available and not forcing federated
    if not FORCE_FEDERATED and not FORCE_RUN:
        saved_J, saved_xi_ref, saved_exposure, saved_xi_true = _try_load_saved_run(out_dir, SEED_IDX)
        if saved_J is not None or saved_xi_ref is not None or saved_exposure is not None:
            J_server = saved_J
            xi_ref = saved_xi_ref
            exposure = saved_exposure
            xi_true = saved_xi_true
            # report what will be reused
            used = []
            if J_server is not None:
                used.append("J_server")
            if xi_ref is not None:
                used.append("xi_ref")
            if exposure is not None:
                used.append("exposure")
            if xi_true is not None:
                used.append("xi_true")
            print(f"[INFO] Articoli caricati dal precedente run: {', '.join(used)}")
        else:
            # diagnostics: explain what was searched and what exists in out_dir
            print(f"[DEBUG] Nessun artefatto J/xi/exposure trovato in {out_dir}. Verifico contenuto della cartella:")
            try:
                for p in sorted(Path(out_dir).glob('*')):
                    print("  -", p.name)
            except Exception as e:
                print("  [DEBUG] Impossibile listare directory:", e)

    # If still missing and not load_only, run federated
    if not load_only and (J_server is None or xi_ref is None or exposure is None):
        missing = []
        if J_server is None:
            missing.append('J_server')
        if xi_ref is None:
            missing.append('xi_ref')
        if exposure is None:
            missing.append('exposure')
        print(f"[INFO] Eseguo run federato perché mancano: {', '.join(missing)} (FORCE_FEDERATED={FORCE_FEDERATED}, FORCE_RUN={FORCE_RUN})")
        fed = run_federated_and_extract(HP, SEED_IDX)
        J_server = fed["J_server"]
        xi_ref = fed["xi_final"]
        exposure = fed["exposure"]
        xi_true = fed["xi_true"]
        # Salva gli artefatti utili per eseguire solo la valutazione Hopfield in futuro
        try:
            # per-seed files
            seed = SEED_IDX
            np.save(out_dir / f"J_server_seed_{seed}.npy", np.asarray(J_server))
            print(f"[INFO] Salvato J_server in: {out_dir / f'J_server_seed_{seed}.npy'}")
            if xi_ref is not None:
                np.save(out_dir / f"xi_ref_seed_{seed}.npy", np.asarray(xi_ref))
                print(f"[INFO] Salvato xi_ref in: {out_dir / f'xi_ref_seed_{seed}.npy'}")
            if xi_true is not None:
                np.save(out_dir / f"xi_true_seed_{seed}.npy", np.asarray(xi_true))
                print(f"[INFO] Salvato xi_true in: {out_dir / f'xi_true_seed_{seed}.npy'} (per caching fedele)")
            # aggregate files (single-seed stack)
            try:
                np.savez_compressed(str(out_dir / "final_J_list.npz"), final_J=np.expand_dims(np.asarray(J_server), axis=0))
                print(f"[INFO] Salvato aggregate final_J_list.npz in: {out_dir / 'final_J_list.npz'}")
            except Exception:
                pass
            if xi_ref is not None:
                try:
                    np.savez_compressed(str(out_dir / "final_xi_list.npz"), np.asarray(xi_ref))
                    print(f"[INFO] Salvato aggregate final_xi_list.npz in: {out_dir / 'final_xi_list.npz'}")
                except Exception:
                    pass
            if exposure is not None:
                try:
                    np.save(out_dir / "exposure_list.npy", np.asarray([exposure]))
                    print(f"[INFO] Salvato exposure in: {out_dir / 'exposure_list.npy'}")
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARN] Impossibile salvare artefatti federati in {out_dir}: {e}")

    # 2) Hopfield eval
    if load_only:
        hop_data = load_hopfield_eval(hop_eval_dir)
    else:
        if exposure is None:
            exposure = np.ones(HP.K, dtype=int)
        reference_xi = xi_true if (USE_XI_TRUE and xi_true is not None) else xi_ref
        if reference_xi is None:
            raise SystemExit("Archetipi di riferimento non disponibili.")
        # Diagnostica: se usiamo xi_ref invece di xi_true (o viceversa) logghiamo e calcoliamo overlap medio
        def _digest(a: np.ndarray) -> str:
            try:
                import hashlib
                h = hashlib.sha256(a.tobytes()).hexdigest()[:10]
                return h
            except Exception:
                return "NA"
        if xi_true is not None and xi_ref is not None and xi_true.shape == xi_ref.shape:
            # overlap matrix (K,K) e max matching naive
            try:
                ov = (xi_true @ xi_ref.T) / xi_true.shape[1]
                best = ov.max(axis=1).mean()
                print(f"[DEBUG] Overlap medio migliore true↔final = {best:.3f}")
            except Exception:
                pass
        print("[INFO] Reference patterns usati:", "xi_true" if (reference_xi is xi_true) else "xi_ref",
              f"shape={reference_xi.shape}, digest={_digest(reference_xi)}")
        # Allinea lunghezze (può succedere che dopo disentangle K_eff < K originale)
        if reference_xi.shape[0] < HP.K:
            # Se il disentangle ha prodotto meno di K pattern, prova a recuperare i veri archetipi
            if xi_true is not None and xi_true.shape[0] == HP.K:
                print(f"[WARN] K_eff disentangled={reference_xi.shape[0]} < K atteso={HP.K}. Uso xi_true come reference.")
                reference_xi = xi_true
            else:
                print(f"[WARN] K_eff disentangled={reference_xi.shape[0]} < K atteso={HP.K}. Proseguo comunque.")
        # Allinea exposure alla dimensione finale dei reference_xi (semplice tronco/pad se necessario)
        if exposure.shape[0] != reference_xi.shape[0]:
            if exposure.shape[0] > reference_xi.shape[0]:
                exposure = exposure[:reference_xi.shape[0]]
            else:
                pad_val = int(exposure.min()) if exposure.size else 1
                exposure = np.pad(exposure, (0, reference_xi.shape[0]-exposure.shape[0]), constant_values=pad_val)
        hop_data = run_or_load_hopfield_eval(
            output_dir=str(hop_eval_dir),
            J_server=J_server,
            xi_true=reference_xi,
            exposure_counts=np.asarray(exposure),
            beta=HOPFIELD_BETA,
            updates=HOPFIELD_UPDATES,
            reps_per_archetype=HOPFIELD_REPS,
            start_overlap=HOPFIELD_START_OVERLAP,
            force_run=force_hopfield_effective,
            stochastic=HOPFIELD_STOCHASTIC,
        )

    eval_stats = hop_data["eval"]
    exposure_counts = hop_data.get("exposure_counts")

    # 3) Plotting
    figs_dir = ensure_dir(out_dir / "figs")
    fmts = [f.strip() for f in SAVE_FORMATS.split(',') if f.strip()]

    # (Figure singole disabilitate temporaneamente: abilita decommentando il blocco sotto)
    # # Figura 1
    # fig1, ax1 = plt.subplots(figsize=(6, 4))
    # plot_magnetization_distribution(eval_stats, ax=ax1, palette=PALETTE_DIST, title="Distribuzione magnetizzazioni")
    # fig1.tight_layout()
    # for fmt in fmts:
    #     fig1.savefig(figs_dir / f"magnetization_distribution.{fmt}", dpi=DPI)
    # # Figura 2
    # fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    # plot_mean_vs_exposure(eval_stats, exposure_counts=exposure_counts, ax=ax2, palette=PALETTE_SCATTER, title="Mean vs Exposure")
    # fig2.tight_layout()
    # for fmt in fmts:
    #     fig2.savefig(figs_dir / f"mean_vs_exposure.{fmt}", dpi=DPI)

    # Figura 3 pannello
    fig3, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4))
    plot_magnetization_distribution(eval_stats, ax=axA, palette=PALETTE_DIST, title="Distribuzione")
    plot_mean_vs_exposure(eval_stats, exposure_counts=exposure_counts, ax=axB, palette=PALETTE_SCATTER, title="Mean vs Exposure")
    fig3.tight_layout()
    for fmt in fmts:
        fig3.savefig(figs_dir / f"panel.{fmt}", dpi=DPI)

    # 4) Summary JSON
    summary = {
        "beta": HOPFIELD_BETA,
        "updates": HOPFIELD_UPDATES,
        "reps": HOPFIELD_REPS,
        "start_overlap": HOPFIELD_START_OVERLAP,
        "stochastic": HOPFIELD_STOCHASTIC,
        "pearson": eval_stats.get("pearson"),
        "spearman": eval_stats.get("spearman"),
        "overall_mean": eval_stats.get("overall_mean"),
        "overall_std": eval_stats.get("overall_std"),
        "K": int(len(eval_stats.get("mean_by_mu", {})))
    }
    (out_dir / "summary_hopfield.json").write_text(json.dumps(summary, indent=2))

    if not NO_SHOW:
        plt.show()
    else:
        plt.close('all')

    print("[OK] Valutazione Hopfield completata. Cartella:", out_dir)
    print("  Eval dir:", hop_eval_dir)
    print("  Figure:", figs_dir)


if __name__ == "__main__":
    run()
