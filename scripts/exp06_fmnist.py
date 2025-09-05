#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp-06 (single-only) — FMNIST (3 classi)
----------------------------------------
Script CLI per eseguire l'esperimento su Fashion-MNIST *strutturato* (binarizzato in {±1}).

Pipeline:
  1) Caricamento dataset da .npz con X (N_img, H, W) o (N_img, D) e y (N_img,)
  2) Costruzione mixing-schedule π_t (scheduler.make_schedule)
  3) Esecuzione pipeline strutturata (pipeline_fmnist.run_seed_fmnist)
  4) Report finale (reporting.build_run_report)
  5) Creazione figure: pannello 4×, lag–amplitude, forgetting vs plasticity, scatter exposure→mag

Requisiti: PYTHONPATH deve includere la radice del progetto che contiene 'src'.
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from dataclasses import asdict
import urllib.request
import gzip
import struct
from typing import cast

# ---------------------------------------------------------------------
# Aggiunge automaticamente la radice del progetto (contenente 'src') al PYTHONPATH
# ---------------------------------------------------------------------
def _ensure_project_root_in_syspath() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, here.parent.parent, here.parent.parent.parent]:
        if (p / "src").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p
    root = here.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_PROJECT_ROOT = _ensure_project_root_in_syspath()

# ---------------------------------------------------------------------
# Import dai moduli del progetto (preferisci src.mixing.*, fallback su src.exp06_single.*)
# ---------------------------------------------------------------------
from src.unsup.config import HyperParams, SpectralParams, PropagationParams  # noqa: E402

try:
    from src.mixing.scheduler import make_schedule  # type: ignore
    from src.mixing.pipeline_fmnist import run_seed_fmnist  # type: ignore
    from src.mixing.reporting import build_run_report, collect_round_metrics  # type: ignore
    from src.mixing.hopfield_hooks import load_magnetization_matrix_from_run  # type: ignore
    from src.mixing.io import ensure_dir, write_json  # type: ignore
    from src.mixing.plotting import (  # type: ignore
        panel4x,
        plot_lag_amplitude_vs_w,
        forgetting_vs_plasticity_triptych,
        scatter_exposure_vs_magnetization,
    )
except Exception:
    from src.exp06_single.scheduler import make_schedule  # type: ignore
    from src.exp06_single.pipeline_fmnist import run_seed_fmnist  # type: ignore
    from src.exp06_single.reporting import build_run_report, collect_round_metrics  # type: ignore
    from src.exp06_single.hopfield_hooks import load_magnetization_matrix_from_run  # type: ignore
    from src.exp06_single.io import ensure_dir, write_json  # type: ignore
    from src.exp06_single.plotting import (  # type: ignore
        panel4x,
        plot_lag_amplitude_vs_w,
        forgetting_vs_plasticity_triptych,
        scatter_exposure_vs_magnetization,
    )

import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exp-06 (single-only) — FMNIST (3 classi)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dati
    p.add_argument("--data-npz", type=str, default=None,
                   help="Percorso a .npz con chiavi: X (N_img, H, W) o (N_img, D), y (N_img,) int. Se omesso, cerco automaticamente in data/.")
    p.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2],
                   help="Tripletta di classi FMNIST da usare (K deve coincidere)")

    # Output
    p.add_argument("--outdir", type=str, default="out_06/fmnist",
                   help="Cartella radice per i risultati")

    # Problem scale / federated
    p.add_argument("--L", type=int, default=3, help="Numero client/layer")
    p.add_argument("--K", type=int, default=3, help="Numero archetipi (deve essere 3 per Δ2)")
    p.add_argument("--rounds", type=int, default=12, help="Numero di round (T)")
    p.add_argument("--M-total", type=int, default=1200, help="Numero totale di esempi per la run")
    p.add_argument("--w", type=float, default=0.6, help="Peso 'unsup' nel blend con memoria ebraica")
    p.add_argument("--binarize-thresh", type=float, default=None,
                   help="Soglia per binarizzazione {±1}; None => mediana globale")

    # Propagazione / Spettro / TAM (override dei default della codebase)
    p.add_argument("--prop-iters", type=int, default=200, help="Iterazioni propagate_J")
    p.add_argument("--prop-eps", type=float, default=1e-2, help="Parametro eps per propagate_J")
    p.add_argument("--tau", type=float, default=0.2, help="Cut sugli autovalori di J_KS")
    p.add_argument("--rho", type=float, default=0.4, help="Soglia allineamento spettrale")
    p.add_argument("--qthr", type=float, default=0.8, help="Pruning per overlap mutuo")
    p.add_argument("--keff-method", type=str, default="shuffle", choices=("shuffle", "mp"), help="Metodo K_eff")
    p.add_argument("--ema-alpha", type=float, default=0.0, help="EMA su J_unsup (0.0=off)")

    # Schedule
    p.add_argument("--schedule", type=str, default="cyclic",
                   choices=("cyclic", "piecewise_dirichlet", "random_walk"),
                   help="Tipo di mixing-schedule")
    # cyclic
    p.add_argument("--period", type=int, default=12, help="[cyclic] Periodo in round")
    p.add_argument("--gamma", type=float, default=2.0, help="[cyclic] Ampiezza logits")
    p.add_argument("--temp", type=float, default=1.2, help="[cyclic] Temperatura softmax")
    p.add_argument("--center-mix", type=float, default=0.30, help="[cyclic] Blend con uniforme")
    # piecewise_dirichlet
    p.add_argument("--block", type=int, default=4, help="[piecewise_dirichlet] Lunghezza blocco")
    p.add_argument("--alpha", type=float, default=1.0, help="[piecewise_dirichlet] Parametro Dirichlet")
    # random_walk
    p.add_argument("--step-sigma", type=float, default=0.7, help="[random_walk] Deviazione passo logits")
    p.add_argument("--tv-max", type=float, default=0.35, help="[random_walk] TV step-wise massima")

    # Seeds & valutazioni
    p.add_argument("--seed-base", type=int, default=12345, help="Seed base (usato dalla codebase)")
    p.add_argument("--n-seeds", type=int, default=1, help="Numero di seed (0..n_seeds-1)")
    p.add_argument("--eval-hopfield-every", type=int, default=1,
                   help="1=ogni round, 0=off, n>1=ogni n round (eseguito in pipeline)")
    p.add_argument("--no-tqdm", action="store_true", help="Disabilita barre di progresso interne")
    # Plotting
    p.add_argument("--simplex-style", type=str, default="legacy", choices=("modern", "legacy"),
                   help="Stile embedding simplesso per le figure (modern|legacy)")
    p.add_argument("--panel-mode", type=str, default="exposure",
                   choices=("phase", "drift", "exposure"),
                   help="Contenuto quadrante in basso a destra del pannello 4×: phase (diagram), drift (TV drift/mismatch), exposure (cumulative exposure)")

    # --- Adaptive w control (common) ---
    p.add_argument("--w-policy", type=str, default="fixed",
                   choices=("fixed", "threshold", "sigmoid", "pctrl"),
                   help="Policy di aggiornamento w round-wise")
    p.add_argument("--w-init", type=float, default=0.6, help="Valore iniziale w al round 0")
    p.add_argument("--w-min", type=float, default=0.10, help="Limite inferiore per w")
    p.add_argument("--w-max", type=float, default=0.90, help="Limite superiore per w")
    p.add_argument("--alpha-w", type=float, default=0.6, help="Smoothing sul controllo di w")
    p.add_argument("--a-drift", type=float, default=0.5, help="Peso D_t in S_t = a*D_t + b*M_t")
    p.add_argument("--b-mismatch", type=float, default=1.0, help="Peso M_t in S_t = a*D_t + b*M_t")

    # Policy A: threshold
    p.add_argument("--theta-low", type=float, default=0.05, help="Soglia bassa per isteresi su S_t")
    p.add_argument("--theta-high", type=float, default=0.15, help="Soglia alta per isteresi su S_t")
    p.add_argument("--delta-up", type=float, default=0.10, help="Incremento w quando S_t supera theta_high")
    p.add_argument("--delta-down", type=float, default=0.05, help="Decremento w quando S_t scende sotto theta_low")

    # Policy B: sigmoid
    p.add_argument("--theta-mid", type=float, default=0.12, help="Centro della sigmoide su S_t")
    p.add_argument("--beta", type=float, default=10.0, help="Ripidità della sigmoide")

    # Policy C: pctrl
    p.add_argument("--lag-target", type=float, default=0.3, help="Target per |lag|(rad)")
    p.add_argument("--lag-window", type=int, default=8, help="Finestra scorrevole per stimare |lag|")
    p.add_argument("--kp", type=float, default=0.8, help="Guadagno proporzionale")
    p.add_argument("--ki", type=float, default=0.0, help="Guadagno integrale")
    p.add_argument("--kd", type=float, default=0.0, help="Guadagno derivativo")
    p.add_argument("--gate-drift-theta", type=float, default=0.1,
                   help="Se impostato: se S_t < theta, non aumentare w (gating)")

    return p


# ---------------------------------------------------------------------
# Helpers caricamento dati / hparams / schedule
# ---------------------------------------------------------------------
def _load_npz_dataset(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carica un .npz con chiavi:
      - X: (N_img, H, W) oppure (N_img, D)
      - y: (N_img,) int
    Converte X in float32, y in int64.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File .npz non trovato: {path}")
    with np.load(path, allow_pickle=False) as f:
        if "X" not in f or "y" not in f:
            raise KeyError("L'npz deve contenere le chiavi 'X' e 'y'.")
        X = f["X"]
        y = f["y"]
    if y.ndim != 1:
        y = y.reshape(-1)
    X = X.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)
    return X, y


def _discover_npz_in_data() -> Optional[Path]:
    """Cerca file .npz utili nella cartella data/ del progetto.
    Preferisce file che contengono 'fmnist' o 'fashion' nel nome.
    Restituisce il Path della prima corrispondenza preferita, altrimenti None.
    """
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    if not data_dir.exists():
        return None
    candidates = list(data_dir.rglob("*.npz"))
    if not candidates:
        return None
    # preferenza per fmnist / fashion
    for key in ("fmnist", "fashion", "fmnist"):
        for c in candidates:
            if key in c.name.lower():
                return c
    # altrimenti ritorna il primo
    return candidates[0]


def _download_fashion_mnist_npz(dest_dir: Path) -> Optional[Path]:
    """Scarica un file .npz per Fashion-MNIST in dest_dir.
    Prova prima un URL pubblico compatibile con tf-keras-datasets; se fallisce tenta
    di usare tensorflow.keras.datasets per scaricare e salvare come .npz.
    Restituisce il Path del file scaricato o None se entrambi i metodi falliscono.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "fashion-mnist.npz"

    # 1) Try direct npz URL (common pattern)
    url_candidates = [
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/fashion-mnist.npz",
    ]
    for url in url_candidates:
        try:
            print(f"[Exp06-FMNIST] Scarico Fashion-MNIST da {url} → {out_path}")
            urllib.request.urlretrieve(url, str(out_path))
            if out_path.exists():
                return out_path
        except Exception as e:
            print(f"[Exp06-FMNIST] Download da URL fallito: {e}")

    # 2) Try downloading IDX gz files from the official Fashion-MNIST S3 and convert
    idx_base = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    tmp_dir = dest_dir / "_tmp_fmnist_idx"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []
        for name, fname in files.items():
            url = idx_base + fname
            dst = tmp_dir / fname
            try:
                print(f"[Exp06-FMNIST] Scarico {url} -> {dst}")
                urllib.request.urlretrieve(url, str(dst))
                if not dst.exists():
                    raise RuntimeError("file non trovato dopo download")
                downloaded.append(dst)
            except Exception as e:
                print(f"[Exp06-FMNIST] Fallito download IDX {fname}: {e}")
                downloaded = []
                break

        if downloaded:
            def _read_idx_images(path: Path) -> np.ndarray:
                with gzip.open(path, "rb") as fh:
                    magic = struct.unpack(">I", fh.read(4))[0]
                    if magic != 2051:
                        raise RuntimeError(f"Magic number mismatch for images: {magic}")
                    n = struct.unpack(">I", fh.read(4))[0]
                    r = struct.unpack(">I", fh.read(4))[0]
                    c = struct.unpack(">I", fh.read(4))[0]
                    buf = fh.read()
                    arr = np.frombuffer(buf, dtype=np.uint8)
                    return arr.reshape(n, r, c)

            def _read_idx_labels(path: Path) -> np.ndarray:
                with gzip.open(path, "rb") as fh:
                    magic = struct.unpack(">I", fh.read(4))[0]
                    if magic != 2049:
                        raise RuntimeError(f"Magic number mismatch for labels: {magic}")
                    n = struct.unpack(">I", fh.read(4))[0]
                    buf = fh.read()
                    arr = np.frombuffer(buf, dtype=np.uint8)
                    return arr.reshape(n,)

            X_train = _read_idx_images(tmp_dir / files["train_images"])
            y_train = _read_idx_labels(tmp_dir / files["train_labels"])
            X_test = _read_idx_images(tmp_dir / files["test_images"])
            y_test = _read_idx_labels(tmp_dir / files["test_labels"])
            X = np.concatenate([X_train, X_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)
            np.savez_compressed(str(out_path), X=X, y=y)
            # cleanup tmp
            try:
                for f in tmp_dir.iterdir():
                    f.unlink()
                tmp_dir.rmdir()
            except Exception:
                pass
            if out_path.exists():
                return out_path
    except Exception as e:
        print(f"[Exp06-FMNIST] IDX download/convert fallito: {e}")

    # 3) Fallback: try tensorflow/keras if installed
    try:
        print("[Exp06-FMNIST] Tentativo fallback: usare tensorflow.keras.datasets.fashion_mnist")
        try:
            # prefer tensorflow.keras if available
            from tensorflow.keras.datasets import fashion_mnist  # type: ignore
        except Exception:
            from keras.datasets import fashion_mnist  # type: ignore
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        X = np.concatenate([x_train, x_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        np.savez_compressed(str(out_path), X=X, y=y)
        if out_path.exists():
            return out_path
    except Exception as e:
        print(f"[Exp06-FMNIST] Fallback con keras fallito: {e}")

    return None


def _check_classes_available(y: np.ndarray, classes: list[int]) -> None:
    present = set(map(int, np.unique(y)))
    missing = [c for c in classes if int(c) not in present]
    if missing:
        raise ValueError(f"Le classi richieste non sono presenti nel dataset: {missing}")


def make_hparams_from_args(args: argparse.Namespace, N_data: int) -> HyperParams:
    """Popola HyperParams dalla CLI. Imposta hp.N coerente con N_data."""
    hp = HyperParams(
        L=int(args.L),
        K=int(args.K),
        N=int(N_data),
        n_batch=int(args.rounds),
        M_total=int(args.M_total),
        r_ex=1.0,            # non usato in FMNIST ma deve essere in (0,1]
        w=float(args.w),
        estimate_keff_method=args.keff_method,
        ema_alpha=float(getattr(args, "ema_alpha", 0.0)),
        prop=PropagationParams(iters=int(args.prop_iters), eps=float(args.prop_eps)),
        spec=SpectralParams(tau=float(args.tau), rho=float(args.rho), qthr=float(args.qthr)),
    )
    if args.no_tqdm:
        hp.use_tqdm = False
    hp.seed_base = int(args.seed_base)
    return hp


def make_schedule_from_args(hp: HyperParams, args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(hp.seed_base)
    kind = str(args.schedule)
    if kind == "cyclic":
        return make_schedule(
            hp, kind="cyclic", rng=rng,
            period=int(args.period), gamma=float(args.gamma), temp=float(args.temp),
            center_mix=float(args.center_mix),
        )
    if kind == "piecewise_dirichlet":
        return make_schedule(
            hp, kind="piecewise_dirichlet", rng=rng,
            block=int(args.block), alpha=float(args.alpha),
        )
    if kind == "random_walk":
        return make_schedule(
            hp, kind="random_walk", rng=rng,
            step_sigma=float(args.step_sigma), tv_max=float(args.tv_max),
        )
    raise ValueError(f"Schedule '{kind}' non riconosciuta.")


def _format_run_dir(base: Path, seed: int, hp: HyperParams, classes: list[int]) -> Path:
    cls_tag = "c" + "-".join(map(str, classes))
    return base / f"seed_{seed:03d}" / f"{cls_tag}" / f"w_{hp.w:.2f}".replace(".", "p")


# ----------------- PLOTTING HELPERS -----------------
def _load_pi_sequences_with_retrieval_preferred(items: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Versione allineata al sintetico: preferisce 'pi_hat_retrieval' se presente; fallback su 'pi_hat' o 'pi_hat_data'."""
    pi_true_seq, pi_hat_seq = [], []
    for it in items:
        if "pi_true" not in it:
            continue
        pt = np.asarray(it["pi_true"], dtype=float)
        ph_raw = it.get("pi_hat_retrieval")
        if ph_raw is None:
            ph_raw = it.get("pi_hat") or it.get("pi_hat_data")
        if ph_raw is None:
            continue
        ph = np.asarray(ph_raw, dtype=float)
        if pt.shape != ph.shape:
            continue
        pt = pt / (pt.sum() if pt.sum() > 0 else 1.0)
        ph = ph / (ph.sum() if ph.sum() > 0 else 1.0)
        pi_true_seq.append(pt)
        pi_hat_seq.append(ph)
    if not pi_true_seq or not pi_hat_seq:
        raise RuntimeError("Impossibile costruire le sequenze π_t/π̂_t dai metrics.json.")
    return np.stack(pi_true_seq, axis=0), np.stack(pi_hat_seq, axis=0)


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _create_panel4x_and_scatter(run_dir: Path, w: float, simplex_style: str, panel_mode: str) -> None:
    """Crea pannello 4× e scatter exposure→magnetization (con panel_mode e retrieval) per la singola run."""
    results_dir = ensure_dir(run_dir / "results")
    figs_dir = ensure_dir(results_dir / "figs")

    # round metrics -> pi sequences
    items = collect_round_metrics(run_dir)
    pi_true_seq, pi_hat_seq = _load_pi_sequences_with_retrieval_preferred(items)

    # magnetization matrix M(K,T)
    M = load_magnetization_matrix_from_run(run_dir)  # può essere None
    if M is None:
        M_plot = np.zeros((1, pi_true_seq.shape[0]), dtype=float)
        M_sem = None
    else:
        M_plot = M
        M_sem = None  # s.e.m. non disponibile round-wise

    # phase metric locale: mean m (safe: controlla array vuoto/None prima di np.mean)
    phase_ws = [w]
    if M_plot is None or getattr(M_plot, "size", 0) == 0:
        phase_metric = [float("nan")]
    else:
        phase_metric = [float(np.mean(M_plot))]

    # 1) Pannello 4×
    fig, _info = panel4x(
        pi_true_seq=pi_true_seq,
        pi_hat_seq=pi_hat_seq,
        M_mean=M_plot,
        M_sem=M_sem,
        w_values=phase_ws,
        phase_metric=phase_metric,
        phase_metric_label="mean m",
        simplex_style=simplex_style,
        bottom_right_mode=str(panel_mode),
        labels=("μ0", "μ1", "μ2"),
        suptitle=f"Seed panel — w={w:.2f}",
    )
    _save_fig(fig, figs_dir / "panel4x.png")

    # 2) Scatter Exposure → Magnetization (per-μ)
    exp_path = run_dir / "exposure_counts.npy"
    if exp_path.exists() and M is not None:
        exposure = np.load(exp_path).astype(float)  # (K,)
        m_by_mu = M.mean(axis=1)  # media su T
        fig, ax = plt.subplots(figsize=(5, 4))
        _ = scatter_exposure_vs_magnetization(
            ax, exposure=exposure, magnetization=m_by_mu,
            title="Exposure → Magnetization (per μ)"
        )
        _save_fig(fig, figs_dir / "exposure_vs_m.png")


def _collect_sibling_w_runs(seed_dir: Path) -> List[Path]:
    """Trova tutte le run 'w_*' sotto una stessa cartella seed_XXX/…/cA-B-C/."""
    if not seed_dir.exists():
        return []
    return sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("w_")])


def _parse_w_from_dirname(p: Path) -> Optional[float]:
    """Estrae w dal nome 'w_0p40' -> 0.40."""
    name = p.name
    if not name.startswith("w_"):
        return None
    try:
        return float(name[2:].replace("p", "."))
    except Exception:
        return None


def _create_lag_amp_and_forgetting(seed_run_dir: Path) -> None:
    """
    Aggrega più run con w diversi sotto lo stesso seed/classe e produce:
      - Lag–Amplitude vs w
      - Forgetting vs Plasticity triptych
    Se c'è un solo w, salta silenziosamente.
    """
    # seed_run_dir = .../seed_XXX/cA-B-C/w_0p40  → vogliamo il livello '.../seed_XXX/cA-B-C'
    class_dir = seed_run_dir.parent
    w_dirs = _collect_sibling_w_runs(class_dir)
    if len(w_dirs) < 2:
        return

    entries = []
    Ws = []
    lag_rad = []
    amp_ratio = []

    for rd in w_dirs:
        w_val = _parse_w_from_dirname(rd)
        if w_val is None:
            continue
        # pi sequences (retrieval preferito)
        try:
            items = collect_round_metrics(rd)
            pi_true_seq, pi_hat_seq = _load_pi_sequences_with_retrieval_preferred(items)
        except Exception:
            continue

        # magnetization (può mancare)
        M = load_magnetization_matrix_from_run(rd)
        if M is None:
            M = np.zeros((pi_true_seq.shape[1], pi_true_seq.shape[0]), dtype=float)

        # lag/amp sperimentali
        from src.mixing.metrics import lag_and_amplitude as _lagamp  # fallback sicuro
        la = _lagamp(pi_true_seq, pi_hat_seq)
        Ws.append(w_val)
        lag_rad.append(float(la["lag_radians"]))
        amp_ratio.append(float(la["amp_ratio"]))
        entries.append({"w": w_val, "pi_true_seq": pi_true_seq, "pi_hat_seq": pi_hat_seq, "M": M})

    if len(Ws) < 2:
        return

    # Ordina per w crescente
    order = np.argsort(Ws)
    Ws_sorted = [float(Ws[i]) for i in order]
    lag_rad_sorted = [float(lag_rad[i]) for i in order]
    amp_ratio_sorted = [float(amp_ratio[i]) for i in order]
    entries = [entries[i] for i in order]

    figs_dir = ensure_dir(class_dir / "results" / "figs")

    # 1) Lag–Amplitude vs w
    fig = plot_lag_amplitude_vs_w(Ws_sorted, lag_rad_sorted, amp_ratio_sorted, suptitle=f"Lag–Amplitude vs w — {class_dir.name}")
    _save_fig(fig, figs_dir / "lag_amp_vs_w.png")

    # 2) Forgetting vs Plasticity triptych
    fig = forgetting_vs_plasticity_triptych(entries, suptitle=f"Forgetting vs Plasticity — {class_dir.name}")
    _save_fig(fig, figs_dir / "forgetting_vs_plasticity.png")


def _open_image_if_possible(img_path: Path) -> None:
    """Apre (best-effort) un'immagine con il viewer di sistema (silenzioso se fallisce)."""
    if not img_path.exists():
        return
    try:
        system = platform.system().lower()
        if system.startswith("win"):
            os.startfile(str(img_path))  # type: ignore[attr-defined]
        elif system == "darwin":
            subprocess.Popen(["open", str(img_path)])
        else:
            subprocess.Popen(["xdg-open", str(img_path)])
    except Exception:
        pass


# ---------------------------------------------------------------------
# Hopfield: evoluzione di un esempio per TUTTI gli archetipi (K=3)
# ---------------------------------------------------------------------
def _hopfield_evolution_all(run_dir: Path, X: np.ndarray, y: np.ndarray, classes: list[int], *,
                            beta: float = 3.0, updates: int = 30,
                            start_overlap: float = 0.2,
                            snap_ts: Optional[List[int]] = None) -> None:
    """Crea un'unica immagine con l'evoluzione Hopfield da un esempio reale (corrotto)
    per ciascuno dei K archetipi (qui K=3).

    Salva il file:
        run_dir / "results" / "figs" / "hopfield_evolution_all.png"

    Requisiti: esistenza di
        - xi_true.npy (K,N)
        - ultimo round round_XXX/J_KS.npy (oppure J_rec.npy se J_KS assente)
    """
    try:
        from src.unsup.networks import Hopfield_Network  # type: ignore
    except Exception:
        print("[Hopfield-Evol] Impossibile importare Hopfield_Network.")
        return

    try:
        xi_path = run_dir / "xi_true.npy"
        if not xi_path.exists():
            print("[Hopfield-Evol] xi_true.npy assente, skip.")
            return
        xi_true = np.load(xi_path).astype(int)  # (K,N)
        K, N = xi_true.shape
        # determina ultimo round
        rounds = sorted([p for p in run_dir.glob("round_*") if p.is_dir()])
        if not rounds:
            print("[Hopfield-Evol] Nessun round trovato.")
            return
        last_r = rounds[-1]
        J_path = last_r / "J_KS.npy"
        if not J_path.exists():
            J_path = last_r / "J_rec.npy"
        if not J_path.exists():
            print("[Hopfield-Evol] J_KS.npy/J_rec.npy assenti nell'ultimo round.")
            return
        J = np.load(J_path).astype(np.float32)
        if J.shape[0] != N:
            print("[Hopfield-Evol] Dimensioni J non coerenti con xi_true.")
            return

        # seleziona un esempio reale per ogni classe richiesta (nell'ordine di 'classes')
        rng = np.random.default_rng(1234)
        samples = []  # lista di vettori ±1 (N,)
        # Prepara binarizzazione coerente: usa mediana globale se necessario
        if X.ndim == 3:
            H, W = X.shape[1], X.shape[2]
        thresh = float(np.median(X))
        for c in classes:
            idxs = np.where(y == c)[0]
            if idxs.size == 0:
                print(f"[Hopfield-Evol] Nessuna immagine per classe {c}, skip intero pannello.")
                return
            i_sel = int(rng.integers(0, idxs.size))
            x_raw = X[idxs[i_sel]]
            if x_raw.ndim == 2:
                x_vec = x_raw.reshape(-1)
            else:
                x_vec = x_raw.astype(np.float32)
            x_bin = np.where(x_vec > thresh, 1, -1).astype(int)
            samples.append(x_bin)
        samples = np.stack(samples, axis=0)  # (K,N)

        # Corruzione per ottenere overlap iniziale ~ start_overlap con il proprio archetipo
        def corrupt(vec: np.ndarray, target_overlap: float, rng: np.random.Generator) -> np.ndarray:
            # overlap m = (1/N) sum v*xi ; flipping prob p -> expected overlap (1-2p)
            p_flip = max(0.0, min(0.5, 0.5 * (1.0 - target_overlap)))
            mask = rng.random(size=vec.shape) < p_flip
            return np.where(mask, -vec, vec)

        init_states = []
        for mu in range(K):
            v0 = corrupt(samples[mu], start_overlap, rng)
            init_states.append(v0)
        init_states = np.stack(init_states, axis=0)  # (K,N)

        # dinamica custom per registrare snapshot
        if snap_ts is None:
            snap_ts = [0, 1, 2, 5, 10, 20, updates]
        snap_ts = sorted(set([t for t in snap_ts if 0 <= t <= updates]))

        # Prealloc containers: snapshots[mu][time] -> state
        snapshots: list[list[np.ndarray]] = [[init_states[mu].copy()] for mu in range(K)]
        mags: list[list[float]] = [[float(np.dot(init_states[mu], xi_true[mu]) / N)] for mu in range(K)]

        σ = init_states.copy()  # (K,N)
        for t in range(1, updates + 1):
            # parallel Glauber step
            h = J @ σ.T  # (N,K)
            h = h.T  # (K,N)
            p = (1.0 + np.tanh(beta * h)) * 0.5
            draw = np.random.random(size=p.shape)
            σ = np.where(draw < p, 1, -1).astype(np.int8)
            # registra se t in snap_ts
            if t in snap_ts:
                for mu in range(K):
                    snapshots[mu].append(σ[mu].copy())
                    mags[mu].append(float(np.dot(σ[mu], xi_true[mu]) / N))

        # --- plotting ---
        import matplotlib.pyplot as _plt
        from matplotlib import gridspec as _gridspec
        ncols = len(snapshots[0])  # include t0
        fig = _plt.figure(figsize=(1.6 * ncols + 2.5, 1.9 * K + 1.8))
        outer = _gridspec.GridSpec(K, ncols + 1, width_ratios=[1]*ncols + [1.8], hspace=0.35, wspace=0.15)

        # funzione helper per reshaping immagine se quadrata
        side = int(np.sqrt(N))
        can_reshape = side * side == N
        for mu in range(K):
            # snapshot cells
            for j, state in enumerate(snapshots[mu]):
                ax = _plt.subplot(outer[mu, j])
                img = state.reshape(side, side) if can_reshape else state
                ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
                ax.axis("off")
                if mu == 0:
                    t_lab = snap_ts[j] if j < len(snap_ts) else "?"
                    ax.set_title(f"t={t_lab}", fontsize=8)
                if j == 0:
                    ax.text(0.02, 0.08, f"μ{mu}", color="lime", fontsize=9, transform=ax.transAxes,
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))
            # magnetization curve (ultima colonna)
            axc = _plt.subplot(outer[mu, -1])
            ts = snap_ts
            axc.plot(ts, mags[mu], marker="o", ms=4)
            axc.set_ylim(-1.05, 1.05)
            axc.axhline(0, color="#888", lw=0.6)
            if mu == K - 1:
                axc.set_xlabel("t")
            axc.set_ylabel(f"m_μ{mu}")
            axc.grid(alpha=0.3, axis="y")
        fig.suptitle("Hopfield evolution per archetipo (esempi reali corrotti)", fontsize=11)
        figs_dir = ensure_dir(run_dir / "results" / "figs")
        out_path = figs_dir / "hopfield_evolution_all.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        _plt.close(fig)
        print(f"[Hopfield-Evol] Salvato {out_path}")
    except Exception as e:
        print(f"[Hopfield-Evol] Errore: {e}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = build_argparser().parse_args()
    base_out = ensure_dir(Path(args.outdir))

    # 1) Caricamento dataset
    if args.data_npz is None:
        found = _discover_npz_in_data()
        if found is None:
            print("[Exp06-FMNIST] Nessun .npz trovato in data/ — provo a scaricare Fashion-MNIST automaticamente...")
            dl = _download_fashion_mnist_npz(Path(__file__).resolve().parent.parent / "data")
            if dl is None:
                raise FileNotFoundError(
                    "--data-npz non specificato, nessun .npz trovato in data/ e download automatico fallito. "
                    "Aggiungi un file .npz con chiavi X,y in data/ o usa --data-npz PATH"
                )
            found = dl
        print(f"[Exp06-FMNIST] Usando dataset trovato automaticamente: {found}")
        args.data_npz = str(found)
    X, y = _load_npz_dataset(args.data_npz)
    # Appiattisci per N_data (la binarizzazione avviene in pipeline)
    if X.ndim == 3:
        N_data = int(X.shape[1] * X.shape[2])
    elif X.ndim == 2:
        N_data = int(X.shape[1])
    else:
        raise ValueError("X deve essere (N_img, H, W) oppure (N_img, D).")

    classes = list(map(int, args.classes))
    if len(classes) != int(args.K):
        raise ValueError(f"--K={args.K} ma --classes ha {len(classes)} elementi. Devono coincidere.")
    _check_classes_available(y, classes)

    # 2) Imposta M_total coerente con il dataset, poi HyperParams e schedule
    n_images = int(X.shape[0])
    if args.M_total is None:
        args.M_total = n_images
    else:
        # Se l'utente ha richiesto più esempi del dataset, cappiamo al massimo disponibile
        if int(args.M_total) > n_images:
            print(f"--M-total ({args.M_total}) > n_images ({n_images}), verrà usato {n_images} come limite.")
            args.M_total = n_images

    hp = make_hparams_from_args(args, N_data=N_data)
    pis = make_schedule_from_args(hp, args)

    # Meta run-level
    write_json(base_out / "hyperparams.json", asdict(hp))
    np.save(base_out / "pis_template.npy", pis.astype(np.float32))
    write_json(base_out / "schedule_meta.json", {
        "kind": args.schedule,
        "params": {
            "period": args.period, "gamma": args.gamma, "temp": args.temp, "center_mix": args.center_mix,
            "block": args.block, "alpha": args.alpha,
            "step_sigma": args.step_sigma, "tv_max": args.tv_max,
        },
    })
    write_json(base_out / "data_meta.json", {
        "data_npz": str(Path(args.data_npz).resolve()),
        "n_images": int(X.shape[0]),
        "N_data": int(N_data),
        "classes": classes,
    })

    # 3) Esecuzione per seed
    for s in range(int(args.n_seeds)):
        run_dir = ensure_dir(_format_run_dir(base_out, s, hp, classes))
        print(f"[Exp06-FMNIST] Seed={s} → {run_dir}")

        # Salva localmente la schedule (per riproducibilità)
        np.save(run_dir / "pis.npy", pis.astype(np.float32))

        # Esecuzione pipeline strutturata (include valutazione Hopfield se richiesto)
        summary = run_seed_fmnist(
            hp=hp,
            seed=int(s),
            outdir=str(run_dir),
            X=X,
            y=y,
            classes=classes,
            pis=pis,
            binarize_thresh=None if args.binarize_thresh is None else float(args.binarize_thresh),
            eval_hopfield_every=int(args.eval_hopfield_every),
            # controllo w
            w_policy=str(args.w_policy),
            w_init=float(args.w_init),
            w_min=float(args.w_min), w_max=float(args.w_max),
            alpha_w=float(args.alpha_w),
            a_drift=float(args.a_drift), b_mismatch=float(args.b_mismatch),
            theta_low=float(args.theta_low), theta_high=float(args.theta_high),
            delta_up=float(args.delta_up), delta_down=float(args.delta_down),
            theta_mid=float(args.theta_mid), beta=float(args.beta),
            lag_target=float(args.lag_target), lag_window=int(args.lag_window),
            kp=float(args.kp), ki=float(args.ki), kd=float(args.kd),
            gate_drift_theta=None if args.gate_drift_theta is None else float(args.gate_drift_theta),
        )
        write_json(run_dir / "summary_cli.json", {
            "seed": int(s),
            "args": vars(args),
            "summary_out": summary,
        })

        # 4) Report finale (CSV + JSON)
        report = build_run_report(
            run_dir=str(run_dir),
            write_json_report=True,
            write_csv_round_metrics=True,
            write_csv_magnetization=True,
            exposure_mask=None,  # opzionale
        )
        write_json(run_dir / "results" / "report_cli.json", report)
        print(f"[Exp06-FMNIST] Seed={s} — report:", json.dumps(report, indent=2))

        # === CREAZIONE FIGURE PER QUESTA RUN ===
        try:
            _create_panel4x_and_scatter(run_dir, w=float(hp.w), simplex_style=str(args.simplex_style), panel_mode=str(args.panel_mode))
            panel_path = run_dir / "results" / "figs" / "panel4x.png"
            _open_image_if_possible(panel_path)
        except Exception as e:
            print(f"[Plot] Pannello/Scatter non creati per {run_dir}: {e}")

        # === Hopfield evolution (tutti gli archetipi) ===
        try:
            _hopfield_evolution_all(run_dir, X=X, y=y, classes=classes,
                                     beta=3.0, updates=30, start_overlap=0.2)
        except Exception as e:
            print(f"[Plot] Hopfield evolution (all) non creata per {run_dir}: {e}")

        # === AGGREGAZIONE SU W (stesso seed e stesse classi): lag–amp e trittico forgetting/plasticity ===
        try:
            _create_lag_amp_and_forgetting(run_dir)
        except Exception as e:
            print(f"[Plot] Aggregati lag/amp & forgetting/plasticity non creati per {run_dir.parent}: {e}")

    print("[Exp06-FMNIST] Completato.")


if __name__ == "__main__":
    main()
