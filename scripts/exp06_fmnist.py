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
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

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
    p.add_argument("--data-npz", type=str, required=True,
                   help="Percorso a .npz con chiavi: X (N_img, H, W) o (N_img, D), y (N_img,) int")
    p.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2],
                   help="Tripletta di classi FMNIST da usare (K deve coincidere)")

    # Output
    p.add_argument("--outdir", type=str, default="runs/exp06/fmnist",
                   help="Cartella radice per i risultati")

    # Problem scale / federated
    p.add_argument("--L", type=int, default=3, help="Numero client/layer")
    p.add_argument("--K", type=int, default=3, help="Numero archetipi (deve essere 3 per Δ2)")
    p.add_argument("--rounds", type=int, default=24, help="Numero di round (T)")
    p.add_argument("--M-total", type=int, default=2400, help="Numero totale di esempi per la run")
    p.add_argument("--w", type=float, default=0.4, help="Peso 'unsup' nel blend con memoria ebraica")
    p.add_argument("--binarize-thresh", type=float, default=None,
                   help="Soglia per binarizzazione {±1}; None => mediana globale")

    # Propagazione / Spettro / TAM (override dei default della codebase)
    p.add_argument("--prop-iters", type=int, default=200, help="Iterazioni propagate_J")
    p.add_argument("--prop-eps", type=float, default=1e-2, help="Parametro eps per propagate_J")
    p.add_argument("--tau", type=float, default=0.5, help="Cut sugli autovalori di J_KS")
    p.add_argument("--rho", type=float, default=0.6, help="Soglia allineamento spettrale")
    p.add_argument("--qthr", type=float, default=0.4, help="Pruning per overlap mutuo")
    p.add_argument("--keff-method", type=str, default="shuffle", choices=("shuffle", "mp"), help="Metodo K_eff")

    # Schedule
    p.add_argument("--schedule", type=str, default="cyclic",
                   choices=("cyclic", "piecewise_dirichlet", "random_walk"),
                   help="Tipo di mixing-schedule")
    # cyclic
    p.add_argument("--period", type=int, default=12, help="[cyclic] Periodo in round")
    p.add_argument("--gamma", type=float, default=3.0, help="[cyclic] Ampiezza logits")
    p.add_argument("--temp", type=float, default=1.0, help="[cyclic] Temperatura softmax")
    p.add_argument("--center-mix", type=float, default=0.0, help="[cyclic] Blend con uniforme")
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
    p.add_argument("--simplex-style", type=str, default="modern", choices=("modern", "legacy"),
                   help="Stile embedding simplesso per le figure (modern|legacy)")

    # --- Adaptive w control (common) ---
    p.add_argument("--w-policy", type=str, default="pctrl",
                   choices=("fixed", "threshold", "sigmoid", "pctrl"),
                   help="Policy di aggiornamento w round-wise")
    p.add_argument("--w-init", type=float, default=0.8, help="Valore iniziale w al round 0")
    p.add_argument("--w-min", type=float, default=0.05, help="Limite inferiore per w")
    p.add_argument("--w-max", type=float, default=0.95, help="Limite superiore per w")
    p.add_argument("--alpha-w", type=float, default=0.3, help="Smoothing sul controllo di w")
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
    p.add_argument("--gate-drift-theta", type=float, default=None,
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
        r_ex=0.0,            # non usato in FMNIST
        w=float(args.w),
        estimate_keff_method=args.keff_method,
        ema_alpha=0.0,
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
def _load_pi_sequences_from_items(items: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Estrae (pi_true_seq, pi_hat_seq) come (T,3) da metrics.json round-wise."""
    pi_true_seq, pi_hat_seq = [], []
    for it in items:
        if "pi_true" in it and "pi_hat" in it:
            pt = np.asarray(it["pi_true"], dtype=float)
            ph = np.asarray(it["pi_hat"], dtype=float)
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


def _create_panel4x_and_scatter(run_dir: Path, w: float, simplex_style: str) -> None:
    """Crea pannello 4× e scatter exposure→magnetization per la singola run."""
    results_dir = ensure_dir(run_dir / "results")
    figs_dir = ensure_dir(results_dir / "figs")

    # round metrics -> pi sequences
    items = collect_round_metrics(run_dir)
    pi_true_seq, pi_hat_seq = _load_pi_sequences_from_items(items)

    # magnetization matrix M(K,T)
    M = load_magnetization_matrix_from_run(run_dir)  # può essere None
    if M is None:
        M_plot = np.zeros((1, pi_true_seq.shape[0]), dtype=float)
        M_sem = None
    else:
        M_plot = M
        M_sem = None  # s.e.m. non disponibile round-wise

    # phase metric locale: mean m
    phase_ws = [w]
    phase_metric = [float(np.mean(M_plot))] if M_plot is not None else [np.nan]

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
        # pi sequences
        try:
            items = collect_round_metrics(rd)
            pi_true_seq, pi_hat_seq = _load_pi_sequences_from_items(items)
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = build_argparser().parse_args()
    base_out = ensure_dir(Path(args.outdir))

    # 1) Caricamento dataset
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

    # 2) HyperParams e schedule
    hp = make_hparams_from_args(args, N_data=N_data)
    pis = make_schedule_from_args(hp, args)

    # Meta run-level
    write_json(base_out / "hyperparams.json", hp.__dict__)
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
            _create_panel4x_and_scatter(run_dir, w=float(hp.w), simplex_style=str(args.simplex_style))
        except Exception as e:
            print(f"[Plot] Pannello/Scatter non creati per {run_dir}: {e}")

        # === AGGREGAZIONE SU W (stesso seed e stesse classi): lag–amp e trittico forgetting/plasticity ===
        try:
            _create_lag_amp_and_forgetting(run_dir)
        except Exception as e:
            print(f"[Plot] Aggregati lag/amp & forgetting/plasticity non creati per {run_dir.parent}: {e}")

    print("[Exp06-FMNIST] Completato.")


if __name__ == "__main__":
    main()
