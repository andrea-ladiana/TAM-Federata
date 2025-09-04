#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp-06 (single-only) — REPORT AGGREGATO
---------------------------------------
Usa i moduli:
  - src/exp06_single/reporting.build_run_report
  - src/exp06_single/hopfield_hooks.load_magnetization_matrix_from_run
  - src/exp06_single/io (ensure_dir, read_json, write_json, find_files, list_round_dirs)

Funzioni:
  1) Scansiona una o più radici (--roots) per trovare run contenenti 'round_XXX/'
  2) Per ogni run, costruisce/ricarica un report locale (JSON + CSV round metrics + CSV magnetizzazione)
  3) Aggrega i report su più run in un riassunto globale (JSON + CSV)
  4) (Opzionale) Media le matrici di magnetizzazione compatibili e salva 'mean_magnetization.csv'

Esempi:
  # Aggrega tutte le run sotto 'runs/exp06/synth' e 'runs/exp06/fmnist'
  python scripts/exp06_single_report.py \
      --roots runs/exp06/synth runs/exp06/fmnist \
      --rebuild-per-run \
      --merge-magnetization

  # Aggrega solo le run con pattern 'seed_*/w_*' sotto la radice indicata
  python scripts/exp06_single_report.py \
      --roots runs/exp06/synth \
      --glob "seed_*/w_*" \
      --rebuild-per-run
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Sequence

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
# Import dai moduli del progetto
# ---------------------------------------------------------------------
from src.mixing.reporting import build_run_report  # noqa: E402
from src.mixing.hopfield_hooks import load_magnetization_matrix_from_run  # noqa: E402
from src.mixing.io import (  # noqa: E402
    ensure_dir, read_json, write_json, find_files, list_round_dirs
)


# ---------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exp-06 (single-only) — Report Aggregato",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--roots", type=str, nargs="+", required=True,
                   help="Una o più cartelle radice da scansionare per trovare run (contengono round_XXX/).")
    p.add_argument("--glob", type=str, default=None,
                   help="Pattern glob opzionale per filtrare le sottocartelle di run (es. 'seed_*/w_*').")
    p.add_argument("--max-depth", type=int, default=4,
                   help="Profondità massima di scansione ricorsiva sotto ciascuna radice.")
    p.add_argument("--rebuild-per-run", action="store_true",
                   help="Ricostruisce il report per ciascuna run (JSON/CSV locali) anche se già presenti.")
    p.add_argument("--merge-magnetization", action="store_true",
                   help="Se presente, calcola la media delle matrici di magnetizzazione compatibili e la salva.")
    p.add_argument("--outdir", type=str, default=None,
                   help="Cartella di output aggregato. Default: '<prima_root>/results_aggregate'.")

    return p


# ---------------------------------------------------------------------
# Scoperta run
# ---------------------------------------------------------------------
def _is_run_dir(p: Path) -> bool:
    """Una 'run' è una directory che contiene almeno una sottocartella 'round_XXX'."""
    if not p.exists() or not p.is_dir():
        return False
    rounds = list_round_dirs(p)
    return len(rounds) > 0


def _iter_candidate_dirs(root: Path, max_depth: int) -> List[Path]:
    """
    Ritorna tutte le directory fino a 'max_depth' livelli sotto 'root'.
    """
    cand: List[Path] = []
    root = Path(root)
    if not root.exists():
        return cand
    # livello 0: la root stessa
    cand.append(root)
    # walk manuale con controllo profondità
    queue = [(root, 0)]
    while queue:
        base, d = queue.pop(0)
        if d >= max_depth:
            continue
        for child in base.iterdir():
            if child.is_dir():
                cand.append(child)
                queue.append((child, d + 1))
    return cand


def discover_runs(roots: Sequence[str | Path], glob: Optional[str], max_depth: int) -> List[Path]:
    runs: List[Path] = []
    for r in roots:
        base = Path(r)
        if not base.exists():
            continue
        cands = _iter_candidate_dirs(base, max_depth=max_depth)
        if glob:
            import fnmatch
            cands = [p for p in cands if fnmatch.fnmatch(p.name, glob)]
        for p in cands:
            if _is_run_dir(p):
                runs.append(p)
    # univocizza preservando ordine
    seen = set()
    uniq = []
    for p in runs:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


# ---------------------------------------------------------------------
# Lettura meta/iperparametri utili per il report aggregato
# ---------------------------------------------------------------------
def _read_hparams(run_dir: Path) -> Dict[str, Any]:
    hp_file = run_dir / "hyperparams.json"
    if hp_file.exists():
        try:
            return read_json(hp_file)
        except Exception:
            pass
    # a volte salvato da CLI come summary_cli.json nella root della run
    cli_file = run_dir / "summary_cli.json"
    if cli_file.exists():
        try:
            obj = read_json(cli_file)
            if isinstance(obj, dict) and "summary_out" in obj and "hp" in obj["summary_out"]:
                return obj["summary_out"]["hp"]
        except Exception:
            pass
    return {}


def _read_schedule_meta(run_dir: Path) -> Dict[str, Any]:
    sm = run_dir.parent.parent / "schedule_meta.json"  # spesso in radice per seed multi-run
    if sm.exists():
        try:
            return read_json(sm)
        except Exception:
            pass
    sm2 = run_dir / "schedule_meta.json"
    if sm2.exists():
        try:
            return read_json(sm2)
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------
# Aggregazione
# ---------------------------------------------------------------------
def _aggregate_rows_to_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # normalizza chiavi (unione di tutte)
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    keys = sorted(all_keys)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _summarize_reports(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Crea statistiche aggregate semplici (mean/median/std per alcune metriche).
    """
    def _collect(field: str) -> List[float]:
        vals = []
        for r in rows:
            if field in r and r[field] is not None:
                try:
                    vals.append(float(r[field]))
                except Exception:
                    pass
        return vals

    tv_mean = _collect("TV_pi_mean")
    tv_median = _collect("TV_pi_median")
    keff_mean = _collect("K_eff_mean")
    keff_median = _collect("K_eff_median")
    lag_rounds = _collect("lag_rounds")
    lag_radians = _collect("lag_radians")
    amp_ratio = _collect("amp_ratio")

    def agg(x: List[float]) -> Dict[str, float]:
        if not x:
            return {"mean": np.nan, "median": np.nan, "std": np.nan, "n": 0}
        x_arr = np.asarray(x, dtype=float)
        return {
            "mean": float(np.mean(x_arr)),
            "median": float(np.median(x_arr)),
            "std": float(np.std(x_arr, ddof=1)) if x_arr.size > 1 else 0.0,
            "n": int(x_arr.size),
        }

    return {
        "TV_pi_mean_stats": agg(tv_mean),
        "TV_pi_median_stats": agg(tv_median),
        "K_eff_mean_stats": agg(keff_mean),
        "K_eff_median_stats": agg(keff_median),
        "lag_rounds_stats": agg(lag_rounds),
        "lag_radians_stats": agg(lag_radians),
        "amp_ratio_stats": agg(amp_ratio),
        "n_runs": int(len(rows)),
    }


def _merge_magnetizations(run_dirs: List[Path], out_csv_path: Path) -> Optional[Tuple[int, int]]:
    """
    Carica le matrici M(K,T) da ciascuna run, ne calcola la media sui run compatibili
    (stesso K e T; oppure si allinea al minimo T comune tagliando in coda), e salva in CSV.
    Ritorna (K, T_agg) oppure None se non trovate matrici compatibili.
    """
    mats: List[np.ndarray] = []
    K_common: Optional[int] = None
    T_min: Optional[int] = None

    for rd in run_dirs:
        M = load_magnetization_matrix_from_run(rd)
        if M is None:
            continue
        K, T = M.shape
        if K_common is None:
            K_common = K
        elif K != K_common:
            # se K differisce, salta la run per omogeneità
            continue
        T_min = T if T_min is None else min(T_min, T)
        mats.append(M)

    if not mats or K_common is None or T_min is None or T_min <= 0:
        return None

    # tronca tutte le matrici a T_min
    trunc = [M[:, :T_min] for M in mats]
    mean_M = np.mean(np.stack(trunc, axis=0), axis=0)  # (K, T_min)

    # salva CSV
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["mu"] + [f"t{t:03d}" for t in range(T_min)]
        w.writerow(header)
        for mu in range(K_common):
            w.writerow([mu] + [f"{float(mean_M[mu, t]):.6f}" for t in range(T_min)])

    return int(K_common), int(T_min)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = build_argparser().parse_args()

    roots = [Path(r) for r in args.roots]
    run_dirs = discover_runs(roots, glob=args.glob, max_depth=int(args.max_depth))
    if not run_dirs:
        print("[Exp06-REPORT] Nessuna run trovata.", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else ensure_dir(roots[0] / "results_aggregate")
    ensure_dir(outdir)

    # Per ogni run: ricostruisci o carica report locale
    rows: List[Dict[str, Any]] = []
    for rd in run_dirs:
        print(f"[Exp06-REPORT] Run: {rd}")
        try:
            rep = build_run_report(
                run_dir=str(rd),
                write_json_report=bool(args.rebuild_per_run),
                write_csv_round_metrics=bool(args.rebuild_per_run),
                write_csv_magnetization=bool(args.rebuild_per_run),
                exposure_mask=None,
            )
        except Exception as e:
            print(f"  ! Errore build_run_report: {e}", file=sys.stderr)
            continue

        # Metadati utili (hp, schedule)
        hp = _read_hparams(rd)
        sch = _read_schedule_meta(rd)

        row: Dict[str, Any] = {
            "run_dir": str(rd),
            "L": hp.get("L", None),
            "K": hp.get("K", None),
            "N": hp.get("N", None),
            "T_rounds": hp.get("n_batch", None),
            "M_total": hp.get("M_total", None),
            "r_ex": hp.get("r_ex", None),
            "w": hp.get("w", None),
            "TV_pi_mean": rep.get("TV_pi_mean", None),
            "TV_pi_median": rep.get("TV_pi_median", None),
            "K_eff_mean": rep.get("K_eff_mean", None),
            "K_eff_median": rep.get("K_eff_median", None),
            "lag_rounds": rep.get("lag_rounds", None),
            "lag_radians": rep.get("lag_radians", None),
            "amp_ratio": rep.get("amp_ratio", None),
            "magnetization_shape": rep.get("magnetization_shape", None),
            "schedule_kind": sch.get("kind", None),
        }
        rows.append(row)

    # Salva CSV con una riga per run
    _aggregate_rows_to_csv(rows, outdir / "runs_summary.csv")

    # Salva JSON aggregato con statistiche
    summary = _summarize_reports(rows)
    write_json(outdir / "aggregate_summary.json", summary)

    # (Opzionale) Media delle magnetizzazioni tra run compatibili
    if args.merge_magnetization:
        mk = _merge_magnetizations(run_dirs, outdir / "mean_magnetization.csv")
        if mk is not None:
            Kc, Tc = mk
            print(f"[Exp06-REPORT] Media magnetizzazioni salvata ({Kc}x{Tc}).")
        else:
            print("[Exp06-REPORT] Nessuna magnetizzazione compatibile da aggregare.")

    print(f"[Exp06-REPORT] Completato. Output → {outdir}")


if __name__ == "__main__":
    main()
