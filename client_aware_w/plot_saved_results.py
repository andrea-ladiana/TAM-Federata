#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Saved Results — Client-Aware Adaptive Weight
==================================================

Re-generate plots from saved .npy files without re-running simulations.

Usage:

    python client_aware_w/plot_saved_results.py

or specify custom paths:

    python client_aware_w/plot_saved_results.py --data-dir client_aware_w/output --out-path custom_panel.png

Set magnetization correction vector with --mag-correction:

    python client_aware_w/plot_saved_results.py --mag-correction 0.1 0.2 0.15 0.05 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── Ensure project root is on sys.path ──
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from client_aware_w.visualization import plot_panel, plot_panel_multiseed
from client_aware_w.simulation import MultiSeedResult, SimConfig


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-plot saved client-aware-w results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir", type=str, default="output",
        help="Directory containing saved .npy files",
    )
    p.add_argument(
        "--out-path", type=str, default=None,
        help="Output figure path (default: data-dir/panel_client_aware_w.png)",
    )
    p.add_argument(
        "--mag-correction", type=float, nargs="+", default=None,
        help="Magnetization correction vector (length T). Default: zeros(T)",
    )
    p.add_argument(
        "--mode", type=str, default="auto", choices=["auto", "single", "multi"],
        help="Plot mode: 'single' (single-seed), 'multi' (multi-seed), 'auto' (detect)",
    )
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    return p


# ──────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────

def load_config(data_dir: Path) -> dict:
    """Load config.json from data directory."""
    cfg_path = data_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return json.load(f)


def detect_mode(data_dir: Path) -> str:
    """Auto-detect if data is single-seed or multi-seed."""
    multi_files = ["w_good_mean.npy", "w_att_mean.npy", "retr_mean.npy"]
    if all((data_dir / f).exists() for f in multi_files):
        return "multi"
    single_files = ["w_history.npy", "retrieval_history.npy"]
    if all((data_dir / f).exists() for f in single_files):
        return "single"
    raise FileNotFoundError(
        f"Could not detect mode (missing expected .npy files in {data_dir})"
    )


def load_single_seed_data(data_dir: Path) -> dict:
    """Load single-seed simulation results."""
    return {
        "w_history": np.load(data_dir / "w_history.npy"),
        "retrieval_history": np.load(data_dir / "retrieval_history.npy"),
        "mag_history": np.load(data_dir / "mag_history.npy"),
    }


def load_multi_seed_data(data_dir: Path) -> dict:
    """Load multi-seed aggregated results."""
    return {
        "w_good_mean": np.load(data_dir / "w_good_mean.npy"),
        "w_good_se": np.load(data_dir / "w_good_se.npy"),
        "w_att_mean": np.load(data_dir / "w_att_mean.npy"),
        "w_att_se": np.load(data_dir / "w_att_se.npy"),
        "retr_mean": np.load(data_dir / "retr_mean.npy"),
        "retr_se": np.load(data_dir / "retr_se.npy"),
        "mag_mean": np.load(data_dir / "mag_mean.npy"),
        "mag_se": np.load(data_dir / "mag_se.npy"),
    }


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)

    # Load config
    cfg_dict = load_config(data_dir)
    print("=" * 60)
    print("  Plot Saved Results — Client-Aware Adaptive Weight")
    print("=" * 60)
    print(f"  Data directory: {data_dir}")
    print(f"  N={cfg_dict['N']}  K={cfg_dict['K']}  L={cfg_dict['L']}  T={cfg_dict['T']}")
    print(f"  r_good={cfg_dict['r_good']}  r_bad={cfg_dict['r_bad']}")
    print("=" * 60)

    # Detect or use specified mode
    mode = args.mode if args.mode != "auto" else detect_mode(data_dir)
    print(f"  Mode: {mode}")

    # Prepare magnetization correction vector
    T = cfg_dict["T"]
    if args.mag_correction is not None:
        mag_correction = np.array(args.mag_correction, dtype=np.float32)
        if len(mag_correction) != T:
            print(f"Error: mag_correction length ({len(mag_correction)}) != T ({T})")
            sys.exit(1)
        print(f"  Magnetization correction: {mag_correction}")
    else:
        mag_correction = np.zeros(T, dtype=np.float32)
        print(f"  Magnetization correction: zeros({T})")

    # Output path
    if args.out_path is not None:
        out_path = Path(args.out_path)
    else:
        out_path = data_dir / "panel_client_aware_w.png"

    print(f"  Output: {out_path}")
    print("=" * 60)

    # ── Plot based on mode ──
    if mode == "single":
        data = load_single_seed_data(data_dir)
        plot_panel(
            w_history=data["w_history"],
            retrieval_history=data["retrieval_history"],
            mag_history=data["mag_history"],
            attacker_idx=cfg_dict["attacker_idx"],
            mag_correction=mag_correction,
            out_path=out_path,
            dpi=args.dpi,
        )
        print("\n✓ Single-seed plot saved.")

    elif mode == "multi":
        data = load_multi_seed_data(data_dir)
        n_seeds = cfg_dict.get("n_seeds", 1)
        
        # Reconstruct MultiSeedResult
        ms = MultiSeedResult(
            w_good_mean=data["w_good_mean"],
            w_good_se=data["w_good_se"],
            w_att_mean=data["w_att_mean"],
            w_att_se=data["w_att_se"],
            retr_mean=data["retr_mean"],
            retr_se=data["retr_se"],
            mag_mean=data["mag_mean"],
            mag_se=data["mag_se"],
            n_seeds=n_seeds,
        )

        plot_panel_multiseed(
            ms=ms,
            mag_correction=mag_correction,
            out_path=out_path,
            dpi=args.dpi,
            r_good_label=cfg_dict["r_good"],
        )
        print(f"\n✓ Multi-seed plot saved ({n_seeds} seeds).")

    print("=" * 60)


if __name__ == "__main__":
    main()
