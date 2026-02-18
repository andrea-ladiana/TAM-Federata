#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point — Client-Aware Adaptive Weight Experiment
=====================================================

Run from project root:

    python client_aware_w/run.py

or with custom parameters:

    python client_aware_w/run.py --T 30 --r-good 0.7 --alpha-ema 0.3

Outputs are saved to  client_aware_w/output/ .
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Ensure project root is on sys.path ──
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from client_aware_w.simulation import SimConfig, run_simulation, run_multi_seed
from client_aware_w.visualization import plot_panel, plot_panel_multiseed


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Client-Aware Adaptive Weight — Federated Unsupervised Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--N", type=int, default=1000, help="Neuron dimension")
    p.add_argument("--K", type=int, default=3, help="Number of archetypes")
    p.add_argument("--L", type=int, default=5, help="Number of clients (last one is attacker)")
    p.add_argument("--T", type=int, default=10, help="Number of rounds")
    p.add_argument("--M-per", type=int, default=800, help="Examples per client per round")
    p.add_argument("--r-good", type=float, default=0.90, help="Signal quality for good clients")
    p.add_argument("--r-bad", type=float, default=0.1, help="Signal quality for attacker (0 = pure noise)")
    p.add_argument("--alpha-ema", type=float, default=0.5, help="EMA coefficient for w smoothing")
    p.add_argument("--prop-iters", type=int, default=30, help="Propagation iterations")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--n-seeds", type=int, default=20,
                   help="Number of independent seeds for mean±SE aggregation (1 = single-seed mode)")
    p.add_argument("--outdir", type=str, default=None, help="Output directory (default: client_aware_w/output/)")
    return p


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    # Resolve output directory
    if args.outdir is not None:
        out_dir = Path(args.outdir)
    else:
        out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build config
    cfg = SimConfig(
        N=args.N,
        K=args.K,
        L=args.L,
        T=args.T,
        M_per=args.M_per,
        r_good=args.r_good,
        r_bad=args.r_bad,
        attacker_idx=args.L - 1,  # last client is attacker
        alpha_ema=args.alpha_ema,
        prop_iters=args.prop_iters,
        seed=args.seed,
    )

    print("=" * 60)
    print("  Client-Aware Adaptive Weight Experiment")
    print("=" * 60)
    print(f"  N={cfg.N}  K={cfg.K}  L={cfg.L}  T={cfg.T}  M/client/round={cfg.M_per}")
    print(f"  r_good={cfg.r_good}  r_bad={cfg.r_bad}  attacker=client_{cfg.attacker_idx}")
    print(f"  alpha_ema={cfg.alpha_ema}  seed={cfg.seed}  n_seeds={args.n_seeds}")
    print(f"  output → {out_dir}")
    print("=" * 60)

    n_seeds = args.n_seeds

    if n_seeds <= 1:
        # ── Single-seed mode ──
        result = run_simulation(cfg)

        np.save(out_dir / "w_history.npy", result.w_history)
        np.save(out_dir / "retrieval_history.npy", result.retrieval_history)
        np.save(out_dir / "mag_history.npy", result.mag_history)
        np.save(out_dir / "H_history.npy", result.H_history)
        np.save(out_dir / "p_history.npy", result.p_history)

        cfg_dict = {
            "N": cfg.N, "K": cfg.K, "L": cfg.L, "T": cfg.T,
            "M_per": cfg.M_per, "r_good": cfg.r_good, "r_bad": cfg.r_bad,
            "attacker_idx": cfg.attacker_idx, "alpha_ema": cfg.alpha_ema,
            "prop_iters": cfg.prop_iters, "seed": cfg.seed, "n_seeds": 1,
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(cfg_dict, f, indent=2)

        fig_path = out_dir / "panel_client_aware_w.png"
        plot_panel(
            w_history=result.w_history,
            retrieval_history=result.retrieval_history,
            mag_history=result.mag_history,
            attacker_idx=cfg.attacker_idx,
            out_path=fig_path,
        )

        print("\n" + "=" * 60)
        print("  Done!  (single seed)")
        print(f"  w_good (final mean) = {result.w_history[:cfg.attacker_idx, -1].mean():.4f}")
        print(f"  w_attacker (final)  = {result.w_history[cfg.attacker_idx, -1]:.4f}")
        print(f"  retrieval (final)   = {result.retrieval_history[-1]:.4f}")
        print("=" * 60)

    else:
        # ── Multi-seed mode ──
        ms = run_multi_seed(cfg, n_seeds=n_seeds, seed_base=cfg.seed, verbose=True)

        # Save aggregated arrays
        np.save(out_dir / "w_good_mean.npy", ms.w_good_mean)
        np.save(out_dir / "w_good_se.npy", ms.w_good_se)
        np.save(out_dir / "w_att_mean.npy", ms.w_att_mean)
        np.save(out_dir / "w_att_se.npy", ms.w_att_se)
        np.save(out_dir / "retr_mean.npy", ms.retr_mean)
        np.save(out_dir / "retr_se.npy", ms.retr_se)
        np.save(out_dir / "mag_mean.npy", ms.mag_mean)
        np.save(out_dir / "mag_se.npy", ms.mag_se)

        cfg_dict = {
            "N": cfg.N, "K": cfg.K, "L": cfg.L, "T": cfg.T,
            "M_per": cfg.M_per, "r_good": cfg.r_good, "r_bad": cfg.r_bad,
            "attacker_idx": cfg.attacker_idx, "alpha_ema": cfg.alpha_ema,
            "prop_iters": cfg.prop_iters, "seed_base": cfg.seed,
            "n_seeds": n_seeds,
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(cfg_dict, f, indent=2)

        fig_path = out_dir / "panel_client_aware_w.png"
        plot_panel_multiseed(
            ms, out_path=fig_path, r_good_label=cfg.r_good,
        )

        print("\n" + "=" * 60)
        print(f"  Done!  ({n_seeds} seeds)")
        print(f"  w_good (final, mean±se) = {ms.w_good_mean[-1]:.4f} ± {ms.w_good_se[-1]:.4f}")
        print(f"  w_att  (final, mean±se) = {ms.w_att_mean[-1]:.4f} ± {ms.w_att_se[-1]:.4f}")
        print(f"  retr   (final, mean±se) = {ms.retr_mean[-1]:.4f} ± {ms.retr_se[-1]:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
