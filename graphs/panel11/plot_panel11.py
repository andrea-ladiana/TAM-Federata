"""
Panel 11 plotting script: Byzantine attacks and persistence defenses.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

COLORS = {
    "baseline": "#d62728",
    "persistence": "#1f77b4",
    "combo": "#2ca02c",
}

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.linewidth": 1.0,
})


def load_data(path: Path):
    data = np.load(path, allow_pickle=True)
    defenses = [str(d) for d in data["defenses"]]
    roc = data["roc"].item()
    fpr_table = data["fpr_table"].item()
    delta_keff = data["delta_keff"].item()
    heatmap = data["heatmap"].item()
    return defenses, roc, fpr_table, delta_keff, heatmap


def plot_roc(ax, defenses: List[str], roc: Dict[str, Dict[str, np.ndarray]]):
    for name in defenses:
        curve = roc[name]
        ax.plot(curve["fpr"], curve["tpr"], label=name.title(), color=COLORS[name], linewidth=2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("11A) ROC – attack detection")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()


def plot_fpr_bars(ax, defenses: List[str], fpr_table: Dict[str, Dict[str, float]]):
    scenario_keys = sorted({k for table in fpr_table.values() for k in table if not k.startswith("c=0")})
    x = np.arange(len(scenario_keys))
    width = 0.25
    for idx, name in enumerate(defenses):
        offsets = x + (idx - (len(defenses) - 1) / 2) * width
        values = [fpr_table[name].get(sk, np.nan) for sk in scenario_keys]
        ax.bar(offsets, values, width=width, color=COLORS[name], label=name.title(), alpha=0.8)
    ax.set_xticks(x, scenario_keys, rotation=30, ha="right")
    ax.set_ylabel("FPR @ TPR=0.9")
    ax.set_title("11B) Operating-point FPR (by scenario)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()


def plot_delta_keff(ax, defenses: List[str], delta_keff: Dict[str, np.ndarray]):
    data = [delta_keff[name] for name in defenses]
    box = ax.boxplot(data, tick_labels=[name.title() for name in defenses],
                     patch_artist=True, showfliers=False)
    for patch, name in zip(box["boxes"], defenses):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.5)
        patch.set_edgecolor(COLORS[name])
    for median in box["medians"]:
        median.set_color("#333333")
    ax.set_ylabel(r"$\Delta K_{\mathrm{eff}}$")
    ax.set_title("11C) Effective rank error")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


def plot_heatmap(ax, heatmap: Dict):
    scores = np.asarray(heatmap.get("scores", []), dtype=np.float64)
    labels = np.asarray(heatmap.get("labels", []), dtype=np.int32)
    if scores.size == 0:
        ax.set_visible(False)
        return
    img = scores[None, :]
    im = ax.imshow(img, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticks([])
    ax.set_xlabel("Round")
    ax.set_title("11D) Cross-shard score $S_t$")
    positives = np.where(labels == 1)[0]
    ax.vlines(positives, 0, 0.9, color="white", linewidth=0.8, linestyles=":")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("S_t")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Panel 11 Byzantine defenses.")
    parser.add_argument("--input", type=Path,
                        default=Path("graphs/panel11/output/panel11_byzantine_data.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("graphs/panel11/output/panel11_byzantine.pdf"))
    return parser.parse_args()


def main():
    args = parse_args()
    defenses, roc, fpr_table, delta_keff, heatmap = load_data(args.input)

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    plot_roc(axA, defenses, roc)
    plot_fpr_bars(axB, defenses, fpr_table)
    plot_delta_keff(axC, defenses, delta_keff)
    plot_heatmap(axD, heatmap)

    fig.suptitle("Panel 11 – Fake spikes vs persistence defenses", fontsize=15, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[Panel11] Figure saved to {args.output} (+ PNG)")


if __name__ == "__main__":
    main()
