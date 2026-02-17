"""
Panel 9 - Separable noise / deformed-MP (plotting)
==================================================

Renders the four sub-panels described in the protocol:
  9A) Distribution of bulk λ_max with MP vs deformed edges.
  9B) False-outlier rate (MP vs deformed) grouped by noise setting.
  9C) Mean K_eff (±IQR) for MP vs deformed with reference lines.
  9D) Effect of whitening on K_eff compared to the true # spikes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

COLORS = {
    "mp": "#CC6677",
    "def": "#4477AA",
    "white": "#117733",
    "grid": "#B0B0B0",
    "bulk": "#8888AA",
}

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})


def load_results(path: Path) -> List[Dict]:
    blob = np.load(path, allow_pickle=True)
    return [dict(item) for item in blob["results"]]


def label_condition(cond: Dict) -> str:
    noise = cond["condition"]["noise"]
    if noise["kind"] == "ar":
        label = rf"AR($\rho={noise['rho']:.1f}$)"
    else:
        label = "Block-var"
    q = cond["condition"]["q"]
    K = cond["condition"]["K"]
    return f"{label}, q={q:.2f}, K={K}"


def aggregate(results: List[Dict]) -> Dict[str, Dict]:
    grouped: Dict[str, Dict] = {}
    for cond in results:
        label = label_condition(cond)
        entry = grouped.setdefault(label, {
            "mp": {},
            "def": {},
            "white": {},
            "mp_edges": [],
            "def_edges": [],
            "K": cond["condition"]["K"],
        })
        for method in ("mp", "def", "white"):
            for metric, values in cond["metrics"][method].items():
                entry[method].setdefault(metric, []).append(np.asarray(values))
        entry["mp_edges"].append(np.full_like(cond["metrics"]["mp"]["bulk_edge"], cond["thresholds"]["mp_edge_cushioned"]))
        entry["def_edges"].append(np.full_like(cond["metrics"]["def"]["bulk_edge"], cond["thresholds"]["def_edge_cushioned"]))

    aggregated = {}
    for label, entry in grouped.items():
        agg_entry = {"K": entry["K"], "mp_edge": None, "def_edge": None}
        for method in ("mp", "def", "white"):
            agg_entry[method] = {}
            for metric, chunks in entry[method].items():
                agg_entry[method][metric] = np.concatenate(chunks) if chunks else np.empty(0)
        agg_entry["mp_edge"] = float(np.mean(np.concatenate(entry["mp_edges"])))
        agg_entry["def_edge"] = float(np.mean(np.concatenate(entry["def_edges"])))
        aggregated[label] = agg_entry
    return aggregated


def plot_violin_bulk(ax, aggregated: Dict[str, Dict]):
    labels = sorted(aggregated.keys())
    y_positions = np.arange(len(labels))
    for idx, label in enumerate(labels):
        vals = aggregated[label]["mp"]["bulk_edge"]
        vp = ax.violinplot([vals], positions=[idx], vert=False, widths=0.6, showmeans=False)
        for body in vp["bodies"]:
            body.set_facecolor(COLORS["bulk"])
            body.set_edgecolor("black")
            body.set_alpha(0.35)
        mp_edge = aggregated[label]["mp_edge"]
        def_edge = aggregated[label]["def_edge"]
        ax.vlines(mp_edge, idx - 0.3, idx + 0.3, colors=COLORS["mp"], linewidth=2, label="MP edge" if idx == 0 else "")
        ax.vlines(def_edge, idx - 0.3, idx + 0.3, colors=COLORS["def"], linewidth=2, label="Deformed edge" if idx == 0 else "", linestyles="--")

    ax.set_yticks(y_positions, labels)
    ax.set_xlabel(r"$\lambda_{\max}$ (bulk)")
    ax.set_title("9A) Bulk spectrum vs thresholds")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4, color=COLORS["grid"])
    ax.legend(loc="lower right")


def grouped_bar(ax, aggregated: Dict[str, Dict], metric: str, ylabel: str, title: str):
    labels = sorted(aggregated.keys())
    x = np.arange(len(labels))
    width = 0.3
    mp_vals = [np.mean(aggregated[label]["mp"][metric]) for label in labels]
    def_vals = [np.mean(aggregated[label]["def"][metric]) for label in labels]
    ax.bar(x - width/2, mp_vals, width=width, color=COLORS["mp"], label="MP")
    ax.bar(x + width/2, def_vals, width=width, color=COLORS["def"], label="Deformed")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color=COLORS["grid"])
    ax.legend()


def keff_bar(ax, aggregated: Dict[str, Dict]):
    labels = sorted(aggregated.keys())
    x = np.arange(len(labels))
    width = 0.25
    for method_idx, method in enumerate(("mp", "def")):
        centers = x + (method_idx - 0.5) * width
        means = [np.mean(aggregated[label][method]["keff"]) for label in labels]
        q25 = [np.percentile(aggregated[label][method]["keff"], 25) for label in labels]
        q75 = [np.percentile(aggregated[label][method]["keff"], 75) for label in labels]
        err_low = np.maximum(0.0, np.array(means) - np.array(q25))
        err_high = np.maximum(0.0, np.array(q75) - np.array(means))
        ax.bar(centers, means, width=width, color=COLORS[method], label=f"{method.upper()} K_eff")
        ax.errorbar(centers, means, yerr=[err_low, err_high], fmt="none", ecolor="#333333", capsize=3, linewidth=1)

    for idx, label in enumerate(labels):
        ax.hlines(
            aggregated[label]["K"],
            idx - 0.45,
            idx + 0.45,
            colors="#333333",
            linestyles=":",
            linewidth=1.1,
        )

    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel(r"$K_{\mathrm{eff}}$")
    ax.set_title("9C) Effective rank (mean ± IQR)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color=COLORS["grid"])
    ax.legend()


def plot_whitening(ax, aggregated: Dict[str, Dict]):
    labels = sorted(aggregated.keys())
    x = np.arange(len(labels))
    width = 0.4
    white_keff = [np.mean(aggregated[label]["white"]["keff"]) for label in labels]
    ax.bar(x, white_keff, width=width, color=COLORS["white"], label="Whitened K_eff")
    for idx, label in enumerate(labels):
        ax.hlines(
            aggregated[label]["K"],
            idx - 0.45,
            idx + 0.45,
            colors="#333333",
            linestyles=":",
            linewidth=1.1,
        )
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel(r"$K_{\mathrm{eff}}$")
    ax.set_title("9D) Whitening ablation")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color=COLORS["grid"])
    ax.legend()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Panel 9 figure.")
    parser.add_argument("--input", type=Path, default=Path("graphs/panel9/output/panel9_separable_data.npz"))
    parser.add_argument("--output", type=Path, default=Path("graphs/panel9/output/panel9_separable.pdf"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.input)
    aggregated = aggregate(results)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    plot_violin_bulk(axA, aggregated)
    grouped_bar(axB, aggregated, "false_outliers", "False outliers per run", "9B) False outlier rate")
    keff_bar(axC, aggregated)
    plot_whitening(axD, aggregated)

    #fig.suptitle("Panel 9 – Correlated noise: MP vs deformed edge", fontsize=15, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[Panel9] Figure saved to {args.output} (+ PNG)")


if __name__ == "__main__":
    main()
