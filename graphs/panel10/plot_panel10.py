"""
Panel 10 plotting script: DP-PCA vs privacy budget ε.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

COLORS = {
    "auc": "#1f77b4",
    "delta": "#d62728",
    "overlap": "#2ca02c",
    "spectra": {
        "1.0": "#9467bd",
        "4.0": "#8c564b",
        "inf": "#7f7f7f",
    },
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
    epsilons = data["epsilons"]
    auc = data["auc"]
    delta = data["delta_keff"]
    overlap = data["overlap"]
    spectra = data["spectra_example"].item()
    return epsilons, auc, delta, overlap, spectra


def aggregate_stats(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=1)
    q25 = np.nanpercentile(values, 25, axis=1)
    q75 = np.nanpercentile(values, 75, axis=1)
    return mean, q25, q75


def format_eps_labels(epsilons: np.ndarray):
    labels = []
    for eps in epsilons:
        if np.isinf(eps):
            labels.append(r"$\infty$")
        else:
            labels.append(f"{eps:g}")
    return labels


def plot_curve(ax, epsilons, stats, color, ylabel, title):
    x = np.arange(len(epsilons))
    mean, q25, q75 = stats
    ax.plot(x, mean, color=color, marker="o", linewidth=2)
    ax.fill_between(x, q25, q75, color=color, alpha=0.2)
    ax.set_xticks(x, format_eps_labels(epsilons))
    ax.set_xlabel(r"Privacy budget $\varepsilon_{\mathrm{tot}}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


def plot_spectra(ax, spectra: Dict):
    epsilons = spectra.get("epsilons", [])
    eigs_dict = spectra.get("eigs", {})
    for eps in epsilons:
        key = str(eps)
        if key not in eigs_dict:
            continue
        eigs = eigs_dict[key]
        top = np.sort(eigs)[::-1][:25]
        ax.plot(np.arange(1, len(top) + 1), top, label=f"$\\varepsilon={key}$",
                color=COLORS["spectra"].get(key, "#555555"))
    lam_plus = spectra.get("lambda_plus")
    if lam_plus is not None:
        ax.axhline((1 + 0) * lam_plus, color="#444444", linestyle="--", linewidth=1.0,
                   label=r"$\lambda_+^{MP}$")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title("10D) Spectral slice")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Panel 10 DP-PCA results.")
    parser.add_argument("--input", type=Path,
                        default=Path("graphs/panel10/output/panel10_dp_pca_data.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("graphs/panel10/output/panel10_dp_pca.pdf"))
    return parser.parse_args()


def main():
    args = parse_args()
    epsilons, auc, delta, overlap, spectra = load_data(args.input)

    stats_auc = aggregate_stats(auc)
    stats_delta = aggregate_stats(delta)
    stats_overlap = aggregate_stats(overlap)

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax_auc = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[0, 1])
    ax_overlap = fig.add_subplot(gs[1, 0])
    ax_spectrum = fig.add_subplot(gs[1, 1])

    plot_curve(ax_auc, epsilons, stats_auc, COLORS["auc"],
               "Novelty AUC", "10A) AUC vs ε")
    plot_curve(ax_delta, epsilons, stats_delta, COLORS["delta"],
               r"$\Delta K_{\mathrm{eff}}$", "10B) Effective-rank error")
    plot_curve(ax_overlap, epsilons, stats_overlap, COLORS["overlap"],
               "Mean overlap", "10C) Eigenvector overlap")
    plot_spectra(ax_spectrum, spectra)

    fig.suptitle("Panel 10 – DP-PCA privacy/utility trade-off", fontsize=15, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[Panel10] Figure saved to {args.output} (+ PNG)")


if __name__ == "__main__":
    main()
