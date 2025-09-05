"""Visualizza i 9 archetipi (cifre 1..9) usati negli esperimenti MNIST.

Genera un pannello estetico (seaborn + matplotlib) con:
- Griglia 3x3 dei prototipi binari (sign(mean) per classe)
- Heatmap con scala colore divergente (-1=nero/blu, +1=giallo)
- Variante opzionale: overlay della densità media (valore medio in [0,1])

Output predefinito: out_01/mnist_single/archetypes_grid.png
Avvio: `python scripts/utilities/visualize_mnist_archetypes.py`

Requisiti: torchvision (per caricare MNIST) oppure file già scaricati.
"""
from __future__ import annotations

import sys, os
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Sequence

# ---------------------------------------------------------------------------
# Path root progetto per import (assume struttura con src/ alla root)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]  # .../UNSUP
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.unsup.mnist_hfl import load_mnist, binarize_images, class_prototypes_sign_mean  # type: ignore

# ---------------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------------
CLASSES: Sequence[int] = list(range(1, 10))  # 1..9
THRESHOLD = 0.5
OUT_DIR = Path("scripts/utilities")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = OUT_DIR / "archetypes_grid.png"
SAVE_PATH_ALT = OUT_DIR / "archetypes_grid_density.png"
DPI = 170
CMAP_BIN = sns.color_palette("icefire", as_cmap=True)  # divergente gradevole
CMAP_DENS = "magma"


def compute_archetypes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restituisce (X_bin, y, archetypes) dove archetypes=(9,784)."""
    X, y = load_mnist("./data", train=True)
    X_bin = binarize_images(X, THRESHOLD)
    arch = class_prototypes_sign_mean(X_bin, y, classes=CLASSES)
    return X_bin, y, arch


def plot_grid(archetypes: np.ndarray) -> None:
    """Griglia 3x3 con titoli; valori in {±1}."""
    sns.set_style("white")
    fig, axes = plt.subplots(3, 3, figsize=(6.5, 6.3))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        a = archetypes[i].reshape(28, 28)
        im = ax.imshow(a, cmap=CMAP_BIN, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"Classe {CLASSES[i]}", fontsize=10)
        ax.axis("off")
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    fig.colorbar(im, cax=cbar_ax, label="valore binario")
    fig.suptitle("Archetipi MNIST (sign(mean), cifre 1..9)", fontsize=13)
    plt.tight_layout(rect=(0, 0, 0.9, 0.97))
    fig.savefig(SAVE_PATH, dpi=DPI)
    plt.close(fig)


def plot_with_density(X_bin: np.ndarray, y: np.ndarray, archetypes: np.ndarray) -> None:
    """Crea una seconda visualizzazione: per ogni classe mostra a sinistra il prototipo binario e a destra la densità (mean pixel in [0,1])."""
    sns.set_style("white")
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.4))  # 2 colonne per classe -> 18 assi
    for row in range(3):
        for col in range(3):
            idx = row*3 + col
            c = CLASSES[idx]
            a_bin = archetypes[idx].reshape(28, 28)
            # maschera immagini classe corrente e densità (mean raw binarizzato -> rimappiamo -1/+1 a 0/1)
            mask = (y == c)
            class_imgs = X_bin[mask]
            # densità: media in [0,1]
            dens = (class_imgs.mean(axis=0)+1.0)/2.0  # {±1} -> {0,1}
            dens = dens.reshape(28, 28)
            ax_bin = axes[row, col*2]
            ax_den = axes[row, col*2+1]
            ax_bin.imshow(a_bin, cmap=CMAP_BIN, vmin=-1, vmax=1)
            ax_bin.set_title(f"{c} prot", fontsize=9)
            ax_bin.axis('off')
            im = ax_den.imshow(dens, cmap=CMAP_DENS, vmin=0.0, vmax=1.0)
            ax_den.set_title("dens", fontsize=9)
            ax_den.axis('off')
    # colorbar densità
    cbar_ax = fig.add_axes((0.93, 0.15, 0.015, 0.7))
    fig.colorbar(im, cax=cbar_ax, label="pixel mean")
    fig.suptitle("Archetipi vs densità media per classe", fontsize=14)
    plt.tight_layout(rect=(0, 0, 0.92, 0.97))
    fig.savefig(SAVE_PATH_ALT, dpi=DPI)
    plt.close(fig)


def main() -> None:
    print("[INFO] Calcolo archetipi MNIST (1..9)...")
    X_bin, y, arch = compute_archetypes()
    print("[INFO] Archetipi shape:", arch.shape)
    plot_grid(arch)
    print(f"[PLOT] Salvato {SAVE_PATH}")
    plot_with_density(X_bin, y, arch)
    print(f"[PLOT] Salvato {SAVE_PATH_ALT}")
    print("[DONE] Visualizzazione completata.")


if __name__ == "__main__":
    main()
