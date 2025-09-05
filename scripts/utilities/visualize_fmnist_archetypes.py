"""Visualizza i 3 archetipi (classi 0,1,2) di Fashion-MNIST.

Genera due figure salvate in questa stessa cartella:
- fmnist_archetypes_grid.png            : griglia 1x3 dei prototipi binari (sign(mean) per classe)
- fmnist_archetypes_grid_density.png    : per ogni classe (colonne 0..2) mostra (prototipo binario | densità media)

Approccio analogo a `visualize_mnist_archetypes.py`, riusando le funzioni generiche di binarizzazione
(`binarize_images`) e costruzione archetipi (`class_prototypes_sign_mean`).

Avvio: `python scripts/utilities/visualize_fmnist_archetypes.py`

Requisiti: torchvision (per caricare FashionMNIST) + seaborn, matplotlib.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path root progetto (assume struttura con src/ alla root)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]  # .../UNSUP
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Riuso funzioni da mnist_hfl (sono agnostiche rispetto al dataset)
from src.unsup.mnist_hfl import binarize_images, class_prototypes_sign_mean  # type: ignore

# ---------------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------------
CLASSES: Sequence[int] = [0, 1, 2]  # prime tre classi Fashion-MNIST
THRESHOLD: float = 0.5
SAVE_PATH = _THIS.parent / "fmnist_archetypes_grid.png"
SAVE_PATH_ALT = _THIS.parent / "fmnist_archetypes_grid_density.png"
DPI = 170
CMAP_BIN = sns.color_palette("icefire", as_cmap=True)
CMAP_DENS = "magma"

# ---------------------------------------------------------------------------
# Caricamento Fashion-MNIST
# ---------------------------------------------------------------------------

def load_fmnist(root: str, train: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Carica Fashion-MNIST tramite torchvision; restituisce (X, y) con X uint8 (num,28,28)."""
    try:
        from torchvision.datasets import FashionMNIST  # type: ignore
    except Exception as e:  # pragma: no cover - dipendenza opzionale
        raise ImportError(
            "torchvision non disponibile. Installa torchvision per usare questo script."
        ) from e
    ds = FashionMNIST(root=root, train=train, download=True)
    X = ds.data.numpy()
    y = ds.targets.numpy()
    return X, y


def compute_archetypes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restituisce (X_bin, y, archetypes) per le classi richieste.

    Usa solo il TRAIN set (come per MNIST nello script originale)."""
    X_train, y_train = load_fmnist(str(_PROJECT_ROOT / "data"), train=True)
    # Binarizza e crea archetipi
    X_bin = binarize_images(X_train, THRESHOLD)
    arch = class_prototypes_sign_mean(X_bin, y_train, classes=CLASSES)
    return X_bin, y_train, arch


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grid(archetypes: np.ndarray) -> None:
    sns.set_style("white")
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.4))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        a = archetypes[i].reshape(28, 28)
        im = ax.imshow(a, cmap=CMAP_BIN, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"Classe {CLASSES[i]}", fontsize=10)
        ax.axis("off")
    cbar_ax = fig.add_axes((0.92, 0.15, 0.015, 0.7))
    fig.colorbar(im, cax=cbar_ax, label="valore binario")
    fig.suptitle("Archetipi Fashion-MNIST (sign(mean), classi 0..2)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 0.9, 0.95))
    fig.savefig(SAVE_PATH, dpi=DPI)
    plt.close(fig)


def plot_with_density(X_bin: np.ndarray, y: np.ndarray, archetypes: np.ndarray) -> None:
    # Layout: per ciascuna classe due colonne (prot, dens) => 1 x 6 figure
    sns.set_style("white")
    fig, axes = plt.subplots(1, 6, figsize=(11.5, 2.6))
    for col, c in enumerate(CLASSES):
        a_bin = archetypes[col].reshape(28, 28)
        mask = (y == c)
        class_imgs = X_bin[mask]
        dens = (class_imgs.mean(axis=0) + 1.0) / 2.0  # {±1} -> {0,1}
        dens = dens.reshape(28, 28)
        ax_bin = axes[col * 2]
        ax_den = axes[col * 2 + 1]
        ax_bin.imshow(a_bin, cmap=CMAP_BIN, vmin=-1, vmax=1)
        ax_bin.set_title(f"{c} prot", fontsize=9)
        ax_bin.axis("off")
        im = ax_den.imshow(dens, cmap=CMAP_DENS, vmin=0.0, vmax=1.0)
        ax_den.set_title("dens", fontsize=9)
        ax_den.axis("off")
    cbar_ax = fig.add_axes((0.93, 0.2, 0.012, 0.6))
    fig.colorbar(im, cax=cbar_ax, label="pixel mean")
    fig.suptitle("Archetipi vs densità (Fashion-MNIST 0..2)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 0.92, 0.95))
    fig.savefig(SAVE_PATH_ALT, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[INFO] Calcolo archetipi Fashion-MNIST (classi 0,1,2)...")
    X_bin, y, arch = compute_archetypes()
    print("[INFO] Archetipi shape:", arch.shape)
    plot_grid(arch)
    print(f"[PLOT] Salvato {SAVE_PATH}")
    plot_with_density(X_bin, y, arch)
    print(f"[PLOT] Salvato {SAVE_PATH_ALT}")
    print("[DONE] Visualizzazione completata.")


if __name__ == "__main__":
    main()
