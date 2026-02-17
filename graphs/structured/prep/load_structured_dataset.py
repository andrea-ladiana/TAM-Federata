"""
Load structured dataset from PNG images.

This module loads 9 grayscale PNG images (28x28) from data/structured-dataset
and converts them to binary patterns {-1, +1} for use as archetypi in the
federated TAM pipeline.

Convention:
- White pixels (high intensity) → -1
- Black pixels (low intensity) → +1
"""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple


def load_structured_archetypes(
    data_dir: str = "data/structured-dataset",
    threshold: float = 0.5
) -> Tuple[np.ndarray, list]:
    """
    Load 9 structured archetypes from PNG images.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the PNG images
    threshold : float
        Threshold for binarization (0-1). Values above threshold → black (+1)
        
    Returns
    -------
    archetypi : np.ndarray
        Array of shape (K, N) where K=9, N=784 (28×28)
        Values are {-1, +1}
    filenames : list
        List of filenames used (for reference)
    """
    data_path = Path(data_dir)
    
    # Get all PNG files and sort them
    png_files = sorted([f for f in data_path.glob("*.png")])
    
    if len(png_files) != 9:
        raise ValueError(f"Expected 9 PNG files, found {len(png_files)}")
    
    archetypi = []
    filenames = []
    
    for png_file in png_files:
        # Load image as grayscale
        img = Image.open(png_file).convert('L')
        
        # Convert to numpy array (28, 28)
        img_array = np.array(img, dtype=float)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Binarize: values > threshold → black (+1), else white (-1)
        pattern = np.where(img_array > threshold, 1.0, -1.0)
        
        # Flatten to (784,)
        pattern_flat = pattern.flatten()
        
        archetypi.append(pattern_flat)
        filenames.append(png_file.name)
    
    # Stack to (K=9, N=784)
    archetypi = np.array(archetypi)
    
    print(f"Loaded {len(archetypi)} archetypes from {data_dir}")
    print(f"Shape: {archetypi.shape}")
    print(f"Files: {filenames}")
    
    # Verify binary values
    unique_vals = np.unique(archetypi)
    print(f"Unique values: {unique_vals}")
    
    return archetypi, filenames


def visualize_archetypes(archetypi: np.ndarray, filenames: list = None):
    """
    Visualize loaded archetypes (for debugging).
    
    Parameters
    ----------
    archetypi : np.ndarray
        Array of shape (K, N) with values {-1, +1}
    filenames : list, optional
        Filenames for titles
    """
    import matplotlib.pyplot as plt
    
    K = archetypi.shape[0]
    N = archetypi.shape[1]
    
    # Assume square images
    side = int(np.sqrt(N))
    
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()
    
    for k in range(K):
        pattern = archetypi[k].reshape(side, side)
        axes[k].imshow(pattern, cmap='gray', vmin=-1, vmax=1)
        axes[k].axis('off')
        
        if filenames is not None:
            # Extract short name
            short_name = filenames[k].replace('.png', '')[:15]
            axes[k].set_title(short_name, fontsize=8)
        else:
            axes[k].set_title(f"Archetype {k+1}")
    
    plt.tight_layout()
    plt.savefig("structured_archetypes_debug.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved debug visualization to structured_archetypes_debug.png")


if __name__ == "__main__":
    # Test loading
    archetypi, filenames = load_structured_archetypes()
    
    # Visualize
    visualize_archetypes(archetypi, filenames)
    
    # Save as .npy for quick loading
    np.save("data/structured-dataset/archetypi.npy", archetypi)
    print("Saved archetypi to data/structured-dataset/archetypi.npy")
