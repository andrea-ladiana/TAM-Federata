"""
Interactive archetype selector for structured dataset.

This script shows all 9 archetypes with their pairwise overlaps,
allowing manual selection of K=6 and K=3 subsets.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_path))

from graphs.structured.prep.load_structured_dataset import load_structured_archetypes


def visualize_all_archetypes(archetypi, filenames):
    """Display all archetypes in a grid with indices."""
    K, N = archetypi.shape
    side = int(np.sqrt(N))
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for k in range(K):
        pattern = archetypi[k].reshape(side, side)
        axes[k].imshow(pattern, cmap='gray', vmin=-1, vmax=1)
        axes[k].axis('off')
        
        # Title with index and filename
        short_name = filenames[k].replace('.png', '').replace('_', ' ')
        axes[k].set_title(f"[{k}] {short_name}", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("archetype_selector_preview.png", dpi=150, bbox_inches='tight')
    print("Saved preview to archetype_selector_preview.png")
    plt.show()


def show_overlap_matrix(archetypi, filenames):
    """Display overlap matrix with labels."""
    K, N = archetypi.shape
    overlaps = (archetypi @ archetypi.T) / float(N)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(overlaps, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Labels
    short_names = [f"[{i}] {f.replace('.png', '')[:10]}" for i, f in enumerate(filenames)]
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    
    # Add values
    for i in range(K):
        for j in range(K):
            text = ax.text(j, i, f'{overlaps[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if abs(overlaps[i, j]) > 0.5 else "black",
                          fontsize=7)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Overlap', fontsize=10)
    
    ax.set_title('Pairwise Overlap Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("archetype_overlap_matrix.png", dpi=150, bbox_inches='tight')
    print("Saved overlap matrix to archetype_overlap_matrix.png")
    plt.show()


def interactive_selection():
    """Interactive archetype selection."""
    
    # Load all archetypes
    archetypi, filenames = load_structured_archetypes()
    K_full, N = archetypi.shape
    
    print("\n" + "="*70)
    print("INTERACTIVE ARCHETYPE SELECTOR")
    print("="*70)
    print(f"\nTotal archetypes available: {K_full}")
    print("\nArchetypes:")
    for i, fname in enumerate(filenames):
        print(f"  [{i}] {fname.replace('.png', '')}")
    
    # Compute and display overlaps
    overlaps = (archetypi @ archetypi.T) / float(N)
    print("\nPairwise overlaps (off-diagonal):")
    for i in range(K_full):
        for j in range(i+1, K_full):
            print(f"  [{i}] ↔ [{j}]: {overlaps[i,j]:+.3f}")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_all_archetypes(archetypi, filenames)
    show_overlap_matrix(archetypi, filenames)
    
    print("\n" + "="*70)
    print("SELECTION PHASE")
    print("="*70)
    
    # Select K=6
    print("\n--- SELECT K=6 ARCHETYPES ---")
    print("Enter 6 indices separated by spaces (e.g., '0 1 2 3 4 5')")
    print("Tip: Choose archetypes with LOW mutual overlaps")
    
    while True:
        try:
            input_k6 = input("\nK=6 selection: ").strip()
            selected_k6 = [int(x) for x in input_k6.split()]
            
            if len(selected_k6) != 6:
                print(f"ERROR: Need exactly 6 indices, got {len(selected_k6)}")
                continue
            
            if any(x < 0 or x >= K_full for x in selected_k6):
                print(f"ERROR: Indices must be in range [0, {K_full-1}]")
                continue
            
            if len(set(selected_k6)) != 6:
                print("ERROR: Indices must be unique")
                continue
            
            # Compute average overlap in subset
            selected_k6_sorted = sorted(selected_k6)
            subset_overlaps = overlaps[np.ix_(selected_k6_sorted, selected_k6_sorted)]
            np.fill_diagonal(subset_overlaps, 0.0)
            avg_overlap = np.abs(subset_overlaps).mean()
            
            print(f"\nSelected K=6: {selected_k6_sorted}")
            print(f"Files: {[filenames[i].replace('.png', '') for i in selected_k6_sorted]}")
            print(f"Average absolute overlap: {avg_overlap:.3f}")
            
            confirm = input("Confirm? (y/n): ").strip().lower()
            if confirm == 'y':
                break
        except ValueError:
            print("ERROR: Invalid input. Use space-separated integers.")
    
    # Select K=3 (subset of K=6)
    print("\n--- SELECT K=3 ARCHETYPES ---")
    print(f"Choose 3 from your K=6 selection: {selected_k6_sorted}")
    print("Enter 3 indices (e.g., '0 2 5')")
    
    while True:
        try:
            input_k3 = input("\nK=3 selection: ").strip()
            selected_k3_idx = [int(x) for x in input_k3.split()]
            
            if len(selected_k3_idx) != 3:
                print(f"ERROR: Need exactly 3 indices, got {len(selected_k3_idx)}")
                continue
            
            if any(x not in selected_k6_sorted for x in selected_k3_idx):
                print(f"ERROR: Indices must be from K=6 selection: {selected_k6_sorted}")
                continue
            
            if len(set(selected_k3_idx)) != 3:
                print("ERROR: Indices must be unique")
                continue
            
            # Compute average overlap
            selected_k3_sorted = sorted(selected_k3_idx)
            subset_overlaps_k3 = overlaps[np.ix_(selected_k3_sorted, selected_k3_sorted)]
            np.fill_diagonal(subset_overlaps_k3, 0.0)
            avg_overlap_k3 = np.abs(subset_overlaps_k3).mean()
            
            print(f"\nSelected K=3: {selected_k3_sorted}")
            print(f"Files: {[filenames[i].replace('.png', '') for i in selected_k3_sorted]}")
            print(f"Average absolute overlap: {avg_overlap_k3:.3f}")
            
            confirm = input("Confirm? (y/n): ").strip().lower()
            if confirm == 'y':
                break
        except ValueError:
            print("ERROR: Invalid input. Use space-separated integers.")
    
    # Save selections
    np.save("data/structured-dataset/selected_k6.npy", np.array(selected_k6_sorted))
    np.save("data/structured-dataset/selected_k3.npy", np.array(selected_k3_sorted))
    
    print("\n" + "="*70)
    print("SAVED!")
    print("="*70)
    print(f"K=6 selection saved to: data/structured-dataset/selected_k6.npy")
    print(f"K=3 selection saved to: data/structured-dataset/selected_k3.npy")
    print("\nYou can now run the federated experiment with these selections.")
    print("="*70)


if __name__ == "__main__":
    interactive_selection()
