"""
Batch runner for multiple BBP experiments.

Creates multiple configurations and runs them sequentially.
"""
import json
import subprocess
from pathlib import Path
import numpy as np


def create_config_variations():
    """Generate multiple config variations for parameter sweep."""
    
    base_config = {
        "N": 400,
        "K": 3,
        "r_ex": 0.95,
        "M_values": [50, 100, 200, 400, 800, 1600],
        "alpha": [0.5, 0.3, 0.2],
        "sharp_alpha": 0.5,
        "seed": 2025
    }
    
    configs = []
    
    # Variation 1: Different r_ex values
    for r_ex in [0.85, 0.90, 0.95, 0.99]:
        config = base_config.copy()
        config['r_ex'] = r_ex
        config['seed'] = 2025 + int(r_ex * 100)
        configs.append((f"rex_{int(r_ex*100)}", config))
    
    # Variation 2: Different K values
    for K in [2, 3, 5]:
        config = base_config.copy()
        config['K'] = K
        # Uniform exposure
        config['alpha'] = [1.0/K] * K
        config['seed'] = 2025 + K
        configs.append((f"K_{K}_uniform", config))
    
    # Variation 3: Different exposure patterns
    exposure_patterns = {
        "uniform": [1/3, 1/3, 1/3],
        "skewed": [0.7, 0.2, 0.1],
        "extreme": [0.9, 0.08, 0.02]
    }
    for name, alpha in exposure_patterns.items():
        config = base_config.copy()
        config['alpha'] = alpha
        config['seed'] = 2025 + hash(name) % 1000
        configs.append((f"exposure_{name}", config))
    
    # Variation 4: Different sharpening strengths
    for sharp_alpha in [0.1, 0.5, 0.9]:
        config = base_config.copy()
        config['sharp_alpha'] = sharp_alpha
        config['seed'] = 2025 + int(sharp_alpha * 100)
        configs.append((f"sharp_{int(sharp_alpha*10)}", config))
    
    return configs


def run_batch_experiments(output_dir: Path = None):
    """Run all config variations."""
    
    if output_dir is None:
        output_dir = Path(__file__).parent / 'output' / 'batch_experiments'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs = create_config_variations()
    
    print("="*70)
    print(f"BATCH EXPERIMENT RUNNER")
    print("="*70)
    print(f"Total experiments: {len(configs)}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    print()
    
    # Save all configs
    configs_dir = output_dir / 'configs'
    configs_dir.mkdir(exist_ok=True)
    
    for i, (name, config) in enumerate(configs, 1):
        config_path = configs_dir / f"{name}.json"
        
        # Convert numpy arrays to lists
        config_json = {}
        for key, val in config.items():
            if isinstance(val, (list, np.ndarray)):
                config_json[key] = list(val) if isinstance(val, np.ndarray) else val
            else:
                config_json[key] = val
        
        with open(config_path, 'w') as f:
            json.dump(config_json, f, indent=2)
        
        print(f"[{i}/{len(configs)}] Running: {name}")
        print(f"  Config: {config_path}")
        
        # Run experiment
        cmd = [
            'python',
            str(Path(__file__).parent / 'run_panel2_experiment.py'),
            '--config', str(config_path),
            '--output', str(output_dir / 'results')
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ Success")
        else:
            print(f"  ❌ Failed")
            print(f"  Error: {result.stderr}")
        
        print()
    
    print("="*70)
    print("✅ BATCH EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"\nResults saved in: {output_dir / 'results'}")
    print(f"Configs saved in: {configs_dir}")


if __name__ == '__main__':
    run_batch_experiments()
