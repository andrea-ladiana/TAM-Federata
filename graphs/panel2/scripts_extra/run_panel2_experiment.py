"""
Interactive Wrapper for BBP Panel 2 Generation

Permette di:
1. Configurare parametri (N, K, r_ex, M_values, exposure α, sharpening α)
2. Generare dati con quei parametri
3. Creare visualizzazioni con naming automatico basato su parametri

Usage:
    python run_panel2_experiment.py --interactive
    python run_panel2_experiment.py --config config.json
    python run_panel2_experiment.py --quick  # Default params
"""
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Import generation and plotting functions
from bbp_theorem_demo import run_exposure_sweep
from plot_panel2 import create_panel2_figure, generate_summary


def generate_experiment_name(params: dict) -> str:
    """
    Genera nome esperimento basato sui parametri.
    
    Formato: bbp_N{N}_K{K}_r{rex}_M{Mmin}-{Mmax}_a{alpha_str}_{timestamp}
    """
    N = params['N']
    K = params['K']
    r_ex = params['r_ex']
    M_values = params['M_values']
    alpha = params['alpha']
    
    # Converti r_ex in stringa senza punto (0.95 -> 95)
    rex_str = str(int(r_ex * 100))
    
    # Range M
    M_min, M_max = M_values[0], M_values[-1]
    
    # Alpha come stringa (es: [0.5,0.3,0.2] -> "50-30-20")
    alpha_str = "-".join([str(int(a*100)) for a in alpha])
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    name = f"bbp_N{N}_K{K}_r{rex_str}_M{M_min}-{M_max}_a{alpha_str}_{timestamp}"
    return name


def load_config_from_file(config_path: Path) -> dict:
    """Carica configurazione da file JSON."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Convert lists to numpy arrays
    if 'M_values' in config:
        config['M_values'] = np.array(config['M_values'])
    if 'alpha' in config:
        config['alpha'] = np.array(config['alpha'])
    
    return config


def save_config_to_file(params: dict, output_path: Path):
    """Salva configurazione in JSON."""
    # Convert numpy arrays to lists for JSON serialization
    config_json = {}
    for key, val in params.items():
        if isinstance(val, np.ndarray):
            config_json[key] = val.tolist()
        else:
            config_json[key] = val
    
    with open(output_path, 'w') as f:
        json.dump(config_json, f, indent=2)
    
    print(f"✅ Config saved to: {output_path}")


def interactive_config() -> dict:
    """Configurazione interattiva dei parametri."""
    print("="*70)
    print("INTERACTIVE CONFIGURATION")
    print("="*70)
    print()
    
    def get_input(prompt, default, type_func=str):
        val = input(f"{prompt} [default: {default}]: ").strip()
        if not val:
            return default
        try:
            return type_func(val)
        except:
            print(f"Invalid input, using default: {default}")
            return default
    
    # Basic parameters
    print("--- Basic Parameters ---")
    N = get_input("Dimension N", 400, int)
    K = get_input("Number of archetypes K", 3, int)
    r_ex = get_input("Rademacher parameter r_ex", 0.95, float)
    
    # M values
    print("\n--- Exposure Values ---")
    print("Enter M values (comma-separated, e.g., 50,100,200,400)")
    M_input = get_input("M values", "50,100,200,400,800,1600", str)
    M_values = np.array([int(m.strip()) for m in M_input.split(',')])
    
    # Alpha (exposure probabilities)
    print("\n--- Archetype Exposure ---")
    print(f"Enter {K} exposure probabilities (comma-separated, will be normalized)")
    alpha_input = get_input("Alpha values", "0.5,0.3,0.2", str)
    alpha = np.array([float(a.strip()) for a in alpha_input.split(',')])
    
    if len(alpha) != K:
        print(f"⚠️  Warning: Expected {K} values, got {len(alpha)}. Using default.")
        alpha = np.array([1.0/K] * K)
    
    alpha = alpha / alpha.sum()  # Normalize
    
    # Sharpening
    print("\n--- Sharpening Parameters ---")
    sharp_alpha = get_input("Sharpening strength alpha (0=none, 1=strong)", 0.5, float)
    
    # Trials
    print("\n--- Statistical Robustness ---")
    n_trials = get_input("Number of independent trials per M (for mean/std)", 10, int)
    
    # Seed
    print("\n--- Random Seed ---")
    seed = get_input("Random seed", 2025, int)
    
    params = {
        'N': N,
        'K': K,
        'r_ex': r_ex,
        'M_values': M_values,
        'alpha': alpha,
        'sharp_alpha': sharp_alpha,
        'n_trials': n_trials,
        'seed': seed
    }
    
    print()
    print("="*70)
    print("CONFIGURATION SUMMARY:")
    print("="*70)
    for key, val in params.items():
        print(f"  {key:15s} = {val}")
    print("="*70)
    print()
    
    confirm = input("Proceed with this configuration? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Aborted.")
        sys.exit(0)
    
    return params


def quick_config() -> dict:
    """Configurazione rapida con valori di default."""
    return {
        'N': 400,
        'K': 3,
        'r_ex': 0.95,
        'M_values': np.array([50, 100, 200, 400, 800, 1600]),
        'alpha': np.array([0.5, 0.3, 0.2]),
        'sharp_alpha': 0.5,
        'n_trials': 10,
        'seed': 2025
    }


def run_experiment(params: dict, output_dir: Path):
    """
    Esegue esperimento completo:
    1. Genera dati
    2. Salva dati
    3. Genera plot
    4. Salva configurazione
    """
    exp_name = generate_experiment_name(params)
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print("="*70)
    print()
    
    # 1. Generate data
    print("📊 Generating data...")
    data = run_exposure_sweep(
        N=params['N'],
        K=params['K'],
        r_ex=params['r_ex'],
        M_values=params['M_values'],
        alpha=params['alpha'],
        seed=params['seed'],
        sharp_alpha=params['sharp_alpha'],
        n_trials=params['n_trials']
    )
    
    # 2. Save data
    data_path = exp_dir / 'bbp_data.npz'
    np.savez(
        data_path,
        results=data['results'],
        xi=data['xi'],
        N=data['N'],
        K=data['K'],
        r_ex=data['r_ex'],
        M_values=data['M_values'],
        n_trials=data['n_trials'],
        allow_pickle=True
    )
    print(f"✅ Data saved to: {data_path}")
    
    # 3. Generate plot
    print("\n📈 Generating visualizations...")
    plot_path = exp_dir / 'panel2_visualization'
    
    # Load data (to use same interface as plot_panel2.py)
    loaded_data = np.load(data_path, allow_pickle=True)
    create_panel2_figure(loaded_data, plot_path)
    summary_lines = generate_summary(loaded_data)
    
    # 4. Save configuration
    config_path = exp_dir / 'config.json'
    save_config_to_file(params, config_path)
    
    # 5. Create summary report
    summary_path = exp_dir / 'SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write(f"BBP PANEL 2 EXPERIMENT\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"PARAMETERS:\n")
        for key, val in params.items():
            f.write(f"  {key:15s} = {val}\n")
        f.write(f"\n{'='*70}\n\n")
        f.write(f"OUTPUTS:\n")
        f.write(f"  - Data:   {data_path.name}\n")
        f.write(f"  - Plot:   {plot_path.name}.pdf / .png\n")
        f.write(f"  - Config: {config_path.name}\n")
        f.write(f"\n{'='*70}\n\n")
        f.write(f"QUICK VALIDATION (M={params['M_values'][0]}):\n")
        res = data['results'][params['M_values'][0]]
        
        # Handle both new format (with mean/std) and old format (backward compatibility)
        emp_pre = res['empirical']
        emp_post = res['empirical_sharp']
        
        if 'overlap_mean' in emp_pre:
            # New format with multiple trials
            f.write(f"  n_trials: {params['n_trials']}\n")
            f.write(f"  Empirical overlap (pre):  {emp_pre['overlap_mean']} ± {emp_pre['overlap_std']}\n")
            f.write(f"  Empirical overlap (post): {emp_post['overlap_mean']} ± {emp_post['overlap_std']}\n")
        else:
            # Old format (single trial)
            f.write(f"  Empirical overlap (pre):  {emp_pre['overlap']}\n")
            f.write(f"  Empirical overlap (post): {emp_post['overlap']}\n")
        
        f.write(f"  Theory gamma:             {res['theory']['gamma']}\n")
        f.write(f"\n{'='*70}\n\n")
        f.write("DETAILED PANEL SUMMARY\n")
        f.write(f"{'='*70}\n\n")
        for line in summary_lines:
            f.write(f"- {line}\n")
        f.write(f"\n{'='*70}\n")
    
    print(f"✅ Summary saved to: {summary_path}")
    
    print()
    print("="*70)
    print("✅ EXPERIMENT COMPLETED!")
    print("="*70)
    print(f"\nResults saved in: {exp_dir}")
    print(f"\nFiles generated:")
    print(f"  - bbp_data.npz           : Raw data")
    print(f"  - panel2_visualization.pdf : Figure (PDF)")
    print(f"  - panel2_visualization.png : Figure (PNG)")
    print(f"  - config.json            : Configuration")
    print(f"  - SUMMARY.txt            : Summary report")
    print("="*70)
    
    return exp_dir


def main():
    parser = argparse.ArgumentParser(
        description='BBP Panel 2 Experiment Wrapper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python run_panel2_experiment.py --interactive
  
  Quick run with defaults:
    python run_panel2_experiment.py --quick
  
  From config file:
    python run_panel2_experiment.py --config experiments/my_config.json
  
  Custom output directory:
    python run_panel2_experiment.py --quick --output my_experiments/
        """
    )
    
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive configuration mode')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick run with default parameters')
    parser.add_argument('--config', '-c', type=str,
                       help='Load configuration from JSON file')
    parser.add_argument('--output', '-o', type=str, default='output/experiments',
                       help='Output directory for results (default: output/experiments)')
    
    args = parser.parse_args()
    
    # Determine configuration mode
    if args.interactive:
        params = interactive_config()
    elif args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Error: Config file not found: {config_path}")
            sys.exit(1)
        print(f"Loading configuration from: {config_path}")
        params = load_config_from_file(config_path)
    elif args.quick:
        print("Using quick configuration (default parameters)")
        params = quick_config()
    else:
        print("No configuration mode specified. Use --interactive, --quick, or --config")
        parser.print_help()
        sys.exit(1)
    
    # Output directory
    output_dir = Path(__file__).parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    exp_dir = run_experiment(params, output_dir)
    
    print(f"\n🎉 Done! Open {exp_dir / 'panel2_visualization.png'} to see results.")


if __name__ == '__main__':
    main()
