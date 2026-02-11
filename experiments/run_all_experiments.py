#!/usr/bin/env python3
"""
================================================================================
Script: run_all_experiments_2layers.py
Description: Orchestrateur pour exécuter toutes les expériences et générer les
             tableaux pour l'article USENIX Security

ARCHITECTURE: 2 GCN Layers (optimal per ablation study - +3.28% forward accuracy)
FAIR COMPARISON: Baseline 60 epochs = FL 20 rounds × 3 local epochs

SCRIPTS UTILISÉS (tous à la racine du projet):
  - p2_train_baseline_2layers.py        (Baseline centralisé, 2 GCN layers)
  - p3_train_federated_2layers.py       (FL sans sécurité, 2 GCN layers)
  - p3_train_federated_secure_2layers.py (FL avec DP+FLAME, 2 GCN layers)
  - p5_security_evaluation_2layers.py   (Évaluation sécurité, 2 GCN layers)

EXPÉRIENCES:
  1. main_comparison   - Table 1 & 2: Baseline vs FL-NoSec vs FL-Secure
  2. privacy_tradeoff  - Table 3: Privacy-utility trade-off (ε values)
  3. byzantine_attack  - Table 4: Byzantine resilience
  4. ablation_layers   - Table 5: GCN layer depth (1-5 layers)
  5. ablation_epochs   - Table 6: Local epochs per round

USAGE:
    python run_all_experiments_2layers.py                    # Tout exécuter
    python run_all_experiments_2layers.py --only main_comparison
    python run_all_experiments_2layers.py --skip byzantine_attack
    python run_all_experiments_2layers.py --dry-run          # Voir sans exécuter
    python run_all_experiments_2layers.py --list             # Lister les expériences
================================================================================
"""

import os
import sys
import subprocess
import argparse
import json
import time
from datetime import datetime
from typing import List, Dict, Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))  # experiments/
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)                 # bitcoin_fl_project/
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'experiments')

# Créer répertoires
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# CHEMINS DES SCRIPTS (tous à la racine du projet, version 2 layers)
# ============================================================================
BASELINE_SCRIPT = os.path.join(PROJECT_DIR, 'p2_train_baseline_2layers.py')
FL_NOSEC_SCRIPT = os.path.join(PROJECT_DIR, 'p3_train_federated_2layers.py')
FL_SECURE_SCRIPT = os.path.join(PROJECT_DIR, 'p3_train_federated_secure_2layers.py')
SECURITY_EVAL_SCRIPT = os.path.join(PROJECT_DIR, 'p5_security_evaluation_2layers.py')

# Pour ablation layers, on utilise le script flexible s'il existe
FL_FLEXIBLE_SCRIPT = os.path.join(SCRIPTS_DIR, 'scripts', 'p3_train_federated_flexible.py')


# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

EXPERIMENTS = {
    # ==========================================================================
    # Table 1 & 2: Main Comparison (2-layer architecture, 60 epoch-equivalent)
    # ==========================================================================
    'main_comparison': {
        'description': 'Compare Baseline vs FL-NoSec vs FL-Secure (all 2 GCN layers)',
        'runs': [
            {
                'name': 'baseline',
                'script': 'baseline',
                'args': ['--epochs', '60'],
                'estimated_time': 10
            },
            {
                'name': 'fl_nosec',
                'script': 'fl_nosec',
                'args': [],  # Default: 20 rounds × 3 local epochs
                'estimated_time': 8
            },
            {
                'name': 'fl_secure',
                'script': 'fl_secure',
                'args': [],  # Default: DP + FLAME + adversarial
                'estimated_time': 10
            }
        ]
    },
    
    # ==========================================================================
    # Table 3: Privacy-Utility Trade-off (2-layer architecture)
    # NOTE: Nécessite que p3_train_federated_secure_2layers.py accepte --epsilon
    # ==========================================================================
    'privacy_tradeoff': {
        'description': 'Evaluate different epsilon values (server-side DP)',
        'runs': [
            {
                'name': 'eps_inf',
                'script': 'fl_secure',
                'args': ['--no-server-dp'],
                'estimated_time': 8
            },
            {
                'name': 'eps_16',
                'script': 'fl_secure',
                'args': ['--epsilon', '16.0'],
                'estimated_time': 8
            },
            {
                'name': 'eps_8',
                'script': 'fl_secure',
                'args': ['--epsilon', '8.0'],
                'estimated_time': 8
            },
            {
                'name': 'eps_4',
                'script': 'fl_secure',
                'args': ['--epsilon', '4.0'],
                'estimated_time': 8
            },
            {
                'name': 'eps_2',
                'script': 'fl_secure',
                'args': ['--epsilon', '2.0'],
                'estimated_time': 8
            },
            {
                'name': 'eps_1',
                'script': 'fl_secure',
                'args': ['--epsilon', '1.0'],
                'estimated_time': 8
            }
        ]
    },
    
    # ==========================================================================
    # Table 4: Byzantine Attack Resilience
    # ==========================================================================
    'byzantine_attack': {
        'description': 'Evaluate FLAME defense against Byzantine attacks',
        'runs': [
            {
                'name': 'byzantine_eval',
                'script': 'security_eval',
                'args': ['--attack', 'byzantine'],
                'estimated_time': 15
            }
        ]
    },
    
    # ==========================================================================
    # Table 5: Ablation - GCN Layers
    # NOTE: Utilise p3_train_federated_flexible.py avec --num_layers variable
    # ==========================================================================
    'ablation_layers': {
        'description': 'Evaluate different number of GCN layers (1-5)',
        'runs': [
            {
                'name': 'layers_1',
                'script': 'fl_flexible',
                'args': ['--num_layers', '1', '--experiment_name', 'ablation_layers_1'],
                'estimated_time': 6
            },
            {
                'name': 'layers_2',
                'script': 'fl_flexible',
                'args': ['--num_layers', '2', '--experiment_name', 'ablation_layers_2'],
                'estimated_time': 7
            },
            {
                'name': 'layers_3',
                'script': 'fl_flexible',
                'args': ['--num_layers', '3', '--experiment_name', 'ablation_layers_3'],
                'estimated_time': 8
            },
            {
                'name': 'layers_4',
                'script': 'fl_flexible',
                'args': ['--num_layers', '4', '--experiment_name', 'ablation_layers_4'],
                'estimated_time': 9
            },
            {
                'name': 'layers_5',
                'script': 'fl_flexible',
                'args': ['--num_layers', '5', '--experiment_name', 'ablation_layers_5'],
                'estimated_time': 10
            }
        ]
    },
    
    # ==========================================================================
    # Table 6: Ablation - Local Epochs (2-layer architecture)
    # ==========================================================================
    'ablation_epochs': {
        'description': 'Evaluate different local epochs per round',
        'runs': [
            {
                'name': 'epochs_1',
                'script': 'fl_secure',
                'args': ['--local_epochs', '1', '--rounds', '60'],
                'estimated_time': 8
            },
            {
                'name': 'epochs_3',
                'script': 'fl_secure',
                'args': ['--local_epochs', '3', '--rounds', '20'],
                'estimated_time': 8
            },
            {
                'name': 'epochs_5',
                'script': 'fl_secure',
                'args': ['--local_epochs', '5', '--rounds', '12'],
                'estimated_time': 8
            }
        ]
    },
}


# ============================================================================
# RUNNER FUNCTIONS
# ============================================================================

def get_script_path(script_type: str) -> str:
    """Get the path to the script based on type."""
    scripts = {
        'baseline': BASELINE_SCRIPT,
        'fl_nosec': FL_NOSEC_SCRIPT,
        'fl_secure': FL_SECURE_SCRIPT,
        'fl_flexible': FL_FLEXIBLE_SCRIPT,
        'security_eval': SECURITY_EVAL_SCRIPT,
    }
    
    if script_type not in scripts:
        raise ValueError(f"Unknown script type: {script_type}")
    
    return scripts[script_type]


def check_scripts_exist():
    """Verify all required scripts exist."""
    scripts = {
        'Baseline (2 layers)': BASELINE_SCRIPT,
        'FL NoSec (2 layers)': FL_NOSEC_SCRIPT,
        'FL Secure (2 layers)': FL_SECURE_SCRIPT,
        'Security Eval (2 layers)': SECURITY_EVAL_SCRIPT,
    }
    
    missing = []
    for name, path in scripts.items():
        if not os.path.exists(path):
            missing.append(f"  ✗ {name}: {path}")
    
    if missing:
        print("\n⚠️  MISSING SCRIPTS:")
        for m in missing:
            print(m)
        print("\nPlease ensure all 2-layer scripts are in the project root.")
        return False
    
    return True


def run_single_experiment(run_config: Dict, dry_run: bool = False) -> Dict:
    """Run a single experiment and return results."""
    
    name = run_config['name']
    script_type = run_config['script']
    args = run_config.get('args', [])
    
    script_path = get_script_path(script_type)
    cmd = ['python', script_path] + args
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Script:  {os.path.basename(script_path)}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return {'status': 'skipped', 'name': name}
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"  ✗ Script not found: {script_path}")
        return {'status': 'failed', 'name': name, 'error': 'Script not found'}
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {name} completed in {elapsed/60:.1f} minutes")
        
        return {
            'status': 'success',
            'name': name,
            'elapsed_seconds': elapsed
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {name} FAILED after {elapsed/60:.1f} minutes")
        print(f"  Error: {e}")
        
        return {
            'status': 'failed',
            'name': name,
            'elapsed_seconds': elapsed,
            'error': str(e)
        }


def run_experiment_group(group_name: str, dry_run: bool = False) -> List[Dict]:
    """Run all experiments in a group."""
    
    if group_name not in EXPERIMENTS:
        print(f"Unknown experiment group: {group_name}")
        return []
    
    group = EXPERIMENTS[group_name]
    print(f"\n{'#'*70}")
    print(f"EXPERIMENT GROUP: {group_name}")
    print(f"Description: {group['description']}")
    print(f"Runs: {len(group['runs'])}")
    
    total_time = sum(r.get('estimated_time', 10) for r in group['runs'])
    print(f"Estimated time: ~{total_time} minutes")
    print(f"{'#'*70}")
    
    results = []
    for run_config in group['runs']:
        result = run_single_experiment(run_config, dry_run)
        results.append(result)
    
    return results


def estimate_total_time(experiments: List[str]) -> int:
    """Estimate total runtime in minutes."""
    total = 0
    for exp_name in experiments:
        if exp_name in EXPERIMENTS:
            for run in EXPERIMENTS[exp_name]['runs']:
                total += run.get('estimated_time', 10)
    return total


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run all experiments for USENIX Security paper (2-layer GCN)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_experiments_2layers.py                      # Run all
  python run_all_experiments_2layers.py --only main_comparison
  python run_all_experiments_2layers.py --skip byzantine_attack
  python run_all_experiments_2layers.py --dry-run            # Show commands
  python run_all_experiments_2layers.py --list               # List experiments
        """
    )
    
    parser.add_argument('--only', nargs='+', default=None,
                        help='Run only these experiment groups')
    parser.add_argument('--skip', nargs='+', default=None,
                        help='Skip these experiment groups')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show commands without running')
    parser.add_argument('--list', action='store_true',
                        help='List all available experiments')
    parser.add_argument('--generate-tables', action='store_true',
                        help='Generate LaTeX tables from existing results')
    
    args = parser.parse_args()
    
    # List experiments
    if args.list:
        print("\n" + "="*60)
        print("AVAILABLE EXPERIMENTS (2-layer GCN architecture)")
        print("="*60)
        for name, config in EXPERIMENTS.items():
            n_runs = len(config['runs'])
            est_time = sum(r.get('estimated_time', 10) for r in config['runs'])
            print(f"\n  {name}")
            print(f"    Description: {config['description']}")
            print(f"    Runs: {n_runs}")
            print(f"    Estimated time: ~{est_time} minutes")
            print(f"    Run names: {', '.join(r['name'] for r in config['runs'])}")
        
        print("\n" + "="*60)
        print("SCRIPTS USED:")
        print("="*60)
        print(f"  Baseline:      {BASELINE_SCRIPT}")
        print(f"  FL NoSec:      {FL_NOSEC_SCRIPT}")
        print(f"  FL Secure:     {FL_SECURE_SCRIPT}")
        print(f"  Security Eval: {SECURITY_EVAL_SCRIPT}")
        print(f"  FL Flexible:   {FL_FLEXIBLE_SCRIPT}")
        return
    
    # Generate tables only
    if args.generate_tables:
        print("Generating LaTeX tables from existing results...")
        generate_script = os.path.join(SCRIPTS_DIR, 'generate_latex_tables.py')
        if os.path.exists(generate_script):
            subprocess.run(['python', generate_script])
        else:
            print(f"Table generation script not found: {generate_script}")
        return
    
    # Determine which experiments to run
    if args.only:
        experiments_to_run = args.only
    else:
        experiments_to_run = list(EXPERIMENTS.keys())
    
    if args.skip:
        experiments_to_run = [e for e in experiments_to_run if e not in args.skip]
    
    # Estimate time
    total_time = estimate_total_time(experiments_to_run)
    
    print("\n" + "="*70)
    print("EXPERIMENT ORCHESTRATOR (2-LAYER GCN ARCHITECTURE)")
    print("="*70)
    print(f"\nArchitecture: 2 GCN layers (optimal per ablation study)")
    print(f"Fair comparison: Baseline 60 epochs = FL 20 rounds × 3 epochs")
    print(f"\nScripts (all 2-layer versions):")
    print(f"  Baseline: {os.path.basename(BASELINE_SCRIPT)}")
    print(f"  FL NoSec: {os.path.basename(FL_NOSEC_SCRIPT)}")
    print(f"  FL Secure: {os.path.basename(FL_SECURE_SCRIPT)}")
    print(f"\nExperiments to run: {experiments_to_run}")
    print(f"Estimated total time: ~{total_time} minutes ({total_time/60:.1f} hours)")
    print(f"Dry run: {args.dry_run}")
    
    # Check scripts exist
    if not args.dry_run:
        if not check_scripts_exist():
            print("\n⚠️  Some scripts are missing. Aborting.")
            sys.exit(1)
        
        print("\n⚠️  Starting experiments. Press Ctrl+C to abort.")
        time.sleep(3)
    
    # Run experiments
    all_results = {}
    start_time = time.time()
    
    for exp_name in experiments_to_run:
        results = run_experiment_group(exp_name, args.dry_run)
        all_results[exp_name] = results
    
    total_elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")
        for r in results:
            status = r['status']
            name = r['name']
            
            if status == 'success':
                elapsed = r.get('elapsed_seconds', 0)
                print(f"  ✓ {name} ({elapsed/60:.1f} min)")
                success_count += 1
            elif status == 'failed':
                print(f"  ✗ {name} FAILED")
                fail_count += 1
            else:
                print(f"  ○ {name} (skipped)")
                skip_count += 1
    
    print(f"\n" + "-"*40)
    print(f"Total: {success_count} succeeded, {fail_count} failed, {skip_count} skipped")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'architecture': '2_layers',
        'experiments': experiments_to_run,
        'results': all_results,
        'total_elapsed_seconds': total_elapsed,
        'counts': {
            'success': success_count,
            'failed': fail_count,
            'skipped': skip_count
        }
    }
    
    summary_path = os.path.join(RESULTS_DIR, 'experiment_summary_2layers.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved: {summary_path}")
    
    # Generate tables if all succeeded
    if fail_count == 0 and not args.dry_run:
        print("\n" + "="*70)
        print("GENERATING LATEX TABLES")
        print("="*70)
        generate_script = os.path.join(SCRIPTS_DIR, 'generate_latex_tables.py')
        if os.path.exists(generate_script):
            subprocess.run(['python', generate_script])
        else:
            print("Table generation script not found. Run separately.")


if __name__ == "__main__":
    main()
