#!/usr/bin/env python3
"""
================================================================================
Script: run_all_experiments.py
Description: Orchestrateur pour exécuter toutes les expériences et générer les
             tableaux pour l'article USENIX Security

EXPÉRIENCES:
  1. main_comparison   - Table 1 & 2: Baseline vs FL-NoSec vs FL-Full
  2. privacy_tradeoff  - Table 3: Privacy-utility trade-off (ε values)
  3. byzantine_attack  - Table 4: Byzantine resilience (via p5_security_evaluation.py)
  4. ablation_layers   - Table 5: GCN layer depth
  5. ablation_epochs   - Table 6: Local epochs per round
  6. ablation_dist     - Table 7: Label distribution

USAGE:
    python run_all_experiments.py                    # Tout exécuter
    python run_all_experiments.py --only main_comparison
    python run_all_experiments.py --only privacy_tradeoff
    python run_all_experiments.py --skip byzantine_attack
    python run_all_experiments.py --generate-tables  # Générer tableaux seulement
================================================================================
"""

import os
import sys
import subprocess
import argparse
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'experiments')

# Créer répertoires
os.makedirs(RESULTS_DIR, exist_ok=True)

# Chemins des scripts
FL_FLEXIBLE_SCRIPT = os.path.join(SCRIPTS_DIR, 'scripts', 'p3_train_federated_flexible.py')
BASELINE_SCRIPT = os.path.join(PROJECT_DIR, 'p2_train_baseline.py')
SECURITY_EVAL_SCRIPT = os.path.join(PROJECT_DIR, 'p5_security_evaluation.py')


# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

EXPERIMENTS = {
    # ==========================================================================
    # Table 1 & 2: Main Comparison
    # ==========================================================================
    'main_comparison': {
        'description': 'Compare Baseline, FL-NoSec, FL configurations',
        'runs': [
            {
                'name': 'baseline',
                'script': 'baseline',
                'args': ['--epochs', '150'],
                'estimated_time': 10
            },
            {
                'name': 'fl_nosec',
                'script': 'fl_flexible',
                'args': ['--no-server-dp', '--no-flame', '--no-adversarial',
                        '--experiment_name', 'fl_nosec'],
                'estimated_time': 8
            },
            {
                'name': 'fl_serverDP',
                'script': 'fl_flexible',
                'args': ['--epsilon', '8.0', '--no-flame', '--no-adversarial',
                        '--experiment_name', 'fl_serverDP'],
                'estimated_time': 8
            },
            {
                'name': 'fl_flame',
                'script': 'fl_flexible',
                'args': ['--no-server-dp', '--no-adversarial',
                        '--experiment_name', 'fl_flame'],
                'estimated_time': 8
            },
            {
                'name': 'fl_full',
                'script': 'fl_flexible',
                'args': ['--experiment_name', 'fl_full'],
                'estimated_time': 8
            }
        ]
    },
    
    # ==========================================================================
    # Table 3: Privacy-Utility Trade-off
    # ==========================================================================
    'privacy_tradeoff': {
        'description': 'Evaluate different epsilon values',
        'runs': [
            {
                'name': 'eps_inf',
                'script': 'fl_flexible',
                'args': ['--no-server-dp', '--experiment_name', 'privacy_eps_inf'],
                'estimated_time': 8
            },
            {
                'name': 'eps_16',
                'script': 'fl_flexible',
                'args': ['--epsilon', '16.0', '--experiment_name', 'privacy_eps_16'],
                'estimated_time': 8
            },
            {
                'name': 'eps_8',
                'script': 'fl_flexible',
                'args': ['--epsilon', '8.0', '--experiment_name', 'privacy_eps_8'],
                'estimated_time': 8
            },
            {
                'name': 'eps_4',
                'script': 'fl_flexible',
                'args': ['--epsilon', '4.0', '--experiment_name', 'privacy_eps_4'],
                'estimated_time': 8
            },
            {
                'name': 'eps_2',
                'script': 'fl_flexible',
                'args': ['--epsilon', '2.0', '--experiment_name', 'privacy_eps_2'],
                'estimated_time': 8
            },
            {
                'name': 'eps_1',
                'script': 'fl_flexible',
                'args': ['--epsilon', '1.0', '--experiment_name', 'privacy_eps_1'],
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
    # ==========================================================================
    'ablation_layers': {
        'description': 'Evaluate different number of GCN layers',
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
    # Table 6: Ablation - Local Epochs
    # ==========================================================================
    'ablation_epochs': {
        'description': 'Evaluate different local epochs per round',
        'runs': [
            {
                'name': 'epochs_1',
                'script': 'fl_flexible',
                'args': ['--local_epochs', '1', '--rounds', '35',
                        '--experiment_name', 'ablation_epochs_1'],
                'estimated_time': 6
            },
            {
                'name': 'epochs_3',
                'script': 'fl_flexible',
                'args': ['--local_epochs', '3', '--rounds', '20',
                        '--experiment_name', 'ablation_epochs_3'],
                'estimated_time': 8
            },
            {
                'name': 'epochs_5',
                'script': 'fl_flexible',
                'args': ['--local_epochs', '5', '--rounds', '15',
                        '--experiment_name', 'ablation_epochs_5'],
                'estimated_time': 8
            },
            {
                'name': 'epochs_7',
                'script': 'fl_flexible',
                'args': ['--local_epochs', '7', '--rounds', '12',
                        '--experiment_name', 'ablation_epochs_7'],
                'estimated_time': 8
            }
        ]
    },
    
    # ==========================================================================
    # Table 7: Ablation - Label Distribution
    # NOTE: Cette expérience nécessite une implémentation différente
    # La distribution de labels affecte la partition des données entre exchanges,
    # pas le split train/val/test. Désactivée pour l'instant.
    # ==========================================================================
    # 'ablation_distribution': {
    #     'description': 'Evaluate different label distributions',
    #     'runs': [...]
    # }
}


# ============================================================================
# RUNNER FUNCTIONS
# ============================================================================

def get_script_path(script_type: str) -> str:
    """Get the path to the script based on type."""
    if script_type == 'baseline':
        return BASELINE_SCRIPT
    elif script_type == 'fl_flexible':
        return FL_FLEXIBLE_SCRIPT
    elif script_type == 'security_eval':
        return SECURITY_EVAL_SCRIPT
    else:
        raise ValueError(f"Unknown script type: {script_type}")


def run_single_experiment(run_config: Dict, dry_run: bool = False) -> Dict:
    """Run a single experiment and return results."""
    
    name = run_config['name']
    script_type = run_config['script']
    args = run_config.get('args', [])
    
    script_path = get_script_path(script_type)
    cmd = ['python', script_path] + args
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return {'status': 'skipped', 'name': name}
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Afficher la sortie en temps réel
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
        description='Run all experiments for USENIX Security paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_experiments.py                      # Run all experiments
  python run_all_experiments.py --only main_comparison
  python run_all_experiments.py --only privacy_tradeoff ablation_layers
  python run_all_experiments.py --skip byzantine_attack
  python run_all_experiments.py --dry-run            # Show what would run
  python run_all_experiments.py --list               # List all experiments
        """
    )
    
    parser.add_argument('--only', nargs='+', default=None,
                        help='Run only these experiment groups')
    parser.add_argument('--skip', nargs='+', default=None,
                        help='Skip these experiment groups')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be executed without running')
    parser.add_argument('--list', action='store_true',
                        help='List all available experiments')
    parser.add_argument('--generate-tables', action='store_true',
                        help='Only generate LaTeX tables from existing results')
    
    args = parser.parse_args()
    
    # List experiments
    if args.list:
        print("\nAvailable experiment groups:")
        print("="*60)
        for name, config in EXPERIMENTS.items():
            n_runs = len(config['runs'])
            est_time = sum(r.get('estimated_time', 10) for r in config['runs'])
            print(f"\n  {name}")
            print(f"    Description: {config['description']}")
            print(f"    Runs: {n_runs}")
            print(f"    Estimated time: ~{est_time} minutes")
            print(f"    Run names: {', '.join(r['name'] for r in config['runs'])}")
        return
    
    # Generate tables only
    if args.generate_tables:
        print("Generating LaTeX tables from existing results...")
        generate_script = os.path.join(SCRIPTS_DIR, 'generate_latex_tables.py')
        subprocess.run(['python', generate_script])
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
    print("EXPERIMENT ORCHESTRATOR")
    print("="*70)
    print(f"\nExperiments to run: {experiments_to_run}")
    print(f"Estimated total time: ~{total_time} minutes ({total_time/60:.1f} hours)")
    print(f"Dry run: {args.dry_run}")
    
    if not args.dry_run:
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
        'experiments': experiments_to_run,
        'results': all_results,
        'total_elapsed_seconds': total_elapsed,
        'counts': {
            'success': success_count,
            'failed': fail_count,
            'skipped': skip_count
        }
    }
    
    summary_path = os.path.join(RESULTS_DIR, 'experiment_summary.json')
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
