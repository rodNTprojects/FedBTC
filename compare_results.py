#!/usr/bin/env python3
"""
================================================================================
Script: compare_results.py (V2 - WITH VALIDITY CHECKS)
Description: Compare results with validation of experimental conditions
================================================================================

CHECKS PERFORMED:
1. Hyperparameter consistency (LR, hidden_dim, dropout, etc.)
2. Data split consistency (same seed, same ratio)
3. Feature count consistency
4. Epochs equivalence fairness

USAGE:
    python compare_results.py
    python compare_results.py --strict    # Fail on any inconsistency
    python compare_results.py --latex     # Generate LaTeX table
================================================================================
"""

import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Result file paths
RESULT_FILES = {
    'baseline': 'results/evaluations/baseline_results.json',
    'fl_nosec': 'results/evaluations/federated_results.json',
    'fl_secure': 'results/evaluations/federated_secure_results.json'
}

# Expected values for valid comparison
EXPECTED_CONFIG = {
    'split_seed': 42,
    'split_ratio': '60/20/20',
    'hidden_dim': 128,
    'dropout': 0.3,
    'learning_rate': 0.002,
    'num_categories': 6,
    'num_exchanges': 3,
    'focal_gamma': 2.0,
    'alpha_forward': 1.0,
    'alpha_backward': 1.5
}


class ValidationResult:
    def __init__(self):
        self.passed = []
        self.warnings = []
        self.errors = []
    
    def add_pass(self, msg: str):
        self.passed.append(msg)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def add_error(self, msg: str):
        self.errors.append(msg)
    
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    def print_report(self):
        print("\n" + "="*70)
        print("VALIDITY CHECK REPORT".center(70))
        print("="*70)
        
        if self.passed:
            print("\n✅ PASSED CHECKS:")
            for msg in self.passed:
                print(f"   ✓ {msg}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for msg in self.warnings:
                print(f"   ⚠ {msg}")
        
        if self.errors:
            print("\n❌ ERRORS (Comparison may be INVALID):")
            for msg in self.errors:
                print(f"   ✗ {msg}")
        
        print("\n" + "-"*70)
        if self.is_valid():
            print("VERDICT: ✅ Comparison is VALID (no critical errors)")
        else:
            print("VERDICT: ❌ Comparison may be INVALID - review errors above")
        print("-"*70)


def load_results(filepath: str) -> Optional[Dict]:
    """Load results from JSON file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def get_nested(d: Dict, *keys, default=None):
    """Safely get nested dictionary value"""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


def extract_config_for_validation(results: Dict, approach: str) -> Dict:
    """Extract configuration values for validation"""
    
    config = {}
    
    # Try new format first, then old format
    
    # Split seed
    config['split_seed'] = get_nested(results, 'data_config', 'split_seed') or \
                           get_nested(results, 'config', 'split_seed')
    
    # Split ratio
    config['split_ratio'] = get_nested(results, 'data_config', 'split_ratio') or \
                            get_nested(results, 'config', 'split_ratio')
    
    # Feature count
    config['feature_count'] = get_nested(results, 'data_config', 'feature_count') or \
                              get_nested(results, 'model_config', 'in_channels') or \
                              get_nested(results, 'config', 'in_channels')
    
    # Hidden dim
    config['hidden_dim'] = get_nested(results, 'model_config', 'hidden_dim') or \
                           get_nested(results, 'config', 'hidden_dim') or \
                           get_nested(results, 'config', 'model', 'hidden_dim')
    
    # Dropout
    config['dropout'] = get_nested(results, 'model_config', 'dropout') or \
                        get_nested(results, 'config', 'dropout') or \
                        get_nested(results, 'config', 'model', 'dropout')
    
    # Learning rate
    config['learning_rate'] = get_nested(results, 'training_config', 'learning_rate') or \
                              get_nested(results, 'config', 'training', 'lr') or \
                              get_nested(results, 'fl_config', 'lr')
    
    # Alpha forward/backward
    config['alpha_forward'] = get_nested(results, 'training_config', 'alpha_forward') or \
                              get_nested(results, 'config', 'loss', 'alpha_forward') or \
                              get_nested(results, 'fl_config', 'alpha_forward')
    
    config['alpha_backward'] = get_nested(results, 'training_config', 'alpha_backward') or \
                               get_nested(results, 'config', 'loss', 'alpha_backward') or \
                               get_nested(results, 'fl_config', 'alpha_backward')
    
    # Epochs equivalent
    if approach == 'baseline':
        config['epochs'] = get_nested(results, 'training_config', 'epochs') or \
                          get_nested(results, 'config', 'training', 'epochs')
        config['epochs_equivalent'] = config['epochs']
    else:
        rounds = get_nested(results, 'training_config', 'rounds') or \
                get_nested(results, 'fl_config', 'rounds') or 20
        local_epochs = get_nested(results, 'training_config', 'local_epochs') or \
                      get_nested(results, 'fl_config', 'local_epochs') or 3
        config['rounds'] = rounds
        config['local_epochs'] = local_epochs
        config['epochs_equivalent'] = rounds * local_epochs
    
    # Data sizes
    config['total_transactions'] = get_nested(results, 'data_config', 'total_transactions')
    config['train_size'] = get_nested(results, 'data_config', 'train_size')
    config['test_size'] = get_nested(results, 'data_config', 'test_size')
    
    return config


def check_hyperparameter_consistency(all_configs: Dict[str, Dict], validation: ValidationResult):
    """Check 1: Hyperparameter consistency across approaches"""
    
    print("\n" + "-"*70)
    print("CHECK 1: HYPERPARAMETER CONSISTENCY")
    print("-"*70)
    
    params_to_check = ['hidden_dim', 'dropout', 'learning_rate', 'alpha_forward', 'alpha_backward']
    
    for param in params_to_check:
        values = {}
        for approach, config in all_configs.items():
            if config.get(param) is not None:
                values[approach] = config[param]
        
        if len(set(values.values())) > 1:
            validation.add_error(f"{param}: INCONSISTENT - {values}")
            print(f"  ❌ {param}: {values}")
        elif len(values) > 0:
            validation.add_pass(f"{param}: {list(values.values())[0]} (consistent)")
            print(f"  ✓ {param}: {list(values.values())[0]}")
        else:
            validation.add_warning(f"{param}: Not found in results")
            print(f"  ⚠ {param}: Not found")


def check_data_split_consistency(all_configs: Dict[str, Dict], validation: ValidationResult):
    """Check 2: Data split consistency (seed and ratio)"""
    
    print("\n" + "-"*70)
    print("CHECK 2: DATA SPLIT CONSISTENCY")
    print("-"*70)
    
    # Check split seed
    seeds = {}
    for approach, config in all_configs.items():
        if config.get('split_seed') is not None:
            seeds[approach] = config['split_seed']
    
    if len(set(seeds.values())) > 1:
        validation.add_error(f"Split seed INCONSISTENT: {seeds} - POTENTIAL DATA LEAKAGE!")
        print(f"  ❌ Split seed: {seeds}")
        print(f"     ⚠️  CRITICAL: Different seeds mean different train/test splits!")
        print(f"     ⚠️  Results may not be comparable!")
    elif len(seeds) > 0:
        seed_val = list(seeds.values())[0]
        if seed_val == EXPECTED_CONFIG['split_seed']:
            validation.add_pass(f"Split seed: {seed_val} (consistent, matches expected)")
            print(f"  ✓ Split seed: {seed_val}")
        else:
            validation.add_warning(f"Split seed: {seed_val} (consistent but not default 42)")
            print(f"  ⚠ Split seed: {seed_val} (expected 42)")
    else:
        validation.add_warning("Split seed: Not recorded in results")
        print(f"  ⚠ Split seed: Not found - cannot verify!")
    
    # Check split ratio
    ratios = {}
    for approach, config in all_configs.items():
        if config.get('split_ratio') is not None:
            ratios[approach] = config['split_ratio']
    
    if len(set(ratios.values())) > 1:
        validation.add_error(f"Split ratio INCONSISTENT: {ratios}")
        print(f"  ❌ Split ratio: {ratios}")
    elif len(ratios) > 0:
        validation.add_pass(f"Split ratio: {list(ratios.values())[0]} (consistent)")
        print(f"  ✓ Split ratio: {list(ratios.values())[0]}")
    else:
        validation.add_warning("Split ratio: Not recorded")
        print(f"  ⚠ Split ratio: Not found")
    
    # Check data sizes
    print("\n  Data sizes:")
    for approach, config in all_configs.items():
        total = config.get('total_transactions', 'N/A')
        train = config.get('train_size', 'N/A')
        test = config.get('test_size', 'N/A')
        print(f"    {approach}: total={total}, train={train}, test={test}")


def check_feature_consistency(all_configs: Dict[str, Dict], validation: ValidationResult):
    """Check 3: Feature count consistency"""
    
    print("\n" + "-"*70)
    print("CHECK 3: FEATURE CONSISTENCY")
    print("-"*70)
    
    features = {}
    for approach, config in all_configs.items():
        if config.get('feature_count') is not None:
            features[approach] = config['feature_count']
    
    if len(set(features.values())) > 1:
        validation.add_error(f"Feature count INCONSISTENT: {features}")
        print(f"  ❌ Feature count: {features}")
    elif len(features) > 0:
        validation.add_pass(f"Feature count: {list(features.values())[0]} (consistent)")
        print(f"  ✓ Feature count: {list(features.values())[0]}")
    else:
        validation.add_warning("Feature count: Not recorded")
        print(f"  ⚠ Feature count: Not found")


def check_epochs_fairness(all_configs: Dict[str, Dict], validation: ValidationResult):
    """Check 4: Epochs equivalence for fair comparison"""
    
    print("\n" + "-"*70)
    print("CHECK 4: TRAINING EFFORT COMPARISON")
    print("-"*70)
    
    epochs = {}
    for approach, config in all_configs.items():
        eq = config.get('epochs_equivalent')
        if eq is not None:
            epochs[approach] = eq
            
            if approach == 'baseline':
                print(f"  {approach}: {eq} epochs")
            else:
                rounds = config.get('rounds', '?')
                local = config.get('local_epochs', '?')
                print(f"  {approach}: {rounds} rounds × {local} local epochs = {eq} equivalent")
    
    if 'baseline' in epochs:
        baseline_epochs = epochs['baseline']
        for approach, eq in epochs.items():
            if approach != 'baseline':
                ratio = baseline_epochs / eq if eq > 0 else float('inf')
                if ratio > 3:
                    validation.add_warning(
                        f"Baseline trained {ratio:.1f}x longer than {approach} "
                        f"({baseline_epochs} vs {eq} epochs)"
                    )
                    print(f"\n  ⚠ Baseline trained {ratio:.1f}x longer than {approach}")
                    print(f"    Consider comparing at similar epoch counts")


def check_reproducibility(all_results: Dict[str, Dict], validation: ValidationResult):
    """Check 5: Reproducibility indicators"""
    
    print("\n" + "-"*70)
    print("CHECK 5: REPRODUCIBILITY")
    print("-"*70)
    
    for approach, results in all_results.items():
        timestamp = get_nested(results, 'metadata', 'timestamp') or \
                   get_nested(results, 'timestamp')
        device = get_nested(results, 'metadata', 'device')
        
        print(f"  {approach}:")
        print(f"    Timestamp: {timestamp or 'Not recorded'}")
        print(f"    Device: {device or 'Not recorded'}")
        
        if not timestamp:
            validation.add_warning(f"{approach}: No timestamp recorded")


def extract_comparable_metrics(results: Dict) -> Dict:
    """Extract standardized metrics for comparison"""
    
    test_metrics = results.get('test_metrics', {})
    security = results.get('security_config', {})
    
    # Handle both old and new format
    if 'forward' in test_metrics and isinstance(test_metrics['forward'], dict):
        # Baseline format
        forward_acc = test_metrics['forward'].get('accuracy')
        backward_acc = test_metrics['backward'].get('accuracy')
        per_category = test_metrics['backward'].get('per_category', {})
    else:
        # FL format
        forward_acc = test_metrics.get('forward_accuracy')
        backward_acc = test_metrics.get('backward_accuracy')
        per_category = test_metrics.get('backward_per_category', {})
    
    return {
        'forward_accuracy': forward_acc,
        'backward_accuracy': backward_acc,
        'per_category': per_category,
        'dp_enabled': security.get('differential_privacy') or security.get('server_dp_enabled', False),
        'epsilon': security.get('epsilon'),
        'flame_enabled': security.get('flame_enabled', False),
        'adversarial_training': security.get('adversarial_training') or \
                               security.get('adversarial_enabled', False),
    }


def compare_metrics(all_results: Dict[str, Dict], validation: ValidationResult) -> Dict:
    """Compare metrics across approaches"""
    
    comparison = {
        'approaches': {},
        'rankings': {},
        'improvements': {}
    }
    
    all_metrics = {}
    for name, results in all_results.items():
        metrics = extract_comparable_metrics(results)
        all_metrics[name] = metrics
        comparison['approaches'][name] = metrics
    
    # Calculate improvements vs baseline
    if 'baseline' in all_metrics and all_metrics['baseline'].get('forward_accuracy'):
        baseline_fwd = all_metrics['baseline']['forward_accuracy']
        baseline_bwd = all_metrics['baseline']['backward_accuracy']
        
        for name, metrics in all_metrics.items():
            if name != 'baseline':
                fwd_acc = metrics.get('forward_accuracy')
                bwd_acc = metrics.get('backward_accuracy')
                
                if fwd_acc and bwd_acc:
                    comparison['improvements'][name] = {
                        'forward_delta': fwd_acc - baseline_fwd,
                        'forward_relative_pct': (fwd_acc - baseline_fwd) / baseline_fwd * 100,
                        'backward_delta': bwd_acc - baseline_bwd,
                        'backward_relative_pct': (bwd_acc - baseline_bwd) / baseline_bwd * 100,
                    }
    
    return comparison


def print_comparison(comparison: Dict, all_configs: Dict[str, Dict]):
    """Print comparison results"""
    
    print("\n" + "="*70)
    print("RESULTS COMPARISON".center(70))
    print("="*70)
    
    # Main metrics table
    print("\n┌─────────────────────┬────────────┬────────────┬────────────┬───────────┐")
    print("│ Approach            │ Forward    │ Backward   │ Privacy    │ Epochs Eq │")
    print("├─────────────────────┼────────────┼────────────┼────────────┼───────────┤")
    
    for name, metrics in comparison.get('approaches', {}).items():
        fwd = metrics.get('forward_accuracy')
        bwd = metrics.get('backward_accuracy')
        
        priv = "ε=" + str(metrics.get('epsilon')) if metrics.get('dp_enabled') else "None"
        epochs = all_configs.get(name, {}).get('epochs_equivalent', 'N/A')
        
        fwd_str = f"{fwd*100:.2f}%" if fwd else "N/A"
        bwd_str = f"{bwd*100:.2f}%" if bwd else "N/A"
        
        print(f"│ {name:19s} │ {fwd_str:>10s} │ {bwd_str:>10s} │ {str(priv):>10s} │ {str(epochs):>9s} │")
    
    print("└─────────────────────┴────────────┴────────────┴────────────┴───────────┘")
    
    # Improvements vs baseline
    if comparison.get('improvements'):
        print("\n" + "-"*70)
        print("IMPROVEMENTS VS BASELINE")
        print("-"*70)
        
        for name, impr in comparison['improvements'].items():
            fwd_delta = impr.get('forward_delta', 0)
            bwd_delta = impr.get('backward_delta', 0)
            
            fwd_sign = "+" if fwd_delta >= 0 else ""
            bwd_sign = "+" if bwd_delta >= 0 else ""
            
            print(f"\n  {name}:")
            print(f"    Forward:  {fwd_sign}{fwd_delta*100:.2f}% ({fwd_sign}{impr.get('forward_relative_pct', 0):.1f}% relative)")
            print(f"    Backward: {bwd_sign}{bwd_delta*100:.2f}% ({bwd_sign}{impr.get('backward_relative_pct', 0):.1f}% relative)")
    
    # Per-category comparison
    print("\n" + "-"*70)
    print("PER-CATEGORY BACKWARD ACCURACY")
    print("-"*70)
    
    categories = ['E-commerce', 'Gambling', 'Services', 'Retail', 'Luxury']
    
    print("\n┌─────────────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐")
    print("│ Approach            │ E-commerce │ Gambling   │ Services   │ Retail     │ Luxury     │")
    print("├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤")
    
    for name, metrics in comparison.get('approaches', {}).items():
        per_cat = metrics.get('per_category', {})
        row = f"│ {name:19s} │"
        for cat in categories:
            acc = per_cat.get(cat)
            acc_str = f"{acc*100:.1f}%" if acc else "N/A"
            row += f" {acc_str:>10s} │"
        print(row)
    
    print("└─────────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘")


def generate_latex_table(comparison: Dict, all_configs: Dict[str, Dict], validation: ValidationResult) -> str:
    """Generate LaTeX table with validity note"""
    
    validity_note = "Valid comparison" if validation.is_valid() else "WARNING: See validity report"
    
    latex = r"""\begin{table}[t]
\centering
\caption{Comparison of training approaches. """ + validity_note + r"""}
\label{tab:comparison}
\footnotesize
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Approach} & \textbf{Forward} & \textbf{Backward} & \textbf{Epochs} & \textbf{DP} & \textbf{FLAME} & \textbf{Adv.} \\
\midrule
"""
    
    for name, metrics in comparison.get('approaches', {}).items():
        fwd = metrics.get('forward_accuracy')
        bwd = metrics.get('backward_accuracy')
        epochs = all_configs.get(name, {}).get('epochs_equivalent', 'N/A')
        dp = r"$\varepsilon$=" + str(metrics.get('epsilon')) if metrics.get('dp_enabled') else "--"
        flame = r"\checkmark" if metrics.get('flame_enabled') else "--"
        adv = r"\checkmark" if metrics.get('adversarial_training') else "--"
        
        name_map = {
            'baseline': 'Centralized',
            'fl_nosec': 'FL-NoSec',
            'fl_secure': r'\textbf{FL-Secure}'
        }
        approach_name = name_map.get(name, name)
        
        fwd_str = f"{fwd*100:.2f}\\%" if fwd else "N/A"
        bwd_str = f"{bwd*100:.2f}\\%" if bwd else "N/A"
        
        if name == 'fl_secure':
            fwd_str = r"\textbf{" + fwd_str + "}"
            bwd_str = r"\textbf{" + bwd_str + "}"
        
        latex += f"{approach_name} & {fwd_str} & {bwd_str} & {epochs} & {dp} & {flame} & {adv} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    
    return latex


def main():
    parser = argparse.ArgumentParser(description='Compare training results with validity checks')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for comparison JSON')
    parser.add_argument('--latex', action='store_true',
                        help='Generate LaTeX table')
    parser.add_argument('--strict', action='store_true',
                        help='Exit with error if validation fails')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip validity checks')
    
    args = parser.parse_args()
    
    # Load all results
    print("\n" + "="*70)
    print("LOADING RESULTS".center(70))
    print("="*70)
    
    all_results = {}
    for name, filepath in RESULT_FILES.items():
        results = load_results(filepath)
        if results:
            all_results[name] = results
            print(f"✓ Loaded: {filepath}")
        else:
            print(f"⚠ Not found: {filepath}")
    
    if len(all_results) < 2:
        print("\n❌ Need at least 2 result files for comparison")
        return
    
    # Extract configs for validation
    all_configs = {}
    for name, results in all_results.items():
        all_configs[name] = extract_config_for_validation(results, name)
    
    # Run validation checks
    validation = ValidationResult()
    
    if not args.skip_validation:
        check_hyperparameter_consistency(all_configs, validation)
        check_data_split_consistency(all_configs, validation)
        check_feature_consistency(all_configs, validation)
        check_epochs_fairness(all_configs, validation)
        check_reproducibility(all_results, validation)
        
        validation.print_report()
    
    # Compare metrics
    comparison = compare_metrics(all_results, validation)
    comparison['validity'] = {
        'is_valid': validation.is_valid(),
        'passed_checks': len(validation.passed),
        'warnings': len(validation.warnings),
        'errors': len(validation.errors),
        'error_messages': validation.errors,
        'warning_messages': validation.warnings
    }
    comparison['configs'] = all_configs
    comparison['timestamp'] = datetime.now().isoformat()
    
    # Print comparison
    print_comparison(comparison, all_configs)
    
    # Generate LaTeX if requested
    if args.latex:
        latex = generate_latex_table(comparison, all_configs, validation)
        print("\n" + "="*70)
        print("LATEX TABLE")
        print("="*70)
        print(latex)
    
    # Save results
    os.makedirs('results/evaluations', exist_ok=True)
    
    output_path = args.output or 'results/evaluations/comparison_report.json'
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\n✓ Comparison saved to: {output_path}")
    
    # Exit with error if strict mode and validation failed
    if args.strict and not validation.is_valid():
        print("\n❌ Strict mode: Exiting with error due to validation failures")
        exit(1)


if __name__ == "__main__":
    main()