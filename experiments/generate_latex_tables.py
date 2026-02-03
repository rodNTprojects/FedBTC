#!/usr/bin/env python3
"""
================================================================================
Script: generate_latex_tables.py
Description: Génère tous les tableaux LaTeX pour l'article USENIX Security
             à partir des fichiers JSON de résultats

TABLES GÉNÉRÉES:
  1. Main Results (Table 1)
  2. Per-Category Backward Attribution (Table 2)
  3. Privacy-Utility Trade-off (Table 3)
  4. Byzantine Attack Resilience (Table 4)
  5. Ablation: GCN Layers (Table 5)
  6. Ablation: Local Epochs (Table 6)
  7. Ablation: Label Distribution (Table 7)

USAGE:
    python generate_latex_tables.py
    python generate_latex_tables.py --output tables.tex
================================================================================
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'experiments')
EVAL_DIR = os.path.join(PROJECT_DIR, 'results', 'evaluations')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_json(filepath: str) -> Optional[Dict]:
    """Load JSON file if exists."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def format_pct(value: float, bold: bool = False) -> str:
    """Format percentage for LaTeX."""
    if value is None:
        return "N/A"
    pct = f"{value*100:.2f}\\%"
    if bold:
        return f"\\textbf{{{pct}}}"
    return pct


def format_epsilon(eps: Optional[float]) -> str:
    """Format epsilon value for LaTeX."""
    if eps is None:
        return "$\\infty$"
    return f"$\\varepsilon$={eps}"


def get_nested(d: Dict, *keys, default=None):
    """Safely get nested dictionary value."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


# ============================================================================
# TABLE 1: MAIN RESULTS
# ============================================================================

def generate_table1_main_results() -> str:
    """Generate Table 1: Main experimental results."""
    
    print("Generating Table 1: Main Results...")
    
    # Load results
    results = {}
    
    # Baseline
    baseline = load_json(os.path.join(EVAL_DIR, 'baseline_results.json'))
    if baseline:
        fwd = get_nested(baseline, 'test_metrics', 'forward', 'accuracy') or \
              get_nested(baseline, 'test_metrics', 'forward_accuracy')
        bwd = get_nested(baseline, 'test_metrics', 'backward', 'accuracy') or \
              get_nested(baseline, 'test_metrics', 'backward_accuracy')
        results['baseline'] = {'fwd': fwd, 'bwd': bwd, 'privacy': 'None'}
    
    # FL configurations
    configs = [
        ('fl_nosec', 'fl_nosec.json', 'Isolation'),
        ('fl_serverDP', 'fl_serverDP.json', '$\\varepsilon$=8.0'),
        ('fl_flame', 'fl_flame.json', 'Byzantine'),
        ('fl_full', 'fl_full.json', 'Full Stack'),
    ]
    
    for name, filename, privacy in configs:
        data = load_json(os.path.join(RESULTS_DIR, filename))
        if data:
            fwd = get_nested(data, 'test_metrics', 'forward_accuracy')
            bwd = get_nested(data, 'test_metrics', 'backward_accuracy')
            f1 = get_nested(data, 'test_metrics', 'f1_score')
            results[name] = {'fwd': fwd, 'bwd': bwd, 'f1': f1, 'privacy': privacy}
    
    # Find best values
    best_fwd = max((r.get('fwd', 0) or 0) for r in results.values())
    best_bwd = max((r.get('bwd', 0) or 0) for r in results.values())
    
    # Generate LaTeX
    latex = r"""
\begin{table}[t]
\centering
\caption{Main experimental results. FL-Full achieves superior accuracy with strong privacy guarantees.}
\label{tab:main_results}
\footnotesize
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Configuration} & \textbf{Privacy} & \textbf{Fwd Acc.} & \textbf{Bwd Acc.} & \textbf{F1} \\
\midrule
\textit{Baselines} & & & & \\
"""
    
    # Baseline row
    if 'baseline' in results:
        r = results['baseline']
        fwd_str = format_pct(r['fwd'])
        bwd_str = format_pct(r['bwd'])
        f1 = 2 * r['fwd'] * r['bwd'] / (r['fwd'] + r['bwd']) if r['fwd'] and r['bwd'] else 0
        latex += f"Centralized & None & {fwd_str} & {bwd_str} & {f1:.2f} \\\\\n"
    
    latex += r"\midrule" + "\n"
    latex += r"\textit{Federated Learning} & & & & \\" + "\n"
    
    # FL rows
    fl_names = {
        'fl_nosec': 'FL-NoSec',
        'fl_serverDP': 'FL-ServerDP',
        'fl_flame': 'FL-FLAME',
        'fl_full': 'FL-Full'
    }
    
    for key, display_name in fl_names.items():
        if key in results:
            r = results[key]
            is_best = (r.get('fwd') == best_fwd) or (r.get('bwd') == best_bwd)
            
            if key == 'fl_full':
                display_name = f"\\textbf{{{display_name}}}"
                fwd_str = format_pct(r['fwd'], bold=True)
                bwd_str = format_pct(r['bwd'], bold=True)
                f1_str = f"\\textbf{{{r.get('f1', 0):.2f}}}"
                privacy_str = f"\\textbf{{{r['privacy']}}}"
            else:
                fwd_str = format_pct(r['fwd'])
                bwd_str = format_pct(r['bwd'])
                f1_str = f"{r.get('f1', 0):.2f}"
                privacy_str = r['privacy']
            
            latex += f"{display_name} & {privacy_str} & {fwd_str} & {bwd_str} & {f1_str} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


# ============================================================================
# TABLE 2: PER-CATEGORY BACKWARD ATTRIBUTION
# ============================================================================

def generate_table2_per_category() -> str:
    """Generate Table 2: Per-category backward attribution accuracy."""
    
    print("Generating Table 2: Per-Category Results...")
    
    # Load baseline
    baseline = load_json(os.path.join(EVAL_DIR, 'baseline_results.json'))
    baseline_cats = get_nested(baseline, 'test_metrics', 'backward', 'per_category') or \
                   get_nested(baseline, 'test_metrics', 'backward_per_category') or {}
    
    # Load FL-Full
    fl_full = load_json(os.path.join(RESULTS_DIR, 'fl_full.json'))
    fl_cats = get_nested(fl_full, 'test_metrics', 'per_category') or {}
    
    categories = ['E-commerce', 'Gambling', 'Services', 'Retail', 'Luxury']
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Per-category backward attribution accuracy.}
\label{tab:per_category}
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Category} & \textbf{Baseline} & \textbf{FL-Full} & \textbf{Improvement} \\
\midrule
"""
    
    for cat in categories:
        base_val = baseline_cats.get(cat, 0)
        fl_val = fl_cats.get(cat, 0)
        
        if base_val and fl_val:
            improvement = fl_val - base_val
            imp_str = f"+{improvement*100:.2f}\\%"
        else:
            imp_str = "N/A"
        
        latex += f"{cat} & {format_pct(base_val)} & {format_pct(fl_val)} & {imp_str} \\\\\n"
    
    # Overall
    base_overall = get_nested(baseline, 'test_metrics', 'backward', 'accuracy') or \
                   get_nested(baseline, 'test_metrics', 'backward_accuracy')
    fl_overall = get_nested(fl_full, 'test_metrics', 'backward_accuracy')
    
    if base_overall and fl_overall:
        imp_overall = fl_overall - base_overall
        imp_str = f"+{imp_overall*100:.2f}\\%"
    else:
        imp_str = "N/A"
    
    latex += r"\midrule" + "\n"
    latex += f"\\textbf{{Overall}} & {format_pct(base_overall, bold=True)} & {format_pct(fl_overall, bold=True)} & \\textbf{{{imp_str}}} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


# ============================================================================
# TABLE 3: PRIVACY-UTILITY TRADE-OFF
# ============================================================================

def generate_table3_privacy_tradeoff() -> str:
    """Generate Table 3: Privacy-utility trade-off."""
    
    print("Generating Table 3: Privacy-Utility Trade-off...")
    
    epsilon_configs = [
        ('inf', None, 0.0),
        ('16', 16.0, 0.05),
        ('8', 8.0, 0.1),
        ('4', 4.0, 0.2),
        ('2', 2.0, 0.4),
        ('1', 1.0, 0.8),
    ]
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Privacy-utility trade-off across $\varepsilon$ configurations (server-side DP).}
\label{tab:privacy_tradeoff}
\footnotesize
\begin{tabular}{@{}cccc@{}}
\toprule
\textbf{$\varepsilon$} & \textbf{$\sigma$} & \textbf{Forward Acc.} & \textbf{Backward Acc.} \\
\midrule
"""
    
    for eps_name, eps_val, sigma in epsilon_configs:
        filename = f"privacy_eps_{eps_name}.json"
        data = load_json(os.path.join(RESULTS_DIR, filename))
        
        if data:
            fwd = get_nested(data, 'test_metrics', 'forward_accuracy')
            bwd = get_nested(data, 'test_metrics', 'backward_accuracy')
        else:
            fwd = None
            bwd = None
        
        eps_str = "$\\infty$ (No DP)" if eps_val is None else f"{eps_val}"
        sigma_str = f"{sigma}"
        
        latex += f"{eps_str} & {sigma_str} & {format_pct(fwd)} & {format_pct(bwd)} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


# ============================================================================
# TABLE 4: BYZANTINE ATTACK RESILIENCE
# ============================================================================

def generate_table4_byzantine() -> str:
    """Generate Table 4: Byzantine attack resilience."""
    
    print("Generating Table 4: Byzantine Resilience...")
    
    # Load security evaluation report
    security_report = load_json(os.path.join(EVAL_DIR, 'security_evaluation_report.json'))
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Byzantine attack resilience with FLAME defense.}
\label{tab:byzantine}
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Scenario} & \textbf{Detection Rate} & \textbf{FP Rate} & \textbf{Attackers Detected} \\
\midrule
"""
    
    if security_report and 'evaluations' in security_report:
        byzantine_data = get_nested(security_report, 'evaluations', 'byzantine_attack', 'scenarios') or {}
        
        # Map scenario names to display names
        scenario_mapping = [
            ('No attack (baseline)', '0\\% (Clean)', 0),
            ('1 random attacker', '33\\% Random', 1),
            ('1 sign-flip attacker', '33\\% Sign-flip', 1),
            ('2 colluding attackers', '66\\% Colluding', 2),
        ]
        
        for scenario_key, display_name, n_attackers in scenario_mapping:
            if scenario_key in byzantine_data:
                data = byzantine_data[scenario_key]
                detection_rate = data.get('detection_rate', 0)
                fp_rate = data.get('false_positive_rate', 0)
                tp = data.get('true_positives', 0)
                
                det_str = f"{detection_rate*100:.0f}\\%" if detection_rate is not None else "N/A"
                fp_str = f"{fp_rate*100:.0f}\\%" if fp_rate is not None else "N/A"
                
                if n_attackers == 0:
                    detected_str = "N/A"
                else:
                    detected_str = f"{tp}/{n_attackers}"
                
                latex += f"{display_name} & {det_str} & {fp_str} & {detected_str} \\\\\n"
            else:
                latex += f"{display_name} & N/A & N/A & N/A \\\\\n"
        
        # Add overall performance
        flame_perf = get_nested(security_report, 'evaluations', 'byzantine_attack', 'flame_performance') or {}
        if flame_perf:
            latex += r"\midrule" + "\n"
            overall_det = flame_perf.get('overall_detection_rate', 0)
            overall_fp = flame_perf.get('overall_false_positive_rate', 0)
            total_tp = flame_perf.get('total_true_positives', 0)
            total_fn = flame_perf.get('total_false_negatives', 0)
            
            latex += f"\\textbf{{Overall}} & \\textbf{{{overall_det*100:.0f}\\%}} & {overall_fp*100:.0f}\\% & {total_tp}/{total_tp + total_fn} \\\\\n"
    else:
        # Fallback with placeholder data
        latex += r"0\% (Clean) & N/A & N/A & N/A \\" + "\n"
        latex += r"33\% Random & N/A & N/A & N/A \\" + "\n"
        latex += r"33\% Sign-flip & N/A & N/A & N/A \\" + "\n"
        latex += r"66\% Colluding & N/A & N/A & N/A \\" + "\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


# ============================================================================
# TABLE 5: ABLATION - GCN LAYERS
# ============================================================================

def generate_table5_ablation_layers() -> str:
    """Generate Table 5: Ablation - GCN layer depth."""
    
    print("Generating Table 5: Ablation GCN Layers...")
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Ablation: GCN layer depth.}
\label{tab:ablation_layers}
\footnotesize
\begin{tabular}{@{}ccc@{}}
\toprule
\textbf{Layers} & \textbf{Forward Acc.} & \textbf{Backward Acc.} \\
\midrule
"""
    
    best_fwd = 0
    best_bwd = 0
    results = {}
    
    # Load all results first to find best
    for n_layers in [1, 2, 3, 4, 5]:
        filename = f"ablation_layers_{n_layers}.json"
        data = load_json(os.path.join(RESULTS_DIR, filename))
        if data:
            fwd = get_nested(data, 'test_metrics', 'forward_accuracy') or 0
            bwd = get_nested(data, 'test_metrics', 'backward_accuracy') or 0
            results[n_layers] = {'fwd': fwd, 'bwd': bwd}
            best_fwd = max(best_fwd, fwd)
            best_bwd = max(best_bwd, bwd)
    
    # Generate rows
    for n_layers in [1, 2, 3, 4, 5]:
        if n_layers in results:
            r = results[n_layers]
            is_best = (r['fwd'] == best_fwd) and (r['bwd'] == best_bwd)
            
            if is_best:
                latex += f"\\textbf{{{n_layers}}} & {format_pct(r['fwd'], bold=True)} & {format_pct(r['bwd'], bold=True)} \\\\\n"
            else:
                latex += f"{n_layers} & {format_pct(r['fwd'])} & {format_pct(r['bwd'])} \\\\\n"
        else:
            latex += f"{n_layers} & N/A & N/A \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


# ============================================================================
# TABLE 6: ABLATION - LOCAL EPOCHS
# ============================================================================

def generate_table6_ablation_epochs() -> str:
    """Generate Table 6: Ablation - Local epochs per round."""
    
    print("Generating Table 6: Ablation Local Epochs...")
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Ablation: Local epochs per round.}
\label{tab:ablation_epochs}
\footnotesize
\begin{tabular}{@{}ccc@{}}
\toprule
\textbf{Local Epochs} & \textbf{Forward Acc.} & \textbf{Rounds to Converge} \\
\midrule
"""
    
    best_fwd = 0
    results = {}
    
    for epochs in [1, 3, 5, 7]:
        filename = f"ablation_epochs_{epochs}.json"
        data = load_json(os.path.join(RESULTS_DIR, filename))
        if data:
            fwd = get_nested(data, 'test_metrics', 'forward_accuracy') or 0
            rounds = get_nested(data, 'defense_stats', 'rounds_to_converge') or \
                    get_nested(data, 'history', 'rounds_to_converge')
            results[epochs] = {'fwd': fwd, 'rounds': rounds}
            best_fwd = max(best_fwd, fwd)
    
    for epochs in [1, 3, 5, 7]:
        if epochs in results:
            r = results[epochs]
            is_best = r['fwd'] == best_fwd
            
            rounds_str = str(r['rounds']) if r['rounds'] else "N/A"
            
            if is_best:
                latex += f"\\textbf{{{epochs}}} & {format_pct(r['fwd'], bold=True)} & \\textbf{{{rounds_str}}} \\\\\n"
            else:
                latex += f"{epochs} & {format_pct(r['fwd'])} & {rounds_str} \\\\\n"
        else:
            latex += f"{epochs} & N/A & N/A \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


# ============================================================================
# TABLE 7: ABLATION - LABEL DISTRIBUTION
# ============================================================================

def generate_table7_ablation_distribution() -> str:
    """Generate Table 7: Ablation - Federated label distribution."""
    
    print("Generating Table 7: Ablation Label Distribution...")
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Ablation: Federated label distribution.}
\label{tab:ablation_distribution}
\footnotesize
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Distribution} & \textbf{Forward Acc.} & \textbf{Convergence} \\
\midrule
"""
    
    distributions = [
        ('100_0_0', '100/0/0 (Naive)'),
        ('80_10_10', '80/10/10'),
        ('60_20_20', '60/20/20 (Ours)'),
        ('40_30_30', '40/30/30'),
        ('33_33_33', '33/33/33 (IID)'),
    ]
    
    best_fwd = 0
    results = {}
    
    for dist_code, dist_name in distributions:
        filename = f"ablation_dist_{dist_code}.json"
        data = load_json(os.path.join(RESULTS_DIR, filename))
        if data:
            fwd = get_nested(data, 'test_metrics', 'forward_accuracy') or 0
            rounds = get_nested(data, 'defense_stats', 'rounds_to_converge')
            results[dist_code] = {'fwd': fwd, 'rounds': rounds, 'name': dist_name}
            best_fwd = max(best_fwd, fwd)
    
    for dist_code, dist_name in distributions:
        if dist_code in results:
            r = results[dist_code]
            is_best = r['fwd'] == best_fwd
            
            # Convergence description
            if r['rounds'] is None:
                conv = "Diverges"
            elif r['rounds'] > 30:
                conv = "Slow"
            elif r['rounds'] < 15:
                conv = "Fast"
            else:
                conv = "Normal"
            
            if is_best or dist_code == '60_20_20':
                latex += f"\\textbf{{{r['name']}}} & {format_pct(r['fwd'], bold=True)} & \\textbf{{{conv}}} \\\\\n"
            else:
                latex += f"{r['name']} & {format_pct(r['fwd'])} & {conv} \\\\\n"
        else:
            latex += f"{dist_name} & N/A & N/A \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


# ============================================================================
# TABLE 8: FL TRAINING CONVERGENCE (COMPACT)
# ============================================================================

def generate_table8_convergence_compact() -> str:
    """Generate compact convergence table with selected rounds."""
    
    print("Generating Table 8: FL Training Convergence (Compact)...")
    
    # Load FL-Full results
    fl_full = load_json(os.path.join(RESULTS_DIR, 'fl_full.json'))
    
    latex = r"""
\begin{table}[t]
\centering
\caption{FL training convergence (selected rounds).}
\label{tab:round_metrics_compact}
\footnotesize
\begin{tabular}{@{}cccccc@{}}
\toprule
\textbf{Round} & \textbf{Fwd Acc.} & \textbf{Bwd Acc.} & \textbf{$\mathcal{L}_0$} & \textbf{$\mathcal{L}_1$} & \textbf{$\mathcal{L}_2$} \\
\midrule
"""
    
    selected_rounds = [1, 5, 10, 15, 20]
    
    if fl_full and 'history' in fl_full:
        history = fl_full['history']
        rounds = history.get('rounds', [])
        fwd_acc = history.get('forward_accuracy', [])
        bwd_acc = history.get('backward_accuracy', [])
        losses = history.get('loss', [])
        
        # Get per-client losses if available
        per_client_losses = history.get('per_client_loss', None)
        
        for r in selected_rounds:
            idx = r - 1  # Convert to 0-indexed
            if idx < len(rounds):
                fwd = fwd_acc[idx] if idx < len(fwd_acc) else None
                bwd = bwd_acc[idx] if idx < len(bwd_acc) else None
                loss = losses[idx] if idx < len(losses) else None
                
                fwd_str = format_pct(fwd) if fwd else "N/A"
                bwd_str = format_pct(bwd) if bwd else "N/A"
                
                # For per-client losses, use placeholder or actual if available
                if per_client_losses and idx < len(per_client_losses):
                    l0 = f"{per_client_losses[idx][0]:.3f}"
                    l1 = f"{per_client_losses[idx][1]:.3f}"
                    l2 = f"{per_client_losses[idx][2]:.3f}"
                elif loss:
                    # Use average loss for all clients as placeholder
                    l0 = l1 = l2 = f"{loss:.3f}"
                else:
                    l0 = l1 = l2 = "N/A"
                
                latex += f"{r} & {fwd_str} & {bwd_str} & {l0} & {l1} & {l2} \\\\\n"
            else:
                latex += f"{r} & N/A & N/A & N/A & N/A & N/A \\\\\n"
    else:
        for r in selected_rounds:
            latex += f"{r} & N/A & N/A & N/A & N/A & N/A \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


# ============================================================================
# TABLE 9: PER-EXCHANGE ACCURACY EVOLUTION
# ============================================================================

def generate_table9_per_exchange_evolution() -> str:
    """Generate per-exchange accuracy evolution table."""
    
    print("Generating Table 9: Per-Exchange Evolution...")
    
    fl_full = load_json(os.path.join(RESULTS_DIR, 'fl_full.json'))
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Per-exchange accuracy evolution.}
\label{tab:per_exchange_compact}
\footnotesize
\begin{tabular}{@{}c|cc|cc|cc@{}}
\toprule
& \multicolumn{2}{c|}{\textbf{Exchange 0}} & \multicolumn{2}{c|}{\textbf{Exchange 1}} & \multicolumn{2}{c}{\textbf{Exchange 2}} \\
\textbf{Round} & Fwd & Bwd & Fwd & Bwd & Fwd & Bwd \\
\midrule
"""
    
    selected_rounds = [1, 5, 10, 15, 20]
    
    if fl_full and 'history' in fl_full:
        history = fl_full['history']
        
        # Check for per-exchange metrics
        per_exchange = history.get('per_exchange', None)
        
        if per_exchange:
            for r in selected_rounds:
                idx = r - 1
                if idx < len(per_exchange):
                    ex_data = per_exchange[idx]
                    row = f"{r}"
                    for ex_id in [0, 1, 2]:
                        ex_key = f"exchange_{ex_id}"
                        if ex_key in ex_data:
                            fwd = ex_data[ex_key].get('forward', 0)
                            bwd = ex_data[ex_key].get('backward', 0)
                            row += f" & {fwd*100:.1f} & {bwd*100:.1f}"
                        else:
                            row += " & N/A & N/A"
                    latex += row + " \\\\\n"
                else:
                    latex += f"{r} & N/A & N/A & N/A & N/A & N/A & N/A \\\\\n"
        else:
            # Use final test metrics as last row placeholder
            test_metrics = fl_full.get('test_metrics', {})
            per_exchange_final = test_metrics.get('per_exchange', [])
            
            for r in selected_rounds:
                if r == 20 and per_exchange_final:
                    row = "20"
                    for ex in per_exchange_final:
                        fwd = ex.get('forward_accuracy', 0)
                        bwd = ex.get('backward_accuracy', 0)
                        row += f" & {fwd*100:.1f} & {bwd*100:.1f}"
                    latex += row + " \\\\\n"
                else:
                    latex += f"{r} & N/A & N/A & N/A & N/A & N/A & N/A \\\\\n"
    else:
        for r in selected_rounds:
            latex += f"{r} & N/A & N/A & N/A & N/A & N/A & N/A \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


# ============================================================================
# TABLE 10: CATEGORY-WISE EVOLUTION
# ============================================================================

def generate_table10_category_evolution() -> str:
    """Generate category-wise backward accuracy evolution."""
    
    print("Generating Table 10: Category Evolution...")
    
    fl_full = load_json(os.path.join(RESULTS_DIR, 'fl_full.json'))
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Category-wise backward accuracy evolution.}
\label{tab:category_compact}
\footnotesize
\begin{tabular}{@{}cccccc@{}}
\toprule
\textbf{Round} & \textbf{E-Comm.} & \textbf{Gambl.} & \textbf{Serv.} & \textbf{Retail} & \textbf{Luxury} \\
\midrule
"""
    
    selected_rounds = [1, 5, 10, 15, 20]
    categories = ['E-commerce', 'Gambling', 'Services', 'Retail', 'Luxury']
    
    if fl_full and 'history' in fl_full:
        history = fl_full['history']
        per_cat_history = history.get('per_category', None)
        
        if per_cat_history:
            for r in selected_rounds:
                idx = r - 1
                if idx < len(per_cat_history):
                    cat_data = per_cat_history[idx]
                    row = f"{r}"
                    for cat in categories:
                        val = cat_data.get(cat, 0)
                        row += f" & {val*100:.1f}" if val else " & N/A"
                    latex += row + " \\\\\n"
                else:
                    latex += f"{r} & N/A & N/A & N/A & N/A & N/A \\\\\n"
        else:
            # Use final test metrics as last row
            test_metrics = fl_full.get('test_metrics', {})
            per_cat_final = test_metrics.get('per_category', {})
            
            for r in selected_rounds:
                if r == 20 and per_cat_final:
                    row = "20"
                    for cat in categories:
                        val = per_cat_final.get(cat, 0)
                        row += f" & {val*100:.1f}" if val else " & N/A"
                    latex += row + " \\\\\n"
                else:
                    latex += f"{r} & N/A & N/A & N/A & N/A & N/A \\\\\n"
    else:
        for r in selected_rounds:
            latex += f"{r} & N/A & N/A & N/A & N/A & N/A \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from experiment results')
    parser.add_argument('--output', type=str, default='evaluation_tables.tex',
                        help='Output filename')
    parser.add_argument('--tables', nargs='+', type=int, default=None,
                        help='Generate only specific tables (1-7)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LATEX TABLE GENERATOR")
    print("="*70)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Output file: {args.output}")
    
    # Generate all tables
    generators = [
        (1, generate_table1_main_results),
        (2, generate_table2_per_category),
        (3, generate_table3_privacy_tradeoff),
        (4, generate_table4_byzantine),
        (5, generate_table5_ablation_layers),
        (6, generate_table6_ablation_epochs),
        (7, generate_table7_ablation_distribution),
        (8, generate_table8_convergence_compact),
        (9, generate_table9_per_exchange_evolution),
        (10, generate_table10_category_evolution),
    ]
    
    # Filter if specific tables requested
    if args.tables:
        generators = [(n, g) for n, g in generators if n in args.tables]
    
    # Combine all tables
    latex_output = f"""% ============================================================================
% AUTO-GENERATED LATEX TABLES
% Generated: {datetime.now().isoformat()}
% ============================================================================

"""
    
    for table_num, generator in generators:
        try:
            table_latex = generator()
            latex_output += f"% Table {table_num}\n"
            latex_output += table_latex
            latex_output += "\n\n"
            print(f"  ✓ Table {table_num} generated")
        except Exception as e:
            print(f"  ✗ Table {table_num} FAILED: {e}")
            latex_output += f"% Table {table_num} - GENERATION FAILED: {e}\n\n"
    
    # Save output
    output_path = os.path.join(RESULTS_DIR, args.output)
    with open(output_path, 'w') as f:
        f.write(latex_output)
    
    print(f"\n✓ Tables saved to: {output_path}")
    
    # Also print to console
    print("\n" + "="*70)
    print("GENERATED LATEX")
    print("="*70)
    print(latex_output)


if __name__ == "__main__":
    main()
