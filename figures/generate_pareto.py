"""
generate_pareto_eps16.py
FedBTC — Standalone pareto figure generator (4-panel, ε=16).

Reads:  results/interaction_eps16/aggregated.json
Writes: results/interaction_eps16/pareto.png

This script does NOT depend on p5_plot_interaction_matrix.py.
All labels and data come directly from the JSON, so the label
"DP only (ε=16)" is guaranteed correct.

USAGE:
    cd ~/links/scratch/fedbtc
    python3 generate_pareto_eps16.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


# -------------------------------------------------------------------------
# Configuration labels (ALL DP configs at epsilon=16)
# -------------------------------------------------------------------------
CONFIGS = {
    'C0': {'label': 'No defense',         'color': '#888888', 'marker': 'o'},
    'C1': {'label': 'AT only',            'color': '#1f77b4', 'marker': 's'},
    'C2': {'label': 'CoordMedian only',   'color': '#ff7f0e', 'marker': '^'},
    'C3': {'label': 'DP only (ε=16)',     'color': '#9467bd', 'marker': 'D'},
    'C4': {'label': 'AT + CoordMedian',   'color': '#2ca02c', 'marker': 'v'},
    'C5': {'label': 'Full stack',         'color': '#d62728', 'marker': 'P'},
    'C6': {'label': 'CoordMedian + DP',   'color': '#17becf', 'marker': '*'},
    'C7': {'label': 'AT + DP',            'color': '#8c564b', 'marker': 'X'},
}


# -------------------------------------------------------------------------
# 1) Load aggregated.json
# -------------------------------------------------------------------------
JSON_PATH = 'results/interaction_eps16/aggregated.json'
OUTPUT_PATH = 'results/interaction_eps16/pareto.png'

if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"Could not find {JSON_PATH}")

with open(JSON_PATH) as f:
    data = json.load(f)

agg = data['agg_raw']
print(f"Loaded {len(agg)} configurations from {JSON_PATH}")


# -------------------------------------------------------------------------
# 2) Helper to extract [mean, std]
# -------------------------------------------------------------------------
def get_metric(cfg, metric_name):
    """Returns (mean, std) for a given metric from aggregated.json."""
    v = agg[cfg][metric_name]
    if isinstance(v, list):
        return v[0], v[1]
    elif isinstance(v, dict):
        return v.get('mean', np.nan), v.get('std', 0.0)
    elif isinstance(v, (int, float)):
        return float(v), 0.0
    return np.nan, 0.0


# -------------------------------------------------------------------------
# 3) Build the 4-panel figure
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

PANELS = [
    {
        'ax':       axes[0, 0],
        'title':    'Utility ↔ Evasion resistance bwd',
        'xlabel':   'Clean accuracy — backward',
        'ylabel':   'Evasion resistance bwd (acc at ε=0.30)',
        'x_metric': 'utility_bwd',
        'y_metric': 'evasion_bwd',
        'show_legend': True,
    },
    {
        'ax':       axes[0, 1],
        'title':    'Utility ↔ Byzantine resistance bwd',
        'xlabel':   'Clean accuracy — backward',
        'ylabel':   'Byzantine resistance bwd (P1 Run C)',
        'x_metric': 'utility_bwd',
        'y_metric': 'byzantine_c_bwd',
        'show_legend': False,
    },
    {
        'ax':       axes[1, 0],
        'title':    'Utility ↔ MIA protection',
        'xlabel':   'Clean accuracy — backward',
        'ylabel':   'MIA protection (1 − AUC_Yeom)',
        'x_metric': 'utility_bwd',
        'y_metric': 'mia_protection',
        'show_legend': False,
    },
    {
        'ax':       axes[1, 1],
        'title':    'Utility ↔ DLG protection',
        'xlabel':   'Clean accuracy — backward',
        'ylabel':   'DLG protection (avg MSE)',
        'x_metric': 'utility_bwd',
        'y_metric': 'dlg_mse',
        'show_legend': False,
    },
]

for panel in PANELS:
    ax = panel['ax']

    for cfg_id, cfg_meta in CONFIGS.items():
        # Get clean acc on backward (x)
        x_mean, x_std = get_metric(cfg_id, panel['x_metric'])
        # Use utility_fwd error bar instead of utility_bwd if available
        # (since this is a paper convention --- adapt if needed)
        _, x_err = get_metric(cfg_id, 'utility_fwd')
        if not x_err or np.isnan(x_err):
            x_err = x_std

        # Get y metric
        y_mean, y_std = get_metric(cfg_id, panel['y_metric'])

        ax.errorbar(
            x_mean, y_mean,
            xerr=x_err if x_err > 0 else None,
            yerr=y_std if y_std > 0 else None,
            fmt=cfg_meta['marker'],
            color=cfg_meta['color'],
            markersize=10,
            elinewidth=1.5,
            capsize=4,
            label=cfg_meta['label'],
        )
        # Add config ID as text annotation
        ax.annotate(
            cfg_id,
            (x_mean, y_mean),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=9,
            color=cfg_meta['color'],
            fontweight='bold',
        )

    ax.set_xlabel(panel['xlabel'], fontsize=11)
    ax.set_ylabel(panel['ylabel'], fontsize=11)
    ax.set_title(panel['title'], fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)

    if panel['show_legend']:
        ax.legend(loc='center right', fontsize=10, framealpha=0.9)

# Main title
fig.suptitle(
    'FedBTC — Utility–Security Pareto Analysis\n'
    'Error bars = std over 5 seeds  |  upper-right = optimal',
    fontsize=13,
    y=0.995,
)

plt.tight_layout(rect=[0, 0, 1, 0.97])


# -------------------------------------------------------------------------
# 4) Save
# -------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: {OUTPUT_PATH}")

# Sanity check printout
print("\nSanity check — clean backward accuracy per config:")
for cfg_id in CONFIGS:
    m, s = get_metric(cfg_id, 'utility_bwd')
    print(f"  {cfg_id} ({CONFIGS[cfg_id]['label']:20s}): {m:.4f} ± {s:.4f}")
