"""
generate_dp_epsilon_curve.py
FedBTC — Regenerate fig:dp-epsilon-curve from table_epsilon_sweep.csv.

Plots:
  - Forward accuracy (%) on left y-axis (blue, solid)
  - MIA Advantage (right y-axis, red, dashed)
  - Mean ± std error bars over 5 seeds
  - X-axis: DP budget epsilon in {1, 2, 4, 8, 16, 32}

USAGE:
    python3 generate_dp_epsilon_curve.py

Input:  results/figures/table_epsilon_sweep.csv (or tables/table_epsilon_sweep.csv)
Output: results/figures/fig2_dp_epsilon_curve.png  (300 DPI)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------
# 1) Locate the CSV (try common paths)
# -------------------------------------------------------------------------
CANDIDATE_PATHS = [
    'results/figures/table_epsilon_sweep.csv',
    'tables/table_epsilon_sweep.csv',
    'results/evaluations/table_epsilon_sweep.csv',
    'table_epsilon_sweep.csv',
]

csv_path = None
for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    print(f"[ERROR] Could not find table_epsilon_sweep.csv in any of:")
    for p in CANDIDATE_PATHS:
        print(f"  - {p}")
    sys.exit(1)

print(f"Reading: {csv_path}")
df = pd.read_csv(csv_path)
print(df)
print()

# -------------------------------------------------------------------------
# 2) Sanity check: required columns
# -------------------------------------------------------------------------
REQUIRED = ['epsilon', 'fwd_mean', 'fwd_std', 'mia_mean', 'mia_std']
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    print(f"[ERROR] Missing columns in CSV: {missing}")
    print(f"Found columns: {list(df.columns)}")
    sys.exit(1)

df = df.sort_values('epsilon').reset_index(drop=True)

# -------------------------------------------------------------------------
# 3) Build the figure
# -------------------------------------------------------------------------
fig, ax_left = plt.subplots(figsize=(7, 4.5), dpi=300)

# X positions
eps_vals = df['epsilon'].values
# Slight horizontal offset for MIA points so the two error bars at
# epsilon=8 (where the forward sigma is very large due to bimodal
# convergence) do not overlap visually with the MIA error bar.
eps_vals_mia = eps_vals * 1.08

# Forward accuracy in % (left axis)
fwd_mean_pct = df['fwd_mean'].values * 100.0
fwd_std_pct  = df['fwd_std'].values * 100.0

# MIA advantage (right axis, plotted in raw units, not x10^-3)
mia_mean = df['mia_mean'].values
mia_std  = df['mia_std'].values

# --- LEFT axis: Forward accuracy ---
color_fwd = '#1f77b4'
ax_left.errorbar(
    eps_vals, fwd_mean_pct, yerr=fwd_std_pct,
    fmt='o-', color=color_fwd, ecolor=color_fwd,
    elinewidth=1.2, capsize=4, markersize=6,
    linewidth=2.0,
    label='Forward accuracy',
)
ax_left.set_xlabel(r'DP Budget $\varepsilon$', fontsize=12)
ax_left.set_ylabel(r'Forward Accuracy % ($\uparrow$)',
                   color=color_fwd, fontsize=12)
ax_left.tick_params(axis='y', labelcolor=color_fwd)
ax_left.set_ylim(25, 100)

# Use log-spaced x ticks since epsilon spans 1..32 (doubling)
ax_left.set_xscale('log', base=2)
ax_left.set_xticks(eps_vals)
ax_left.set_xticklabels([str(int(e)) for e in eps_vals])
ax_left.grid(True, axis='y', linestyle='--', alpha=0.4)

# --- RIGHT axis: MIA Advantage ---
ax_right = ax_left.twinx()
color_mia = '#d62728'
ax_right.errorbar(
    eps_vals_mia, mia_mean, yerr=mia_std,
    fmt='s--', color=color_mia, ecolor=color_mia,
    elinewidth=1.2, capsize=4, markersize=6,
    linewidth=2.0,
    label='MIA Advantage',
)
ax_right.set_ylabel(r'MIA Advantage ($\downarrow$)',
                    color=color_mia, fontsize=12)
ax_right.tick_params(axis='y', labelcolor=color_mia)

# Set right-axis range to make the noise floor visible
mia_max = max(mia_mean + mia_std) if len(mia_mean) > 0 else 0.002
ax_right.set_ylim(-0.0005, max(0.002, mia_max * 1.1))

# --- Title and layout ---
plt.title('Privacy-Utility Curve (Server-side DP)', fontsize=13)

# Combine legends from both axes
lines_left, labels_left = ax_left.get_legend_handles_labels()
lines_right, labels_right = ax_right.get_legend_handles_labels()
ax_left.legend(lines_left + lines_right,
               labels_left + labels_right,
               loc='center right', fontsize=10, framealpha=0.9)

plt.tight_layout()

# -------------------------------------------------------------------------
# 4) Save
# -------------------------------------------------------------------------
os.makedirs('results/figures', exist_ok=True)
out_path = 'results/figures/fig2_dp_epsilon_curve.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: {out_path}")

# Also save a copy under tables/ for convenience
out_path2 = 'tables/fig2_dp_epsilon_curve.png'
os.makedirs('tables', exist_ok=True)
import shutil
shutil.copy(out_path, out_path2)
print(f"✓ Copied to: {out_path2}")

# -------------------------------------------------------------------------
# 5) Sanity printout (helps verify the figure matches the CSV)
# -------------------------------------------------------------------------
print("\nSanity check (values plotted in figure):")
print(f"  {'eps':>4}  {'fwd_mean':>10}  {'fwd_std':>8}  {'mia_mean':>10}  {'mia_std':>10}")
for _, row in df.iterrows():
    print(f"  {int(row['epsilon']):>4}  "
          f"{row['fwd_mean']*100:>9.2f}%  "
          f"{row['fwd_std']*100:>7.2f}%  "
          f"{row['mia_mean']:>10.6f}  "
          f"{row['mia_std']:>10.6f}")
