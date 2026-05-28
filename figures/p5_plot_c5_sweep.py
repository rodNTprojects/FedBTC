"""
p5_plot_c5_sweep.py
FedBTC — Aggregate and plot C5 DP sweep results across 5 seeds.

Reads:   results/interaction_c5/c5_sweep_seed*.json
Outputs: results/interaction_c5/c5_sweep_aggregated.json
         results/interaction_c5/c5_sweep_comparison.png
"""

import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIGS_ORDER = ['C0', 'C5_eps8', 'C5_eps16', 'C5_eps32']
LABELS = {
    'C0':       'No defense\n(reference)',
    'C5_eps8':  'Full stack\nε=8  σ=0.10',
    'C5_eps16': 'Full stack\nε=16  σ=0.05',
    'C5_eps32': 'Full stack\nε=32  σ=0.025',
}
COLORS = ['#AAAAAA', '#F44336', '#FF9800', '#4CAF50']

DIMS = [
    ('utility_fwd',     'Utility fwd'),
    ('utility_bwd',     'Utility bwd'),
    ('evasion_fwd',     'Evasion fwd\n(acc ε=0.30)'),
    ('evasion_bwd',     'Evasion bwd\n(acc ε=0.30)'),
    ('byzantine_b_bwd', 'Byzantine\n(no def)'),
    ('byzantine_c_bwd', 'Byzantine\n(+CoordMed)'),
    ('mia_protection',  'MIA\nprotection'),
    ('dlg_mse',         'DLG\nprotection'),
]


def load_all(results_dir):
    data = {}
    for path in sorted(glob.glob(
            os.path.join(results_dir, 'c5_sweep_seed*.json'))):
        with open(path) as f:
            d = json.load(f)
        data[d['seed']] = d
    return data


def aggregate(per_seed):
    vals = {cfg: {d[0]: [] for d in DIMS} for cfg in CONFIGS_ORDER}
    for seed, data in per_seed.items():
        for cfg in CONFIGS_ORDER:
            res = data.get('results', {}).get(cfg)
            if res is None:
                continue
            for d_key, _ in DIMS:
                v = res.get(d_key)
                if v is not None:
                    vals[cfg][d_key].append(v)

    agg = {}
    for cfg in CONFIGS_ORDER:
        agg[cfg] = {}
        for d_key, _ in DIMS:
            vs = vals[cfg][d_key]
            if vs:
                agg[cfg][d_key] = {
                    'mean': round(float(np.mean(vs)), 4),
                    'std':  round(float(np.std(vs)), 4),
                    'n':    len(vs),
                }
    return agg


def plot_comparison(agg, out_path):
    n_dims = len(DIMS)
    n_cfgs = len(CONFIGS_ORDER)
    x = np.arange(n_dims)
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 6))

    for k, cfg in enumerate(CONFIGS_ORDER):
        means = [agg[cfg].get(d[0], {}).get('mean', np.nan) for d in DIMS]
        stds  = [agg[cfg].get(d[0], {}).get('std', 0.0)    for d in DIMS]
        offset = (k - n_cfgs/2 + 0.5) * width
        ax.bar(x + offset, means, width,
               label=LABELS[cfg].replace('\n', ' '),
               color=COLORS[k], alpha=0.85,
               yerr=stds, capsize=3, error_kw={'linewidth': 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels([d[1] for d in DIMS], fontsize=9)
    ax.set_ylabel('Metric value', fontsize=10)
    ax.set_title(
        'FedBTC — C5 DP Epsilon Sweep\n'
        'C5 full stack at ε=8 (σ=0.10), ε=16 (σ=0.05), ε=32 (σ=0.025) vs C0 baseline\n'
        'Error bars = std over 5 seeds',
        fontsize=10, pad=12)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)

    # Add vertical separators between groups
    for x_pos in [1.5, 3.5, 5.5]:
        ax.axvline(x_pos, color='#CCCCCC', linewidth=1, zorder=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='results/interaction_c5')
    p.add_argument('--output_dir',  default='results/interaction_c5')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    per_seed = load_all(args.results_dir)
    if not per_seed:
        sys.exit(f"[ERROR] No JSON files in {args.results_dir}")
    print(f"Seeds loaded: {sorted(per_seed)}")

    agg = aggregate(per_seed)

    # Print summary table
    print("\n" + "="*80)
    print("C5 DP SWEEP — AGGREGATED (mean ± std, 5 seeds)")
    print("="*80)
    print(f"{'Config':14s}  " + "  ".join(f"{d[1][:8]:>9s}" for d in DIMS))
    print("-"*80)
    for cfg in CONFIGS_ORDER:
        row = f"{cfg:14s}  "
        for d_key, _ in DIMS:
            v = agg[cfg].get(d_key)
            if v:
                row += f"{v['mean']:>7.4f}  "
            else:
                row += f"{'—':>9s}  "
        print(row)

    # Save
    agg_path = os.path.join(args.output_dir, 'c5_sweep_aggregated.json')
    with open(agg_path, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregated: {agg_path}")

    plot_comparison(agg, os.path.join(args.output_dir, 'c5_sweep_comparison.png'))


if __name__ == '__main__':
    main()
