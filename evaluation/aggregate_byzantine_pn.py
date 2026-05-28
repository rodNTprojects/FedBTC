"""
aggregate_byzantine_pn.py
FedBTC — Aggregate P1/P2/P3 Run B and Run C from existing
security_eval_*.json files into a clean CSV.

Reads:  results/evaluations/security_eval_{tag}_seed{N}.json  (5 seeds)
Writes: results/evaluations/table_byzantine_full.csv

For each config × attack (P1/P2/P3), computes:
  - Clean fwd/bwd accuracy (from utility section)
  - Run B (no_defense) fwd/bwd accuracy
  - Run C (with_defense) fwd/bwd accuracy
  - Drop_B = clean - Run B
  - Drop_C = clean - Run C

USAGE:
    python3 aggregate_byzantine_pn.py
"""

import os, json, glob
import numpy as np
import pandas as pd

EVAL_DIR = 'results/evaluations'
OUT_PATH = 'results/evaluations/table_byzantine_full.csv'

# Configurations to aggregate (must match security_eval_{tag}_seed*.json
# filenames). FAT-base has no Layer 2 → with_defense will be 'n/a'.
CONFIGS = [
    ('fat_base',         'FAT-base'),
    ('fat_def_eps32.0',  'FAT-def (eps=32)'),
    ('fat_def_eps16.0',  'FAT-def (eps=16)'),
    ('fat_def_eps8.0',   'FAT-def (eps=8)'),
    ('fat_def_eps4.0',   'FAT-def (eps=4)'),
    ('fat_def_eps2.0',   'FAT-def (eps=2)'),
    ('fat_def_eps1.0',   'FAT-def (eps=1)'),
]

SEEDS = [0, 45, 123, 654, 789]
ATTACKS = ['P1', 'P2', 'P3']


def load_seed(tag, seed):
    path = os.path.join(EVAL_DIR, f'security_eval_{tag}_seed{seed}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_attack(report, attack):
    """Extract Run B and Run C for an attack.
    Returns dict with clean_fwd, clean_bwd, runB_fwd, runB_bwd, runC_fwd, runC_bwd.
    Returns None if missing."""
    if 'utility' not in report or 'attacks' not in report:
        return None
    if attack not in report['attacks']:
        return None

    a = report['attacks'][attack]

    # P1 may be missing 'with_defense' for FAT-base (no Layer 2)
    runB = a.get('no_defense', {})
    runC = a.get('with_defense', {})

    return {
        'clean_fwd':  report['utility'].get('forward_acc'),
        'clean_bwd':  report['utility'].get('backward_acc'),
        'runB_fwd':   runB.get('forward_acc'),
        'runB_bwd':   runB.get('backward_acc'),
        'runC_fwd':   runC.get('forward_acc'),
        'runC_bwd':   runC.get('backward_acc'),
    }


def aggregate(tag, attack):
    """Aggregate across seeds. Returns dict of mean/std for each metric."""
    rows = []
    for s in SEEDS:
        rep = load_seed(tag, s)
        if rep is None:
            print(f"  [MISS] {tag} seed={s}: file not found")
            continue
        ex = extract_attack(rep, attack)
        if ex is None:
            print(f"  [MISS] {tag} seed={s} attack={attack}: not in JSON")
            continue
        # Compute drops
        if ex['clean_fwd'] is not None and ex['runB_fwd'] is not None:
            ex['dropB_fwd'] = ex['clean_fwd'] - ex['runB_fwd']
        else:
            ex['dropB_fwd'] = None
        if ex['clean_fwd'] is not None and ex['runC_fwd'] is not None:
            ex['dropC_fwd'] = ex['clean_fwd'] - ex['runC_fwd']
        else:
            ex['dropC_fwd'] = None
        rows.append(ex)

    if not rows:
        return None

    # Compute mean/std for each metric across seeds (ignoring None)
    out = {'n_seeds': len(rows)}
    for k in ['clean_fwd', 'clean_bwd',
              'runB_fwd', 'runB_bwd',
              'runC_fwd', 'runC_bwd',
              'dropB_fwd', 'dropC_fwd']:
        vals = [r[k] for r in rows if r[k] is not None and not np.isnan(r[k])]
        if vals:
            out[f'{k}_mean'] = float(np.mean(vals))
            out[f'{k}_std']  = float(np.std(vals, ddof=0))
        else:
            out[f'{k}_mean'] = float('nan')
            out[f'{k}_std']  = float('nan')
    return out


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    rows = []

    print(f"Aggregating P1/P2/P3 across {len(SEEDS)} seeds for "
          f"{len(CONFIGS)} configs.\n")

    for tag, label in CONFIGS:
        for attack in ATTACKS:
            print(f"--- {label} × {attack} ---")
            agg = aggregate(tag, attack)
            if agg is None:
                print(f"  [SKIP] No data\n")
                continue

            row = {
                'config':  label,
                'tag':     tag,
                'attack':  attack,
                'n_seeds': agg['n_seeds'],
            }
            for k, v in agg.items():
                if k != 'n_seeds':
                    row[k] = v
            rows.append(row)

            # Print summary
            print(f"  Clean fwd:    {agg['clean_fwd_mean']*100:6.2f} "
                  f"± {agg['clean_fwd_std']*100:.2f}%")
            print(f"  Run B fwd:    {agg['runB_fwd_mean']*100:6.2f} "
                  f"± {agg['runB_fwd_std']*100:.2f}%  "
                  f"(drop = {agg['dropB_fwd_mean']*100:+6.2f} "
                  f"± {agg['dropB_fwd_std']*100:.2f}pp)")
            print(f"  Run C fwd:    {agg['runC_fwd_mean']*100:6.2f} "
                  f"± {agg['runC_fwd_std']*100:.2f}%  "
                  f"(drop = {agg['dropC_fwd_mean']*100:+6.2f} "
                  f"± {agg['dropC_fwd_std']*100:.2f}pp)")
            print()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n✓ Saved: {OUT_PATH}")
    print(f"  ({len(df)} rows: {len(CONFIGS)} configs × {len(ATTACKS)} attacks)")


if __name__ == '__main__':
    main()
