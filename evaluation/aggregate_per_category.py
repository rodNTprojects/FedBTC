"""
FedBTC -- Aggregation script for per-category backward attribution metrics
AND per-exchange global metrics.

Reads `test_metrics.backward_per_category` and `test_metrics.per_exchange`
from each JSON file and computes mean +/- std across seeds. Produces:
  1. CSV:    per_category_backward_summary.csv      (existing, unchanged)
  2. LaTeX:  per_category_backward_table.tex        (existing, unchanged)
  3. CSV:    per_exchange_summary.csv               (NEW: per-exchange fwd+bwd)
  4. LaTeX:  per_exchange_table.tex                 (NEW: per-exchange table)
  5. Report: per_category_backward_report.txt       (extended with per-exchange section)

Usage:
    python aggregate_per_category.py \\
        --results_dir results/evaluations \\
        --out_dir results/figures/

Honest notes:
  - NoMerchant (category 0) is absent from backward_per_category in all
    observed JSON files. Reported honestly in the output rather than
    invented.
  - Per-exchange data is extracted from test_metrics.per_exchange. If a
    JSON file lacks this field, the config's per-exchange data is reported
    as MISSING rather than padded.
  - If a config has fewer than the expected number of seeds, the script
    reports the actual count rather than failing or padding.
  - Configurations defined in TARGET_CONFIGS but absent from
    results_dir are reported as MISSING (not silently dropped).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


# ---------------------------------------------------------------------------
# Configuration filter -- ONLY these configs are aggregated
# ---------------------------------------------------------------------------

TARGET_CONFIGS = {
    'cent_noxedge_100ep':  ('C3 -- Centralized NoXEdge',     1),
    'fl_r100e1':           ('F2 -- FL non-IID (R100xE1)',    2),
    'noxedge_cat':         ('CAT -- Centralized Adv. Train.', 3),
    'fat_base':            ('FAT-base (AT + Byzantine)',     4),
    'fat_def_eps1.0':      ('FAT-def $\\varepsilon=1$',       5),
    'fat_def_eps2.0':      ('FAT-def $\\varepsilon=2$',       6),
    'fat_def_eps4.0':      ('FAT-def $\\varepsilon=4$',       7),
    'fat_def_eps8.0':      ('FAT-def $\\varepsilon=8$',       8),
    'fat_def_eps16.0':     ('FAT-def $\\varepsilon=16$',      9),
    'fat_def_eps32.0':     ('FAT-def $\\varepsilon=32$',     10),
}


CANONICAL_ORDER = ['E-commerce', 'Gambling', 'Services', 'Retail', 'Luxury']

# Per-exchange settings
NUM_EXCHANGES = 3  # FedBTC uses K=3 exchanges


# ---------------------------------------------------------------------------
# Parsing config name from file path
# ---------------------------------------------------------------------------

def parse_filename(path):
    """
    Extract (config_tag, seed) from a results JSON filename.
    Returns None if not parseable.
    """
    stem = path.stem
    for prefix in ('federated_secure_results_',
                   'federated_results_',
                   'baseline_results_'):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    else:
        return None

    m = re.match(r'^(.+)_seed(\d+)$', stem)
    if not m:
        return None
    return m.group(1), int(m.group(2))


# ---------------------------------------------------------------------------
# Loading and extracting metrics
# ---------------------------------------------------------------------------

def extract_metrics(json_path):
    """
    Return a dict with:
      - 'bwd_global': float | None       (backward_accuracy)
      - 'per_category': dict | None      (backward_per_category)
      - 'per_exchange': list[dict] | None (per-exchange fwd + bwd)
    """
    try:
        with json_path.open('r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    test_metrics = data.get('test_metrics', {})
    out = {
        'bwd_global':    test_metrics.get('backward_accuracy'),
        'fwd_global':    test_metrics.get('forward_accuracy'),
        'per_category':  test_metrics.get('backward_per_category'),
        'per_exchange':  test_metrics.get('per_exchange'),
    }
    return out


# ---------------------------------------------------------------------------
# Aggregation (filtered)
# ---------------------------------------------------------------------------

def aggregate(results_dir):
    """
    Walk results_dir, group files by config_tag (ONLY for TARGET_CONFIGS),
    compute mean / std across seeds for both per-category and per-exchange.
    """
    grouped = defaultdict(lambda: {
        'global_fwd':  [],
        'global_bwd':  [],
        'per_cat':     defaultdict(list),
        'per_exch':    defaultdict(lambda: {'fwd': [], 'bwd': []}),
        'seeds':       [],
    })
    skipped_configs = set()

    for path in sorted(Path(results_dir).glob('*.json')):
        parsed = parse_filename(path)
        if parsed is None:
            continue
        config_tag, seed = parsed

        if config_tag not in TARGET_CONFIGS:
            skipped_configs.add(config_tag)
            continue

        metrics = extract_metrics(path)
        if metrics is None:
            continue
        if metrics['bwd_global'] is None:
            continue

        bucket = grouped[config_tag]
        bucket['seeds'].append(seed)
        if metrics['fwd_global'] is not None:
            bucket['global_fwd'].append(metrics['fwd_global'])
        bucket['global_bwd'].append(metrics['bwd_global'])

        # Per-category
        if metrics['per_category'] is not None:
            for cat_name, cat_acc in metrics['per_category'].items():
                bucket['per_cat'][cat_name].append(cat_acc)

        # Per-exchange (NEW)
        if metrics['per_exchange'] is not None:
            for ex_entry in metrics['per_exchange']:
                ex_id = ex_entry.get('exchange_id')
                if ex_id is None:
                    continue
                fwd = ex_entry.get('forward_accuracy')
                bwd = ex_entry.get('backward_accuracy')
                if fwd is not None:
                    bucket['per_exch'][ex_id]['fwd'].append(fwd)
                if bwd is not None:
                    bucket['per_exch'][ex_id]['bwd'].append(bwd)

    summary = {}
    for config_tag, data in grouped.items():
        n = len(data['seeds'])
        entry = {
            'n_seeds':    n,
            'seeds_seen': sorted(data['seeds']),
            'forward_global': {
                'mean':   mean(data['global_fwd']) if data['global_fwd'] else None,
                'std':    stdev(data['global_fwd']) if len(data['global_fwd']) > 1 else 0.0,
            },
            'backward_global': {
                'mean':   mean(data['global_bwd']) if data['global_bwd'] else None,
                'std':    stdev(data['global_bwd']) if len(data['global_bwd']) > 1 else 0.0,
            },
            'categories': {},
            'exchanges':  {},
        }
        for cat_name, values in data['per_cat'].items():
            entry['categories'][cat_name] = {
                'mean': mean(values),
                'std':  stdev(values) if len(values) > 1 else 0.0,
            }
        for ex_id, ex_data in data['per_exch'].items():
            entry['exchanges'][ex_id] = {
                'fwd_mean': mean(ex_data['fwd']) if ex_data['fwd'] else None,
                'fwd_std':  stdev(ex_data['fwd']) if len(ex_data['fwd']) > 1 else 0.0,
                'bwd_mean': mean(ex_data['bwd']) if ex_data['bwd'] else None,
                'bwd_std':  stdev(ex_data['bwd']) if len(ex_data['bwd']) > 1 else 0.0,
                'n_seeds':  len(ex_data['fwd']),
            }
        summary[config_tag] = entry

    return summary, skipped_configs


# ---------------------------------------------------------------------------
# Output formatting -- helpers
# ---------------------------------------------------------------------------

def _categories_in_order(summary):
    all_cats = set()
    for entry in summary.values():
        all_cats.update(entry['categories'].keys())
    return ([c for c in CANONICAL_ORDER if c in all_cats]
            + sorted(c for c in all_cats if c not in CANONICAL_ORDER))


def _sorted_target_tags(summary):
    """Return TARGET_CONFIGS tags in display order, only if present in summary."""
    return [tag for tag, (_, _) in
            sorted(TARGET_CONFIGS.items(), key=lambda kv: kv[1][1])
            if tag in summary]


def _plain_label(config_tag):
    """Strip LaTeX markup from paper_label for plain-text outputs."""
    return TARGET_CONFIGS[config_tag][0].replace(
        '$\\varepsilon', 'eps').replace('$', '')


# ---------------------------------------------------------------------------
# Output 1 -- per-category CSV  (unchanged behavior)
# ---------------------------------------------------------------------------

def write_category_csv(summary, out_path):
    cats_ordered = _categories_in_order(summary)

    with out_path.open('w', encoding='utf-8') as f:
        f.write('config_tag,paper_label,n_seeds,'
                'backward_global_mean,backward_global_std')
        for cat in cats_ordered:
            f.write(f',{cat}_mean,{cat}_std')
        f.write('\n')

        for config_tag in _sorted_target_tags(summary):
            entry = summary[config_tag]
            label_plain = _plain_label(config_tag)
            g = entry['backward_global']
            row = [config_tag,
                   f'"{label_plain}"',
                   str(entry['n_seeds']),
                   f'{g["mean"]:.4f}' if g['mean'] is not None else '',
                   f'{g["std"]:.4f}']
            for cat in cats_ordered:
                cd = entry['categories'].get(cat)
                if cd is None:
                    row.extend(['', ''])
                else:
                    row.extend([f'{cd["mean"]:.4f}', f'{cd["std"]:.4f}'])
            f.write(','.join(row) + '\n')

    print(f'CSV (per-category) : {out_path}')


# ---------------------------------------------------------------------------
# Output 2 -- per-category LaTeX  (unchanged behavior)
# ---------------------------------------------------------------------------

def write_category_latex(summary, out_path):
    cats_ordered = _categories_in_order(summary)

    n_cols = 2 + len(cats_ordered)
    col_spec = 'l' + 'c' * (n_cols - 1)

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Per-category backward attribution accuracy '
                 r'(mean $\pm$ std across seeds, in percent). The '
                 r'\textsc{NoMerchant} class is excluded from this '
                 r'breakdown as it lacks a merchant category label. '
                 r'C3 and F2 are the centralized and federated baselines; '
                 r'CAT, FAT-base, and FAT-def are the security stack '
                 r'configurations.}')
    lines.append(r'  \label{tab:per_category_backward}')
    lines.append(r'  \small')
    lines.append(r'  \begin{tabular}{' + col_spec + '}')
    lines.append(r'    \toprule')
    header = ['Configuration', r'\textbf{Global}']
    header += [r'\textbf{' + c + '}' for c in cats_ordered]
    lines.append('    ' + ' & '.join(header) + r' \\')
    lines.append(r'    \midrule')

    for config_tag in _sorted_target_tags(summary):
        entry = summary[config_tag]
        label = TARGET_CONFIGS[config_tag][0]
        g = entry['backward_global']
        row = [label,
               f'${g["mean"]*100:.2f} \\pm {g["std"]*100:.2f}$']
        for cat in cats_ordered:
            cd = entry['categories'].get(cat)
            if cd is None:
                row.append('---')
            else:
                row.append(f'${cd["mean"]*100:.2f} \\pm {cd["std"]*100:.2f}$')
        lines.append('    ' + ' & '.join(row) + r' \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')

    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'LaTeX (per-category) : {out_path}')


# ---------------------------------------------------------------------------
# Output 3 -- per-exchange CSV  (NEW)
# ---------------------------------------------------------------------------

def write_exchange_csv(summary, out_path):
    """
    One row per (config, exchange_id) pair. Columns:
        config_tag, paper_label, exchange_id, n_seeds,
        forward_mean, forward_std, backward_mean, backward_std
    """
    with out_path.open('w', encoding='utf-8') as f:
        f.write('config_tag,paper_label,exchange_id,n_seeds,'
                'forward_mean,forward_std,backward_mean,backward_std\n')

        for config_tag in _sorted_target_tags(summary):
            entry = summary[config_tag]
            label_plain = _plain_label(config_tag)
            for ex_id in sorted(entry['exchanges'].keys()):
                ex = entry['exchanges'][ex_id]
                row = [
                    config_tag,
                    f'"{label_plain}"',
                    str(ex_id),
                    str(ex['n_seeds']),
                    f'{ex["fwd_mean"]:.4f}' if ex['fwd_mean'] is not None else '',
                    f'{ex["fwd_std"]:.4f}',
                    f'{ex["bwd_mean"]:.4f}' if ex['bwd_mean'] is not None else '',
                    f'{ex["bwd_std"]:.4f}',
                ]
                f.write(','.join(row) + '\n')

    print(f'CSV (per-exchange) : {out_path}')


# ---------------------------------------------------------------------------
# Output 4 -- per-exchange LaTeX  (NEW)
# ---------------------------------------------------------------------------

def write_exchange_latex(summary, out_path):
    """
    Wide table: one row per config, columns grouped by exchange.
    Layout:
      Config | E0 fwd | E0 bwd | E1 fwd | E1 bwd | E2 fwd | E2 bwd
    """
    n_cols = 1 + 2 * NUM_EXCHANGES
    col_spec = 'l' + 'cc' * NUM_EXCHANGES

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Per-exchange forward and backward accuracy '
                 r'(mean $\pm$ std across seeds, in percent). The three '
                 r'exchanges (E0, E1, E2) are the simulated clients of the '
                 r'federated consortium; in centralized configurations '
                 r'(C3, CAT), the per-exchange split is preserved at '
                 r'evaluation time on the same partition.}')
    lines.append(r'  \label{tab:per_exchange_global}')
    lines.append(r'  \small')
    lines.append(r'  \begin{tabular}{' + col_spec + '}')
    lines.append(r'    \toprule')
    # First header row: exchange labels
    h1 = ['\\multirow{2}{*}{Configuration}']
    for ex_id in range(NUM_EXCHANGES):
        h1.append(f'\\multicolumn{{2}}{{c}}{{\\textbf{{E{ex_id}}}}}')
    lines.append('    ' + ' & '.join(h1) + r' \\')
    # cmidrules
    cmid_parts = []
    for k in range(NUM_EXCHANGES):
        col_start = 2 + 2 * k
        col_end   = col_start + 1
        cmid_parts.append(f'\\cmidrule(lr){{{col_start}-{col_end}}}')
    lines.append('    ' + ''.join(cmid_parts))
    # Second header row: fwd/bwd labels
    h2 = ['']
    for _ in range(NUM_EXCHANGES):
        h2.extend(['Fwd', 'Bwd'])
    lines.append('    ' + ' & '.join(h2) + r' \\')
    lines.append(r'    \midrule')

    for config_tag in _sorted_target_tags(summary):
        entry = summary[config_tag]
        label = TARGET_CONFIGS[config_tag][0]
        row = [label]
        for ex_id in range(NUM_EXCHANGES):
            ex = entry['exchanges'].get(ex_id)
            if ex is None or ex['fwd_mean'] is None:
                row.extend(['---', '---'])
            else:
                row.append(f'${ex["fwd_mean"]*100:.2f} \\pm {ex["fwd_std"]*100:.2f}$')
                row.append(f'${ex["bwd_mean"]*100:.2f} \\pm {ex["bwd_std"]*100:.2f}$')
        lines.append('    ' + ' & '.join(row) + r' \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')

    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'LaTeX (per-exchange) : {out_path}')


# ---------------------------------------------------------------------------
# Output 5 -- summary report (extended with per-exchange section)
# ---------------------------------------------------------------------------

def write_summary_report(summary, skipped, out_path):
    cats_present = set()
    exchanges_present = set()
    for entry in summary.values():
        cats_present.update(entry['categories'].keys())
        exchanges_present.update(entry['exchanges'].keys())

    lines = []
    lines.append('=' * 72)
    lines.append('FedBTC -- Per-Category Backward + Per-Exchange Aggregation')
    lines.append('=' * 72)
    lines.append('')
    lines.append(f'Categories present in at least one config : '
                 f'{sorted(cats_present)}')
    lines.append(f'Exchanges present in at least one config  : '
                 f'{sorted(exchanges_present)}')

    if 'NoMerchant' not in cats_present:
        lines.append('')
        lines.append('NOTE: NoMerchant (category 0) is absent from the '
                     'per-category breakdown in all evaluated JSON files.')
        lines.append('  -> Consistent with reporting per-category metrics '
                     'only for the 5 substantive merchant categories.')
        lines.append('  -> To be mentioned in the paper caption / footnote.')

    lines.append('')
    lines.append(f'Target configurations  : {len(TARGET_CONFIGS)}')
    lines.append(f'Configurations found   : {len(summary)}')
    lines.append(f'Configurations missing : '
                 f'{len(TARGET_CONFIGS) - len(summary)}')

    missing = set(TARGET_CONFIGS) - set(summary)
    if missing:
        lines.append('')
        lines.append('MISSING configurations (defined in TARGET_CONFIGS '
                     'but no JSON found):')
        for m in sorted(missing):
            lines.append(f'  - {m}')

    # --------- per-config global ---------
    lines.append('')
    lines.append('-' * 72)
    lines.append('Per-config seed counts, global forward + backward accuracy:')
    lines.append('-' * 72)
    for config_tag in _sorted_target_tags(summary):
        entry = summary[config_tag]
        n = entry['n_seeds']
        seeds = entry['seeds_seen']
        gf = entry['forward_global']
        gb = entry['backward_global']
        label = _plain_label(config_tag)
        marker = '' if n == 5 else '  <-- WARNING: not 5 seeds'
        gf_str = (f'{gf["mean"]*100:6.2f}+/-{gf["std"]*100:.2f}%'
                  if gf['mean'] is not None else 'n/a')
        gb_str = (f'{gb["mean"]*100:6.2f}+/-{gb["std"]*100:.2f}%'
                  if gb['mean'] is not None else 'n/a')
        lines.append(f'  {label:35s}  n={n}  '
                     f'fwd={gf_str}  bwd={gb_str}{marker}')
        lines.append(f'      tag : {config_tag}')
        lines.append(f'      seeds: {seeds}')

    # --------- per-exchange section ---------
    lines.append('')
    lines.append('-' * 72)
    lines.append('Per-exchange global accuracy (mean +/- std, percent):')
    lines.append('-' * 72)
    for config_tag in _sorted_target_tags(summary):
        entry = summary[config_tag]
        label = _plain_label(config_tag)
        lines.append(f'  {label}')
        if not entry['exchanges']:
            lines.append('      (no per-exchange data)')
            continue
        for ex_id in sorted(entry['exchanges'].keys()):
            ex = entry['exchanges'][ex_id]
            if ex['fwd_mean'] is None or ex['bwd_mean'] is None:
                lines.append(f'      E{ex_id}: (incomplete)')
                continue
            lines.append(
                f'      E{ex_id} (n={ex["n_seeds"]}): '
                f'fwd={ex["fwd_mean"]*100:6.2f}+/-{ex["fwd_std"]*100:.2f}%  '
                f'bwd={ex["bwd_mean"]*100:6.2f}+/-{ex["bwd_std"]*100:.2f}%')

    if skipped:
        lines.append('')
        lines.append('-' * 72)
        lines.append(f'Skipped configurations (not in TARGET_CONFIGS): '
                     f'{len(skipped)}')
        lines.append('-' * 72)
        for s in sorted(skipped):
            lines.append(f'  - {s}')

    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Report : {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate per-category backward and per-exchange '
                    'metrics across seeds, filtered to paper-relevant '
                    'configs only.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing the evaluation JSON files')
    parser.add_argument('--out_dir', type=str, default='.',
                        help='Directory to write CSV, LaTeX, and summary report')
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        raise SystemExit(f'Error: results_dir does not exist: {results_dir}')

    print(f'Scanning: {results_dir}')
    summary, skipped = aggregate(results_dir)
    print(f'Found {len(summary)}/{len(TARGET_CONFIGS)} target configurations.')
    print(f'Skipped {len(skipped)} non-target configurations.')
    print()

    write_category_csv(summary,
                       out_dir / 'per_category_backward_summary.csv')
    write_category_latex(summary,
                         out_dir / 'per_category_backward_table.tex')
    write_exchange_csv(summary,
                       out_dir / 'per_exchange_summary.csv')
    write_exchange_latex(summary,
                         out_dir / 'per_exchange_table.tex')
    write_summary_report(summary, skipped,
                         out_dir / 'per_category_backward_report.txt')


if __name__ == '__main__':
    main()