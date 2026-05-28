"""
analyze_results.py
FedBTC — Result Aggregation and Analysis
=========================================

Run this script AFTER retrieving results from the cluster.

Handles TWO phases of results:
  Phase 1 (already run): C1/C2/C3, F1/F2/F3/F4, A1–A6 ablations
  Phase 6+ (new):        CAT, FAT-base, FAT-def (ε sweep), security eval

Produces:
  1. Main comparison table (utility)           → console + CSV + LaTeX
  2. Ablation multi-task table                 → console + CSV
  3. DP ε-sweep table (privacy-utility curve)  → console + CSV
  4. Security evaluation table                 → console + CSV
  5. Figures PNG 300dpi                        → save_dir/

Usage
-----
python analyze_results.py --results_dir results/evaluations
python analyze_results.py --results_dir results/evaluations --latex
python analyze_results.py --results_dir results/evaluations --phase1_only
python analyze_results.py --results_dir results/evaluations --phase6_only

Seeds: 0, 45, 123, 654, 789

Notes on P1 reading (interpretation 2 — orthogonal-layer decomposition)
------------------------------------------------------------------------
Per Table 3 of defense_choices.docx, P1 attacks the FL aggregation rule,
not the model. We therefore compute P1-drop directly from the raw attack
JSONs at <attacks_dir>/attack_p1_seed{N}_{N}.json:
  - FAT-Base (no Byzantine defense): drop = run_a.fwd - run_b.fwd
  - FAT-Def  (CoordMedian+NormBound): drop = run_a.fwd - run_c.fwd
  - CAT (centralized): N/A (no FL aggregation to corrupt)

Tag-naming normalization
------------------------
Some training-result and security_eval files use an underscore in the
epsilon suffix (e.g. 'fat_def_eps16_0' instead of 'fat_def_eps16.0').
We normalize internally so that both conventions map to the same tag
('fat_def_eps16.0'), allowing analyze_results.py to merge results
regardless of the producing script's filename convention.
"""

import os, sys, json, glob, csv, argparse, re
import numpy as np
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Tag → label mapping (matches exact --results_tag values used in scripts)
# ─────────────────────────────────────────────────────────────────────────────

SEEDS = [0, 45, 123, 654, 789]

# Phase 1 — already executed
PHASE1_CONFIGS = {
    # tag                      label (paper)
    'cent_100ep':              'C1 — Centralized 100ep',
    'cent_200ep':              'C2 — Centralized 200ep',
    'cent_noxedge_100ep':      'C3 — Centralized NoXEdge 100ep',
    'fl_r20e3':                'F1 — FL non-IID R20×E3',
    'fl_r100e1':               'F2 — FL non-IID R100×E1',
    'fl_iid_noxedge_r20e3':    'F3 — FL IID NoXEdge R20×E3',
    'fl_iid_noxedge_r100e1':   'F4 — FL IID NoXEdge R100×E1',
}

ABLATION_CONFIGS = {
    'single_fwd':              'A1 — Single-task Fwd (Centralized)',
    'single_bwd':              'A2 — Single-task Bwd (Centralized)',
    'noxedge_single_fwd':      'A3 — Single-task Fwd (NoXEdge)',
    'noxedge_single_bwd':      'A4 — Single-task Bwd (NoXEdge)',
    'fl_r100e1_single_fwd':    'A5 — Single-task Fwd (FL R100×E1)',
    'fl_r100e1_single_bwd':    'A6 — Single-task Bwd (FL R100×E1)',
}

# Phase 6+ — new (CAT / FAT / security)
PHASE6_CONFIGS = {
    'noxedge_cat':             'CAT (NoXEdge + AT)',
    'fat_base':                'FAT-base (AT, no DP)',
    'fat_def_eps32.0':         'FAT-def (eps=32.0)',
    'fat_def_eps16.0':         'FAT-def (eps=16.0)',
    'fat_def_eps8.0':          'FAT-def (eps=8.0)',
    'fat_def_eps4.0':          'FAT-def (eps=4.0)',
    'fat_def_eps2.0':          'FAT-def (eps=2.0)',
    'fat_def_eps1.0':          'FAT-def (eps=1.0)',
}

DP_EPSILONS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

ALL_CONFIGS = {**PHASE1_CONFIGS, **ABLATION_CONFIGS, **PHASE6_CONFIGS}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(d, *keys, default=float('nan')):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    try:
        return float(d)
    except (TypeError, ValueError):
        return default


def _extract_seed(path: str) -> Optional[int]:
    m = re.search(r'_seed(\d+)\.json$', path)
    return int(m.group(1)) if m else None


def _normalize_tag(tag: str) -> str:
    """Normalize underscore-decimal naming in epsilon suffix.

    Handles both 'fat_def_eps16_0' (from some training scripts that
    sanitize filenames) and 'fat_def_eps16.0' (canonical), mapping both
    to 'fat_def_eps16.0'. Idempotent.
    """
    return re.sub(r'(_eps\d+)_(\d+)$', r'\1.\2', tag)


def _mean_std(vals):
    v = [x for x in vals if not np.isnan(x)]
    if not v:
        return float('nan'), float('nan')
    return float(np.mean(v)), float(np.std(v))


def _fmt(mean, std=None, pct=True, n_seeds=None):
    if np.isnan(mean):
        return '—'
    scale = 100 if pct else 1
    if std is not None and not np.isnan(std):
        return f'{mean*scale:.2f} ± {std*scale:.2f}'
    return f'{mean*scale:.2f}'


def extract_utility(result: dict) -> Tuple[float, float]:
    """Handle both p2 and p3 JSON schemas."""
    # Both p2 and p3 use: result['test_metrics']['forward_accuracy']
    fwd = _safe(result, 'test_metrics', 'forward_accuracy')
    bwd = _safe(result, 'test_metrics', 'backward_accuracy')
    return fwd, bwd


def _read_p1_attack_file(seed: int, attacks_dir: str) -> Optional[dict]:
    """Load the raw attack_p1 JSON for a given seed."""
    path = os.path.join(attacks_dir, f'attack_p1_seed{seed}_{seed}.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f'  WARN: could not read {path}: {e}')
        return None


def _compute_p1_drop_from_attack_files(tag: str, seed: int,
                                       attacks_dir: str,
                                       security: Optional[dict] = None) -> float:
    """Compute P1-drop for a FAT config by reading the raw attack_p1 JSON.

    Per design (Table 3, defense_choices.docx, interpretation 2):
      - FAT-Base (no Byzantine defense): drop = run_a.fwd - run_b.fwd
      - FAT-Def  (CoordMedian + NormBound): drop = run_a.fwd - run_c.fwd
      - CAT (centralized) and other tags: N/A (returns NaN)

    P1-PATCH: fallback to security_eval when attack_p1 missing.
    If the raw attack_p1 JSON does not exist for this (tag, seed), and the
    `security` mapping is provided, fall back to:
      - clean_fwd - P1.no_defense.forward_acc   (for fat_base)
      - clean_fwd - P1.with_defense.forward_acc (for fat_def_eps*)
    where clean_fwd = security[tag][seed].utility.forward_acc.
    This makes use of the byzantine_noise-patched security_eval files that
    contain attacks.P1.{no_defense,with_defense}.forward_acc.
    """
    if tag != 'fat_base' and not tag.startswith('fat_def_eps'):
        return float('nan')

    # Primary path: raw attack_p1 JSON in attacks_dir
    d = _read_p1_attack_file(seed, attacks_dir)
    # P1-PATCH-v2: skip primary for fat_def_eps (attack_p1 has no run_c).
    if d is not None and tag == "fat_base":
        run_a_fwd = _safe(d, 'run_a', 'forward_acc')
        if tag == 'fat_base':
            run_b_fwd = _safe(d, 'run_b', 'forward_acc')
            if np.isnan(run_a_fwd + run_b_fwd):
                return float('nan')
            return run_a_fwd - run_b_fwd
        else:  # tag.startswith('fat_def_eps')
            run_c_fwd = _safe(d, 'run_c', 'forward_acc')
            if np.isnan(run_a_fwd + run_c_fwd):
                return float('nan')
            return run_a_fwd - run_c_fwd

    # Fallback: read P1 from security_eval (post byzantine_noise patch)
    if security is None:
        return float('nan')
    sec_for_tag = security.get(tag, {})
    sec = sec_for_tag.get(seed)
    if sec is None:
        return float('nan')
    clean_fwd = _safe(sec, 'utility', 'forward_acc')
    if tag == 'fat_base':
        p1_fwd = _safe(sec, 'attacks', 'P1', 'no_defense', 'forward_acc')
    else:  # tag.startswith('fat_def_eps')
        p1_fwd = _safe(sec, 'attacks', 'P1', 'with_defense', 'forward_acc')
    if np.isnan(clean_fwd + p1_fwd):
        return float('nan')
    return clean_fwd - p1_fwd



# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_training(results_dir: str) -> Dict[str, Dict[int, dict]]:
    """Load all training JSONs keyed by (tag, seed).

    File naming convention:
      p2 scripts  → baseline_results_{tag}_seed{N}.json
      p3 scripts  → federated_results_{tag}_seed{N}.json
      p3_secure   → federated_secure_results_{tag}_seed{N}.json

    Tags are normalized via _normalize_tag to harmonize the two epsilon
    naming conventions (eps16_0 vs eps16.0).
    """
    data: Dict[str, Dict[int, dict]] = {}

    patterns = [
        ('baseline_results_*_seed*.json',          None),
        ('federated_results_*_seed*.json',          None),
        ('federated_secure_results_*_seed*.json',   None),
    ]

    for pattern, _ in patterns:
        for path in glob.glob(os.path.join(results_dir, pattern)):
            seed = _extract_seed(path)
            if seed is None:
                continue
            fname = os.path.basename(path)
            for prefix in ('baseline_results_', 'federated_results_',
                           'federated_secure_results_'):
                if fname.startswith(prefix):
                    tag = fname[len(prefix):]
                    break
            else:
                tag = fname
            tag = re.sub(r'_seed\d+\.json$', '', tag)
            tag = _normalize_tag(tag)

            if tag not in data:
                data[tag] = {}
            try:
                with open(path) as f:
                    data[tag][seed] = json.load(f)
            except Exception as e:
                print(f'  WARN: could not load {path}: {e}')

    return data


def load_security(results_dir: str) -> Dict[str, Dict[int, dict]]:
    """Load security eval JSONs: security_eval_{tag}_seed{N}.json.

    Tags are normalized via _normalize_tag.
    """
    data: Dict[str, Dict[int, dict]] = {}
    for path in glob.glob(os.path.join(results_dir, 'security_eval_*_seed*.json')):
        seed = _extract_seed(path)
        if seed is None:
            continue
        fname = os.path.basename(path)
        tag   = re.sub(r'_seed\d+\.json$', '',
                       fname.replace('security_eval_', ''))
        tag   = _normalize_tag(tag)
        if tag not in data:
            data[tag] = {}
        try:
            with open(path) as f:
                data[tag][seed] = json.load(f)
        except Exception as e:
            print(f'  WARN: {e}')
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def build_utility_table(training: dict, config_map: dict) -> List[dict]:
    rows = []
    for tag, label in config_map.items():
        fwd_vals, bwd_vals = [], []
        for seed, res in training.get(tag, {}).items():
            f, b = extract_utility(res)
            fwd_vals.append(f); bwd_vals.append(b)
        rows.append({
            'tag': tag, 'label': label,
            'n_seeds': len(fwd_vals),
            'fwd_mean': _mean_std(fwd_vals)[0],
            'fwd_std':  _mean_std(fwd_vals)[1],
            'bwd_mean': _mean_std(bwd_vals)[0],
            'bwd_std':  _mean_std(bwd_vals)[1],
        })
    return rows


def build_security_table(training: dict, security: dict,
                         config_map: dict,
                         attacks_dir: str = 'results/attacks') -> List[dict]:
    """Build the per-config security table.

    P1-drop is computed via dispatch on tag:
      - 'noxedge_cat' → NaN (centralized, P1 N/A by design)
      - 'fat_base' / 'fat_def_eps*' → from raw attack_p1 JSONs in attacks_dir
      - other tags → legacy fallback reading the security_eval P1 sub-dict
    """
    rows = []
    for tag, label in config_map.items():
        fwd_vals, e1_drops, e2_drops, p1_drops, mia_advs, dlg_mses = \
            [], [], [], [], [], []

        for seed, res in training.get(tag, {}).items():
            f, _ = extract_utility(res)
            fwd_vals.append(f)

        for seed, sec in security.get(tag, {}).items():
            atk   = sec.get('attacks', {})
            clean = _safe(sec, 'utility', 'forward_acc')
            e1_drops.append(_safe(atk, 'E1', 'acc_drop'))
            e2_drops.append(_safe(atk, 'E2', 'acc_drop'))

            # ── P1 dispatch (interpretation 2: orthogonal layers) ──
            if tag == 'noxedge_cat':
                p1_drops.append(float('nan'))
            elif tag == 'fat_base' or tag.startswith('fat_def_eps'):
                p1_drops.append(_compute_p1_drop_from_attack_files(
                    tag, seed, attacks_dir, security=security))
            else:
                p1_nd = _safe(atk, 'P1', 'no_defense', 'forward_acc')
                p1_drops.append(
                    clean - p1_nd
                    if not np.isnan(clean + p1_nd) else float('nan'))

            mia_advs.append(_safe(atk, 'D1', 'loss_based', 'mia_advantage'))
            dlg_mses.append(_safe(atk, 'D2', 'avg_mse'))

        rows.append({
            'tag': tag, 'label': label,
            'n_seeds_train':  len(fwd_vals),
            'n_seeds_sec':    len(mia_advs),
            'fwd_mean':       _mean_std(fwd_vals)[0],
            'fwd_std':        _mean_std(fwd_vals)[1],
            'e1_drop_mean':   _mean_std(e1_drops)[0],
            'e1_drop_std':    _mean_std(e1_drops)[1],
            'e2_drop_mean':   _mean_std(e2_drops)[0],
            'e2_drop_std':    _mean_std(e2_drops)[1],
            'p1_drop_mean':   _mean_std(p1_drops)[0],
            'p1_drop_std':    _mean_std(p1_drops)[1],
            'mia_adv_mean':   _mean_std(mia_advs)[0],
            'mia_adv_std':    _mean_std(mia_advs)[1],
            'dlg_mse_mean':   _mean_std(dlg_mses)[0],
            'dlg_mse_std':    _mean_std(dlg_mses)[1],
        })
    return rows


def build_epsilon_table(training: dict, security: dict) -> List[dict]:
    rows = []
    for eps in DP_EPSILONS:
        tag = f'fat_def_eps{eps}'
        fwd_v, bwd_v, mia_v, dlg_v = [], [], [], []
        for seed, res in training.get(tag, {}).items():
            f, b = extract_utility(res)
            fwd_v.append(f); bwd_v.append(b)
        for seed, sec in security.get(tag, {}).items():
            mia_v.append(_safe(sec, 'attacks', 'D1', 'loss_based', 'mia_advantage'))
            dlg_v.append(_safe(sec, 'attacks', 'D2', 'avg_mse'))
        rows.append({
            'epsilon': eps, 'n_seeds': len(fwd_v),
            'fwd_mean': _mean_std(fwd_v)[0], 'fwd_std': _mean_std(fwd_v)[1],
            'bwd_mean': _mean_std(bwd_v)[0], 'bwd_std': _mean_std(bwd_v)[1],
            'mia_mean': _mean_std(mia_v)[0], 'mia_std': _mean_std(mia_v)[1],
            'dlg_mean': _mean_std(dlg_v)[0], 'dlg_std': _mean_std(dlg_v)[1],
        })
    return rows


def build_defense_table(security: dict,
                        attacks_dir: str = 'results/attacks') -> List[dict]:
    """Defense effectiveness table: no-defense vs with-defense drop.

    For P1, both columns are read from the raw attack_p1 JSONs.
    For P2/P3, the legacy security_eval format is used.
    """
    rows = []
    for tag in ['fat_base', 'fat_def_eps8.0']:
        label = PHASE6_CONFIGS.get(tag, tag)
        for atk in ['P1', 'P2', 'P3']:
            nd_v, wd_v = [], []
            for seed, sec in security.get(tag, {}).items():
                clean = _safe(sec, 'utility', 'forward_acc')

                if atk == 'P1':
                    # DEFENSE-TABLE-PATCH: split P1 dispatch by tag
                    # - fat_base: attack_p1 JSON (run_a / run_b protocol)
                    # - fat_def_eps*: read directly from security_eval
                    if tag == 'fat_base':
                        pd = _read_p1_attack_file(seed, attacks_dir)
                        if pd is not None:
                            run_a = _safe(pd, 'run_a', 'forward_acc')
                            run_b = _safe(pd, 'run_b', 'forward_acc')
                            if not np.isnan(run_a + run_b):
                                nd_v.append(run_a - run_b)
                            # fat_base has no Byzantine defense by design
                            # -> wd column intentionally left empty
                    else:  # tag.startswith('fat_def_eps')
                        entry = sec.get('attacks', {}).get('P1', {})
                        nd = _safe(entry, 'no_defense',   'forward_acc')
                        wd = _safe(entry, 'with_defense', 'forward_acc')
                        if not np.isnan(clean + nd):
                            nd_v.append(clean - nd)
                        if not np.isnan(clean + wd):
                            wd_v.append(clean - wd)
                else:
                    entry = sec.get('attacks', {}).get(atk, {})
                    nd    = _safe(entry, 'no_defense',   'forward_acc')
                    wd    = _safe(entry, 'with_defense', 'forward_acc')
                    if not np.isnan(clean + nd):
                        nd_v.append(clean - nd)
                    if not np.isnan(clean + wd):
                        wd_v.append(clean - wd)

            rows.append({
                'config': label, 'attack': atk,
                'nd_drop_mean': _mean_std(nd_v)[0],
                'nd_drop_std':  _mean_std(nd_v)[1],
                'wd_drop_mean': _mean_std(wd_v)[0],
                'wd_drop_std':  _mean_std(wd_v)[1],
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

W = 82

def _header(title):
    print(f'\n{"="*W}\n  {title}\n{"─"*W}')


def print_utility_table(rows, title='UTILITY', latex=False):
    _header(title)
    hdr = f"{'Configuration':42s}  {'N':3s}  {'Fwd Acc %':15s}  {'Bwd Acc %':15s}"
    print(hdr); print('─'*len(hdr))
    for r in rows:
        fwd = _fmt(r['fwd_mean'], r['fwd_std'])
        bwd = _fmt(r['bwd_mean'], r['bwd_std'])
        print(f"{r['label']:42s}  {r['n_seeds']:3d}  {fwd:15s}  {bwd:15s}")
    if latex:
        print(f'\n-- LaTeX --')
        for r in rows:
            print(f"  {r['label']} & {_fmt(r['fwd_mean'],r['fwd_std'])} "
                  f"& {_fmt(r['bwd_mean'],r['bwd_std'])} \\\\")


def print_security_table(rows, latex=False):
    _header('SECURITY EVALUATION (Phase 6+)')
    hdr = (f"{'Config':30s}  {'Fwd %':12s}  {'E1-drop':9s}  "
           f"{'E2-drop':9s}  {'P1-drop':9s}  {'MIA-Adv':9s}  {'DLG-MSE':9s}")
    print(hdr); print('─'*len(hdr))
    for r in rows:
        print(
            f"{r['label']:30s}  "
            f"{_fmt(r['fwd_mean'],r['fwd_std']):12s}  "
            f"{_fmt(r['e1_drop_mean'],r['e1_drop_std']):9s}  "
            f"{_fmt(r['e2_drop_mean'],r['e2_drop_std']):9s}  "
            f"{_fmt(r['p1_drop_mean'],r['p1_drop_std']):9s}  "
            f"{_fmt(r['mia_adv_mean'],r['mia_adv_std'],pct=False):9s}  "
            f"{_fmt(r['dlg_mse_mean'],r['dlg_mse_std'],pct=False):9s}"
        )


def print_epsilon_table(rows, latex=False):
    _header('DP ε-SWEEP — PRIVACY-UTILITY CURVE')
    hdr = f"{'ε':5s}  {'N':3s}  {'Fwd %':15s}  {'Bwd %':15s}  {'MIA-Adv':12s}  {'DLG-MSE':10s}"
    print(hdr); print('─'*len(hdr))
    for r in rows:
        print(
            f"{r['epsilon']:<5.1f}  {r['n_seeds']:3d}  "
            f"{_fmt(r['fwd_mean'],r['fwd_std']):15s}  "
            f"{_fmt(r['bwd_mean'],r['bwd_std']):15s}  "
            f"{_fmt(r['mia_mean'],r['mia_std'],pct=False):12s}  "
            f"{_fmt(r['dlg_mean'],r['dlg_std'],pct=False):10s}"
        )
    if latex:
        print('\n-- LaTeX --')
        for r in rows:
            print(f"  {r['epsilon']:.1f} & {_fmt(r['fwd_mean'],r['fwd_std'])} "
                  f"& {_fmt(r['bwd_mean'],r['bwd_std'])} "
                  f"& {_fmt(r['mia_mean'],r['mia_std'],pct=False)} "
                  f"& {_fmt(r['dlg_mean'],r['dlg_std'],pct=False)} \\\\")


def print_defense_table(rows):
    _header('DEFENSE EFFECTIVENESS (forward acc drop no-def vs with-def)')
    hdr = f"{'Config':30s}  {'Attack':6s}  {'No defense %':15s}  {'With defense %':15s}"
    print(hdr); print('─'*len(hdr))
    for r in rows:
        print(
            f"{r['config']:30s}  {r['attack']:6s}  "
            f"{_fmt(r['nd_drop_mean'],r['nd_drop_std']):15s}  "
            f"{_fmt(r['wd_drop_mean'],r['wd_drop_std']):15s}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CSV + Figures
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(rows, path):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f'  CSV: {path}')


def generate_figures(p1_rows, security_rows, eps_rows, save_dir):
    try:
        import matplotlib.pyplot as plt, matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print('matplotlib not available — skipping figures'); return

    os.makedirs(save_dir, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Fig 1: Phase 1 — Fwd vs Bwd accuracy (bar chart)
    valid = [(r['label'], r['fwd_mean'], r['fwd_std'],
               r['bwd_mean'], r['bwd_std'])
              for r in p1_rows if not np.isnan(r['fwd_mean'])]
    if valid:
        labels, fw, fs, bw, bs = zip(*valid)
        x = np.arange(len(labels)); w = 0.35
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x - w/2, [v*100 for v in fw], w, yerr=[v*100 for v in fs],
               label='Forward Acc', capsize=3, color='steelblue')
        ax.bar(x + w/2, [v*100 for v in bw], w, yerr=[v*100 for v in bs],
               label='Backward Acc', capsize=3, color='darkorange')
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy % (mean ± std, 5 seeds)', fontsize=10)
        ax.set_title('FedBTC — Phase 1: Utility Comparison', fontsize=12)
        ax.legend(); plt.tight_layout()
        p = os.path.join(save_dir, 'fig1_phase1_utility.png')
        plt.savefig(p, dpi=300); plt.close(); print(f'  {p}')

    # Fig 2: DP epsilon curve
    valid_eps = [r for r in eps_rows if not np.isnan(r['fwd_mean'])]
    if valid_eps:
        eps_v = [r['epsilon']  for r in valid_eps]
        fwd_v = [r['fwd_mean'] for r in valid_eps]
        fwd_s = [r['fwd_std']  for r in valid_eps]
        mia_v = [r['mia_mean'] for r in valid_eps]
        mia_s = [r['mia_std']  for r in valid_eps]
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.errorbar(eps_v, [v*100 for v in fwd_v],
                     yerr=[v*100 for v in fwd_s],
                     fmt='o-', color='tab:blue', capsize=4, label='Fwd Acc %')
        ax1.set_xlabel('DP Budget ε', fontsize=11)
        ax1.set_ylabel('Forward Accuracy % (↑)', color='tab:blue', fontsize=11)
        ax2 = ax1.twinx()
        ax2.errorbar(eps_v, mia_v, yerr=mia_s,
                     fmt='s--', color='tab:red', capsize=4, label='MIA Advantage')
        ax2.set_ylabel('MIA Advantage (↓)', color='tab:red', fontsize=11)
        plt.title('Privacy-Utility Curve (Server-side DP)', fontsize=12)
        fig.tight_layout()
        p = os.path.join(save_dir, 'fig2_dp_epsilon_curve.png')
        plt.savefig(p, dpi=300); plt.close(); print(f'  {p}')

    # Fig 3: Security — E1 robustness drop
    sec_valid = [(r['label'], r['e1_drop_mean'], r['e1_drop_std'])
                  for r in security_rows if not np.isnan(r['e1_drop_mean'])]
    if sec_valid:
        labels, drops, stds = zip(*sec_valid)
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x, [v*100 for v in drops], yerr=[v*100 for v in stds],
               capsize=4, color='steelblue', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('Forward Acc Drop at ε=0.2 pp (↓ = more robust)', fontsize=9)
        ax.set_title('Robustness under PGD Evasion E1', fontsize=11)
        plt.tight_layout()
        p = os.path.join(save_dir, 'fig3_evasion_robustness.png')
        plt.savefig(p, dpi=300); plt.close(); print(f'  {p}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='FedBTC Result Analysis')
    ap.add_argument('--results_dir',  default='results/evaluations')
    ap.add_argument('--save_dir',     default='results/figures')
    ap.add_argument('--attacks_dir',  default='results/attacks',
                    help='Directory containing raw attack JSONs '
                         '(used for FAT P1 reading).')
    ap.add_argument('--latex',        action='store_true')
    ap.add_argument('--no_figures',   action='store_true')
    ap.add_argument('--phase1_only',  action='store_true',
                    help='Only show Phase 1 results (C/F/A configs)')
    ap.add_argument('--phase6_only',  action='store_true',
                    help='Only show Phase 6+ results (CAT/FAT/security)')
    args = ap.parse_args()

    print(f'\nLoading from: {args.results_dir}')
    print(f'Attack JSONs: {args.attacks_dir}')
    training = load_all_training(args.results_dir)
    security = load_security(args.results_dir)

    found_tags = sorted(training.keys())
    print(f'  Found {len(found_tags)} training result tags: {found_tags}')
    print(f'  Found {len(security)} security result tags: {sorted(security.keys())}')

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Phase 1 tables ───────────────────────────────────────────────────────
    if not args.phase6_only:
        p1_rows  = build_utility_table(training, PHASE1_CONFIGS)
        abl_rows = build_utility_table(training, ABLATION_CONFIGS)
        print_utility_table(p1_rows,  title='PHASE 1 — MAIN CONFIGS (C/F)', latex=args.latex)
        print_utility_table(abl_rows, title='PHASE 1 — ABLATION MULTI-TASK (A1–A6)', latex=args.latex)
        save_csv(p1_rows,  os.path.join(args.save_dir, 'table_phase1_main.csv'))
        save_csv(abl_rows, os.path.join(args.save_dir, 'table_phase1_ablation.csv'))
    else:
        p1_rows = []

    # ── Phase 6 tables ───────────────────────────────────────────────────────
    if not args.phase1_only:
        p6_util   = build_utility_table(training, PHASE6_CONFIGS)
        p6_sec    = build_security_table(training, security, PHASE6_CONFIGS,
                                         attacks_dir=args.attacks_dir)
        eps_rows  = build_epsilon_table(training, security)
        def_rows  = build_defense_table(security, attacks_dir=args.attacks_dir)
        print_utility_table(p6_util, title='PHASE 6 — UTILITY (CAT/FAT)', latex=args.latex)
        print_security_table(p6_sec, latex=args.latex)
        print_epsilon_table(eps_rows, latex=args.latex)
        print_defense_table(def_rows)
        save_csv(p6_util,  os.path.join(args.save_dir, 'table_phase6_utility.csv'))
        save_csv(p6_sec,   os.path.join(args.save_dir, 'table_phase6_security.csv'))
        save_csv(eps_rows, os.path.join(args.save_dir, 'table_epsilon_sweep.csv'))
        save_csv(def_rows, os.path.join(args.save_dir, 'table_defense.csv'))
    else:
        p6_sec = []; eps_rows = []

    # ── Figures ──────────────────────────────────────────────────────────────
    if not args.no_figures:
        print('\nGenerating figures...')
        generate_figures(p1_rows, p6_sec, eps_rows, args.save_dir)

    print(f'\nOutputs in: {args.save_dir}/')


if __name__ == '__main__':
    main()