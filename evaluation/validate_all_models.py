"""
validate_all_models.py — Phase 5 (post-retraining global validation)
====================================================================

After all 4 SLURM arrays have completed (130 jobs total), run this
script to verify that all expected models and results files exist
and are valid.

Aligned with the corrected SLURM v2/v3 (matches analyze_results.py
PHASE1+ABLATION+PHASE6 + p5_interaction_matrix.py CONFIGS).

It produces a clear pass/fail report listing exactly which models
or evaluation results are missing or invalid, so you can re-submit
only the failing array indices instead of re-running everything.

Notes
-----
- Phase 5 models (coord_only, dp_only, at_coord, coord_dp, at_dp)
  are produced by p5_train_defense_configs.py which does NOT save
  a separate evaluation JSON. So we check only the .pt + .canonical
  marker for those 25 models.
- All other 105 models (centralized, FL standard, FAT) produce both
  .pt + .json + .canonical.

Usage
-----
    cd ~/links/scratch/fedbtc
    python validate_all_models.py
"""

import os
import sys
import json
from pathlib import Path

SEEDS = [0, 45, 123, 654, 789]

# ─────────────────────────────────────────────────────────────────────
# Expected models, organized by category.
# Format: (tag, model_filename_pattern, json_filename_pattern_or_None)
# ─────────────────────────────────────────────────────────────────────

EXPECTED = {
    'centralized (40 = 8 × 5)': [
        ('cent_100ep',
         'baseline_dual_attribution_cent_100ep_seed{seed}.pt',
         'baseline_results_cent_100ep_seed{seed}.json'),
        ('cent_200ep',
         'baseline_dual_attribution_cent_200ep_seed{seed}.pt',
         'baseline_results_cent_200ep_seed{seed}.json'),
        ('cent_noxedge_100ep',
         'baseline_dual_attribution_cent_noxedge_100ep_seed{seed}.pt',
         'baseline_results_cent_noxedge_100ep_seed{seed}.json'),
        ('noxedge_cat',
         'baseline_dual_attribution_noxedge_cat_seed{seed}.pt',
         'baseline_results_noxedge_cat_seed{seed}.json'),
        ('single_fwd',
         'baseline_dual_attribution_single_fwd_seed{seed}.pt',
         'baseline_results_single_fwd_seed{seed}.json'),
        ('single_bwd',
         'baseline_dual_attribution_single_bwd_seed{seed}.pt',
         'baseline_results_single_bwd_seed{seed}.json'),
        ('noxedge_single_fwd',
         'baseline_dual_attribution_noxedge_single_fwd_seed{seed}.pt',
         'baseline_results_noxedge_single_fwd_seed{seed}.json'),
        ('noxedge_single_bwd',
         'baseline_dual_attribution_noxedge_single_bwd_seed{seed}.pt',
         'baseline_results_noxedge_single_bwd_seed{seed}.json'),
    ],
    'fl_standard (30 = 6 × 5)': [
        ('fl_r20e3',
         'federated_dual_attribution_fl_r20e3_seed{seed}.pt',
         'federated_results_fl_r20e3_seed{seed}.json'),
        ('fl_r100e1',
         'federated_dual_attribution_fl_r100e1_seed{seed}.pt',
         'federated_results_fl_r100e1_seed{seed}.json'),
        ('fl_iid_noxedge_r20e3',
         'federated_dual_attribution_fl_iid_noxedge_r20e3_seed{seed}.pt',
         'federated_results_fl_iid_noxedge_r20e3_seed{seed}.json'),
        ('fl_iid_noxedge_r100e1',
         'federated_dual_attribution_fl_iid_noxedge_r100e1_seed{seed}.pt',
         'federated_results_fl_iid_noxedge_r100e1_seed{seed}.json'),
        ('fl_r100e1_single_fwd',
         'federated_dual_attribution_fl_r100e1_single_fwd_seed{seed}.pt',
         'federated_results_fl_r100e1_single_fwd_seed{seed}.json'),
        ('fl_r100e1_single_bwd',
         'federated_dual_attribution_fl_r100e1_single_bwd_seed{seed}.pt',
         'federated_results_fl_r100e1_single_bwd_seed{seed}.json'),
    ],
    'fat (35 = 7 × 5)': [
        ('fat_base',
         'federated_secure_dual_attribution_fat_base_seed{seed}.pt',
         'federated_secure_results_fat_base_seed{seed}.json'),
        ('fat_def_eps1.0',
         'federated_secure_dual_attribution_fat_def_eps1.0_seed{seed}.pt',
         'federated_secure_results_fat_def_eps1.0_seed{seed}.json'),
        ('fat_def_eps2.0',
         'federated_secure_dual_attribution_fat_def_eps2.0_seed{seed}.pt',
         'federated_secure_results_fat_def_eps2.0_seed{seed}.json'),
        ('fat_def_eps4.0',
         'federated_secure_dual_attribution_fat_def_eps4.0_seed{seed}.pt',
         'federated_secure_results_fat_def_eps4.0_seed{seed}.json'),
        ('fat_def_eps8.0',
         'federated_secure_dual_attribution_fat_def_eps8.0_seed{seed}.pt',
         'federated_secure_results_fat_def_eps8.0_seed{seed}.json'),
        ('fat_def_eps16.0',
         'federated_secure_dual_attribution_fat_def_eps16.0_seed{seed}.pt',
         'federated_secure_results_fat_def_eps16.0_seed{seed}.json'),
        ('fat_def_eps32.0',
         'federated_secure_dual_attribution_fat_def_eps32.0_seed{seed}.pt',
         'federated_secure_results_fat_def_eps32.0_seed{seed}.json'),
    ],
    'phase5 (25 = 5 × 5, no JSON expected)': [
        ('coord_only',
         'federated_secure_dual_attribution_coord_only_seed{seed}.pt',
         None),  # No JSON: p5_train_defense_configs.py doesn't save one
        ('dp_only',
         'federated_secure_dual_attribution_dp_only_seed{seed}.pt',
         None),
        ('at_coord',
         'federated_secure_dual_attribution_at_coord_seed{seed}.pt',
         None),
        ('coord_dp',
         'federated_secure_dual_attribution_coord_dp_seed{seed}.pt',
         None),
        ('at_dp',
         'federated_secure_dual_attribution_at_dp_seed{seed}.pt',
         None),
    ],
}


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

def check_model_file(path: str) -> tuple:
    """(exists, size_ok, message)"""
    if not os.path.exists(path):
        return (False, False, 'MISSING')
    size = os.path.getsize(path)
    if size < 100_000:
        return (True, False, f'TOO SMALL ({size} bytes)')
    return (True, True, f'{size:,} bytes')


def check_marker_file(model_path: str) -> bool:
    return os.path.exists(model_path + '.canonical')


def check_json_file(path: str) -> tuple:
    """(exists, valid, fwd_acc)"""
    if not os.path.exists(path):
        return (False, False, None)
    try:
        with open(path) as f:
            d = json.load(f)
        fwd = d.get('test_metrics', {}).get('forward_accuracy')
        if fwd is None:
            return (True, False, None)
        if not (0.0 <= fwd <= 1.0):
            return (True, False, fwd)
        return (True, True, fwd)
    except Exception:
        return (True, False, None)


def main():
    print('=' * 80)
    print('  PHASE 5 — Global validation of all retrained models (Option B-2)')
    print('=' * 80)
    print()

    models_dir = 'results/models'
    eval_dir   = 'results/evaluations'

    if not os.path.isdir(models_dir):
        sys.exit(f'[FATAL] Missing directory: {models_dir}')

    total_pass    = 0
    total_fail    = 0
    total_warn    = 0
    fail_summary  = []

    grand_expected = 0
    for category, configs in EXPECTED.items():
        grand_expected += len(configs) * 5

    print(f'Expected total: {grand_expected} models (130 = 40+30+35+25)\n')

    for category, configs in EXPECTED.items():
        print('─' * 80)
        print(f'  {category}')
        print('─' * 80)

        for tag, model_pat, json_pat in configs:
            for seed in SEEDS:
                model_path = os.path.join(models_dir, model_pat.format(seed=seed))
                json_path  = os.path.join(eval_dir,  json_pat.format(seed=seed)) \
                             if json_pat else None

                # 1. .pt file
                exists, size_ok, msg = check_model_file(model_path)
                # 2. .canonical marker
                has_marker = check_marker_file(model_path)
                # 3. JSON if applicable
                json_exists, json_valid, fwd = check_json_file(json_path) \
                                                if json_path else (None, None, None)

                # Determine status
                if not exists:
                    status = 'FAIL'
                    detail = 'model missing'
                elif not size_ok:
                    status = 'FAIL'
                    detail = msg
                elif not has_marker:
                    status = 'WARN'
                    detail = 'no .canonical marker (stale or interrupted)'
                elif json_path and not json_exists:
                    status = 'FAIL'
                    detail = 'json missing'
                elif json_path and not json_valid:
                    status = 'FAIL'
                    detail = f'json invalid (fwd={fwd})'
                else:
                    status = 'OK'
                    detail = (f'{size_ok and msg or ""}'
                              + (f' fwd={fwd:.4f}' if fwd is not None else ''))

                if status == 'OK':
                    total_pass += 1
                    print(f'  [OK]   {tag:30s} seed={seed:3d}  {detail}')
                elif status == 'WARN':
                    total_warn += 1
                    print(f'  [WARN] {tag:30s} seed={seed:3d}  {detail}')
                else:
                    total_fail += 1
                    fail_summary.append((tag, seed, detail))
                    print(f'  [FAIL] {tag:30s} seed={seed:3d}  {detail}')

        print()

    # ── Summary ─────────────────────────────────────────────────────────────
    print('=' * 80)
    print(f'  TOTAL: {total_pass} OK, {total_warn} WARN, {total_fail} FAIL '
          f'(of {grand_expected} expected)')
    print('=' * 80)

    if fail_summary:
        print()
        print('Failing items (action required):')
        for tag, seed, detail in fail_summary:
            print(f'  - {tag} seed={seed}: {detail}')

    if total_warn > 0:
        print()
        print('WARN means the model exists but has no .canonical marker.')
        print('This typically happens when a job was interrupted between')
        print('the training step and the marker-writing step.')
        print('Action: delete the model and the SLURM will re-train it on next launch.')

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == '__main__':
    main()
