"""
p5_evaluate_c5_sweep.py
FedBTC — Evaluate C5 at different DP epsilon values and compare with C0.

This script is standalone — it does not modify p5_interaction_matrix.py.
It loads models by explicit path and runs the same 5 risk dimensions.

Configurations tested:
  C0        : No defense (reference)          fl_r100e1_seed{N}.pt
  C5_eps8   : AT + CoordMedian + DP ε=8  σ=0.10  fat_def_eps8.0_seed{N}.pt   (existing)
  C5_eps16  : AT + CoordMedian + DP ε=16 σ=0.05  fat_def_eps16.0_seed{N}.pt  (new)
  C5_eps32  : AT + CoordMedian + DP ε=32 σ=0.025 fat_def_eps32.0_seed{N}.pt  (new)

USAGE:
    python p5_evaluate_c5_sweep.py --seed 0
    python p5_evaluate_c5_sweep.py --seed 0 --fast
"""

import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from attack_benchmark import (
        load_all_exchanges, load_model,
        _collect_losses_and_features, _mia_score,
        _pgd, _dlg_one_batch,
        run_fl_attack_experiment, _p1_malicious_fn,
    )
    from p5_interaction_matrix import (
        eval_utility, eval_evasion, eval_byzantine,
        eval_mia_lira, eval_dlg,
    )
except ImportError as e:
    sys.exit(f"[ERROR] Cannot import dependencies: {e}")


# ── Model paths ───────────────────────────────────────────────────────────────

SWEEP_CONFIGS = {
    'C0': {
        'label':   'No defense',
        'epsilon':  None,
        'sigma':    None,
        'filename': 'federated_dual_attribution_fl_r100e1_seed{seed}.pt',
    },
    'C5_eps8': {
        'label':   'Full stack ε=8  σ=0.10',
        'epsilon':  8.0,
        'sigma':    0.10,
        'filename': 'federated_secure_dual_attribution_fat_def_eps8.0_seed{seed}.pt',
    },
    'C5_eps16': {
        'label':   'Full stack ε=16 σ=0.05',
        'epsilon': 16.0,
        'sigma':    0.05,
        'filename': 'federated_secure_dual_attribution_fat_def_eps16.0_seed{seed}.pt',
    },
    'C5_eps32': {
        'label':   'Full stack ε=32 σ=0.025',
        'epsilon': 32.0,
        'sigma':    0.025,
        'filename': 'federated_secure_dual_attribution_fat_def_eps32.0_seed{seed}.pt',
    },
}


def model_path(cfg_key: str, seed: int) -> str:
    fname = SWEEP_CONFIGS[cfg_key]['filename'].format(seed=seed)
    return os.path.join('results', 'models', fname)


def evaluate_config(cfg_key: str, seed: int,
                    exchange_data, device,
                    num_rounds: int = 100) -> dict:
    path = model_path(cfg_key, seed)
    cfg  = SWEEP_CONFIGS[cfg_key]

    if not os.path.exists(path):
        print(f"  [MISS] {cfg_key} ({cfg['label']}): {path}")
        return None

    model, model_config = load_model(path, device)
    model.eval()
    print(f"\n  [{cfg_key}] {cfg['label']}")

    print("    Utility...", end=' ', flush=True)
    u_fwd, u_bwd = eval_utility(model, exchange_data, device)
    print(f"fwd={u_fwd:.4f}  bwd={u_bwd:.4f}")

    print("    Evasion ε=0.30...", end=' ', flush=True)
    e_fwd, e_bwd = eval_evasion(model, exchange_data, device)
    print(f"fwd={e_fwd:.4f}  bwd={e_bwd:.4f}")

    print(f"    Byzantine P1 ({num_rounds}r)...", end=' ', flush=True)
    b_fwd, b_bwd, c_fwd, c_bwd = eval_byzantine(
        model_config, exchange_data, device, num_rounds)
    print(f"RunB bwd={b_bwd:.4f}  RunC bwd={c_bwd:.4f}")

    print("    MIA (LiRA K=128)...", end=' ', flush=True)
    mia_auc = eval_mia_lira(model, exchange_data, device)
    print(f"AUC={mia_auc:.4f}")

    print("    DLG batch=1...", end=' ', flush=True)
    dlg_mse = eval_dlg(model, exchange_data, device)
    print(f"MSE={dlg_mse:.4f}")

    return {
        'config':    cfg_key,
        'label':     cfg['label'],
        'epsilon':   cfg['epsilon'],
        'sigma':     cfg['sigma'],
        'seed':      seed,
        'utility_fwd':      u_fwd,
        'utility_bwd':      u_bwd,
        'evasion_fwd':      e_fwd,
        'evasion_bwd':      e_bwd,
        'byzantine_b_bwd':  b_bwd,
        'byzantine_c_bwd':  c_bwd,
        'mia_auc':          mia_auc,
        'mia_protection':   round(1 - mia_auc, 4),
        'dlg_mse':          dlg_mse,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed',       type=int, required=True)
    p.add_argument('--data_dir',   type=str, default='data/federated_enriched')
    p.add_argument('--output_dir', type=str, default='results/interaction_c5')
    p.add_argument('--rounds',     type=int, default=100)
    p.add_argument('--fast',       action='store_true')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cpu')

    print(f"FedBTC C5 DP Sweep | seed={args.seed} | "
          f"{'fast' if args.fast else 'full'}")
    print(f"ε sweep: 8 (σ=0.10)  →  16 (σ=0.05)  →  32 (σ=0.025)")

    exchange_data = load_all_exchanges(args.data_dir, args.seed, fast=args.fast)

    results = {}
    for cfg_key in ['C0', 'C5_eps8', 'C5_eps16', 'C5_eps32']:
        results[cfg_key] = evaluate_config(
            cfg_key, args.seed, exchange_data, device,
            num_rounds=10 if args.fast else args.rounds)

    # Compute deltas vs C0
    ref = results.get('C0')
    dims = ['utility_fwd', 'utility_bwd', 'evasion_fwd', 'evasion_bwd',
            'byzantine_b_bwd', 'byzantine_c_bwd', 'mia_protection', 'dlg_mse']

    print("\n" + "="*72)
    print("C5 DP SWEEP — Δ vs C0 baseline")
    print("="*72)
    print(f"{'Config':14s} {'ε':>5s} {'σ':>6s}  "
          + "  ".join(f"{d[:8]:>9s}" for d in dims))
    print("-"*72)

    for cfg_key in ['C0', 'C5_eps8', 'C5_eps16', 'C5_eps32']:
        res = results.get(cfg_key)
        if res is None:
            print(f"{cfg_key:14s}  [not available]")
            continue
        eps_str = f"{res['epsilon']:.0f}" if res['epsilon'] else 'ref'
        sig_str = f"{res['sigma']:.3f}" if res['sigma'] else '—'
        row = f"{cfg_key:14s} {eps_str:>5s} {sig_str:>6s}  "
        for d in dims:
            if ref and ref.get(d) is not None and res.get(d) is not None:
                delta = res[d] - ref[d]
                sign = '+' if delta >= 0 else ''
                row += f"{sign}{delta:>+8.4f}  "
            else:
                row += f"{'ref':>9s}  "
        print(row)

    # Save JSON
    out_path = os.path.join(
        args.output_dir, f'c5_sweep_seed{args.seed}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'seed': args.seed,
            'configs': {k: {'label': v['label'], 'epsilon': v['epsilon'],
                            'sigma': v['sigma']} for k,v in SWEEP_CONFIGS.items()},
            'results': {k: v for k,v in results.items() if v},
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
