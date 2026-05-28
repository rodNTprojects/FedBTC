"""
p5_interaction_matrix_eps16.py
FedBTC — Defense Interaction × Risk Matrix (epsilon = 16 variant)

This file is a SIBLING of p5_interaction_matrix.py, with two changes:
  1. --models_dir flag (default: 'results/models'). When set to
     'results/models_eps16', the script loads .pt files from the
     isolated eps16 directory instead of the original results/models/.
  2. C5 mapping points to fat_def_eps16.0 (not fat_def_eps8.0). This
     reflects the headline operational regime epsilon=16 used in the
     paper. The eps=8 variant remains available via the original
     p5_interaction_matrix.py.

All other logic is identical to the original script.

Evaluates 8 defense configurations against 5 risk dimensions:

  C0  No defense       fl_r100e1                (Phase 1)
  C1  AT only          fat_base                 (Phase 6b)
  C2  CoordMedian only coord_only               (Phase 5)
  C3  DP only          dp_only                  (Phase 5)
  C4  AT + CoordMedian at_coord                 (Phase 5)
  C5  Full stack       fat_def_eps16.0          (Phase 6c, eps=16 headline)
  C6  CoordMedian + DP coord_dp                 (Phase 5)
  C7  AT + DP          at_dp                    (Phase 5)

Risk dimensions:
  1. Utility           clean forward + backward accuracy
  2. Evasion           accuracy at PGD eps=0.30 (E1 fwd / E2 bwd)
  3. Byzantine         accuracy under P1 noise (Run B / Run C)
  4. MIA protection    1 - AUC_Yeom (higher = better privacy)
  5. DLG protection    avg MSE batch=1 (higher = harder to reconstruct)

Interaction score:
  I(defense D, risk R) = metric(D, R) - metric(C0, R)
  Positive  = D helps against R  (intended or positive unintended)
  Negative  = D hurts on R       (unintended degradation)

USAGE:
    python p5_interaction_matrix_eps16.py --seed 0 \\
        --models_dir results/models_eps16 \\
        --output_dir results/interaction_eps16
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── Import from attack_benchmark ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from attack_benchmark import (
        DualAttributionModel, load_all_exchanges, load_model,
        _collect_losses_and_features, _mia_score,
        _pgd, _dlg_one_batch,
        run_fl_attack_experiment, _p1_malicious_fn,
        coord_median_norm_bound,
    )
except ImportError as e:
    sys.exit(f"[ERROR] Cannot import attack_benchmark: {e}")

from torch_geometric.data import Data


# ── Defense configurations ────────────────────────────────────────────────────

CONFIGS = {
    'C0': 'No defense',
    'C1': 'AT only',
    'C2': 'CoordMedian only',
    'C3': 'DP only (ε=16)',
    'C4': 'AT + CoordMedian',
    'C5': 'Full stack (AT + CoordMedian + DP, ε=16)',
    'C6': 'CoordMedian + DP',
    'C7': 'AT + DP',
}


# Module-level constant; overwritten from args.models_dir in main().
# Default matches the original script for backward compatibility if the
# script is invoked without --models_dir.
_MODELS_DIR = os.path.join('results', 'models')


def model_filename(config: str, seed: int) -> str:
    names = {
        'C0': f'federated_dual_attribution_fl_r100e1_seed{seed}.pt',
        'C1': f'federated_secure_dual_attribution_fat_base_seed{seed}.pt',
        'C2': f'federated_secure_dual_attribution_coord_only_seed{seed}.pt',
        'C3': f'federated_secure_dual_attribution_dp_only_seed{seed}.pt',
        'C4': f'federated_secure_dual_attribution_at_coord_seed{seed}.pt',
        # C5 uses the eps=16 model file (the headline operational regime).
        'C5': f'federated_secure_dual_attribution_fat_def_eps16.0_seed{seed}.pt',
        'C6': f'federated_secure_dual_attribution_coord_dp_seed{seed}.pt',
        'C7': f'federated_secure_dual_attribution_at_dp_seed{seed}.pt',
    }
    return os.path.join(_MODELS_DIR, names[config])


# ── Evaluators ────────────────────────────────────────────────────────────────

def eval_utility(model, exchange_data, device) -> Tuple[float, float]:
    """Clean accuracy on test set, per-exchange macro-average."""
    model.eval()
    fwd_per_exchange = []
    bwd_per_exchange = []

    for gd in exchange_data:
        gd = gd.to(device)
        mask = gd.test_mask
        if mask.sum() == 0:
            continue

        with torch.no_grad():
            out = model(gd.x, gd.edge_index)

        # Forward accuracy on this exchange
        pred = out['forward'][mask].argmax(1)
        fwd_acc = (pred == gd.y_forward[mask]).float().mean().item()
        fwd_per_exchange.append(fwd_acc)

        # Backward accuracy on this exchange (ignore class 0 = NO_MERCHANT)
        bwd_true = gd.y_backward[mask]
        bwd_mask = bwd_true > 0
        if bwd_mask.sum() > 0:
            bwd_pred = out['backward'][mask][bwd_mask].argmax(1)
            bwd_acc = (bwd_pred == bwd_true[bwd_mask]).float().mean().item()
            bwd_per_exchange.append(bwd_acc)

    fwd_macro = float(np.mean(fwd_per_exchange)) if fwd_per_exchange else 0.0
    bwd_macro = float(np.mean(bwd_per_exchange)) if bwd_per_exchange else 0.0
    return round(fwd_macro, 4), round(bwd_macro, 4)


def eval_evasion(model, exchange_data, device,
                 epsilon: float = 0.30, steps: int = 7) -> Tuple[float, float]:
    """Accuracy under PGD at epsilon on fwd and bwd heads,
    per-exchange macro-average."""
    model.eval()
    fwd_per_exchange = []
    bwd_per_exchange = []

    for gd in exchange_data:
        gd = gd.to(device)
        mask = gd.test_mask
        if mask.sum() == 0:
            continue

        # Forward head under PGD
        x_adv = _pgd(model, gd.x, gd.edge_index,
                     gd.y_forward, mask, 'forward', epsilon, steps)
        with torch.no_grad():
            out = model(x_adv, gd.edge_index)
        fwd_pred = out['forward'][mask].argmax(1)
        fwd_acc  = (fwd_pred == gd.y_forward[mask]).float().mean().item()
        fwd_per_exchange.append(fwd_acc)

        # Backward head under PGD (ignore class 0)
        x_adv2 = _pgd(model, gd.x, gd.edge_index,
                      gd.y_backward, mask, 'backward', epsilon, steps)
        with torch.no_grad():
            out2 = model(x_adv2, gd.edge_index)
        bwd_true = gd.y_backward[mask]
        bwd_m    = bwd_true > 0
        if bwd_m.sum() > 0:
            bwd_pred = out2['backward'][mask][bwd_m].argmax(1)
            bwd_acc  = (bwd_pred == bwd_true[bwd_m]).float().mean().item()
            bwd_per_exchange.append(bwd_acc)

    fwd_macro = float(np.mean(fwd_per_exchange)) if fwd_per_exchange else 0.0
    bwd_macro = float(np.mean(bwd_per_exchange)) if bwd_per_exchange else 0.0
    return round(fwd_macro, 4), round(bwd_macro, 4)


def eval_byzantine(model_config: Dict, exchange_data: List[Data],
                   device, num_rounds: int = 100,
                   local_epochs: int = 1, lr: float = 0.002
                   ) -> Tuple[float, float, float, float]:
    """P1 Byzantine noise — Run B (attack, no defense) and Run C (attack + CoordMedian)."""
    run_b = run_fl_attack_experiment(
        exchange_data, model_config, device, num_rounds, local_epochs, lr,
        malicious_fn=_p1_malicious_fn, use_defense=False)
    run_c = run_fl_attack_experiment(
        exchange_data, model_config, device, num_rounds, local_epochs, lr,
        malicious_fn=_p1_malicious_fn, use_defense=True)
    return (round(run_b['forward_acc'],  4), round(run_b['backward_acc'], 4),
            round(run_c['forward_acc'],  4), round(run_c['backward_acc'], 4))


def eval_mia_lira(model, exchange_data, device,
                   n_shadows: int = 128, shadow_epochs: int = 15) -> float:
    """LiRA offline K=128 (Carlini et al., S&P 2022)."""
    from attack_benchmark import _d1_lira
    result = _d1_lira(model, exchange_data, device,
                      n_shadows=n_shadows, shadow_epochs=shadow_epochs)
    return round(result['auc'], 4)


def eval_dlg(model, exchange_data, device,
             batch_size: int = 1, n_iters: int = 50) -> float:
    """DLG gradient inversion — avg MSE across exchanges."""
    mses = []
    for gd in exchange_data:
        gd = gd.to(device)
        res = _dlg_one_batch(model, gd, batch_size, n_iters, device)
        mses.append(res['reconstruction_mse'])
    return round(float(np.mean(mses)), 4)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate_one(config: str, seed: int,
                 exchange_data: List[Data], device,
                 num_rounds: int = 100, fast: bool = False) -> Optional[Dict]:
    path = model_filename(config, seed)
    if not os.path.exists(path):
        print(f"  [MISS] {config}: {path}")
        return None

    model, model_config = load_model(path, device)
    model.eval()
    rounds = 10 if fast else num_rounds
    print(f"\n  [{config}] {CONFIGS[config]}")

    print("    Utility...", end=' ', flush=True)
    u_fwd, u_bwd = eval_utility(model, exchange_data, device)
    print(f"fwd={u_fwd:.4f}  bwd={u_bwd:.4f}")

    print("    Evasion ε=0.30...", end=' ', flush=True)
    e_fwd, e_bwd = eval_evasion(model, exchange_data, device)
    print(f"fwd={e_fwd:.4f}  bwd={e_bwd:.4f}")

    print(f"    Byzantine P1 Run B ({rounds} rounds, no defense)...", end=' ', flush=True)
    b_fwd, b_bwd, c_fwd, c_bwd = eval_byzantine(
        model_config, exchange_data, device, rounds)
    print(f"RunB fwd={b_fwd:.4f} bwd={b_bwd:.4f} | RunC fwd={c_fwd:.4f} bwd={c_bwd:.4f}")

    print(f"    MIA (LiRA K=128)...", end=' ', flush=True)
    mia_auc = eval_mia_lira(model, exchange_data, device)
    print(f"AUC={mia_auc:.4f}  MIA-protection={1-mia_auc:.4f}")

    print("    DLG batch=1...", end=' ', flush=True)
    dlg_mse = eval_dlg(model, exchange_data, device)
    print(f"MSE={dlg_mse:.4f}")

    return {
        'config': config, 'description': CONFIGS[config], 'seed': seed,
        'utility_fwd':    u_fwd, 'utility_bwd':    u_bwd,
        'evasion_fwd':    e_fwd, 'evasion_bwd':    e_bwd,
        'byzantine_b_fwd': b_fwd, 'byzantine_b_bwd': b_bwd,
        'byzantine_c_fwd': c_fwd, 'byzantine_c_bwd': c_bwd,
        'mia_auc':        mia_auc,
        'mia_protection': round(1 - mia_auc, 4),
        'dlg_mse':        dlg_mse,
    }


def compute_scores(results: Dict[str, Optional[Dict]]) -> Dict:
    """Interaction score = metric(D) - metric(C0) for each defense D and risk R."""
    if 'C0' not in results or results['C0'] is None:
        return {}

    ref = results['C0']
    dims = ['utility_fwd', 'utility_bwd',
            'evasion_fwd', 'evasion_bwd',
            'byzantine_b_fwd', 'byzantine_b_bwd',
            'byzantine_c_fwd', 'byzantine_c_bwd',
            'mia_protection', 'dlg_mse']

    scores = {}
    for cfg, res in results.items():
        if res is None:
            continue
        scores[cfg] = {}
        for d in dims:
            v, r = res.get(d), ref.get(d)
            scores[cfg][d] = round(v - r, 4) if (v is not None and r is not None) else None
    return scores


def print_matrix(scores: Dict):
    dims = ['utility_fwd', 'utility_bwd',
            'evasion_fwd', 'evasion_bwd',
            'byzantine_b_bwd', 'byzantine_c_bwd',
            'mia_protection', 'dlg_mse']
    labels = ['Util_fwd','Util_bwd','Eva_fwd','Eva_bwd',
              'ByzB_bwd','ByzC_bwd','MIA_prot','DLG_mse']

    print("\n" + "="*80)
    print("INTERACTION MATRIX  (Δ vs C0 no-defense; + = defense helps, – = hurts)")
    print("="*80)
    print(f"{'Config':8s} " + " ".join(f"{l:>10s}" for l in labels))
    print("-"*80)
    for cfg in ['C0','C1','C2','C3','C4','C5','C6','C7']:
        if cfg not in scores:
            continue
        row = f"{cfg:8s} "
        for d in dims:
            v = scores[cfg].get(d)
            if v is None:
                row += f"{'—':>10s} "
            else:
                sign = '+' if v > 0 else ''
                row += f"{sign}{v:>9.4f} "
        print(row)
    print("-"*80)


def main():
    global _MODELS_DIR
    p = argparse.ArgumentParser()
    p.add_argument('--seed',        type=int,   required=True)
    p.add_argument('--data_dir',    type=str,   default='data/federated_enriched')
    p.add_argument('--output_dir',  type=str,   default='results/interaction_eps16')
    p.add_argument('--models_dir',  type=str,   default='results/models',
                   help='Directory where .pt files are loaded from '
                        '(default: results/models). Set to results/models_eps16 '
                        'to use the isolated epsilon=16 model set.')
    p.add_argument('--configs',     type=str,   default='C0,C1,C2,C3,C4,C5,C6,C7')
    p.add_argument('--rounds',      type=int,   default=100)
    p.add_argument('--fast',        action='store_true',
                   help='Fast mode: 10 FL rounds, 200 test nodes')
    args = p.parse_args()

    # Propagate models_dir to the module-level constant used by model_filename()
    _MODELS_DIR = args.models_dir

    os.makedirs(args.output_dir, exist_ok=True)
    configs = [c.strip() for c in args.configs.split(',')]
    device  = torch.device('cpu')

    print(f"FedBTC Interaction Matrix (eps=16 variant) | seed={args.seed} | "
          f"{'fast' if args.fast else 'full'} | models_dir={_MODELS_DIR}")

    exchange_data = load_all_exchanges(args.data_dir, args.seed, fast=args.fast)
    print(f"Exchanges: {[gd.num_nodes for gd in exchange_data]} nodes")

    raw = {}
    for cfg in configs:
        raw[cfg] = evaluate_one(
            cfg, args.seed, exchange_data, device,
            num_rounds=args.rounds, fast=args.fast)

    scores = compute_scores(raw)
    print_matrix(scores)

    out_path = os.path.join(
        args.output_dir, f'interaction_matrix_seed{args.seed}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'seed':        args.seed,
            'configs':     CONFIGS,
            'models_dir':  _MODELS_DIR,
            'raw_results': {k: v for k, v in raw.items()    if v},
            'scores':      {k: v for k, v in scores.items() if v},
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
