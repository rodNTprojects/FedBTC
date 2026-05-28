"""
run_security_evaluation.py
FedBTC — Security Evaluation Orchestrator
==========================================

Entry point for the full attack benchmark. Runs the chosen attacks
against a trained model and writes structured JSON results.

Three evaluation modes
----------------------
  attacks   : run all integrity/privacy attacks, no active defense
  defenses  : run clean + under-attack with Byzantine defense enabled
  both      : run attacks with and without defenses and compare

Usage
-----
python run_security_evaluation.py \\
    --model_path results/models/baseline_dual_attribution_noxedge_cat_seed0.pt \\
    --mode both --attacks all --seed 0 --output_tag cat_seed0

python run_security_evaluation.py \\
    --model_path results/models/federated_secure_dual_attribution_fat_def_seed0.pt \\
    --mode both --attacks all --seed 0 --output_tag fat_def_seed0

Output schema (results/evaluations/security_eval_{output_tag}.json)
-------------------------------------------------------------------
{
  "config": {"mode": "both", "seed": 0, "model_path": "...", ...},
  "utility": {"forward_acc": 0.852, "backward_acc": 0.883},
  "attacks": {
    "E1": {"eps_0_00": 0.85, "eps_0_05": 0.79, "eps_0_20": 0.62, "acc_drop": 0.23},
    "E2": { ... },
    "P1": {"no_defense": {...}, "with_defense": {...}},
    "P2": {"no_defense": {...}, "with_defense": {...}},
    "P3": {"no_defense": {...}, "with_defense": {...}},
    "D1": {"loss_based": {...}, "shadow_model": {...}},
    "D2": {"avg_mse": 0.45, "dp_protected": false, ...}
  }
}

Seeds used for FedBTC experiments: 0 45 123 654 789
"""

import os
import sys
import json
import copy
import random
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

import torch

# ── Resolve imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fedbtc_attacks.integrity import EvasionAttack, PoisoningAttack
from fedbtc_attacks.privacy   import MembershipInferenceAttack, GradientLeakage


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities (data loading + model loading)
# These mirror p3_train_federated_secure_2layers_FAT.py to avoid duplication
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_and_config(model_path: str, device: torch.device):
    """Load a trained DualAttributionModel from checkpoint."""
    # Import the model class from FAT script
    try:
        from p3_train_federated_secure_2layers_FAT import DualAttributionModel
    except ImportError:
        from p3_train_federated_secure_2layers import DualAttributionModel

    ckpt   = torch.load(model_path, map_location=device)
    config = ckpt['config']
    model  = DualAttributionModel(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, config


def _load_exchange_data(data_dir: str, seed: int) -> List:
    """Load PyG Data for all 3 exchanges using the same protocol as FAT."""
    try:
        from p3_train_federated_secure_2layers_FAT import (
            load_exchange_data, create_pyg_data_for_exchange)
    except ImportError:
        from p3_train_federated_secure_2layers import (
            load_exchange_data, create_pyg_data_for_exchange)

    exchange_data = []
    for i in range(3):
        data, edges = load_exchange_data(i, data_dir)
        gd = create_pyg_data_for_exchange(data, edges, exchange_id=i, seed=seed)
        exchange_data.append(gd)
    return exchange_data


def _clean_accuracy(model, exchange_data: List, device: torch.device) -> Dict:
    fwd_accs, bwd_accs = [], []
    for gd in exchange_data:
        gd = gd.to(device)
        with torch.no_grad():
            out  = model(gd.x, gd.edge_index)
            mask = gd.test_mask
        fwd_accs.append(
            (out['forward'][mask].argmax(1) ==
             gd.y_forward[mask]).float().mean().item())
        bwd  = gd.y_backward[mask]
        cm   = bwd > 0
        bwd_accs.append(
            (out['backward'][mask][cm].argmax(1) ==
             bwd[cm]).float().mean().item() if cm.sum() > 0 else 0.0)
    return {'forward_acc':  round(float(np.mean(fwd_accs)), 4),
            'backward_acc': round(float(np.mean(bwd_accs)), 4)}


def _byzantine_defense_fn(client_updates):
    """CoordMedian + NormBound — matches FAT implementation."""
    if not client_updates:
        return []
    norms = np.array([
        float(np.sqrt(sum(np.sum(p ** 2) for p in u)))
        for u in client_updates
    ])
    tau = 2.0
    med_norm  = float(np.median(norms))
    threshold = tau * med_norm + 1e-8
    bounded = []
    for upd, n in zip(client_updates, norms):
        if n > threshold:
            s = threshold / n
            bounded.append([p * s for p in upd])
        else:
            bounded.append(upd)
    return [np.median(np.stack([b[i] for b in bounded], axis=0), axis=0)
            for i in range(len(bounded[0]))]


# ─────────────────────────────────────────────────────────────────────────────
# Attack runners
# ─────────────────────────────────────────────────────────────────────────────

def run_evasion(model, exchange_data, device, head: str) -> Dict:
    attack = EvasionAttack(epsilon=0.2, steps=7)
    print(f"\n  Running {'E1' if head=='forward' else 'E2'}: "
          f"PGD evasion → {head} head")
    return attack.evaluate_robustness(model, exchange_data, device, head=head)


def run_poisoning(model, exchange_data, model_config, device,
                  attack_id: str, with_defense: bool) -> Dict:
    """Run one poisoning attack with/without Byzantine defense."""
    get_params = lambda m: [v.cpu().numpy().copy()
                             for v in m.state_dict().values()]
    global_params = get_params(model)
    defense_fn    = _byzantine_defense_fn if with_defense else None

    if attack_id == 'P1':
        honest_update = [np.zeros_like(p) for p in global_params]
        mal_update    = PoisoningAttack.byzantine_noise(honest_update, sigma=1.0)
    elif attack_id == 'P2':
        poisoned_data  = PoisoningAttack.backward_label_flip(
            exchange_data[0], flip_pct=0.30, from_class=2, to_class=1)
        poisoned_exch  = [poisoned_data] + exchange_data[1:]
        print(f"\n  Running P2: backward label-flip "
              f"({'w/ defense' if with_defense else 'no defense'})")
        try:
            return PoisoningAttack.evaluate_one_round(
                global_params, poisoned_exch, model_config, device,
                malicious_idx=0,
                malicious_update=[np.zeros_like(p) for p in global_params],
                defense_fn=defense_fn)
        except Exception as e:
            print(f"    P2 evaluation failed: {e}")
            return {'error': str(e)}
    elif attack_id == 'P3':
        # Approximate honest update with zero delta, then flip
        honest_update = [np.zeros_like(p) for p in global_params]
        mal_update    = PoisoningAttack.sign_flip(honest_update, scale=5.0)
    else:
        return {}

    print(f"\n  Running {attack_id}: "
          f"{'w/ defense' if with_defense else 'no defense'}")
    try:
        return PoisoningAttack.evaluate_one_round(
            global_params, exchange_data, model_config, device,
            malicious_idx=1, malicious_update=mal_update,
            defense_fn=defense_fn)
    except Exception as e:
        print(f"    {attack_id} evaluation failed: {e}")
        return {'error': str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

ATTACK_IDS = ['e1', 'e2', 'p1', 'p2', 'p3', 'd1', 'd2']


def run_evaluation(args) -> Dict:
    # ── Seed ──────────────────────────────────────────────────────────────────
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print(f"FedBTC Security Evaluation")
    print(f"  model  : {args.model_path}")
    print(f"  mode   : {args.eval_mode}")
    print(f"  attacks: {args.attacks}")
    print(f"  seed   : {args.seed}")
    print(f"{'='*65}")

    # ── Load ─────────────────────────────────────────────────────────────────
    model, model_config = _load_model_and_config(args.model_path, device)
    exchange_data       = _load_exchange_data(args.data_dir, args.seed)

    # ── Resolve attack list ───────────────────────────────────────────────────
    attacks = ATTACK_IDS if args.attacks == 'all' \
              else [a.lower() for a in args.attacks.split(',')]

    run_defense = args.eval_mode in ('defenses', 'both')
    run_no_def  = args.eval_mode in ('attacks',  'both')

    # ── Baseline ─────────────────────────────────────────────────────────────
    utility = _clean_accuracy(model, exchange_data, device)
    print(f"\nClean accuracy: fwd={utility['forward_acc']:.4f}  "
          f"bwd={utility['backward_acc']:.4f}")

    report = {
        'config': {
            'model_path':  args.model_path,
            'eval_mode':   args.eval_mode,
            'attacks':     attacks,
            'seed':        args.seed,
            'timestamp':   datetime.now().isoformat(),
        },
        'utility':  utility,
        'attacks':  {},
    }

    # ── E1 / E2 ──────────────────────────────────────────────────────────────
    if 'e1' in attacks:
        report['attacks']['E1'] = run_evasion(model, exchange_data, device, 'forward')
    if 'e2' in attacks:
        report['attacks']['E2'] = run_evasion(model, exchange_data, device, 'backward')

    # ── P1 / P2 / P3 ─────────────────────────────────────────────────────────
    for atk in ['p1', 'p2', 'p3']:
        if atk not in attacks:
            continue
        key  = atk.upper()
        entry: Dict = {}
        if run_no_def:
            entry['no_defense'] = run_poisoning(
                model, exchange_data, model_config, device, key, False)
        if run_defense:
            entry['with_defense'] = run_poisoning(
                model, exchange_data, model_config, device, key, True)
        report['attacks'][key] = entry

    # ── D1 — MIA ─────────────────────────────────────────────────────────────
    if 'd1' in attacks:
        print("\n  Running D1: Membership Inference Attack (both variants)")
        mia = MembershipInferenceAttack()
        report['attacks']['D1'] = mia.evaluate(model, exchange_data, device)

    # ── D2 — Gradient Leakage ────────────────────────────────────────────────
    if 'd2' in attacks:
        dlg = GradientLeakage(n_iters=50, batch_size=16)
        report['attacks']['D2'] = dlg.evaluate(model, exchange_data, device)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs('results/evaluations', exist_ok=True)
    tag      = f'_{args.output_tag}' if args.output_tag else ''
    out_path = f'results/evaluations/security_eval{tag}.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✔ Report saved: {out_path}")
    return report


def parse_args():
    p = argparse.ArgumentParser(
        description='FedBTC Security Evaluation Orchestrator')
    p.add_argument('--model_path', required=True,
                   help='Path to .pt checkpoint')
    p.add_argument('--data_dir', default='data/federated_enriched',
                   help='Enriched exchange data directory')
    p.add_argument('--eval_mode', default='both',
                   choices=['attacks', 'defenses', 'both'],
                   help=(
                       'attacks  : run all attacks, no Byzantine defense active; '
                       'defenses : run with CoordMedian+NormBound defense; '
                       'both     : compare no-defense vs defense side-by-side'
                   ))
    p.add_argument('--attacks', default='all',
                   help='Comma-separated attack IDs or "all". '
                        'Choices: e1,e2,p1,p2,p3,d1,d2')
    p.add_argument('--seed', type=int, default=0,
                   help='Random seed (FedBTC seeds: 0 45 123 654 789)')
    p.add_argument('--output_tag', default='',
                   help='Tag appended to output filename')
    return p.parse_args()


if __name__ == '__main__':
    run_evaluation(parse_args())
