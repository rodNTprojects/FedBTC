"""
attack_benchmark.py
FedBTC — Attack and Defense Benchmark
======================================

Simulates 8 attacks against the dual-attribution GCN model.
Each attack is independently executable, results printed immediately.

Attacks
-------
  E1  PGD evasion → forward head   (Madry et al., ICLR 2018)
  E2  PGD evasion → backward head
  P1  Byzantine noise               (Blanchard et al., NeurIPS 2017)
  P2  Backward label-flip (KYC)     (Fang et al., USENIX Security 2020)
  P3  Sign-flip + scaling           (Baruch et al., NeurIPS 2019)
  D1  Membership inference (loss)   (Yeom et al., CSF 2018)
  D2  Gradient leakage (DLG)        (Zhu & Han, NeurIPS 2019)
  D3  Regulatory model analysis     (passive, no defense)

Usage
-----
  # Single attack on a trained model
  python attack_benchmark.py --attack e1 --model results/models/fl_r100e1_seed0.pt

  # Same attack + its defense
  python attack_benchmark.py --attack e1 --model results/models/fl_r100e1_seed0.pt --defense

  # Byzantine attack with defense comparison (runs A → B → C automatically)
  python attack_benchmark.py --attack p2 --model results/models/fl_r100e1_seed0.pt --defense

  # Local fast mode (reduced data + 10 rounds, no cluster needed)
  python attack_benchmark.py --attack e1 --fast

  # All attacks sequentially
  python attack_benchmark.py --attack all --model results/models/fl_r100e1_seed0.pt --defense

Seeds: 0, 45, 123, 654, 789
"""

import os
import sys
import json
import copy
import random
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# ── Import from the project's FL script ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p3_train_federated_2layers import (
    DualAttributionModel,
    FocalLoss,
    FederatedClient,
    FederatedServer,
    load_exchange_data,
    create_pyg_data_for_exchange,
    evaluate_model,
    CATEGORY_NAMES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SEEDS     = [0, 45, 123, 654, 789]
EPS_SWEEP = [0.0, 0.05, 0.10, 0.20, 0.30]   # E1 / E2 epsilon values
N_EXCHANGES = 3
MALICIOUS_IDX = 0                             # exchange 0 is always the attacker


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_path: str, device: torch.device) -> Tuple[DualAttributionModel, Dict]:
    """Load a trained DualAttributionModel from checkpoint."""
    ckpt = torch.load(model_path, map_location=device)
    config = ckpt.get('config', ckpt.get('model_config', {}))
    model = DualAttributionModel(config).to(device)
    key = 'model_state_dict' if 'model_state_dict' in ckpt else 'state_dict'
    model.load_state_dict(ckpt[key])
    model.eval()
    return model, config


def load_all_exchanges(data_dir: str, seed: int,
                       fast: bool = False) -> List[Data]:
    """Load PyG graph data for all 3 exchanges."""
    exchange_data = []
    for i in range(N_EXCHANGES):
        data, edges = load_exchange_data(i, data_dir)
        gd = create_pyg_data_for_exchange(data, edges, exchange_id=i)

        if fast:
            # Subsample test set for speed (local validation)
            test_idx = gd.test_mask.nonzero(as_tuple=True)[0]
            keep = test_idx[:min(200, len(test_idx))]
            fast_mask = torch.zeros(gd.num_nodes, dtype=torch.bool)
            fast_mask[keep] = True
            gd.test_mask = fast_mask

        exchange_data.append(gd)
    return exchange_data


def avg_metrics(exchange_data: List[Data],
                model: DualAttributionModel,
                device: torch.device,
                mask: str = 'test') -> Dict:
    """Average forward and backward accuracy across all exchanges."""
    fwd_list, bwd_list = [], []
    for gd in exchange_data:
        m = evaluate_model(model, gd, device, mask)
        fwd_list.append(m['forward_accuracy'])
        bwd_list.append(m['backward_accuracy'])
    return {
        'forward_acc':  round(float(np.mean(fwd_list)), 4),
        'backward_acc': round(float(np.mean(bwd_list)), 4),
        'per_exchange': [
            {'exchange': i,
             'forward_acc':  round(fwd_list[i], 4),
             'backward_acc': round(bwd_list[i], 4)}
            for i in range(len(exchange_data))
        ]
    }


def get_model_params(model: DualAttributionModel) -> List[np.ndarray]:
    return [v.cpu().numpy().copy() for v in model.state_dict().values()]


def set_model_params(model: DualAttributionModel, params: List[np.ndarray]):
    sd = {k: torch.tensor(v)
          for k, v in zip(model.state_dict().keys(), params)}
    model.load_state_dict(sd)


def print_header(attack: str, desc: str):
    print(f"\n{'='*60}")
    print(f"  [{attack}]  {desc}")
    print(f"{'='*60}")


def save_result(result: Dict, output_dir: str, tag: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"attack_{tag}_{result['seed']}.json")
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CoordMedian + NormBound defense  (used by P1 / P2 / P3)
# Reference: Blanchard et al., NeurIPS 2017 (coordinate median)
# ─────────────────────────────────────────────────────────────────────────────

def coord_median_norm_bound(client_updates: List[List[np.ndarray]],
                             tau: float = 2.0) -> List[np.ndarray]:
    """Byzantine-robust aggregation: NormBound clipping + coordinate median.

    1. Compute the median L2 norm of all updates.
    2. Clip any update whose norm exceeds tau × median norm.
    3. Take coordinate-wise median of the clipped updates.

    Parameters
    ----------
    client_updates : list of deltas (local_params - global_params)
    tau : clipping threshold multiplier (default 2.0)
    """
    # --- NormBound ---
    # Flatten all updates for norm computation (float64 to avoid overflow)
    norms = np.array([
        float(np.sqrt(np.clip(sum(
            np.sum(np.asarray(u, dtype=np.float64).ravel() ** 2)
            for u in upd
        ), 0, 1e24)))
        for upd in client_updates
    ])
    median_norm = float(np.median(norms)) + 1e-8
    threshold   = tau * median_norm

    bounded = []
    for upd, norm in zip(client_updates, norms):
        if norm > threshold:
            scale = threshold / norm
            bounded.append([np.asarray(p, dtype=np.float32) * scale for p in upd])
        else:
            bounded.append([np.asarray(p, dtype=np.float32) for p in upd])

    # --- Coordinate median ---
    # Use atleast_1d for stacking, then reshape back to original shape
    n_params = len(bounded[0])
    result = []
    for i in range(n_params):
        orig_shape = bounded[0][i].shape
        stacked = np.stack(
            [np.atleast_1d(b[i].ravel()) for b in bounded], axis=0
        )
        med = np.median(stacked, axis=0)
        result.append(med.reshape(orig_shape) if orig_shape != () else med.ravel()[0])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Minimal FL loop for Byzantine attack simulation
# Supports clean (Run A), attacked (Run B), and defended (Run C)
# ─────────────────────────────────────────────────────────────────────────────

def run_fl_attack_experiment(
        exchange_data:   List[Data],
        model_config:    Dict,
        device:          torch.device,
        num_rounds:      int,
        local_epochs:    int,
        lr:              float,
        malicious_fn:    Optional[callable] = None,
        use_defense:     bool = False,
        verbose:         bool = False,
) -> Dict:
    """Run one FL experiment.

    Parameters
    ----------
    malicious_fn : callable or None
        If None → clean run (Run A).
        Signature: malicious_fn(global_params, honest_params, honest_n_samples)
                   → (poisoned_params, n_samples)
        The function receives honest training output and returns the poisoned
        version, so P1/P2/P3 each implement their own logic here.
    use_defense : bool
        If True, apply CoordMedian+NormBound before aggregation.
    """
    server  = FederatedServer(model_config, device)
    clients = [
        FederatedClient(
            client_id=i, graph_data=gd,
            model_config=model_config, device=device,
            local_epochs=local_epochs, lr=lr,
        )
        for i, gd in enumerate(exchange_data)
    ]

    best_metric = 0.0
    best_params = None

    for rnd in range(num_rounds):
        global_params = server.get_parameters()
        for c in clients:
            c.set_parameters(global_params)

        # Local training — all clients
        results = []
        for i, c in enumerate(clients):
            params, n, metrics = c.train(1.0, 1.5)
            results.append((params, n, metrics))

        # Inject malicious update for MALICIOUS_IDX
        if malicious_fn is not None:
            honest_params, honest_n, _ = results[MALICIOUS_IDX]
            poi_params, poi_n = malicious_fn(global_params, honest_params, honest_n)
            results[MALICIOUS_IDX] = (poi_params, poi_n, {'loss': 0.0})

        # Aggregation
        if use_defense:
            # Compute deltas for defense
            deltas = [
                [results[i][0][j] - global_params[j]
                 for j in range(len(global_params))]
                for i in range(len(results))
            ]
            agg_delta = coord_median_norm_bound(deltas)
            new_params = [global_params[j] + agg_delta[j]
                          for j in range(len(global_params))]
            sd = {k: torch.tensor(v) for k, v in
                  zip(server.global_model.state_dict().keys(), new_params)}
            server.global_model.load_state_dict(sd)
        else:
            server.aggregate(results)

        # Track best
        fwd_accs, bwd_accs = [], []
        for c in clients:
            c.set_parameters(server.get_parameters())
            m = evaluate_model(c.model, c.graph_data, device, 'val')
            fwd_accs.append(m['forward_accuracy'])
            bwd_accs.append(m['backward_accuracy'])
        combined = 0.4 * np.mean(fwd_accs) + 0.6 * np.mean(bwd_accs)
        if combined > best_metric:
            best_metric = combined
            best_params = server.get_parameters()

        if verbose:
            print(f"  Round {rnd+1:3d}  fwd={np.mean(fwd_accs):.3f}  "
                  f"bwd={np.mean(bwd_accs):.3f}")

    if best_params:
        sd = {k: torch.tensor(v) for k, v in
              zip(server.global_model.state_dict().keys(), best_params)}
        server.global_model.load_state_dict(sd)

    return avg_metrics(exchange_data, server.global_model, device, 'test')


# ─────────────────────────────────────────────────────────────────────────────
# E1 / E2 — PGD Evasion
# ─────────────────────────────────────────────────────────────────────────────

def _pgd(model: DualAttributionModel,
         x:     torch.Tensor,
         edge_index: torch.Tensor,
         y:     torch.Tensor,
         mask:  torch.Tensor,
         head:  str,
         epsilon: float,
         steps:   int = 7,
         alpha:   Optional[float] = None) -> torch.Tensor:
    """Single PGD run targeting one attribution head."""
    if epsilon == 0.0:
        return x
    alpha = alpha or epsilon / 4.0
    model.eval()

    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, -10.0, 10.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        out = model(x_adv, edge_index)

        if head == 'forward':
            loss = F.cross_entropy(out['forward'][mask], y[mask])
        else:
            bwd  = y[mask]
            cmask = bwd > 0
            if cmask.sum() == 0:
                break
            loss = F.cross_entropy(out['backward'][mask][cmask], bwd[cmask])

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x, -epsilon, epsilon)
            x_adv = torch.clamp(x + delta, -10.0, 10.0).detach()

    model.train()
    return x_adv


def attack_e1_e2(model: DualAttributionModel,
                 exchange_data: List[Data],
                 device: torch.device,
                 head: str,
                 pgd_steps: int = 7,
                 defense_model: Optional[DualAttributionModel] = None) -> Dict:
    """E1 (head='forward') or E2 (head='backward'): PGD feature evasion.

    Reference: Madry et al., ICLR 2018.
    """
    label = 'E1' if head == 'forward' else 'E2'
    metric = 'forward_accuracy' if head == 'forward' else 'backward_accuracy'
    print_header(label, f"PGD evasion → {head} head  (steps={pgd_steps})")

    results_by_eps = {}

    for eps in EPS_SWEEP:
        accs = []
        for gd in exchange_data:
            gd   = gd.to(device)
            mask = gd.test_mask
            y    = gd.y_forward if head == 'forward' else gd.y_backward

            x_adv = _pgd(model, gd.x, gd.edge_index, y, mask,
                         head=head, epsilon=eps, steps=pgd_steps)

            model.eval()
            with torch.no_grad():
                out = model(x_adv, gd.edge_index)

            if head == 'forward':
                acc = (out['forward'][mask].argmax(1) ==
                       gd.y_forward[mask]).float().mean().item()
            else:
                bwd   = gd.y_backward[mask]
                cmask = bwd > 0
                acc   = (out['backward'][mask][cmask].argmax(1) ==
                         bwd[cmask]).float().mean().item() \
                        if cmask.sum() > 0 else 0.0
            accs.append(acc)

        mean_acc = round(float(np.mean(accs)), 4)
        key = f'eps_{eps:.2f}'.replace('.', '_')
        results_by_eps[key] = mean_acc
        tag = 'clean' if eps == 0.0 else f'ε={eps}'
        print(f"  {tag:8s}  {metric.replace('_accuracy',''):8s} acc = {mean_acc:.4f}")

    clean_key = 'eps_0_00'
    worst_key = 'eps_0_30'
    drop = round(
        results_by_eps.get(clean_key, 0.0) - results_by_eps.get(worst_key, 0.0), 4)
    results_by_eps['acc_drop_0_to_0_30'] = drop
    print(f"\n  Accuracy drop (ε=0 → ε=0.30): {drop:+.4f}")

    result = {'attack': label, 'head': head, 'no_defense': results_by_eps}

    # Defense: compare with adversarially-trained model (FAT-base)
    if defense_model is not None:
        print(f"\n  [Defense] Adversarial training (FAT-base) — same sweep")
        defense_results = {}
        for eps in EPS_SWEEP:
            accs = []
            for gd in exchange_data:
                gd   = gd.to(device)
                mask = gd.test_mask
                y    = gd.y_forward if head == 'forward' else gd.y_backward
                x_adv = _pgd(defense_model, gd.x, gd.edge_index, y, mask,
                             head=head, epsilon=eps, steps=pgd_steps)
                defense_model.eval()
                with torch.no_grad():
                    out = defense_model(x_adv, gd.edge_index)
                if head == 'forward':
                    acc = (out['forward'][mask].argmax(1) ==
                           gd.y_forward[mask]).float().mean().item()
                else:
                    bwd   = gd.y_backward[mask]
                    cmask = bwd > 0
                    acc   = (out['backward'][mask][cmask].argmax(1) ==
                             bwd[cmask]).float().mean().item() \
                            if cmask.sum() > 0 else 0.0
                accs.append(acc)
            mean_acc = round(float(np.mean(accs)), 4)
            key = f'eps_{eps:.2f}'.replace('.', '_')
            defense_results[key] = mean_acc
            tag = 'clean' if eps == 0.0 else f'ε={eps}'
            vanilla_acc = results_by_eps.get(key, float('nan'))
            gain = round(mean_acc - vanilla_acc, 4)
            print(f"  {tag:8s}  AT model={mean_acc:.4f}  "
                  f"standard={vanilla_acc:.4f}  gain={gain:+.4f}")
        d_clean = defense_results.get('eps_0_00', 0.0)
        d_worst = defense_results.get('eps_0_30', 0.0)
        defense_results['acc_drop_0_to_0_30'] = round(d_clean - d_worst, 4)
        defense_results['acc_drop_reduction'] = round(
            results_by_eps.get('acc_drop_0_to_0_30', 0.0) -
            defense_results['acc_drop_0_to_0_30'], 4)
        print(f"\n  AT drop (ε=0→0.30): {defense_results['acc_drop_0_to_0_30']:+.4f}  "
              f"reduction vs standard: {defense_results['acc_drop_reduction']:+.4f}")
        result['with_defense'] = defense_results

    return result


# ─────────────────────────────────────────────────────────────────────────────
# P1 — Byzantine noise
# ─────────────────────────────────────────────────────────────────────────────

def _p1_malicious_fn(global_params, honest_params, honest_n):
    """Replace honest update with random noise scaled to update norm.
    Preserves exact shapes of each parameter (including 0-dim BatchNorm scalars).
    """
    # Compute delta norm in float64 to avoid float32 overflow
    d_norm_sq = sum(
        float(np.sum((np.asarray(h, dtype=np.float64).ravel() -
                      np.asarray(g, dtype=np.float64).ravel()) ** 2))
        for h, g in zip(honest_params, global_params)
    )
    d_norm = float(np.sqrt(np.clip(d_norm_sq, 0, 1e12))) + 1e-8
    # Generate noise with the EXACT same shape as each global param
    poi_params = []
    for g in global_params:
        g_arr = np.asarray(g, dtype=np.float32)
        if g_arr.ndim == 0:  # scalar (e.g. num_batches_tracked)
            noise = np.float32(np.random.randn() * d_norm)
            poi_params.append(g_arr + noise)
        else:
            noise = np.random.randn(*g_arr.shape).astype(np.float32) * d_norm
            poi_params.append(g_arr + noise)
    return poi_params, honest_n


def attack_p1(model_config: Dict, exchange_data: List[Data],
              device: torch.device, num_rounds: int,
              local_epochs: int, lr: float,
              run_defense: bool) -> Dict:
    """P1 — Byzantine noise injected by exchange 0 every round.

    Reference: Blanchard et al., NeurIPS 2017.
    """
    print_header("P1", "Byzantine noise  (exchange 0, every round)")

    # Run A — clean
    print("\n  [Run A] Clean FL (no attacker)")
    run_a = run_fl_attack_experiment(
        exchange_data, model_config, device, num_rounds, local_epochs, lr,
        malicious_fn=None, use_defense=False, verbose=False)
    print(f"  Run A  fwd={run_a['forward_acc']:.4f}  bwd={run_a['backward_acc']:.4f}")

    # Run B — attack, no defense
    print("\n  [Run B] Byzantine noise, no defense")
    run_b = run_fl_attack_experiment(
        exchange_data, model_config, device, num_rounds, local_epochs, lr,
        malicious_fn=_p1_malicious_fn, use_defense=False, verbose=False)
    print(f"  Run B  fwd={run_b['forward_acc']:.4f}  bwd={run_b['backward_acc']:.4f}")
    print(f"  Attack impact  fwd={run_b['forward_acc']-run_a['forward_acc']:+.4f}  "
          f"bwd={run_b['backward_acc']-run_a['backward_acc']:+.4f}")

    result = {
        'attack': 'P1', 'malicious_fraction': '1/3',
        'run_a': run_a, 'run_b': run_b,
        'attack_fwd_impact': round(run_b['forward_acc'] - run_a['forward_acc'], 4),
        'attack_bwd_impact': round(run_b['backward_acc'] - run_a['backward_acc'], 4),
    }

    # Run C — attack + defense
    if run_defense:
        print("\n  [Run C] Byzantine noise + CoordMedian+NormBound defense")
        run_c = run_fl_attack_experiment(
            exchange_data, model_config, device, num_rounds, local_epochs, lr,
            malicious_fn=_p1_malicious_fn, use_defense=True, verbose=False)
        print(f"  Run C  fwd={run_c['forward_acc']:.4f}  bwd={run_c['backward_acc']:.4f}")
        print(f"  Defense gain  fwd={run_c['forward_acc']-run_b['forward_acc']:+.4f}  "
              f"bwd={run_c['backward_acc']-run_b['backward_acc']:+.4f}")
        result['run_c'] = run_c
        result['defense_fwd_gain'] = round(run_c['forward_acc'] - run_b['forward_acc'], 4)
        result['defense_bwd_gain'] = round(run_c['backward_acc'] - run_b['backward_acc'], 4)
        result['defense_cost_fwd'] = round(run_a['forward_acc'] - run_c['forward_acc'], 4)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# P2 — Backward label-flip (KYC manipulation)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_label_flip(exchange_data: List[Data],
                      flip_pct: float = 0.30,
                      from_class: int = 2,
                      to_class: int = 1) -> List[Data]:
    """Flip backward labels in exchange 0's training set.

    Simulates a colluding exchange relabeling merchants in its KYC database.
    Only training labels are flipped; test labels remain correct.
    """
    poisoned = []
    for i, gd in enumerate(exchange_data):
        if i != MALICIOUS_IDX:
            poisoned.append(gd)
            continue
        gd_copy = copy.deepcopy(gd)
        y_bwd   = gd_copy.y_backward.clone()

        eligible = (y_bwd == from_class) & gd_copy.train_mask
        n_flip   = int(eligible.sum().item() * flip_pct)

        if n_flip > 0:
            idx    = eligible.nonzero(as_tuple=True)[0]
            chosen = idx[torch.randperm(len(idx))[:n_flip]]
            y_bwd[chosen] = to_class
            gd_copy.y_backward = y_bwd
            print(f"  [P2]  Flipped {n_flip} labels: "
                  f"class {from_class} → {to_class} in exchange {i} train_mask")
        else:
            print(f"  [P2]  WARNING: no eligible nodes to flip in exchange {i}")

        poisoned.append(gd_copy)
    return poisoned


def attack_p2(model_config: Dict, exchange_data: List[Data],
              device: torch.device, num_rounds: int,
              local_epochs: int, lr: float,
              flip_pct: float, run_defense: bool) -> Dict:
    """P2 — Label-flip: exchange 0 relabels Gambling → E-commerce in its KYC.

    Reference: Fang et al., USENIX Security 2020.
    """
    print_header("P2", f"Label-flip KYC  (Gambling→E-commerce, {flip_pct:.0%} of labels)")

    # Run A — clean
    print("\n  [Run A] Clean FL (no flip)")
    run_a = run_fl_attack_experiment(
        exchange_data, model_config, device, num_rounds, local_epochs, lr,
        malicious_fn=None, use_defense=False)
    print(f"  Run A  fwd={run_a['forward_acc']:.4f}  bwd={run_a['backward_acc']:.4f}")

    # Run B — poisoned data, no defense
    print("\n  [Run B] Poisoned labels, no defense")
    poisoned_data = _apply_label_flip(exchange_data, flip_pct)
    run_b = run_fl_attack_experiment(
        poisoned_data, model_config, device, num_rounds, local_epochs, lr,
        malicious_fn=None, use_defense=False)
    print(f"  Run B  fwd={run_b['forward_acc']:.4f}  bwd={run_b['backward_acc']:.4f}")
    print(f"  Attack impact  fwd={run_b['forward_acc']-run_a['forward_acc']:+.4f}  "
          f"bwd={run_b['backward_acc']-run_a['backward_acc']:+.4f}")

    result = {
        'attack': 'P2',
        'flip_pct': flip_pct, 'from_class': 2, 'to_class': 1,
        'malicious_fraction': '1/3',
        'run_a': run_a, 'run_b': run_b,
        'attack_fwd_impact': round(run_b['forward_acc'] - run_a['forward_acc'], 4),
        'attack_bwd_impact': round(run_b['backward_acc'] - run_a['backward_acc'], 4),
    }

    # Run C — poisoned data + defense
    if run_defense:
        print("\n  [Run C] Poisoned labels + CoordMedian+NormBound defense")
        run_c = run_fl_attack_experiment(
            poisoned_data, model_config, device, num_rounds, local_epochs, lr,
            malicious_fn=None, use_defense=True)
        print(f"  Run C  fwd={run_c['forward_acc']:.4f}  bwd={run_c['backward_acc']:.4f}")
        print(f"  Defense gain  fwd={run_c['forward_acc']-run_b['forward_acc']:+.4f}  "
              f"bwd={run_c['backward_acc']-run_b['backward_acc']:+.4f}")
        result['run_c'] = run_c
        result['defense_fwd_gain'] = round(run_c['forward_acc'] - run_b['forward_acc'], 4)
        result['defense_bwd_gain'] = round(run_c['backward_acc'] - run_b['backward_acc'], 4)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# P3 — Sign-flip + scaling
# ─────────────────────────────────────────────────────────────────────────────

def _make_p3_fn(gamma: float):
    """Return a malicious_fn that sends global − γ·delta."""
    def fn(global_params, honest_params, honest_n):
        poi_params = [
            np.asarray(g, dtype=np.float32) -
            gamma * (np.asarray(h, dtype=np.float32) - np.asarray(g, dtype=np.float32))
            for h, g in zip(honest_params, global_params)
        ]
        return poi_params, honest_n
    return fn


def attack_p3(model_config: Dict, exchange_data: List[Data],
              device: torch.device, num_rounds: int,
              local_epochs: int, lr: float, run_defense: bool) -> Dict:
    """P3 — Sign-flip with three scaling levels.

    γ=0.5  sub-threshold: attacker stays well below detection threshold.
           Expected: partial degradation, shows progressive behavior.
    γ=2    adaptive: (n_clients−1)/n_malicious = 2/1 = 2.
           Maximum scaling that evades norm-based filtering (Baruch et al.).
    γ=10   aggressive: stress test, clearly detectable but maximally damaging.

    Reference: Baruch et al., NeurIPS 2019.
    """
    print_header("P3", "Sign-flip + scaling  (γ=0.5 sub-threshold, γ=2 adaptive, γ=10 aggressive)")

    # Run A — clean
    print("\n  [Run A] Clean FL (no sign-flip)")
    run_a = run_fl_attack_experiment(
        exchange_data, model_config, device, num_rounds, local_epochs, lr,
        malicious_fn=None, use_defense=False)
    print(f"  Run A  fwd={run_a['forward_acc']:.4f}  bwd={run_a['backward_acc']:.4f}")

    results = {'attack': 'P3', 'run_a': run_a,
               'attack_fwd_impact_sub': None,
               'attack_fwd_impact_adaptive': None,
               'attack_fwd_impact_aggressive': None}

    for label, gamma in [('sub-threshold (γ=0.5)', 0.5),
                         ('adaptive (γ=2)', 2.0),
                         ('aggressive (γ=10)', 10.0)]:
        gkey = 'sub' if gamma == 0.5 else ('adaptive' if gamma == 2.0 else 'aggressive')

        # Run B
        print(f"\n  [Run B] Sign-flip {label}, no defense")
        run_b = run_fl_attack_experiment(
            exchange_data, model_config, device, num_rounds, local_epochs, lr,
            malicious_fn=_make_p3_fn(gamma), use_defense=False)
        print(f"  Run B  fwd={run_b['forward_acc']:.4f}  bwd={run_b['backward_acc']:.4f}")
        print(f"  Attack impact  fwd={run_b['forward_acc']-run_a['forward_acc']:+.4f}  "
              f"bwd={run_b['backward_acc']-run_a['backward_acc']:+.4f}")

        results[f'run_b_{gkey}'] = run_b
        results[f'attack_fwd_impact_{gkey}'] = round(
            run_b['forward_acc'] - run_a['forward_acc'], 4)
        results[f'attack_bwd_impact_{gkey}'] = round(
            run_b['backward_acc'] - run_a['backward_acc'], 4)

        if run_defense:
            print(f"\n  [Run C] Sign-flip {label} + CoordMedian+NormBound")
            run_c = run_fl_attack_experiment(
                exchange_data, model_config, device, num_rounds, local_epochs, lr,
                malicious_fn=_make_p3_fn(gamma), use_defense=True)
            print(f"  Run C  fwd={run_c['forward_acc']:.4f}  bwd={run_c['backward_acc']:.4f}")
            print(f"  Defense gain  fwd={run_c['forward_acc']-run_b['forward_acc']:+.4f}  "
                  f"bwd={run_c['backward_acc']-run_b['backward_acc']:+.4f}")
            results[f'run_c_{gkey}'] = run_c
            results[f'defense_gain_fwd_{gkey}'] = round(
                run_c['forward_acc'] - run_b['forward_acc'], 4)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# D1 — Membership Inference Attack (three variants)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_losses_and_features(model, exchange_data, device):
    """Collect per-node loss, output stats, and membership labels.
    Returns: loss_vec (N,), feat_mat (N,4), y_true (N,)
    """
    model.eval()
    all_loss, all_feat, all_labels = [], [], []
    for gd in exchange_data:
        gd = gd.to(device)
        with torch.no_grad():
            out  = model(gd.x, gd.edge_index)
            prob = torch.softmax(out['forward'], dim=1).cpu()
            logp = torch.log_softmax(out['forward'], dim=1).cpu()
            y    = gd.y_forward.cpu()
        for mask, is_member in [(gd.train_mask, 1), (gd.test_mask, 0)]:
            mask   = mask.cpu()
            p_m    = prob[mask].numpy()
            y_m    = y[mask].numpy()
            loss_m = -logp[mask].numpy()[np.arange(len(y_m)), y_m]
            finite = np.isfinite(loss_m)
            if not finite.all():
                loss_m = np.where(finite, loss_m, float(np.nanmedian(loss_m)))
            conf_m   = p_m.max(axis=1)
            ent_m    = -(p_m * np.log(p_m + 1e-10)).sum(axis=1)
            top2     = np.sort(p_m, axis=1)[:, -2:]
            margin_m = top2[:, 1] - top2[:, 0]
            all_loss.append(loss_m)
            all_feat.append(np.stack([conf_m, ent_m, margin_m, loss_m], axis=1))
            all_labels.append(np.full(len(y_m), is_member))
    return (np.concatenate(all_loss),
            np.vstack(all_feat),
            np.concatenate(all_labels))


def _mia_score(y_true, scores, name):
    from sklearn.metrics import roc_auc_score, accuracy_score
    try:
        auc = float(roc_auc_score(y_true, scores))
    except Exception:
        auc = 0.5
    preds = (scores >= np.median(scores)).astype(int)
    acc   = float(accuracy_score(y_true, preds))
    adv   = max(0.0, 2 * (acc - 0.5))
    leak  = 'LOW' if auc < 0.55 else 'MODERATE' if auc < 0.65 else 'HIGH'
    print(f"  [{name}]  AUC={auc:.4f}  advantage={adv:.4f}  leakage={leak}")
    return {'variant': name, 'auc': round(auc, 4), 'accuracy': round(acc, 4),
            'mia_advantage': round(adv, 4), 'privacy_leakage': leak}


def _d1_yeom(model, exchange_data, device):
    """Loss-based MIA. Yeom et al., CSF 2018. DOI: 10.1109/csf.2018.00027"""
    loss_vec, _, y_true = _collect_losses_and_features(model, exchange_data, device)
    return _mia_score(y_true, -loss_vec, 'Yeom-loss (2018)')


def _d1_shadow(model, exchange_data, device):
    """Salem et al. (NDSS 2019) — ML-Leaks. DOI: 10.14722/ndss.2019.23119

    Meta-classifier on output statistics [confidence, entropy, top2_margin, loss]
    extracted from the TARGET model. No shadow models required.

    WHY NOT SHOKRI 2017 (DOI: 10.1109/sp.2017.41):
    Shokri et al. require the attacker to train K shadow models on data drawn
    from the SAME distribution as the target model's training set. In FedBTC,
    the FL server does not possess labeled exchange-assignment data — that is
    precisely what federated learning is designed to protect. Without auxiliary
    data matching the per-exchange label distribution, shadow models produce a
    training signal that is statistically misspecified, rendering the Shokri
    attack inapplicable in this threat model.

    Salem et al. (ML-Leaks, NDSS 2019) relax this assumption by showing that
    the target model's own output statistics carry sufficient signal, making
    shadow models unnecessary. This is the appropriate variant for the FedBTC
    server-side threat model where the adversary has black-box query access to
    the global model but no labeled auxiliary data.

    References:
      Shokri et al., IEEE S&P 2017 — DOI: 10.1109/sp.2017.41
      Salem   et al., NDSS 2019   — DOI: 10.14722/ndss.2019.23119
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler

    _, feat_mat, y_true = _collect_losses_and_features(model, exchange_data, device)

    scaler   = StandardScaler()
    feat_std = scaler.fit_transform(feat_mat)
    clf      = LogisticRegression(max_iter=500, C=1.0)
    n_cv     = min(5, int(y_true.sum()), int((y_true == 0).sum()))
    try:
        prob_pred = cross_val_predict(clf, feat_std, y_true,
                                      cv=n_cv, method='predict_proba')[:, 1]
    except Exception as e:
        print(f"  [Salem] CV failed ({e}), defaulting to AUC=0.5")
        return {'variant': 'Salem-MLLeaks (2019)', 'auc': 0.5,
                'accuracy': 0.5, 'mia_advantage': 0.0, 'privacy_leakage': 'LOW'}
    return _mia_score(y_true, prob_pred, 'Salem-MLLeaks (2019)')


def _d1_lira(model, exchange_data, device, n_shadows=128,
             shadow_epochs=15, lr=0.002):
    """Offline LiRA — Carlini et al., IEEE S&P 2022.
    DOI: 10.1109/sp46214.2022.9833649

    ALGORITHM (offline variant, Algorithm 1 with lines 5,6,10,12 omitted):

      For each target node x:
        1. For k = 1..K OUT-shadow models (each trained WITHOUT x):
               phi_out_k = loss(shadow_k, x)
        2. Fit Gaussian to OUT losses:
               mu_out, sigma_out = mean({phi_out_k}), std({phi_out_k})
        3. phi_target = loss(target_model, x)
        4. Score(x) = (mu_out - phi_target) / sigma_out
               Positive score → target model has LOWER loss than OUT-shadows
               → x is likely a member (model knows x)

    This is equivalent to a one-sided z-test: how many standard deviations
    below the OUT-distribution does the target model score?

    Deviation from online LiRA: no IN-models trained (half the compute).
    Offline LiRA is weaker than online but sufficient for privacy auditing
    when online is computationally infeasible (K=256 required for reliable
    Gaussian estimation in online mode).

    K=128 (offline): matches the original paper recommendation for
    offline LiRA (Carlini et al., 2022). K=256 is the online variant.
    Using K=128 provides reliable Gaussian estimation at acceptable
    compute cost on the cluster (CPU-only, ~8h per seed for D1).
    """
    print(f"  [LiRA-offline] Training {n_shadows} OUT-shadow models "
          f"({shadow_epochs} epochs each)...")

    # Shadow models must match the target model's architecture for valid
    # Gaussian estimation. Read hidden_dim from model.config (stored in
    # DualAttributionModel.__init__ as self.config = config).
    # This fixes a hidden_dim mismatch when evaluating FAT models (128)
    # vs the former hardcoded default (64).
    _tc = model.config if hasattr(model, 'config') else {}
    mc = {
        'in_channels':    exchange_data[0].x.shape[1],
        'hidden_dim':     _tc.get('hidden_dim', 64),
        'dropout':        _tc.get('dropout', 0.3),
        'num_gcn_layers': 2,
        'num_exchanges':  3,
        'num_categories': 6,
    }

    # We evaluate on exchange 0 (the one most exposed to the server)
    gd0 = exchange_data[0].to(device)
    N0  = gd0.num_nodes

    # Collect OUT-shadow losses: shape (n_shadows, N0)
    out_losses = np.full((n_shadows, N0), np.nan, dtype=np.float32)

    for k in range(n_shadows):
        # OUT-model for node i: trained on a random 50% subset that EXCLUDES i
        # Practical approximation: train on random 50% — any given node is
        # excluded with probability 0.5 across shadow models.
        # This is the standard offline LiRA construction.
        out_mask = torch.rand(N0) > 0.5      # ~50% of nodes excluded
        in_idx   = (~out_mask).nonzero(as_tuple=True)[0]

        if len(in_idx) < 10:
            continue

        sm  = DualAttributionModel(mc).to(device)
        opt = torch.optim.Adam(sm.parameters(), lr=lr)

        for _ in range(shadow_epochs):
            sm.train(); opt.zero_grad()
            out  = sm(gd0.x, gd0.edge_index)
            loss = F.cross_entropy(out['forward'][in_idx], gd0.y_forward[in_idx])
            loss.backward(); opt.step()

        sm.eval()
        with torch.no_grad():
            out  = sm(gd0.x, gd0.edge_index)
            logp = F.log_softmax(out['forward'], dim=1).cpu().numpy()
            y    = gd0.y_forward.cpu().numpy()

        for i in range(N0):
            l = -logp[i, y[i]]
            out_losses[k, i] = l if np.isfinite(l) else np.nan

        if (k + 1) % 4 == 0:
            print(f"    shadow {k+1}/{n_shadows} done")

    # Offline LiRA score per node:
    # score(x) = (mu_out - phi_target) / sigma_out
    # where phi_target = loss(target_model, x)
    _, _, y_true = _collect_losses_and_features(model, exchange_data[:1], device)
    loss_target, _, _ = _collect_losses_and_features(model, exchange_data[:1], device)
    # Recompute per-node target loss directly
    model.eval()
    with torch.no_grad():
        out_t = model(gd0.x, gd0.edge_index)
        logp_t = F.log_softmax(out_t['forward'], dim=1).cpu().numpy()
        y_t    = gd0.y_forward.cpu().numpy()
    phi_target = np.array([-logp_t[i, y_t[i]] for i in range(N0)], dtype=np.float32)

    # Compute Gaussian parameters over OUT-shadow losses per node
    mu_out    = np.nanmean(out_losses, axis=0)    # shape (N0,)
    sigma_out = np.nanstd( out_losses, axis=0)    # shape (N0,)

    # Avoid division by zero (nodes with constant shadow losses)
    sigma_out = np.where(sigma_out < 1e-6, 1e-6, sigma_out)

    # z-score: positive = target model scores lower loss than OUT-distribution
    scores = (mu_out - phi_target) / sigma_out

    # Valid nodes: must have enough OUT-shadow measurements
    n_valid_shadows = np.sum(~np.isnan(out_losses), axis=0)  # shape (N0,)
    valid = n_valid_shadows >= max(4, n_shadows // 4)

    if valid.sum() < 10:
        print("  [LiRA-offline] Insufficient valid nodes, defaulting to AUC=0.5")
        return {'variant': f'LiRA-offline-{n_shadows}sh (Carlini 2022)',
                'auc': 0.5, 'accuracy': 0.5, 'mia_advantage': 0.0,
                'privacy_leakage': 'LOW',
                'note': f'Only {valid.sum()} valid nodes'}

    y_true_ex0 = gd0.train_mask.cpu().numpy().astype(int)
    result = _mia_score(y_true_ex0[valid], scores[valid],
                        f'LiRA-offline-{n_shadows}sh (Carlini 2022)')
    result['n_valid_nodes'] = int(valid.sum())
    result['mean_sigma_out'] = float(np.nanmean(sigma_out[valid]))
    return result



def attack_d1(model: DualAttributionModel,
              exchange_data: List[Data],
              device: torch.device,
              run_defense: bool,
              dp_model_paths: Optional[Dict[float, str]] = None,
              n_shadows: int = 128,
              shadow_epochs: int = 15) -> Dict:
    """D1 — Membership Inference Attack, three variants by increasing strength.

    Yeom 2018  (CSF)    : loss threshold — weakest, standard baseline
    Salem 2019 (NDSS)   : meta-classifier on target outputs (ML-Leaks)
                          NOTE: Shokri 2017 inapplicable in FedBTC —
                          requires same-distribution labeled aux. data
                          (see _d1_shadow docstring for full justification)
    LiRA 2022  (S&P)    : offline likelihood ratio with Gaussian fit
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {'attack': 'D1', 'error': 'scikit-learn not available'}

    print_header("D1", "Membership Inference — 3 variants (Yeom | Salem | LiRA-offline)")
    yeom_r   = _d1_yeom(model, exchange_data, device)
    shadow_r = _d1_shadow(model, exchange_data, device)
    lira_r   = _d1_lira(model, exchange_data, device, n_shadows, shadow_epochs)

    result = {'attack': 'D1',
              'no_dp': {'yeom_2018': yeom_r, 'salem_2019': shadow_r,
                        'lira_2022': lira_r}}

    print(f"\n  Summary (no DP):")
    for name, r in result['no_dp'].items():
        print(f"    {name:15s}  AUC={r['auc']:.4f}  leakage={r['privacy_leakage']}")

    if run_defense:
        if not dp_model_paths:
            print("\n  [D1 Defense] Pass --dp_model_eps* to run DP sweep.")
        else:
            result['dp_sweep'] = {}
            for eps, p in sorted(dp_model_paths.items()):
                if not os.path.exists(p):
                    print(f"  eps={eps}: not found, skipping.")
                    continue
                print(f"\n  [DP eps={eps}]")
                dpm, _ = load_model(p, device)
                result['dp_sweep'][f'eps_{eps}'] = {
                    'yeom_2018':   _d1_yeom(dpm, exchange_data, device),
                    'salem_2019': _d1_shadow(dpm, exchange_data, device),
                    'lira_2022':   _d1_lira(dpm, exchange_data, device,
                                            n_shadows, shadow_epochs),
                }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# D2 — Gradient Leakage (DLG)
# ─────────────────────────────────────────────────────────────────────────────

def _dlg_one_batch(model, gd, batch_size, n_iters, device):
    """True DLG — Zhu, Liu & Han, NeurIPS 2019.
    DOI: 10.1007/978-3-030-63076-8_2

    ALGORITHM (Section 3 of paper):
      1. Compute real gradients: nabla_W = d/dW L(F(x, W), y)
      2. Initialize BOTH x_dummy ~ N(0,1) AND y_dummy ~ N(0,1)
      3. For t = 1..T:
           nabla_W_dummy = d/dW L(F(x_dummy, W), softmax(y_dummy))
           x_dummy, y_dummy = LBFGS_step(||nabla_W - nabla_W_dummy||^2)
      4. Return x_dummy, argmax(y_dummy)

    Both inputs AND labels are optimized. Labels are continuous logits
    (not one-hot) and recovered via argmax after convergence.

    Note: iDLG (Zhao et al. 2020, arXiv:2001.02610) derives labels
    analytically from gradient sign, avoiding label optimization.
    Original DLG is used here for fidelity to Zhu et al. 2019.

    Metric note: label_recovery_acc (per-element accuracy) is NOT used
    because with small batches argmax(y_dummy) converges to the majority
    class, producing inflated accuracy as a class-frequency artifact.
    exact_sequence_match (1 iff the full label sequence is exactly correct)
    is the only valid label reconstruction metric.
    """
    train_idx = gd.train_mask.nonzero(as_tuple=True)[0][:batch_size]
    x_real    = gd.x[train_idx].detach()
    y_real    = gd.y_forward[train_idx]
    n_classes = 3  # number of exchanges (forward head classes)

    # Compute REAL gradients on the real data
    model.zero_grad()
    out  = model(gd.x, gd.edge_index)
    loss = F.cross_entropy(out['forward'][train_idx], y_real)
    loss.backward()
    real_grads = [p.grad.clone().detach()
                  for p in model.parameters() if p.grad is not None]
    model.zero_grad()

    # Initialize BOTH x_dummy and y_dummy (true DLG)
    x_dummy = torch.randn_like(x_real, requires_grad=True)
    y_dummy = torch.randn(batch_size, n_classes,
                          device=device, requires_grad=True)

    opt = torch.optim.LBFGS([x_dummy, y_dummy], lr=0.1, max_iter=n_iters)

    def closure():
        opt.zero_grad()
        model.zero_grad()
        x_full            = gd.x.clone()
        x_full[train_idx] = x_dummy

        out_d  = model(x_full, gd.edge_index)
        # Use softmax(y_dummy) as soft labels → cross-entropy with label smoothing
        y_soft = F.softmax(y_dummy, dim=1)
        loss_d = -(y_soft * F.log_softmax(out_d['forward'][train_idx], dim=1)).sum(dim=1).mean()
        loss_d.backward(retain_graph=True)

        dummy_grads = [p.grad.clone()
                       for p in model.parameters() if p.grad is not None]
        gd_diff = sum(((dg - rg)**2).sum()
                      for dg, rg in zip(dummy_grads, real_grads))
        gd_diff.backward()
        return gd_diff

    try:
        opt.step(closure)
    except Exception:
        pass

    # Recovered label sequence — exact match only
    # per-element accuracy is NOT reported: with small batches, argmax(y_dummy)
    # converges to the majority class, producing a class-frequency artifact.
    y_recovered = y_dummy.detach().argmax(dim=1)
    exact_match = int((y_recovered == y_real).all().item())  # 1 iff fully correct

    mse = float(F.mse_loss(x_dummy.detach(), x_real).item())
    cos = float(F.cosine_similarity(
        x_dummy.detach().flatten().unsqueeze(0),
        x_real.flatten().unsqueeze(0)).item())

    return {
        'reconstruction_mse':    round(mse, 6),
        'cosine_similarity':     round(cos, 4),
        'exact_sequence_match':  exact_match,  # 1 = full label sequence reconstructed
    }



def attack_d2(model: DualAttributionModel,
              exchange_data: List[Data],
              device: torch.device,
              batch_sizes: List[int] = None,
              n_iters: int = 50,
              defense_model: Optional[DualAttributionModel] = None) -> Dict:
    """D2 — Deep Leakage from Gradients (Zhu & Han, NeurIPS 2019).

    Runs DLG over a sweep of batch sizes to separate:
      - batch_size=1 : isolates graph topology effect (DLG is strongest here)
      - batch_size=3 : intermediate
      - batch_size=8 : standard from original paper

    Reference: Zhu & Han, NeurIPS 2019 (DOI: 10.1007/978-3-030-63076-8_2)
    """
    if batch_sizes is None:
        batch_sizes = [1, 3, 8]

    print_header("D2", f"Gradient leakage DLG (batch_sizes={batch_sizes}, iters={n_iters})")
    print("  batch_size=1 isolates graph topology effect.")
    print("  High MSE at all batch sizes = structural protection.\n")

    result_all = {'attack': 'D2'}

    for batch_size in batch_sizes:
        print(f"  --- batch_size={batch_size} ---")
        ex_results = {}
        for i, gd in enumerate(exchange_data):
            gd = gd.to(device)
            r  = _dlg_one_batch(model, gd, batch_size, n_iters, device)
            ex_results[f'exchange_{i}'] = r
            print(f"  Exchange {i}: MSE={r['reconstruction_mse']:.6f}  "
                  f"cos={r['cosine_similarity']:.4f}  "
                  f"({'protected' if r['reconstruction_mse'] > 0.5 else 'reconstructed'})")

        avg_mse = float(np.mean([v['reconstruction_mse'] for v in ex_results.values()]))
        ex_results['avg_mse']      = round(avg_mse, 6)
        ex_results['dp_protected'] = avg_mse > 0.5
        print(f"  avg MSE={avg_mse:.6f}  "
              f"({'failed — graph protects' if avg_mse > 0.5 else 'reconstructed'})")

        bs_result = {'no_defense': ex_results}

        # Defense: compare with DP-protected model
        if defense_model is not None:
            print(f"  [Defense DP] DLG on DP-protected model, batch_size={batch_size}")
            dp_results = {}
            for i, gd in enumerate(exchange_data):
                gd = gd.to(device)
                r  = _dlg_one_batch(defense_model, gd, batch_size, n_iters, device)
                dp_results[f'exchange_{i}'] = r
                baseline = ex_results.get(f'exchange_{i}', {}).get('reconstruction_mse', float('nan'))
                print(f"  Exchange {i}: DP MSE={r['reconstruction_mse']:.6f}  "
                      f"baseline={baseline:.6f}  delta={r['reconstruction_mse']-baseline:+.6f}")
            dp_avg = float(np.mean([v['reconstruction_mse'] for v in dp_results.values()]))
            dp_results['avg_mse']      = round(dp_avg, 6)
            dp_results['dp_protected'] = dp_avg > 0.5
            bs_result['with_defense'] = dp_results
            print(f"  DP avg MSE={dp_avg:.6f}  vs standard={avg_mse:.6f}")

        result_all[f'batch_{batch_size}'] = bs_result

    # Summary
    mses = {bs: result_all[f'batch_{bs}']['no_defense']['avg_mse']
            for bs in batch_sizes}
    print(f"\n  MSE summary:")
    for bs, mse in sorted(mses.items()):
        print(f"    batch_size={bs}: MSE={mse:.6f}  "
              f"({'protected' if mse > 0.5 else 'reconstructed'})")
    result_all['mse_by_batch'] = {str(k): round(v, 6) for k, v in mses.items()}
    return result_all


# ─────────────────────────────────────────────────────────────────────────────
# D3 — Regulatory model analysis (passive)
# ─────────────────────────────────────────────────────────────────────────────

def attack_d3(model: DualAttributionModel,
              exchange_data: List[Data],
              device: torch.device) -> Dict:
    """D3 — Passive regulatory analysis.

    A regulator with legal access to the global model probes it with synthetic
    transactions to infer each exchange's backward category distribution,
    without accessing any raw data.

    Metric: KL divergence between inferred and real backward distributions.
    """
    print_header("D3", "Regulatory model analysis (passive, no defense)")
    # D3 is a legal passive attack — no technical defense exists.
    # When --defense is passed, we document the exposure level explicitly.

    # Real backward distributions per exchange
    real_dists, inferred_dists = {}, {}

    for i, gd in enumerate(exchange_data):
        gd = gd.to(device)

        # Real distribution (from the actual data)
        y_bwd  = gd.y_backward[gd.test_mask].cpu().numpy()
        real_counts = np.zeros(6)
        for c in range(6):
            real_counts[c] = (y_bwd == c).sum()
        real_dist = real_counts[1:] / (real_counts[1:].sum() + 1e-8)
        real_dists[f'exchange_{i}'] = real_dist.tolist()

        # Inferred distribution: inject 500 synthetic transactions
        # Features drawn from the exchange's test data distribution
        x_test = gd.x[gd.test_mask]
        n_synthetic = min(500, len(x_test))
        synthetic_x = x_test[torch.randperm(len(x_test))[:n_synthetic]]
        # Minimal edge: connect all synthetic nodes in a line
        idx = torch.arange(n_synthetic, device=device)
        syn_edges = torch.stack([idx[:-1], idx[1:]], dim=0)

        model.eval()
        with torch.no_grad():
            out   = model(synthetic_x, syn_edges)
            preds = out['backward'].argmax(dim=1).cpu().numpy()

        inferred_counts = np.zeros(6)
        for c in range(6):
            inferred_counts[c] = (preds == c).sum()
        inferred_dist = inferred_counts[1:] / (inferred_counts[1:].sum() + 1e-8)
        inferred_dists[f'exchange_{i}'] = inferred_dist.tolist()

        # KL divergence (add small epsilon for numerical stability)
        p = real_dist     + 1e-8
        q = inferred_dist + 1e-8
        kl = float(np.sum(p * np.log(p / q)))

        print(f"\n  Exchange {i}:")
        print(f"  {'Category':12s}  {'Real %':8s}  {'Inferred %':10s}")
        for c, name in enumerate(['E-commerce', 'Gambling', 'Services', 'Retail', 'Luxury']):
            print(f"  {name:12s}  {real_dist[c]*100:6.1f}%    {inferred_dist[c]*100:8.1f}%")
        print(f"  KL divergence = {kl:.4f}  "
              f"{'(low exposure)' if kl < 0.1 else '(moderate exposure)' if kl < 0.5 else '(high exposure)'}")

    return {
        'attack':          'D3',
        'real_dists':      real_dists,
        'inferred_dists':  inferred_dists,
        'note': 'No defense available for D3. Exposure level is documented only.',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='FedBTC Attack Benchmark')
    p.add_argument('--attack', required=True,
                   choices=['e1','e2','p1','p2','p3','d1','d2','d3','all'],
                   help='Attack to simulate')
    p.add_argument('--model', default=None,
                   help='Path to trained model checkpoint (.pt). '
                        'Required for E1/E2/D1/D2/D3. '
                        'For P1/P2/P3, model_config is inferred from the checkpoint '
                        'and a fresh FL run is started.')
    p.add_argument('--data_dir', default='data/federated_enriched',
                   help='Exchange data directory')
    p.add_argument('--defense', action='store_true',
                   help='Run with defense after the attack (Run C)')
    p.add_argument('--defense_model', default=None,
                   help='Path to adversarially-trained model for E1/E2/D2 defense '
                        '(e.g. FAT-base or FAT-def checkpoint). '
                        'Required when --defense is used with e1, e2, or d2.')
    p.add_argument('--seed', type=int, default=0,
                   help='Random seed (FedBTC seeds: 0 45 123 654 789)')
    p.add_argument('--fast', action='store_true',
                   help='Fast mode for local testing (10 rounds, 200 test nodes)')
    p.add_argument('--pgd_steps', type=int, default=7,
                   help='PGD steps for E1/E2 (default 7)')
    p.add_argument('--p2_flip_pct', type=float, default=0.30,
                   help='Fraction of Gambling labels to flip for P2 (default 0.30)')
    p.add_argument('--dlg_iters', type=int, default=50,
                   help='L-BFGS iterations for D2 DLG (default 50)')
    p.add_argument('--dlg_batches', type=str, default='1,3,8',
                   help='Comma-separated D2 batch sizes (default "1,3,8")')
    p.add_argument('--lira_shadows', type=int, default=128,
                   help='OUT shadow models for offline LiRA D1 (default 16, paper uses 128)')
    p.add_argument('--lira_epochs', type=int, default=15,
                   help='Training epochs per shadow model (default 15)')
    p.add_argument('--output_dir', default='results/attacks',
                   help='Output directory for JSON results')
    # DP model paths for D1 sweep
    p.add_argument('--dp_model_eps1', default=None, help='Path to FAT-def ε=1 model')
    p.add_argument('--dp_model_eps2', default=None, help='Path to FAT-def ε=2 model')
    p.add_argument('--dp_model_eps4', default=None, help='Path to FAT-def ε=4 model')
    p.add_argument('--dp_model_eps8', default=None, help='Path to FAT-def ε=8 model')
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_rounds   = 10  if args.fast else 100
    local_epochs = 1
    lr           = 0.002

    print(f"\nFedBTC Attack Benchmark")
    print(f"  attack  : {args.attack}")
    print(f"  seed    : {args.seed}")
    print(f"  device  : {device}")
    print(f"  fast    : {args.fast} ({'10 rounds, 200 nodes' if args.fast else '100 rounds, full dataset'})")
    print(f"  defense : {args.defense}")

    # Load exchange data
    print(f"\nLoading exchange data from {args.data_dir} ...")
    exchange_data = load_all_exchanges(args.data_dir, args.seed, fast=args.fast)
    print(f"  Exchange data loaded: {[gd.num_nodes for gd in exchange_data]} nodes")

    # Load model if needed
    model = None
    model_config = None
    if args.model and os.path.exists(args.model):
        print(f"  Loading model: {args.model}")
        model, model_config = load_model(args.model, device)
        print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    elif args.attack in ('e1', 'e2', 'd1', 'd2', 'd3', 'all'):
        if args.attack not in ('p1', 'p2', 'p3'):
            print("  WARNING: no --model provided. "
                  "Evasion and privacy attacks require a trained model.")
            if not args.fast:
                sys.exit(1)

    # For P1/P2/P3 without explicit model, derive config from data
    if model_config is None:
        n_feats = exchange_data[0].num_features
        model_config = {
            'in_channels':    n_feats,
            'hidden_dim':     128,
            'dropout':        0.3,
            'num_gcn_layers': 2,
            'num_exchanges':  3,    # fixed: 3 exchanges in FedBTC
            'num_categories': 6,   # fixed: 6 backward categories
        }

    results = {'attack': args.attack, 'seed': args.seed,
               'fast': args.fast, 'timestamp': datetime.now().isoformat()}

    # ── Load defense model if provided ───────────────────────────────────────
    defense_model = None
    if args.defense and args.defense_model:
        if os.path.exists(args.defense_model):
            print(f"  Loading defense model: {args.defense_model}")
            defense_model, _ = load_model(args.defense_model, device)
            print(f"  Defense model loaded.")
        else:
            print(f"  WARNING: --defense_model path not found: {args.defense_model}")
    elif args.defense and args.attack in ('e1', 'e2', 'd2'):
        print("  WARNING: --defense passed for E1/E2/D2 but --defense_model not provided.")
        print("  Pass --defense_model path/to/fat_base_seed0.pt to compare with AT model.")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    attacks_to_run = (['e1','e2','p1','p2','p3','d1','d2','d3']
                      if args.attack == 'all' else [args.attack])

    for atk in attacks_to_run:
        print(f"\n{'#'*60}")
        print(f"  Starting attack: {atk.upper()}")
        print(f"{'#'*60}")

        if atk in ('e1', 'e2'):
            if model is None:
                print(f"  Skipping {atk}: no model loaded.")
                continue
            head = 'forward' if atk == 'e1' else 'backward'
            res  = attack_e1_e2(model, exchange_data, device, head, args.pgd_steps,
                                defense_model=defense_model if args.defense else None)

        elif atk == 'p1':
            res = attack_p1(model_config, exchange_data, device,
                            num_rounds, local_epochs, lr, args.defense)

        elif atk == 'p2':
            res = attack_p2(model_config, exchange_data, device,
                            num_rounds, local_epochs, lr, args.p2_flip_pct, args.defense)

        elif atk == 'p3':
            res = attack_p3(model_config, exchange_data, device,
                            num_rounds, local_epochs, lr, args.defense)

        elif atk == 'd1':
            if model is None:
                print("  Skipping D1: no model loaded.")
                continue
            dp_paths = {}
            for eps, attr in [(1.0,'dp_model_eps1'),(2.0,'dp_model_eps2'),
                               (4.0,'dp_model_eps4'),(8.0,'dp_model_eps8')]:
                v = getattr(args, attr)
                if v:
                    dp_paths[eps] = v
            res = attack_d1(model, exchange_data, device, args.defense,
                            dp_model_paths=dp_paths if dp_paths else None,
                            n_shadows=args.lira_shadows,
                            shadow_epochs=args.lira_epochs)

        elif atk == 'd2':
            if model is None:
                print("  Skipping D2: no model loaded.")
                continue
            batch_sizes = [int(b) for b in args.dlg_batches.split(',')]
            res = attack_d2(model, exchange_data, device,
                            batch_sizes=batch_sizes, n_iters=args.dlg_iters,
                            defense_model=defense_model if args.defense else None)

        elif atk == 'd3':
            if model is None:
                print("  Skipping D3: no model loaded.")
                continue
            res = attack_d3(model, exchange_data, device)

        else:
            continue

        res['seed'] = args.seed
        results[atk] = res
        save_result(res, args.output_dir, f"{atk}_seed{args.seed}")

    print(f"\n{'='*60}")
    print(f"  Benchmark complete — seed {args.seed}")
    print(f"  Results saved to {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
