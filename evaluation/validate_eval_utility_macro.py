"""
validate_eval_utility_macro.py

Quick validation: call the patched eval_utility() on a few configs/seeds
and print the per-exchange breakdown alongside macro vs micro values.

Runs in ~30 seconds on the login node — no PGD, no FL re-training, no
MIA, no DLG. Just model loading + inference on test masks.

Usage:
    python validate_eval_utility_macro.py
"""

import os, sys
sys.path.insert(0, os.getcwd())

import numpy as np
import torch

from attack_benchmark import load_all_exchanges, load_model
from p5_interaction_matrix import eval_utility, model_filename


def eval_both_conventions(model, exchange_data, device):
    """Compute macro AND micro on the same model+data, side by side.
    Returns dict with per-exchange, macro, and micro values for fwd/bwd.
    """
    model.eval()
    fwd_per_exchange, bwd_per_exchange = [], []
    fwd_correct_global, fwd_total_global = 0, 0
    bwd_correct_global, bwd_total_global = 0, 0
    per_exchange_sizes = []

    for gd in exchange_data:
        gd = gd.to(device)
        mask = gd.test_mask
        n = mask.sum().item()
        per_exchange_sizes.append(n)
        if n == 0:
            continue

        with torch.no_grad():
            out = model(gd.x, gd.edge_index)

        # Forward
        pred = out['forward'][mask].argmax(1)
        c = (pred == gd.y_forward[mask]).sum().item()
        fwd_per_exchange.append(c / n)
        fwd_correct_global += c
        fwd_total_global   += n

        # Backward (ignore class 0)
        bwd_true = gd.y_backward[mask]
        bwd_mask = bwd_true > 0
        n_bwd = bwd_mask.sum().item()
        if n_bwd > 0:
            bwd_pred = out['backward'][mask][bwd_mask].argmax(1)
            c_bwd = (bwd_pred == bwd_true[bwd_mask]).sum().item()
            bwd_per_exchange.append(c_bwd / n_bwd)
            bwd_correct_global += c_bwd
            bwd_total_global   += n_bwd

    return {
        'per_exchange_sizes': per_exchange_sizes,
        'fwd_per_exchange':   fwd_per_exchange,
        'bwd_per_exchange':   bwd_per_exchange,
        'fwd_macro': float(np.mean(fwd_per_exchange)) if fwd_per_exchange else 0.0,
        'bwd_macro': float(np.mean(bwd_per_exchange)) if bwd_per_exchange else 0.0,
        'fwd_micro': fwd_correct_global / fwd_total_global if fwd_total_global else 0.0,
        'bwd_micro': bwd_correct_global / bwd_total_global if bwd_total_global else 0.0,
    }


def main():
    device = torch.device('cpu')
    SEED   = 0  # fast validation on a single seed

    print(f"Validation: eval_utility patched (macro-average)")
    print(f"Seed = {SEED} (full test set, no --fast)")
    print('=' * 78)

    # Load full test set (no --fast)
    exchange_data = load_all_exchanges(
        'data/federated_enriched', SEED, fast=False)
    sizes = [int(gd.test_mask.sum()) for gd in exchange_data]
    print(f"Exchange test-set sizes: {sizes}")
    print(f"Total test nodes       : {sum(sizes)}")
    print()

    for cfg in ['C0', 'C5']:  # C0 baseline + C5 = FAT-Def_eps8
        path = model_filename(cfg, SEED)
        if not os.path.exists(path):
            print(f"[MISS] {cfg}: {path}")
            continue

        print(f"--- Config {cfg} ---")
        print(f"    {os.path.basename(path)}")
        model, cfg_dict = load_model(path, device)

        # Side-by-side comparison
        side = eval_both_conventions(model, exchange_data, device)

        print(f"    Per-exchange forward acc : {[f'{v:.4f}' for v in side['fwd_per_exchange']]}")
        print(f"    Per-exchange backward acc: {[f'{v:.4f}' for v in side['bwd_per_exchange']]}")
        print(f"    Forward  macro = {side['fwd_macro']:.4f}    "
              f"micro = {side['fwd_micro']:.4f}    "
              f"diff = {abs(side['fwd_macro'] - side['fwd_micro']):.4f}")
        print(f"    Backward macro = {side['bwd_macro']:.4f}    "
              f"micro = {side['bwd_micro']:.4f}    "
              f"diff = {abs(side['bwd_macro'] - side['bwd_micro']):.4f}")

        # Confirm patched eval_utility returns the macro value
        u_fwd, u_bwd = eval_utility(model, exchange_data, device)
        print(f"    eval_utility() returns   : fwd={u_fwd:.4f}  bwd={u_bwd:.4f}")
        ok_fwd = abs(u_fwd - side['fwd_macro']) < 1e-3
        ok_bwd = abs(u_bwd - side['bwd_macro']) < 1e-3
        print(f"    Match macro?              : fwd={'OK' if ok_fwd else 'FAIL'}  "
              f"bwd={'OK' if ok_bwd else 'FAIL'}")
        print()

    print("=" * 78)
    print("Expected: eval_utility() outputs match the 'macro' column.")
    print("Expected: macro and micro should DIFFER by ~1-2% for non-i.i.d.")
    print("          partitions of unequal sizes (here all 67923, so closer).")
    print("Expected for C5 seed=0: fwd_macro ≈ 0.8540 (per training_result).")


if __name__ == '__main__':
    main()
