"""
p5_train_defense_configs_eps16.py
FedBTC — Train the missing defense configurations for the interaction matrix.

This file is a SIBLING of p5_train_defense_configs.py, with one addition:
  --models_dir  flag (default: 'results/models') to control where .pt files
                are written. Used by run_interaction_matrix_eps16.slurm to
                isolate the epsilon=16 outputs in results/models_eps16/
                without overwriting the original epsilon=8 runs in
                results/models/.

Lines changed vs the original:
  1. argparse:    new --models_dir argument
  2. model_path:  reads from a module-level _MODELS_DIR variable
  3. main():      sets _MODELS_DIR from args.models_dir
  4. train_config: passes args.models_dir to os.makedirs

Available from Phase 6 (still loaded from results/models/, not the
eps16 directory):
  C0  No defense           federated_dual_attribution_fl_r100e1_seed{N}.pt
  C1  AT only              federated_secure_dual_attribution_fat_base_seed{N}.pt
  C5  AT + CoordMedian + DP  federated_secure_dual_attribution_fat_def_eps16.0_seed{N}.pt

Missing (trained here):
  C2  CoordMedian only     federated_secure_dual_attribution_coord_only_seed{N}.pt
  C3  DP only (eps=16)     federated_secure_dual_attribution_dp_only_seed{N}.pt
  C4  AT + CoordMedian     federated_secure_dual_attribution_at_coord_seed{N}.pt
  C6  CoordMedian + DP     federated_secure_dual_attribution_coord_dp_seed{N}.pt
  C7  AT + DP              federated_secure_dual_attribution_at_dp_seed{N}.pt

USAGE:
    python p5_train_defense_configs_eps16.py --seed 0 --epsilon 16.0 \\
        --models_dir results/models_eps16
"""

import os, sys, json, argparse, torch, numpy as np
from datetime import datetime
import importlib.util

FAT_SCRIPT = 'p3_train_federated_secure_2layers_FAT.py'

try:
    spec = importlib.util.spec_from_file_location('fat', FAT_SCRIPT)
    fat  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fat)
    run_secure_fl                = fat.run_secure_fl
    load_exchange_data           = fat.load_exchange_data
    create_pyg_data_for_exchange = fat.create_pyg_data_for_exchange
    get_noise_multiplier         = fat.get_noise_multiplier
except Exception as e:
    sys.exit(f"[ERROR] Cannot import from {FAT_SCRIPT}: {e}")

CONFIGS = {
    'C2': dict(name='CoordMedian only',      tag='coord_only',
               use_adversarial=False, use_flame=True,  use_server_dp=False),
    'C3': dict(name='Server-side DP only',   tag='dp_only',
               use_adversarial=False, use_flame=False, use_server_dp=True),
    'C4': dict(name='AT + CoordMedian',      tag='at_coord',
               use_adversarial=True,  use_flame=True,  use_server_dp=False),
    'C6': dict(name='CoordMedian + DP',      tag='coord_dp',
               use_adversarial=False, use_flame=True,  use_server_dp=True),
    'C7': dict(name='AT + DP',               tag='at_dp',
               use_adversarial=True,  use_flame=False, use_server_dp=True),
}

# Module-level variable set from args.models_dir in main(). Default matches
# the original script so that running without --models_dir behaves identically.
_MODELS_DIR = 'results/models'


def model_path(tag, seed):
    return f'{_MODELS_DIR}/federated_secure_dual_attribution_{tag}_seed{seed}.pt'


def train_config(cfg_key, seed, data_dir, rounds=100, local_epochs=1,
                 lr=0.002, epsilon=8.0, dry_run=False):
    cfg      = CONFIGS[cfg_key]
    out_path = model_path(cfg['tag'], seed)
    print(f"\n── {cfg_key}: {cfg['name']}")
    print(f"   AT={cfg['use_adversarial']}  CoordMedian={cfg['use_flame']}  DP={cfg['use_server_dp']}")
    print(f"   Output: {out_path}")

    if os.path.exists(out_path):
        print("   [SKIP] Already exists.")
        return True
    if dry_run:
        print(f"   [DRY RUN] Would train {rounds} rounds.")
        return True

    device = torch.device('cpu')
    exchange_data = []
    for i in range(3):
        data, edges = load_exchange_data(i, data_dir)
        exchange_data.append(
            create_pyg_data_for_exchange(data, edges, exchange_id=i, seed=seed))

    model_config = {
        'in_channels':    exchange_data[0].num_features,
        'hidden_dim':     128, 'dropout': 0.3,
        'num_exchanges':  3,   'num_categories': 6,
    }

    print(f"   Training {rounds} rounds × {local_epochs} local epochs...")
    t0 = datetime.now()

    final_model, history = run_secure_fl(
        exchange_data=exchange_data, model_config=model_config, device=device,
        num_rounds=rounds, local_epochs=local_epochs, lr=lr,
        use_client_dp=False,
        use_server_dp=cfg['use_server_dp'],  epsilon=epsilon,
        noise_multiplier=None,
        use_adversarial=cfg['use_adversarial'],
        use_flame=cfg['use_flame'],
        use_fedval=False, alpha_forward=1.0, alpha_backward=1.5,
    )

    elapsed = (datetime.now() - t0).seconds // 60
    os.makedirs(_MODELS_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_config': model_config, 'config_key': cfg_key,
        'seed': seed, **cfg, 'epsilon': epsilon if cfg['use_server_dp'] else None,
    }, out_path)

    fwd = history['forward_accuracy'][-1]  if history.get('forward_accuracy')  else '?'
    bwd = history['backward_accuracy'][-1] if history.get('backward_accuracy') else '?'
    print(f"   [OK] {elapsed}min — fwd={fwd:.4f}  bwd={bwd:.4f}")
    return True


def main():
    global _MODELS_DIR
    p = argparse.ArgumentParser()
    p.add_argument('--seed',         type=int,   required=True)
    p.add_argument('--configs',      type=str,   default='C2,C3,C4,C6,C7')
    p.add_argument('--data_dir',     type=str,   default='data/federated_enriched')
    p.add_argument('--rounds',       type=int,   default=100)
    p.add_argument('--local_epochs', type=int,   default=1)
    p.add_argument('--lr',           type=float, default=0.002)
    p.add_argument('--epsilon',      type=float, default=8.0)
    p.add_argument('--models_dir',   type=str,   default='results/models',
                   help='Directory where .pt files are written '
                        '(default: results/models)')
    p.add_argument('--dry_run',      action='store_true')
    args = p.parse_args()

    # Propagate models_dir to the module-level constant used by model_path()
    _MODELS_DIR = args.models_dir

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    configs = [c.strip() for c in args.configs.split(',')]
    print(f"FedBTC — Training defense configs: {configs} | seed={args.seed} "
          f"| epsilon={args.epsilon} | models_dir={_MODELS_DIR}")

    for k in configs:
        if k not in CONFIGS:
            print(f"[WARN] Unknown config {k}. Valid: {list(CONFIGS)}")
            continue
        train_config(k, args.seed, args.data_dir,
                     args.rounds, args.local_epochs, args.lr, args.epsilon,
                     args.dry_run)
    print("\nDone.")


if __name__ == '__main__':
    main()