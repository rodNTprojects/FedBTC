# FedBTC: Privacy- and Confidentiality-Preserving Bitcoin Transaction Attribution via Federated Graph Learning

> Anonymous submission — ACM AFT 2026 (LIPIcs).
>
> A federated graph-learning framework that lets cryptocurrency exchanges
> collaboratively train a dual-attribution model over the Bitcoin transaction
> graph **without sharing raw transaction data**, defended by a three-layer
> security stack.

## Overview

Exchanges hold fragmented, mutually confidential views of the Bitcoin
transaction graph and cannot pool raw records across institutions. FedBTC
addresses two distinct constraints at once:

- **Confidentiality-preserving** (institutional level): a federated GCN lets
  `K=3` exchanges co-train without exchanging local graphs.
- **Privacy-preserving** (customer level): server-side differential privacy
  perturbs the released aggregate.

The model performs **dual attribution** on each transaction node:

- **Forward attribution** — predict the cash-out exchange (3 classes).
- **Backward attribution** — predict the originating merchant category
  (6 classes: NO_MERCHANT, E-commerce, Gambling, Services, Retail, Luxury);
  NO_MERCHANT is excluded from the backward loss.

Both heads share a 2-layer GCN trunk and couple only through the summed
gradient. The full per-round procedure is given as Algorithm 1 in the paper.

## Defense stack

Three independently toggleable layers, each targeting a distinct adversary:

| Layer | Tier | Mechanism | Adversary |
|------|------|-----------|-----------|
| 1 | Client | PGD adversarial training (ε_adv=0.2, α=0.05, 7 steps, ratio 0.30) | A4 — inference-time evasion |
| 2 | Server | Norm bounding (τ=2.0 × median norm) + coordinate-wise median | A3 — single Byzantine client |
| 3 | Server | Gaussian noise on the aggregate (σ = σ(ε), heuristic) | A1/A2 — honest-but-curious |

> **On the privacy budget.** ε is a **heuristic calibration label**, not a
> formally accounted (ε, δ) bound: the noise multiplier follows a fixed
> schedule (σ=0.1 at ε=8, σ=0.025 at ε=16) and no RDP accountant is used.
> The privacy evidence here is **empirical** — membership-inference advantage
> and gradient-reconstruction (DLG) failure — not a formal guarantee. `opacus`
> appears in `requirements.txt`, but the canonical secure trainer applies the
> Layer-3 noise directly and does not use it for accounting.

## Key results

Mean ± std over 5 seeds {0, 45, 123, 654, 789}.

| Configuration | Forward | Backward | Notes |
|---|---|---|---|
| C3 — Centralized (NoXEdge) | ~100% | 90.82 ± 0.15 | Forward ≈100% is a **structural** effect of disconnected sub-graphs, not generalization |
| F2 — Federated baseline (R100×E1, non-IID) | 88.14 ± 0.04 | 90.65 ± 0.11 | No defenses |
| FAT-base — AT + Byzantine, no DP | 88.44 | 90.93 | |
| **FAT-Def ε=16 — full stack** | **88.56 ± 0.07** | **90.98 ± 0.09** | Headline config |

Server-side DP adds noise once per round, versus client-side DP-SGD which
injects it K·E·T times; the full defended stack stays within < 0.5 pp of the
undefended federated baseline. Membership-inference advantage remains below
2×10⁻³ across defended configurations.

## Datasets

- **Elliptic Bitcoin dataset** (required): 203,769 transaction nodes,
  234,355 edges, 49 temporal snapshots, 165 anonymized features used.
  Download from Kaggle (`ellipticco/elliptic-data-set`) into
  `data/raw/elliptic_bitcoin_dataset/`.
- **BABD-13** (optional, recalibration only): used once to calibrate the
  20 proxy-feature distributions (Xiang et al., IEEE TIFS 2024). A precomputed
  `config/calibration_params.json` is provided, so BABD-13 is not required.

### Feature space (191 total)

| Group | Count | Source |
|---|---|---|
| Elliptic original | 165 | `feat_1 … feat_165` |
| Merchant category | 6 | `cat_*` |
| Calibration proxy | 20 | BABD-13 + literature |

### Merchant categories (500 simulated entities)

| ID | Category | Count |
|---|---|---|
| 0 | NO_MERCHANT | — |
| 1 | E-commerce | 175 |
| 2 | Gambling | 125 |
| 3 | Services | 100 |
| 4 | Retail | 75 |
| 5 | Luxury | 25 |

## Pipeline

1. **Data preparation** — `p1_data_preparation.py` is run **locally** and is
   the **single source of truth** for the train/val/test split (`split_map.pkl`)
   and feature normalization (`normalization_stats.pkl`). Its outputs are
   transferred to the cluster. All training/evaluation scripts read these
   artifacts **read-only** and raise `FileNotFoundError` if absent; no script
   generates its own split or local normalization.
2. **Centralized baselines** — C3 (NoXEdge), CAT.
3. **Federated training** — F2 (baseline FL) and FAT (full defense stack).
4. **Defense interaction & ε sweep** — per-defense / pairwise interaction
   matrix, plus the ε ∈ {1, 2, 4, 8, 16, 32} sweep.
5. **Security evaluation** — MIA, DLG, evasion/poisoning attack suite.

After preparation, `data/` on the cluster has the following layout (generated,
gitignored):

```
data/
├── split_map.pkl                # canonical split (source of truth)
├── normalization_stats.pkl      # canonical normalization (source of truth)
├── backward_weights_global.pt   # class weights for the backward (focal) loss
├── federated/
│   └── exchange_{0,1,2}/{data.pkl, edges.pkl}
└── federated_enriched/
    └── exchange_{0,1,2}_enriched.pkl
```

## Installation

CPU-only SLURM cluster (AMD EPYC). PyTorch Geometric, CPU build.

```bash
python -m venv venvs/fedbtc
source venvs/fedbtc/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements.txt
```

`flwr[simulation]` is listed but optional: the secure trainer falls back to a
manual FedAvg/aggregation path if Flower is absent.

## Reproducing the paper

Each config runs across the 5 seeds via SLURM array jobs; results are
aggregated to mean ± std with `analyze_results.py`.

```bash
# 1. canonical split + normalization (LOCAL, once), then transfer data/ to the cluster
python p1_data_preparation.py

# 2. baselines and federated configs (per-seed SLURM arrays)
sbatch --array=0-4 slurm/retrain_centralized.slurm   # C3
sbatch --array=0-4 slurm/run_cat.slurm               # CAT
sbatch --array=0-4 slurm/retrain_fl_standard.slurm   # F2
sbatch --array=0-4 slurm/retrain_fat.slurm           # FAT

# 3. defense interaction matrix + epsilon sweep
sbatch slurm/run_interaction_matrix.slurm
sbatch slurm/run_c5_sweep.slurm                      # ε ∈ {1,2,4,8,16,32}

# 4. security evaluation
sbatch slurm/run_security_eval_fat.slurm

# 5. aggregate + figures/tables
python evaluation/analyze_results.py
```

SLURM conventions: `--cpus-per-task=4`, no explicit `--mem`; hyphenated mode
names (`fat-base`, not `fat_base`); `--epsilon` (not `--dp_epsilon`); the
`--enable_at` flag must be passed for CAT/FAT configs.

## Configuration reference (verified defaults)

| Parameter | Value | Source |
|---|---|---|
| GCN layers | 2 | `DualAttributionModel` |
| Hidden dim | 128 | FAT l.1486 |
| Dropout | 0.3 | FAT l.1487 |
| Learning rate | 0.002 | FAT l.1491 |
| α_forward / α_backward | 1.0 / 1.5 | FAT l.1493-94 |
| Focal γ | 2.0 | FAT l.832 |
| Clients K | 3 | partitioning |
| Rounds T / local epochs E | 100 / 1 (canonical; script default 20/3) | runs |
| PGD ε_adv / α / steps / ratio | 0.2 / 0.05 / 7 / 0.30 | FAT l.746, 703 |
| Byzantine τ | 2.0 | FAT l.335 |
| DP σ(ε) | {8→0.1, 16→0.025, 32→0.0125} | FAT l.996-1001 |
| Seeds | {0, 45, 123, 654, 789} | runs |

## License

Academic research use only. Author/contact details withheld for anonymous review.
