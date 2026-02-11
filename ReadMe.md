# FedBTC: Privacy-Preserving Bitcoin Transaction Attribution via Federated Learning

> **USENIX Security 2026 Submission**
>
> A federated learning framework enabling cryptocurrency exchanges to collaboratively detect illicit transactions and attribute them to destination exchanges (forward) and merchant categories (backward) — without sharing sensitive KYC data.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Datasets](#datasets)
7. [Pipeline Overview](#pipeline-overview)
8. [Quick Start](#quick-start)
9. [Detailed Usage](#detailed-usage)
10. [Experiments Framework](#experiments-framework)
11. [Reproducing Paper Results](#reproducing-paper-results)
12. [Configuration Reference](#configuration-reference)
13. [Troubleshooting](#troubleshooting)
14. [Citation](#citation)
15. [License](#license)

---

## Overview

Cryptocurrency exchanges hold fragmented views of the Bitcoin transaction graph. Due to GDPR and privacy regulations, they cannot share customer KYC data to trace illicit flows across platforms. **FedBTC** solves this with a federated learning approach where K=3 simulated exchanges collaboratively train a Graph Neural Network (GNN) model while keeping their data local.

The system performs **dual attribution**:

- **Forward attribution**: Given a suspicious transaction, predict which exchange will ultimately receive the funds (3-class classification among Exchange 0, 1, 2).
- **Backward attribution**: Identify the category of criminal merchant that originated the transaction (6-class: No Merchant, E-commerce, Gambling, Services, Retail, Luxury).

The security stack includes server-side differential privacy (ε=8.0, δ=10⁻⁵), FLAME Byzantine defense, and PGD adversarial training.

---

## Key Results

| Configuration | Forward Acc. | Backward Acc. | Privacy |
|---------------|-------------|---------------|---------|
| Centralized Baseline | 71.00% | 76.91% | None |
| FL (No Security) | 85.80% | 88.66% | Data isolation only |
| FL + Adversarial + FLAME | 85.10% | 88.68% | Byzantine defense |
| **FL Secure (Full)** | **86.31%** | **87.96%** | **ε=8.0** |

FL outperforms the centralized baseline by +15.31% (forward) and +11.05% (backward). This counterintuitive result stems from the 60/20/20 label distribution across clients acting as implicit data augmentation — each client sees diverse cross-exchange labels during training, improving generalization.

---

## Repository Structure

```
FedBTC/
├── p1_data_preparation.py              # Phase 1 orchestrator (10 preprocessing steps)
├── p2_train_baseline.py                # Phase 2: centralized baseline training
├── p3_train_federated.py               # Phase 3: federated learning (simple FedAvg)
├── p3_train_federated_secure.py        # Phase 3: secure FL (DP + FLAME + Adversarial)
├── p5_security_evaluation.py           # Phase 5: security evaluation suite
│
├── config/
│   ├── calibration_params.json         # BABD-13 calibrated exchange/merchant profiles
│   └── calibration_report.md           # Calibration documentation & sources
│
├── scripts/
│   ├── preprocessing/
│   │   ├── calibrate_from_babd13.py        # Step 0: BABD-13 distribution extraction
│   │   ├── preprocess_elliptic.py          # Step 1: Elliptic dataset preprocessing
│   │   ├── build_temporal_graph.py         # Step 2: temporal graph construction
│   │   ├── simulate_merchants.py           # Step 3: merchant simulation (500 entities)
│   │   ├── expand_merchants_khop.py        # Step 4: BFS k-hop expansion
│   │   ├── precompute_merchant_embeddings.py  # Step 5: GNN embedding cache
│   │   ├── create_criminal_db.py           # Step 6: criminal entity database
│   │   ├── split_merchants_known_unknown.py   # Step 7: 90/10 known/unknown split
│   │   ├── partition_federated.py          # Step 8: K=3 exchange partitioning
│   │   └── add_hybrid_features_elliptic.py # Step 9: calibrated proxy features (20 feat.)
│   ├── training/
│   │   └── train_attribution_baseline.py   # Core baseline training logic
│   ├── evaluation/
│   │   └── evaluate_fl_vs_baseline.py      # FL vs baseline comparison
│   ├── investigation/
│   │   └── investigate_transaction.py      # Transaction investigation workflow
│   └── utils/
│       ├── inspect_babd13.py               # BABD-13 dataset inspector
│       └── extract_babd13_stats_local.py   # Local stats extraction utility
│
├── experiments/
│   ├── config.py                       # Shared experiment configuration
│   ├── run_all_experiments.py          # Experiment orchestrator (~3h total)
│   ├── generate_latex_tables.py        # LaTeX table generation from results
│   └── scripts/
│       └── p3_train_federated_flexible.py  # Parameterized FL training script
│
├── data/
│   ├── raw/
│   │   └── elliptic_bitcoin_dataset/   # Elliptic dataset (download required)
│   ├── external/
│   │   └── babd13/                     # BABD-13 extracted data (optional)
│   ├── processed/                      # Intermediate preprocessed data
│   └── federated_enriched/             # Final partitioned data (K=3 exchanges)
│       ├── exchange_0_enriched.pkl
│       ├── exchange_1_enriched.pkl
│       └── exchange_2_enriched.pkl
│
├── results/
│   ├── models/                         # Saved model checkpoints
│   │   ├── baseline_dual_attribution.pt
│   │   └── federated_dual_attribution.pt
│   ├── evaluations/                    # JSON evaluation reports
│   └── experiments/                    # Experiment results & LaTeX tables
│
├── requirements.txt
└── README.md
```

---

## Prerequisites

**Hardware (minimum)**:
- CPU: 4+ cores (8 recommended)
- RAM: 16 GB minimum (32 GB recommended)
- Storage: 10 GB free (datasets + models + checkpoints)
- GPU: Optional but recommended (NVIDIA with CUDA support)

**Software**:
- Python 3.10 or 3.11
- Conda or venv for environment management
- Git
- Kaggle account (for Elliptic dataset download)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rodNTprojects/FedBTC.git
cd FedBTC
```

### 2. Create and activate the environment

**Option A — Conda (recommended)**:

```bash
conda create -n fedbtc python=3.10
conda activate fedbtc
```

**Option B — venv**:

```bash
python -m venv fedbtc-env
# Linux/macOS
source fedbtc-env/bin/activate
# Windows PowerShell
.\fedbtc-env\Scripts\Activate.ps1
```

### 3. Install PyTorch

Choose the appropriate command based on your hardware.

**GPU (CUDA 11.8)**:

```bash
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU only**:

```bash
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install PyTorch Geometric

```bash
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

Replace `+cpu` with `+cu118` if using a CUDA GPU.

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not yet generated, install manually:

```bash
pip install flwr==1.5.0          # Federated Learning (Flower)
pip install opacus==1.4.0        # Differential Privacy
pip install pandas numpy scikit-learn networkx
pip install matplotlib seaborn
pip install tqdm jupyterlab
```

### 6. Verify installation

```bash
python -c "
import torch
import torch_geometric
print(f'PyTorch:  {torch.__version__}')
print(f'PyG:      {torch_geometric.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
print('All OK')
"
```

### 7. CPU optimization (optional, for CPU-only setups)

Add to your scripts or shell profile:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

Or in Python:

```python
import torch, os
torch.set_num_threads(8)
os.environ['OMP_NUM_THREADS'] = '8'
```

---

## Datasets

### Elliptic Bitcoin Dataset (required)

The Elliptic dataset contains 203,769 Bitcoin transactions with 166 anonymized features across 49 timesteps.

**Download via Kaggle API**:

```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/ (get it from https://www.kaggle.com/settings)
kaggle datasets download -d ellipticco/elliptic-data-set
unzip elliptic-data-set.zip -d data/raw/elliptic_bitcoin_dataset/
```

**Manual download**: Visit https://www.kaggle.com/datasets/ellipticco/elliptic-data-set, download, and extract into `data/raw/elliptic_bitcoin_dataset/`.

After extraction you should have:

```
data/raw/elliptic_bitcoin_dataset/
├── elliptic_txs_features.csv      # 203,769 × 167 (txId + 166 features)
├── elliptic_txs_classes.csv       # Labels (illicit / licit / unknown)
└── elliptic_txs_edgelist.csv      # 234,355 directed edges
```

### BABD-13 Dataset (optional, for recalibration)

BABD-13 (544,462 labeled Bitcoin addresses across 13 entity types) is used to calibrate the proxy feature distributions. A pre-computed `config/calibration_params.json` is included in the repository, so downloading BABD-13 is only needed if you wish to regenerate calibration from scratch.

**If you want to recalibrate**:

1. Download `babd.zip` (~5.3 GB) from the original source (Xiang et al., IEEE TIFS 2024).
2. Place it at the project root: `FedBTC/babd.zip`.
3. Run the calibration:

```bash
python scripts/utils/inspect_babd13.py          # Inspect contents
python scripts/preprocessing/calibrate_from_babd13.py  # Generate calibration_params.json
```

---

## Pipeline Overview

The project follows a sequential pipeline. Each phase depends on the outputs of the previous one.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FedBTC PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PHASE 1: Data Preparation (p1_data_preparation.py)                  │
│  ├── Step 0: BABD-13 calibration → calibration_params.json          │
│  ├── Step 1: Preprocess Elliptic (203K tx, 166 features)            │
│  ├── Step 2: Build temporal graph (49 timesteps)                    │
│  ├── Step 3: Simulate 500 merchants (5 categories)                  │
│  ├── Step 4: BFS k-hop expansion from merchant seeds               │
│  ├── Step 5: Precompute GNN merchant embeddings                    │
│  ├── Step 6: Create criminal entity database                       │
│  ├── Step 7: Split merchants 90% known / 10% unknown               │
│  ├── Step 8: Partition data for K=3 federated exchanges             │
│  └── Step 9: Add 20 calibrated proxy features                      │
│      OUTPUT → data/federated_enriched/exchange_{0,1,2}_enriched.pkl │
│                                                                      │
│  PHASE 2: Baseline Training (p2_train_baseline.py)                   │
│  ├── Train centralized dual-attribution GNN                         │
│  └── OUTPUT → results/models/baseline_dual_attribution.pt           │
│                                                                      │
│  PHASE 3: Federated Learning                                         │
│  ├── Option A: p3_train_federated.py (simple FedAvg)                │
│  └── Option B: p3_train_federated_secure.py (DP+FLAME+Adversarial) │
│      OUTPUT → results/models/federated_dual_attribution.pt          │
│                                                                      │
│  PHASE 5: Security Evaluation (p5_security_evaluation.py)            │
│  ├── Membership Inference Attack (MIA)                              │
│  ├── Byzantine Attack Simulation (FLAME)                            │
│  ├── Adversarial Robustness (PGD)                                   │
│  └── Confidence Calibration                                         │
│      OUTPUT → results/evaluations/security_evaluation_report.json   │
│                                                                      │
│  EXPERIMENTS: Ablation studies & LaTeX table generation               │
│  └── experiments/run_all_experiments.py                              │
│      OUTPUT → results/experiments/evaluation_tables.tex             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Feature Engineering (20 Proxy Features)

Since Elliptic's 166 features are anonymized and lack exchange-discriminating power, we add 20 scientifically calibrated proxy features:

| Category | Features (count) | Calibration Source |
|----------|------------------|--------------------|
| Fee structure | fee_percentage, fee_tier, ... (5) | BABD-13 + CoinMarketCap 2024 |
| Volume patterns | volume_scale, volume_class, ... (5) | BABD-13 + Zhou 2023 |
| Temporal/hour | synthetic_hour, timezone_proxy, ... (6) | Juhász et al. 2018 |
| Liquidity | liquidity_score, processing_speed, ... (4) | BABD-13 + Kaiko Benchmark |

Total feature dimension: **186** (166 original + 20 proxy).

### Model Architecture

```
Input (186 features)
    │
    ├── GCN Layer 1 (186 → 128) + BatchNorm + ReLU + Dropout(0.3)
    ├── GCN Layer 2 (128 → 128) + BatchNorm + ReLU + Dropout(0.3)
    │
    ├── Forward Head: Linear(128→64) → BN → ReLU → Linear(64→3)   [Exchange classification]
    └── Backward Head: Linear(128→64) → BN → ReLU → Linear(64→6)  [Merchant category]

Loss = α_fwd × CrossEntropy_fwd  +  α_bwd × FocalLoss_bwd
       (α_fwd=1.0, α_bwd=1.5)
```

### Security Stack

| Layer | Technique | Purpose | Reference |
|-------|-----------|---------|-----------|
| Privacy | Server-side DP (ε=8.0, σ=0.1) | Protect membership | McMahan et al. 2018 |
| Byzantine | FLAME (DBSCAN clustering) | Detect malicious updates | Nguyen et al. 2022 |
| Robustness | PGD adversarial training (15%) | Evasion resistance | Madry et al. 2018 |

Server-side DP applies noise once per round (T=20 total) instead of client-side DP-SGD which applies noise K×E×T=180 times, preserving model utility (<1% accuracy degradation vs >40% for client-side).

---

## Quick Start

Once installation and dataset download are complete:

```bash
# Phase 1: Prepare data (~20-30 min)
python p1_data_preparation.py

# Phase 2: Train centralized baseline (~30-60 min)
python p2_train_baseline.py

# Phase 3: Train secure federated model (~45-90 min)
python p3_train_federated_secure.py

# Phase 5: Run security evaluations (~15 min)
python p5_security_evaluation.py --attack all
```

Results will be saved to `results/evaluations/` and `results/models/`.

---

## Detailed Usage

### Phase 1: Data Preparation

```bash
python p1_data_preparation.py
```

This orchestrator runs Steps 0–9 sequentially. If the BABD-13 zip is not found, Step 0 uses the pre-computed `config/calibration_params.json`.

**Individual step execution** (useful for debugging):

```bash
# Run steps individually
python scripts/preprocessing/calibrate_from_babd13.py
python scripts/preprocessing/preprocess_elliptic.py
python scripts/preprocessing/build_temporal_graph.py
python scripts/preprocessing/simulate_merchants.py
python scripts/preprocessing/expand_merchants_khop.py
python scripts/preprocessing/precompute_merchant_embeddings.py
python scripts/preprocessing/create_criminal_db.py
python scripts/preprocessing/split_merchants_known_unknown.py
python scripts/preprocessing/partition_federated.py
python scripts/preprocessing/add_hybrid_features_elliptic.py
```

**Outputs**:
- `data/federated_enriched/exchange_0_enriched.pkl`
- `data/federated_enriched/exchange_1_enriched.pkl`
- `data/federated_enriched/exchange_2_enriched.pkl`
- `config/calibration_params.json`

### Phase 2: Baseline Training

```bash
python p2_train_baseline.py
```

Trains a centralized dual-attribution GNN on all data (no federation, no privacy). Serves as the upper-bound reference for comparison.

Options:

```bash
python p2_train_baseline.py --epochs 100 --lr 0.002 --hidden_dim 128
```

**Output**: `results/models/baseline_dual_attribution.pt`

### Phase 3: Federated Learning

**Option A — Simple FedAvg** (no security, for comparison):

```bash
python p3_train_federated.py
```

**Option B — Secure FL** (recommended, full security stack):

```bash
python p3_train_federated_secure.py
```

To customize security parameters:

```bash
# Custom privacy budget
python p3_train_federated_secure.py --epsilon 4.0

# Disable specific defenses
python p3_train_federated_secure.py --no-server-dp
python p3_train_federated_secure.py --no-flame
python p3_train_federated_secure.py --no-adversarial
```

**Output**: `results/models/federated_dual_attribution.pt`

### Phase 5: Security Evaluation

```bash
# Run all evaluations
python p5_security_evaluation.py --attack all

# Individual evaluations
python p5_security_evaluation.py --attack mia          # Membership Inference
python p5_security_evaluation.py --attack byzantine    # Byzantine attack simulation
python p5_security_evaluation.py --attack adversarial  # Adversarial robustness
python p5_security_evaluation.py --attack confidence   # Confidence calibration
```

**Output**: `results/evaluations/security_evaluation_report.json`

---

## Experiments Framework

The `experiments/` directory contains a fully automated experiment runner and LaTeX table generator for the paper.

### Run all experiments (~3 hours)

```bash
cd experiments/

# Full run
python run_all_experiments.py

# Dry run (preview what will execute)
python run_all_experiments.py --dry-run

# List available experiments
python run_all_experiments.py --list
```

### Run specific experiments

```bash
# Main comparison only (Tables 1 & 2)
python run_all_experiments.py --only main_comparison

# Privacy trade-off study (Table 3)
python run_all_experiments.py --only privacy_tradeoff

# Multiple experiments
python run_all_experiments.py --only main_comparison ablation_layers

# Skip a specific experiment
python run_all_experiments.py --skip byzantine_attack
```

### Manual single runs with flexible parameters

```bash
# Custom epsilon
python scripts/p3_train_federated_flexible.py --epsilon 4.0 --experiment_name privacy_eps_4

# Custom GCN layers
python scripts/p3_train_federated_flexible.py --num_layers 2 --experiment_name ablation_layers_2

# Custom label distribution
python scripts/p3_train_federated_flexible.py --label_dist 80-10-10 --experiment_name ablation_dist_80

# FL without differential privacy
python scripts/p3_train_federated_flexible.py --no-server-dp --experiment_name fl_nosec
```

### Generate LaTeX tables

```bash
# Generate all tables from existing results
python generate_latex_tables.py

# Output to specific file
python generate_latex_tables.py --output my_tables.tex

# Generate only specific tables
python generate_latex_tables.py --tables 1 2 3
```


### Flexible FL script parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_layers` | 3 | Number of GCN layers (1–5) |
| `--label_dist` | 60-20-20 | Own-other1-other2 label distribution |
| `--epsilon` | 8.0 | Differential privacy epsilon |
| `--local_epochs` | 3 | Local epochs per FL round |
| `--rounds` | 20 | Number of FL communication rounds |
| `--no-server-dp` | — | Disable server-side DP |
| `--no-flame` | — | Disable FLAME Byzantine defense |
| `--no-adversarial` | — | Disable adversarial training |
| `--experiment_name` | — | Custom output filename |
| `--split_seed` | 42 | Random seed (keep constant for fair comparison) |

---

## Reproducing Paper Results

To reproduce all results from the USENIX Security submission:

```bash
# 1. Prepare data (run once)
python p1_data_preparation.py

# 2. Train baseline
python p2_train_baseline.py

# 3. Train secure FL
python p3_train_federated_secure.py

# 4. Security evaluations
python p5_security_evaluation.py --attack all

# 5. Run all ablation experiments
cd experiments/
python run_all_experiments.py

# 6. Generate LaTeX tables for the paper
python generate_latex_tables.py --output results/experiments/evaluation_tables.tex
```

**Expected outputs**:

```
results/
├── models/
│   ├── baseline_dual_attribution.pt
│   └── federated_dual_attribution.pt
├── evaluations/
│   ├── baseline_results.json
│   ├── fl_secure_results.json
│   └── security_evaluation_report.json
└── experiments/
    ├── main_comparison_*.json
    ├── privacy_tradeoff_*.json
    ├── ablation_layers_*.json
    ├── ablation_epochs_*.json
    ├── byzantine_attack_*.json
    └── evaluation_tables.tex
```

**Expected performance** (with seed 42):

| Metric | Baseline | FL Secure |
|--------|----------|-----------|
| Forward Accuracy | ~71% | ~86% |
| Backward Accuracy | ~77% | ~88% |
| Privacy ε | ∞ | 8.0 |
| MIA AUC | — | ≤0.55 |

---

## Configuration Reference

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | (auto) | CPU thread count |
| `MKL_NUM_THREADS` | (auto) | MKL thread count |
| `CUDA_VISIBLE_DEVICES` | (auto) | GPU selection |

### Key hyperparameters (hardcoded defaults)

| Parameter | Value | Location |
|-----------|-------|----------|
| GCN hidden dim | 128 | p2/p3 scripts |
| GCN layers | 2 (optimal) | p3 scripts |
| Learning rate | 0.002 | p2/p3 scripts |
| FL rounds (T) | 20 | p3 scripts |
| FL clients (K) | 3 | p1 partitioning |
| Local epochs (E) | 3 | p3 scripts |
| DP epsilon | 8.0 | p3_secure |
| DP delta | 10⁻⁵ | p3_secure |
| DP noise σ | 0.1 | p3_secure |
| FLAME ε_cluster | 0.5 | p3_secure |
| PGD adv. ratio | 15% | p3_secure |
| Batch size | 32 | All training scripts |
| Label distribution | 60-20-20 | p1 partitioning |
| Merchant count | 500 | Step 3 |
| Known/unknown split | 90/10 | Step 7 |

---

## Troubleshooting

### "No data found" or missing `.pkl` files

Phase 2 and 3 require the enriched data from Phase 1. Run `p1_data_preparation.py` first and verify that `data/federated_enriched/exchange_*_enriched.pkl` files exist.

### Out of memory

Reduce batch size or hidden dimension:

```bash
python p2_train_baseline.py --batch_size 16 --hidden_dim 64
```

For CPU-only systems, also reduce the number of data loader workers.

### PyTorch Geometric version mismatch

Ensure `torch-scatter` and `torch-sparse` match your PyTorch + CUDA version:

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### Flower (flwr) import errors

If Flower is unavailable, `p3_train_federated.py` falls back to a manual FedAvg implementation automatically. The secure version (`p3_train_federated_secure.py`) does not require Flower.

### Comparison across experiments is invalid

Always use the same `--split_seed 42` across experiments to ensure identical train/val/test splits. This is the default.

### BABD-13 calibration fails

If `babd.zip` is too large or unavailable, the pipeline uses `config/calibration_params.json` (literature-based calibration) by default. This is the recommended approach and produces equivalent results.

---


### Key references

- Weber et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with GCN for Financial Forensics.* KDD Workshop.
- Xiang et al. (2024). *BABD-13: A Bitcoin Address Behavior Dataset with 13 Categories.* IEEE TIFS.
- Nguyen et al. (2022). *FLAME: Taming Backdoors in Federated Learning.* USENIX Security.
- McMahan et al. (2018). *Learning Differentially Private Recurrent Language Models.* ICLR.
- Madry et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR.
- Juhász et al. (2018). *Bitcoin Transaction Activity Analysis via Circadian Rhythms.* Royal Society Open Science.
- Zhou et al. (2023). *Volume Distribution Patterns in Cryptocurrency Exchanges.* IEEE Access.
- Meiklejohn et al. (2013). *A Fistful of Bitcoins.* ACM IMC.
- Kairouz & McMahan (2021). *Advances and Open Problems in Federated Learning.* FnTML.
- CoinMarketCap (2024). Exchange fee structure data. https://coinmarketcap.com
- Kaiko (2024). Exchange Benchmark Report. https://www.kaiko.com

---

## License


This project is developed for academic research purposes. Please contact the authors before any commercial use.


