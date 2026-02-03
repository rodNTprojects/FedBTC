"""
================================================================================
Script: partition_federated.py (V2 - REALISTIC DISTRIBUTION)
Description: Partition Elliptic dataset into K=3 exchanges with REALISTIC
             forward_label distribution to enable Federated Learning

PROBLÈME RÉSOLU:
----------------
Ancienne version: Chaque exchange ne voyait que forward_label = self (100% non-IID)
  → FL Forward échouait (33% = random)

Nouvelle version: Distribution réaliste
  → Chaque exchange voit ~60% de ses propres transactions + ~40% des autres
  → FL Forward peut apprendre (target: 65-70%)

JUSTIFICATION RÉALISTE:
-----------------------
En réalité, un exchange voit:
1. Transactions ENTRANTES vers lui (dépôts) → forward_label = self
2. Transactions de ses clients SORTANTES (retraits vers autres exchanges)
   → forward_label = autre_exchange
3. L'exchange connaît ces transactions car ce sont SES clients qui envoient

DISTRIBUTION CIBLE:
-------------------
Exchange 0: forward_label = {0: 60%, 1: 20%, 2: 20%}
Exchange 1: forward_label = {0: 20%, 1: 60%, 2: 20%}
Exchange 2: forward_label = {0: 20%, 1: 20%, 2: 60%}

INPUT: data/processed/data_with_merchants.pkl (ou data_preprocessed.pkl)
OUTPUT: data/federated/exchange_X/data.pkl, edges.pkl

USAGE:
    python partition_federated.py
    python partition_federated.py --self_ratio 0.7  # 70% self, 30% others
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from typing import Dict, List, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================

K_EXCHANGES = 3  # Nombre d'exchanges simulés

# Distribution réaliste des forward_labels par exchange
# self_ratio = proportion de transactions avec forward_label = exchange_id
DEFAULT_SELF_RATIO = 0.6  # 60% self, 40% others (20% each)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def print_distribution(name: str, labels: np.ndarray, num_classes: int = 3):
    """Print label distribution"""
    counts = Counter(labels)
    total = len(labels)
    print(f"\n  {name} forward_label distribution:")
    for i in range(num_classes):
        count = counts.get(i, 0)
        pct = 100 * count / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    Label {i}: {count:6,} ({pct:5.1f}%) {bar}")


# ============================================================================
# MAIN PARTITIONING LOGIC
# ============================================================================

def assign_forward_labels_realistic(
    data: pd.DataFrame,
    exchange_partition: np.ndarray,
    self_ratio: float = 0.6,
    seed: int = 42
) -> np.ndarray:
    """
    Assign forward_labels with realistic distribution.
    
    Each exchange sees:
    - self_ratio% of transactions with forward_label = exchange_id (deposits TO this exchange)
    - (1-self_ratio)% of transactions with forward_label = other exchanges (client withdrawals)
    
    Args:
        data: DataFrame with transactions
        exchange_partition: Array indicating which exchange "owns" each transaction
        self_ratio: Proportion of self-destined transactions (default 0.6 = 60%)
        seed: Random seed
    
    Returns:
        forward_labels: Array of forward labels (0, 1, or 2)
    """
    np.random.seed(seed)
    n = len(data)
    forward_labels = np.zeros(n, dtype=int)
    
    other_ratio = (1 - self_ratio) / (K_EXCHANGES - 1)  # Split remaining among others
    
    print(f"\n  Distribution cible par exchange:")
    print(f"    Self (dépôts entrants):     {self_ratio*100:.0f}%")
    print(f"    Autres (retraits sortants): {(1-self_ratio)*100:.0f}% ({other_ratio*100:.0f}% chacun)")
    
    for exchange_id in range(K_EXCHANGES):
        # Indices des transactions de cet exchange
        exchange_mask = exchange_partition == exchange_id
        exchange_indices = np.where(exchange_mask)[0]
        n_exchange = len(exchange_indices)
        
        if n_exchange == 0:
            continue
        
        # Shuffle pour distribution aléatoire
        np.random.shuffle(exchange_indices)
        
        # Calculer les splits
        n_self = int(n_exchange * self_ratio)
        n_per_other = (n_exchange - n_self) // (K_EXCHANGES - 1)
        
        # Assigner forward_label = self pour les premiers n_self
        forward_labels[exchange_indices[:n_self]] = exchange_id
        
        # Assigner forward_label = autres pour le reste
        current_idx = n_self
        for other_id in range(K_EXCHANGES):
            if other_id == exchange_id:
                continue
            
            end_idx = min(current_idx + n_per_other, n_exchange)
            forward_labels[exchange_indices[current_idx:end_idx]] = other_id
            current_idx = end_idx
        
        # Les transactions restantes vont vers un exchange aléatoire (équilibrage)
        if current_idx < n_exchange:
            others = [i for i in range(K_EXCHANGES) if i != exchange_id]
            for idx in exchange_indices[current_idx:]:
                forward_labels[idx] = np.random.choice(others)
    
    return forward_labels


def partition_data_stratified(
    data: pd.DataFrame,
    k: int = 3,
    seed: int = 42
) -> np.ndarray:
    """
    Partition data into K exchanges using stratified sampling.
    
    Preserves:
    - Class distribution (illicit/licit)
    - Temporal distribution (time_steps)
    
    Returns:
        exchange_partition: Array indicating exchange ownership (0, 1, or 2)
    """
    # Create stratification label
    if 'class' in data.columns:
        class_col = data['class'].astype(str)
    else:
        class_col = "0"
    
    if 'time_step' in data.columns:
        time_col = (data['time_step'] // 10).astype(str)  # Bin into groups
    else:
        time_col = "0"
    
    strata = class_col + "_" + time_col
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    
    exchange_partition = np.zeros(len(data), dtype=int)
    
    for exchange_id, (_, partition_idx) in enumerate(skf.split(data, strata)):
        exchange_partition[partition_idx] = exchange_id
    
    return exchange_partition


def partition_edges(
    edges: pd.DataFrame,
    data: pd.DataFrame,
    exchange_partition: np.ndarray
) -> Dict[int, pd.DataFrame]:
    """
    Partition edges by exchange ownership.
    
    Règle: Une edge appartient à l'exchange qui possède txId1 (source).
    Les edges inter-exchange sont dupliquées dans les deux partitions.
    """
    # Map txId → exchange_id
    txid_to_exchange = dict(zip(data['txId'], exchange_partition))
    
    edges_by_exchange = {i: [] for i in range(K_EXCHANGES)}
    
    for _, row in edges.iterrows():
        tx1, tx2 = row['txId1'], row['txId2']
        
        exchange1 = txid_to_exchange.get(tx1, -1)
        exchange2 = txid_to_exchange.get(tx2, -1)
        
        # Ajouter l'edge à l'exchange source
        if exchange1 >= 0:
            edges_by_exchange[exchange1].append(row)
        
        # Si inter-exchange, ajouter aussi à la destination (bidirectionnel)
        if exchange2 >= 0 and exchange2 != exchange1:
            edges_by_exchange[exchange2].append(row)
    
    # Convert to DataFrames
    for i in range(K_EXCHANGES):
        if edges_by_exchange[i]:
            edges_by_exchange[i] = pd.DataFrame(edges_by_exchange[i]).drop_duplicates()
        else:
            edges_by_exchange[i] = pd.DataFrame(columns=['txId1', 'txId2'])
    
    return edges_by_exchange


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def partition_federated(
    input_path: str = None,
    output_dir: str = 'data/federated',
    self_ratio: float = DEFAULT_SELF_RATIO,
    seed: int = 42
):
    """
    Main partitioning function.
    
    Creates K=3 exchange partitions with realistic forward_label distribution.
    """
    print_header("FEDERATED PARTITIONING (V2 - REALISTIC)")
    print(f"  Self ratio: {self_ratio*100:.0f}% (transactions with forward_label = self)")
    print(f"  K exchanges: {K_EXCHANGES}")
    print(f"  Seed: {seed}")
    
    # ===== LOAD DATA =====
    print_header("LOADING DATA")
    
    # Try multiple possible input paths
    possible_paths = [
        input_path,
        'data/processed/data_with_merchants.pkl',
        'data/processed/data_preprocessed.pkl',
        'data/processed/elliptic_processed.pkl',
    ]
    
    data = None
    edges = None
    
    for path in possible_paths:
        if path and os.path.exists(path):
            data = pd.read_pickle(path)
            print(f"  ✓ Loaded data: {path}")
            print(f"    Shape: {data.shape}")
            break
    
    if data is None:
        raise FileNotFoundError(f"No data file found. Tried: {possible_paths}")
    
    # Load edges
    edge_paths = [
        'data/processed/edges_preprocessed.pkl',
        'data/processed/edges.pkl',
        'data/raw/elliptic_txs_edgelist.csv',
    ]
    
    for path in edge_paths:
        if os.path.exists(path):
            if path.endswith('.csv'):
                edges = pd.read_csv(path, names=['txId1', 'txId2'])
            else:
                edges = pd.read_pickle(path)
            print(f"  ✓ Loaded edges: {path}")
            print(f"    Shape: {edges.shape}")
            break
    
    if edges is None:
        print("  ⚠ No edges file found, creating empty edges")
        edges = pd.DataFrame(columns=['txId1', 'txId2'])
    
    # ===== STRATIFIED PARTITIONING =====
    print_header("STRATIFIED PARTITIONING")
    
    exchange_partition = partition_data_stratified(data, k=K_EXCHANGES, seed=seed)
    
    # Statistics
    for i in range(K_EXCHANGES):
        count = (exchange_partition == i).sum()
        print(f"  Exchange {i}: {count:,} transactions ({100*count/len(data):.1f}%)")
    
    # ===== ASSIGN FORWARD LABELS (REALISTIC) =====
    print_header("ASSIGNING FORWARD LABELS (REALISTIC DISTRIBUTION)")
    
    forward_labels = assign_forward_labels_realistic(
        data, 
        exchange_partition, 
        self_ratio=self_ratio, 
        seed=seed
    )
    
    # Add to data
    data['exchange_partition'] = exchange_partition
    data['forward_label'] = forward_labels
    
    # Verify distribution per exchange
    print("\n  Vérification de la distribution:")
    for i in range(K_EXCHANGES):
        mask = exchange_partition == i
        exchange_forward = forward_labels[mask]
        print_distribution(f"Exchange {i}", exchange_forward)
    
    # Global distribution
    print_distribution("GLOBAL", forward_labels)
    
    # ===== PARTITION EDGES =====
    print_header("PARTITIONING EDGES")
    
    edges_by_exchange = partition_edges(edges, data, exchange_partition)
    
    for i in range(K_EXCHANGES):
        print(f"  Exchange {i}: {len(edges_by_exchange[i]):,} edges")
    
    # ===== SAVE PARTITIONS =====
    print_header("SAVING PARTITIONS")
    
    for i in range(K_EXCHANGES):
        exchange_dir = f"{output_dir}/exchange_{i}"
        os.makedirs(exchange_dir, exist_ok=True)
        
        # Filter data for this exchange
        mask = exchange_partition == i
        exchange_data = data[mask].copy()
        exchange_edges = edges_by_exchange[i]
        
        # Save to data/federated/exchange_X/
        exchange_data.to_pickle(f"{exchange_dir}/data.pkl")
        exchange_edges.to_pickle(f"{exchange_dir}/edges.pkl")
        
        print(f"  ✓ Exchange {i}: {len(exchange_data):,} transactions, {len(exchange_edges):,} edges")
        print(f"    → {exchange_dir}/")
    
    # ===== SUMMARY =====
    print_header("PARTITIONING COMPLETE")
    
    print("\n  📊 FORWARD LABEL DISTRIBUTION (per exchange):")
    print("  " + "-" * 50)
    print(f"  {'Exchange':<10} {'Label 0':<12} {'Label 1':<12} {'Label 2':<12}")
    print("  " + "-" * 50)
    
    for i in range(K_EXCHANGES):
        mask = exchange_partition == i
        fwd = forward_labels[mask]
        counts = Counter(fwd)
        total = len(fwd)
        pcts = [100 * counts.get(j, 0) / total for j in range(K_EXCHANGES)]
        print(f"  {i:<10} {pcts[0]:>5.1f}%{' (self)' if i==0 else '':<6} "
              f"{pcts[1]:>5.1f}%{' (self)' if i==1 else '':<6} "
              f"{pcts[2]:>5.1f}%{' (self)' if i==2 else '':<6}")
    
    print("  " + "-" * 50)
    
    print(f"\n  ✅ Chaque exchange voit TOUTES les classes forward")
    print(f"  ✅ Distribution non-IID mais PAS extrême")
    print(f"  ✅ FL Forward Attribution devrait fonctionner")
    
    print(f"\n  Output: {output_dir}/")
    print(f"  Next: python scripts/preprocessing/add_hybrid_features_elliptic.py")
    
    return data, exchange_partition, forward_labels


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Partition Elliptic dataset for Federated Learning (V2 - Realistic)'
    )
    
    parser.add_argument('--input', type=str, default=None,
                        help='Input data path (auto-detect if not specified)')
    parser.add_argument('--output', type=str, default='data/federated',
                        help='Output directory')
    parser.add_argument('--self_ratio', type=float, default=DEFAULT_SELF_RATIO,
                        help=f'Ratio of self-destined transactions (default: {DEFAULT_SELF_RATIO})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    partition_federated(
        input_path=args.input,
        output_dir=args.output,
        self_ratio=args.self_ratio,
        seed=args.seed
    )