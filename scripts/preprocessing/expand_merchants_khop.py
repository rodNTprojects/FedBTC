"""
Script: expand_merchants_khop.py
Description: Expand merchant interactions via BFS k-hop neighborhood


BFS k-hop exploration to find merchants in transaction neighborhood.
This simulates the investigation workflow where we explore k-hop neighbors
to find merchant interactions.
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def merchant_id_to_category(merchant_id):
    """
    Convert merchant_id (1-500) to category_label (0-5)
    
    Categories:
        0 = NO_MERCHANT
        1 = E-commerce (merchants 1-175)
        2 = Gambling (merchants 176-300)
        3 = Services (merchants 301-400)
        4 = Retail (merchants 401-475)
        5 = Luxury (merchants 476-500)
    """
    if merchant_id == 0 or pd.isna(merchant_id):
        return 0  # NO_MERCHANT
    
    merchant_id = int(merchant_id)
    
    if 1 <= merchant_id <= 175:
        return 1  # E-commerce
    elif 176 <= merchant_id <= 300:
        return 2  # Gambling
    elif 301 <= merchant_id <= 400:
        return 3  # Services
    elif 401 <= merchant_id <= 475:
        return 4  # Retail
    else:  # 476-500
        return 5  # Luxury

def load_data():
    """Load data, graph, and merchants"""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load transaction data with merchants
    data = pd.read_pickle('data/processed/data_with_merchants.pkl')
    print(f"✓ Data: {data.shape}")
    
    # Load graph
    with open('data/processed/graph.pkl', 'rb') as f:
        G = pickle.load(f)
    print(f"✓ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Load txid mapping
    with open('data/processed/txid_to_idx.pkl', 'rb') as f:
        txid_to_idx = pickle.load(f)
    print(f"✓ TxID mapping: {len(txid_to_idx)} transactions")
    
    return data, G, txid_to_idx


def bfs_khop_exploration(txid, G, txid_to_idx, k_values=[3, 7, 10]):
    """
    BFS k-hop exploration from transaction
    
    Args:
        txid: Transaction ID
        G: NetworkX graph
        txid_to_idx: Transaction ID to index mapping
        k_values: List of k-hop distances to explore
    
    Returns:
        neighbors_by_k: Dict {k: [neighbor_txids]}
    """
    # Get node index
    if txid not in txid_to_idx:
        return {}
    
    idx = txid_to_idx[txid]
    
    if idx not in G:
        return {}
    
    # BFS from idx
    try:
        # Compute shortest path lengths to all reachable nodes
        path_lengths = nx.single_source_shortest_path_length(
            G, idx, cutoff=max(k_values)
        )
    except:
        return {}
    
    # Group by distance
    neighbors_by_k = {k: [] for k in k_values}
    
    # Reverse mapping (idx to txid)
    idx_to_txid = {v: k for k, v in txid_to_idx.items()}
    
    for neighbor_idx, distance in path_lengths.items():
        if neighbor_idx == idx:
            continue  # Skip self
        
        # Check if distance matches any k
        for k in k_values:
            if distance <= k:
                neighbor_txid = idx_to_txid.get(neighbor_idx)
                if neighbor_txid:
                    neighbors_by_k[k].append((neighbor_txid, distance))
    
    return neighbors_by_k


def expand_merchants_via_khop(data, G, txid_to_idx, k_values=[3, 7, 10], sample_size=5000):
    """
    Expand merchant interactions via k-hop exploration
    
    For transactions WITHOUT direct merchant:
    - Explore k-hop neighborhood
    - Check if any neighbor has merchant
    - Assign merchant with distance
    
    Args:
        data: Transaction data
        G: Graph
        txid_to_idx: Mapping
        k_values: k-hop distances to explore
        sample_size: Number of transactions to process (for performance)
    
    Returns:
        data: Enhanced with k-hop merchant assignments
    """
    print("\n" + "="*60)
    print(f"EXPANDING MERCHANTS VIA K-HOP (k={k_values})")
    print("="*60)
    
    # Transactions without direct merchant
    no_merchant_txs = data[data['merchant_distance'] == -1].copy()
    
    print(f"Transactions without merchant: {len(no_merchant_txs):,}")
    print(f"Processing sample: {min(sample_size, len(no_merchant_txs)):,}")
    
    # Sample for performance
    if len(no_merchant_txs) > sample_size:
        no_merchant_txs = no_merchant_txs.sample(n=sample_size, random_state=42)
    
    # Track assignments
    khop_assignments = 0
    distance_counts = {k: 0 for k in k_values}
    
    for txid in tqdm(no_merchant_txs['txId'].values, desc="BFS k-hop exploration"):
        # Skip if already assigned in previous iteration
        if pd.notna(data.loc[data['txId'] == txid, 'merchant_id'].values[0]):
            continue
        
        # BFS exploration
        neighbors_by_k = bfs_khop_exploration(txid, G, txid_to_idx, k_values)
        
        if not neighbors_by_k:
            continue
        
        # Check each k-hop level (start with smallest k)
        for k in sorted(k_values):
            neighbors = neighbors_by_k.get(k, [])
            
            if not neighbors:
                continue
            
            # Check if any neighbor has merchant
            for neighbor_txid, distance in neighbors:
                neighbor_merchant = data.loc[
                    data['txId'] == neighbor_txid, 'merchant_id'
                ].values
                
                if len(neighbor_merchant) > 0 and pd.notna(neighbor_merchant[0]):
                    # Merchant found in k-hop!
                    merchant_id = neighbor_merchant[0]
                    
                    # Assign to current transaction
                    data.loc[data['txId'] == txid, 'merchant_id'] = merchant_id
                    data.loc[data['txId'] == txid, 'merchant_distance'] = distance
                    
                    khop_assignments += 1
                    distance_counts[k] += 1
                    
                    break  # Stop after first merchant found
            
            # If merchant found, stop searching farther k
            if pd.notna(data.loc[data['txId'] == txid, 'merchant_id'].values[0]):
                break
    
    print(f"\n✓ K-hop merchant assignments: {khop_assignments}")
    print(f"\nAssignments by distance:")
    for k in sorted(k_values):
        count = distance_counts[k]
        print(f"  ≤{k}-hop: {count} ({count/khop_assignments*100:.1f}%)" if khop_assignments > 0 else f"  ≤{k}-hop: 0")
    
    return data


def update_backward_labels(data):
    """
    Update backward labels after k-hop expansion
    
    Some transactions now have merchants (via k-hop)
    """
    print("\n" + "="*60)
    print("UPDATING BACKWARD LABELS")
    print("="*60)
    
    # Reload label encoder
    with open('data/processed/merchant_label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    # Fill NaN with 'NO_MERCHANT'
    data['merchant_id_filled'] = data['merchant_id'].fillna('NO_MERCHANT')
    
    # Re-encode (some new merchants may have been assigned)
    # For merchants not in original encoder, use 'NO_MERCHANT' label
    data['backward_label'] = data['merchant_id_filled'].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else 0
    )
    
    print(f"✓ Updated backward labels")
    print(f"  Label 0 (NO_MERCHANT): {(data['backward_label'] == 0).sum():,}")
    print(f"  Labels 1-500 (merchants): {(data['backward_label'] > 0).sum():,}")
    
    return data


def compute_final_statistics(data):
    """Compute final statistics after k-hop expansion"""
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    
    # Merchant distribution by distance
    print("Merchant interactions by distance:")
    distance_dist = data['merchant_distance'].value_counts().sort_index()
    
    for distance, count in distance_dist.items():
        if distance == -1:
            label = "No merchant"
        elif distance == 0:
            label = "Direct (0-hop)"
        else:
            label = f"{distance}-hop"
        
        print(f"  {label}: {count:,} ({count/len(data)*100:.1f}%)")
    
    # Total with merchant
    has_merchant = data['merchant_id'].notna()
    print(f"\nTotal transactions with merchant: {has_merchant.sum():,} ({has_merchant.sum()/len(data)*100:.1f}%)")


def save_expanded_data(data):
    """Save expanded data"""
    print("\n" + "="*60)
    print("SAVING EXPANDED DATA")
    print("="*60)
    
    data_path = 'data/processed/data_with_merchants.pkl'
    data.to_pickle(data_path)
    print(f"✓ Saved: {data_path}")


def main():
    """Main k-hop expansion pipeline"""
    print("\n" + "="*70)
    print("K-HOP MERCHANT EXPANSION (BFS)")
    print("="*70)
    
    # Step 1: Load data
    data, G, txid_to_idx = load_data()
    
    # Step 2: Expand merchants via k-hop
    data = expand_merchants_via_khop(
        data, G, txid_to_idx,
        k_values=[3, 7, 10],
        sample_size=5000  # Adjust based on computational resources
    )
    
    # Step 3: Update backward labels
    data = update_backward_labels(data)
    
    # Step 4: Compute final statistics
    compute_final_statistics(data)
    
    # Convert backward_label (merchant_id) to category_label
    print("\n" + "="*60)
    print("CREATING CATEGORY LABELS")
    print("="*60)
    
    data['category_label'] = data['backward_label'].apply(merchant_id_to_category)
    
    # Statistics
    print("Category distribution:")
    cat_names = {0: 'NO_MERCHANT', 1: 'E-commerce', 2: 'Gambling', 
                 3: 'Services', 4: 'Retail', 5: 'Luxury'}
    for cat_id in range(6):
        count = (data['category_label'] == cat_id).sum()
        pct = 100 * count / len(data)
        print(f"  {cat_id} ({cat_names[cat_id]:12s}): {count:>7,} ({pct:5.1f}%)")
    # Step 5: Compute merchant categories
    # Step 5: Save expanded data
    save_expanded_data(data)
    
    # Summary
    print("\n" + "="*70)
    print("K-HOP EXPANSION COMPLETE")
    print("="*70)
    print(f"Enhanced {len(data):,} transactions")
    print(f"\nMerchant detection via:")
    print(f"  - Direct (0-hop): {(data['merchant_distance'] == 0).sum():,}")
    print(f"  - K-hop (1-10): {(data['merchant_distance'] > 0).sum():,}")
    print(f"  - No merchant: {(data['merchant_distance'] == -1).sum():,}")
    
    print(f"\nNext steps:")
    print(f"  1. Run create_criminal_db.py to create criminal entities")
    print(f"  2. Proceed to model training (Phase 3)")


if __name__ == "__main__":
    main()