"""
Script: preprocess_elliptic.py
Description: Preprocess Elliptic Bitcoin dataset with feature engineering

Enhanced preprocessing with:
- Temporal encoding (sinusoidal)
- Topological features (centrality, clustering)
- Feature normalization
- Data quality validation
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def load_elliptic_data(data_dir='data/raw'):
    """
    Load Elliptic dataset files
    
    Returns:
        features_df, classes_df, edges_df
    """
    print("\n" + "="*60)
    print("LOADING ELLIPTIC DATASET")
    print("="*60)
    
    # Load features (no header)
    features_path = os.path.join(data_dir, 'elliptic_txs_features.csv')
    features_df = pd.read_csv(features_path, header=None)
    print(f"✓ Loaded features: {features_df.shape}")
    
    # Load classes
    classes_path = os.path.join(data_dir, 'elliptic_txs_classes.csv')
    classes_df = pd.read_csv(classes_path)
    print(f"✓ Loaded classes: {classes_df.shape}")
    
    # Load edges
    edges_path = os.path.join(data_dir, 'elliptic_txs_edgelist.csv')
    edges_df = pd.read_csv(edges_path)
    print(f"✓ Loaded edges: {edges_df.shape}")
    
    return features_df, classes_df, edges_df


# def merge_features_classes(features_df, classes_df):
#     """
#     Merge features with class labels
#     """
#     print("\n" + "="*60)
#     print("MERGING FEATURES & CLASSES")
#     print("="*60)
    
#     # Rename columns in features
#     # Column 0: txId
#     # Column 1: time_step
#     # Columns 2-166: features (165 features)
#     feature_cols = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, 166)]
#     features_df.columns = feature_cols
    
#     # Merge with classes
#     data = features_df.merge(classes_df, on='txId', how='left')
    
#     # Fill missing classes with -1 (unknown)
#     data['class'] = (
#         pd.to_numeric(data['class'], errors='coerce')
#         .fillna(-1)
#         .astype(int)
#     )

    
#     print(f"✓ Merged data shape: {data.shape}")
#     print(f"\nClass distribution:")
#     print(data['class'].value_counts(normalize=True))
    
#     return data

def merge_features_classes(features_df, classes_df):
    """
    Merge features with class labels
    """
    print("\n" + "="*60)
    print("MERGING FEATURES & CLASSES")
    print("="*60)
    
    # Rename columns in features
    # Column 0: txId
    # Column 1: time_step
    # Columns 2-166: features (165 features)
    feature_cols = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, 166)]
    features_df.columns = feature_cols
    
    # Merge with classes
    data = features_df.merge(classes_df, on='txId', how='left')
    
    # Handle classes: "" or "unknown" → -1, "1" → 1 (illicit), "2" → 2 (licit)
    data['class'] = (
        pd.to_numeric(data['class'], errors='coerce')
        .fillna(-1)
        .astype(int)
    )
    
    print(f"✓ Merged data shape: {data.shape}")
    
    # Class distribution with correct labels
    print(f"\nClass distribution:")
    class_names = {-1: 'unknown', 1: 'illicit', 2: 'licit'}
    for cls in sorted(data['class'].unique()):
        count = (data['class'] == cls).sum()
        pct = count / len(data) * 100
        name = class_names.get(cls, f'unexpected_{cls}')
        print(f"  {name}: {count:,} ({pct:.1f}%)")
    
    return data


def create_graph(edges_df, data):
    """
    Create NetworkX graph from edges
    """
    print("\n" + "="*60)
    print("CREATING GRAPH")
    print("="*60)
    
    # Create mapping txId -> index
    txid_to_idx = {txid: idx for idx, txid in enumerate(data['txId'])}
    
    # Create graph
    G = nx.Graph()
    
    # Add edges (only if both nodes in dataset)
    valid_edges = 0
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Adding edges"):
        if row['txId1'] in txid_to_idx and row['txId2'] in txid_to_idx:
            idx1 = txid_to_idx[row['txId1']]
            idx2 = txid_to_idx[row['txId2']]
            G.add_edge(idx1, idx2)
            valid_edges += 1
    
    print(f"✓ Graph created:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Connected components: {nx.number_connected_components(G)}")
    
    # Check if connected
    if nx.is_connected(G):
        print(f"  Diameter: {nx.diameter(G)}")
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"  Largest component: {len(largest_cc)} nodes")
    
    return G, txid_to_idx


def engineer_temporal_features(data):
    """
    Add temporal encoding features
    
    Sinusoidal encoding of timestep (1-49)
    """
    print("\n" + "="*60)
    print("TEMPORAL FEATURE ENGINEERING")
    print("="*60)
    
    # Sinusoidal encoding
    data['sin_time'] = np.sin(2 * np.pi * data['time_step'] / 49)
    data['cos_time'] = np.cos(2 * np.pi * data['time_step'] / 49)
    
    print(f"✓ Added temporal features:")
    print(f"  - sin_time (range: [{data['sin_time'].min():.3f}, {data['sin_time'].max():.3f}])")
    print(f"  - cos_time (range: [{data['cos_time'].min():.3f}, {data['cos_time'].max():.3f}])")
    
    return data


# def engineer_topological_features(data, G, txid_to_idx):
#     """
#     Add topological features from graph
    
#     Features:
#     - Degree centrality
#     - Clustering coefficient
#     - Betweenness centrality (expensive, optional)
#     """
#     print("\n" + "="*60)
#     print("TOPOLOGICAL FEATURE ENGINEERING")
#     print("="*60)
    
#     # Initialize features
#     data['degree_centrality'] = 0.0
#     data['clustering_coefficient'] = 0.0
#     data['betweenness_centrality'] = 0.0
    
#     # Compute centrality measures
#     print("Computing degree centrality...")
#     degree_centrality = nx.degree_centrality(G)
    
#     print("Computing clustering coefficient...")
#     clustering = nx.clustering(G)
    
#     # Betweenness is expensive - sample or skip
#     print("Computing betweenness centrality (sampled 10%)...")
#     # Sample 10% of nodes for betweenness
#     sample_nodes = np.random.choice(
#         list(G.nodes()),
#         size=int(0.1 * G.number_of_nodes()),
#         replace=False
#     )
#     betweenness = nx.betweenness_centrality(G, k=len(sample_nodes))
    
#     # Map to dataframe
#     for txid, idx in tqdm(txid_to_idx.items(), desc="Mapping features"):
#         if idx in G:
#             data.loc[data['txId'] == txid, 'degree_centrality'] = degree_centrality.get(idx, 0)
#             data.loc[data['txId'] == txid, 'clustering_coefficient'] = clustering.get(idx, 0)
#             data.loc[data['txId'] == txid, 'betweenness_centrality'] = betweenness.get(idx, 0)
    
#     print(f"✓ Added topological features:")
#     print(f"  - degree_centrality (mean: {data['degree_centrality'].mean():.6f})")
#     print(f"  - clustering_coefficient (mean: {data['clustering_coefficient'].mean():.6f})")
#     print(f"  - betweenness_centrality (mean: {data['betweenness_centrality'].mean():.6f})")
    
#     return data

def engineer_topological_features(data, G, txid_to_idx):
    """
    Add topological features from graph
    """
    print("\n" + "="*60)
    print("TOPOLOGICAL FEATURE ENGINEERING")
    print("="*60)
    
    # Initialize features
    data['degree_centrality'] = 0.0
    data['clustering_coefficient'] = 0.0
    data['betweenness_centrality'] = 0.0
    
    # Compute centrality measures
    print("Computing degree centrality...")
    degree_centrality = nx.degree_centrality(G)
    
    print("Computing clustering coefficient...")
    clustering = nx.clustering(G)
    
    # Betweenness with progress tracking
    print("Computing betweenness centrality (sampled 10%)...")
    
    # Sample nodes
    all_nodes = list(G.nodes())
    sample_size = int(0.1 * len(all_nodes))
    sample_nodes = np.random.choice(all_nodes, size=sample_size, replace=False)
    
    print(f"  Sample size: {sample_size:,} nodes (10% of {len(all_nodes):,})")
    
    # Initialize betweenness
    betweenness = {node: 0.0 for node in G.nodes()}
    
    # Calculate betweenness in batches with progress
    batch_size = 500  # Process 500 nodes at a time
    num_batches = (len(sample_nodes) + batch_size - 1) // batch_size
    
    print(f"  Processing in {num_batches} batches of ~{batch_size} nodes...")
    
    for i in tqdm(range(0, len(sample_nodes), batch_size), 
                  desc="  Betweenness batches", 
                  unit="batch"):
        
        batch = sample_nodes[i:i+batch_size]
        
        # Calculate betweenness for this batch
        batch_betweenness = nx.betweenness_centrality_subset(
            G, 
            sources=batch,
            targets=all_nodes,
            normalized=True
        )
        
        # Accumulate results
        for node, value in batch_betweenness.items():
            betweenness[node] += value
    
    # Normalize (since we summed over batches)
    max_betweenness = max(betweenness.values()) if betweenness else 1.0
    if max_betweenness > 0:
        betweenness = {k: v/max_betweenness for k, v in betweenness.items()}
    
    print(f"✓ Betweenness computed")
    
    # Map to dataframe
    print("\nMapping features to transactions...")
    for txid, idx in tqdm(txid_to_idx.items(), desc="  Mapping", unit="tx"):
        if idx in G:
            data.loc[data['txId'] == txid, 'degree_centrality'] = degree_centrality.get(idx, 0)
            data.loc[data['txId'] == txid, 'clustering_coefficient'] = clustering.get(idx, 0)
            data.loc[data['txId'] == txid, 'betweenness_centrality'] = betweenness.get(idx, 0)
    
    print(f"\n✓ Added topological features:")
    print(f"  - degree_centrality (mean: {data['degree_centrality'].mean():.6f})")
    print(f"  - clustering_coefficient (mean: {data['clustering_coefficient'].mean():.6f})")
    print(f"  - betweenness_centrality (mean: {data['betweenness_centrality'].mean():.6f})")
    
    return data

def normalize_features(data):
    """
    Normalize all features using StandardScaler
    """
    print("\n" + "="*60)
    print("FEATURE NORMALIZATION")
    print("="*60)
    
    # Select feature columns (exclude txId, time_step, class)
    feature_cols = [col for col in data.columns if col.startswith('feat_')]
    feature_cols += ['sin_time', 'cos_time', 'degree_centrality', 'clustering_coefficient', 'betweenness_centrality']
    
    print(f"Normalizing {len(feature_cols)} features...")
    
    # Fit scaler
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    
    # Verify normalization
    print(f"✓ Features normalized:")
    print(f"  Mean (abs): {np.abs(data[feature_cols].mean().mean()):.6f} (should be ~0)")
    print(f"  Std (mean): {data[feature_cols].std().mean():.6f} (should be ~1)")
    
    # Save scaler
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler to data/processed/feature_scaler.pkl")
    
    return data, scaler


def validate_data_quality(data):
    """
    Validate preprocessed data quality
    """
    print("\n" + "="*60)
    print("DATA QUALITY VALIDATION")
    print("="*60)
    
    # Check timesteps
    unique_timesteps = data['time_step'].nunique()
    print(f"✓ Timesteps: {unique_timesteps} (expected: 49)")
    assert unique_timesteps == 49, "Should have 49 timesteps!"
    
    # Check features normalized
    feature_cols = [col for col in data.columns if col.startswith('feat_')]
    mean_abs = np.abs(data[feature_cols].mean().mean())
    print(f"✓ Features mean (abs): {mean_abs:.6f} (should be < 0.01)")
    assert mean_abs < 0.01, "Features should be normalized!"
    
    # Check class distribution
    class_dist = data['class'].value_counts(normalize=True)
    print(f"✓ Class distribution:")
    for cls, ratio in class_dist.items():
        cls_name = {-1: 'unknown', 1: 'illicit', 2: 'licit'}[cls]
        print(f"  {cls_name}: {ratio:.1%}")
    
    # Check for NaN
    nan_count = data.isnull().sum().sum()
    print(f"✓ NaN values: {nan_count} (should be 0)")
    assert nan_count == 0, "No NaN values allowed!"
    
    print("\n✅ Data quality validation PASSED")


def save_preprocessed_data(data, edges_df, G, txid_to_idx):
    """
    Save preprocessed data
    """
    print("\n" + "="*60)
    print("SAVING PREPROCESSED DATA")
    print("="*60)
    
    os.makedirs('data/processed', exist_ok=True)
    
    # Save data
    data_path = 'data/processed/data_with_graph_features.pkl'
    data.to_pickle(data_path)
    print(f"✓ Saved data: {data_path} ({data.shape})")
    
    # Save edges
    edges_path = 'data/processed/edges_preprocessed.pkl'
    edges_df.to_pickle(edges_path)
    print(f"✓ Saved edges: {edges_path} ({edges_df.shape})")
    
    # Save graph
    graph_path = 'data/processed/graph.pkl'
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"✓ Saved graph: {graph_path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    
    # Save txid mapping
    mapping_path = 'data/processed/txid_to_idx.pkl'
    with open(mapping_path, 'wb') as f:
        pickle.dump(txid_to_idx, f)
    print(f"✓ Saved txId mapping: {mapping_path}")
    
    # Summary statistics
    stats = {
        'num_transactions': len(data),
        'num_timesteps': data['time_step'].nunique(),
        'num_features': len([col for col in data.columns if col.startswith('feat_')]) + 5,  # +5 for temporal & topological
        'num_edges': len(edges_df),
        'num_graph_nodes': G.number_of_nodes(),
        'num_graph_edges': G.number_of_edges(),
        'class_distribution': data['class'].value_counts().to_dict()
    }
    
    stats_path = 'data/processed/preprocessing_stats.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"✓ Saved statistics: {stats_path}")
    
    return stats


def main():
    """
    Main preprocessing pipeline
    """
    print("\n" + "="*70)
    print("ELLIPTIC BITCOIN DATASET PREPROCESSING")
    print("="*70)
    
    # Step 1: Load data
    features_df, classes_df, edges_df = load_elliptic_data()
    
    # Step 2: Merge features & classes
    data = merge_features_classes(features_df, classes_df)
    
    # Step 3: Create graph
    G, txid_to_idx = create_graph(edges_df, data)
    
    # Step 4: Feature engineering - Temporal
    data = engineer_temporal_features(data)
    
    # Step 5: Feature engineering - Topological
    data = engineer_topological_features(data, G, txid_to_idx)
    
    # Step 6: Normalize features
    data, scaler = normalize_features(data)
    
    # Step 7: Validate data quality
    validate_data_quality(data)
    
    # Step 8: Save preprocessed data
    stats = save_preprocessed_data(data, edges_df, G, txid_to_idx)
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Total features: {stats['num_features']}")
    print(f"  - Original Elliptic: 165")
    print(f"  - Temporal: 2 (sin_time, cos_time)")
    print(f"  - Topological: 3 (degree, clustering, betweenness)")
    print(f"  - Total: {stats['num_features']}")
    print(f"\nTransactions: {stats['num_transactions']:,}")
    print(f"Timesteps: {stats['num_timesteps']}")
    print(f"Edges: {stats['num_edges']:,}")
    print(f"\nOutput files:")
    print(f"  - data/processed/data_with_graph_features.pkl")
    print(f"  - data/processed/edges_preprocessed.pkl")
    print(f"  - data/processed/graph.pkl")
    print(f"  - data/processed/txid_to_idx.pkl")
    print(f"  - data/processed/feature_scaler.pkl")
    print(f"  - data/processed/preprocessing_stats.pkl")


if __name__ == "__main__":
    main()