"""
Script: build_temporal_graph.py
Description: Build temporal graph snapshots (49 timesteps)

Creates graph snapshot for each timestep for LSTM processing
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


def load_data():
    """Load preprocessed data and edges"""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    data = pd.read_pickle('data/processed/data_with_graph_features.pkl')
    edges = pd.read_pickle('data/processed/edges_preprocessed.pkl')
    
    print(f"✓ Data: {data.shape}")
    print(f"✓ Edges: {edges.shape}")
    
    return data, edges


def build_temporal_snapshots(data, edges):
    """
    Build graph snapshot for each timestep
    
    Returns:
        temporal_graphs: Dict {timestep: NetworkX Graph}
    """
    print("\n" + "="*60)
    print("BUILDING TEMPORAL GRAPH SNAPSHOTS")
    print("="*60)
    
    # Create txId to index mapping
    txid_to_idx = {txid: idx for idx, txid in enumerate(data['txId'])}
    
    # Initialize temporal graphs
    temporal_graphs = {}
    
    # For each timestep (1-49)
    for t in tqdm(range(1, 50), desc="Creating snapshots"):
        # Transactions at timestep t
        txs_at_t = data[data['time_step'] == t]['txId'].tolist()
        
        if len(txs_at_t) == 0:
            print(f"  ⚠️ Timestep {t}: No transactions")
            continue
        
        # Filter edges: both nodes must be at timestep t
        edges_at_t = edges[
            edges['txId1'].isin(txs_at_t) & 
            edges['txId2'].isin(txs_at_t)
        ]
        
        # Create graph
        G_t = nx.Graph()
        
        # Add nodes
        for txid in txs_at_t:
            idx = txid_to_idx[txid]
            G_t.add_node(idx, txId=txid)
        
        # Add edges
        for _, row in edges_at_t.iterrows():
            idx1 = txid_to_idx[row['txId1']]
            idx2 = txid_to_idx[row['txId2']]
            G_t.add_edge(idx1, idx2)
        
        temporal_graphs[t] = G_t
    
    print(f"\n✓ Created {len(temporal_graphs)} temporal snapshots")
    
    # Statistics
    print(f"\nSnapshot statistics:")
    nodes_per_timestep = [G.number_of_nodes() for G in temporal_graphs.values()]
    edges_per_timestep = [G.number_of_edges() for G in temporal_graphs.values()]
    
    print(f"  Nodes per timestep:")
    print(f"    Min: {min(nodes_per_timestep):,}")
    print(f"    Max: {max(nodes_per_timestep):,}")
    print(f"    Mean: {np.mean(nodes_per_timestep):.0f}")
    
    print(f"  Edges per timestep:")
    print(f"    Min: {min(edges_per_timestep):,}")
    print(f"    Max: {max(edges_per_timestep):,}")
    print(f"    Mean: {np.mean(edges_per_timestep):.0f}")
    
    return temporal_graphs


def analyze_temporal_evolution(temporal_graphs):
    """
    Analyze how graph evolves over time
    """
    print("\n" + "="*60)
    print("TEMPORAL EVOLUTION ANALYSIS")
    print("="*60)
    
    timesteps = sorted(temporal_graphs.keys())
    
    # Track metrics over time
    metrics = {
        'nodes': [],
        'edges': [],
        'density': [],
        'avg_degree': []
    }
    
    for t in timesteps:
        G = temporal_graphs[t]
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        metrics['nodes'].append(num_nodes)
        metrics['edges'].append(num_edges)
        
        # Density
        if num_nodes > 1:
            density = nx.density(G)
            metrics['density'].append(density)
        else:
            metrics['density'].append(0)
        
        # Average degree
        if num_nodes > 0:
            avg_degree = 2 * num_edges / num_nodes
            metrics['avg_degree'].append(avg_degree)
        else:
            metrics['avg_degree'].append(0)
    
    # Summary
    print(f"Temporal evolution (timesteps {timesteps[0]}-{timesteps[-1]}):")
    print(f"  Nodes growth: {metrics['nodes'][0]:,} → {metrics['nodes'][-1]:,}")
    print(f"  Edges growth: {metrics['edges'][0]:,} → {metrics['edges'][-1]:,}")
    print(f"  Avg density: {np.mean(metrics['density']):.6f}")
    print(f"  Avg degree: {np.mean(metrics['avg_degree']):.2f}")
    
    # Detect sudden changes
    node_deltas = np.diff(metrics['nodes'])
    edge_deltas = np.diff(metrics['edges'])
    
    max_node_increase = np.max(node_deltas)
    max_edge_increase = np.max(edge_deltas)
    
    print(f"\nMax growth per timestep:")
    print(f"  Nodes: +{max_node_increase:,}")
    print(f"  Edges: +{max_edge_increase:,}")
    
    return metrics


def save_temporal_graphs(temporal_graphs):
    """Save temporal graphs to disk"""
    print("\n" + "="*60)
    print("SAVING TEMPORAL GRAPHS")
    print("="*60)
    
    os.makedirs('data/processed', exist_ok=True)
    
    output_path = 'data/processed/temporal_graphs.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(temporal_graphs, f)
    
    print(f"✓ Saved temporal graphs: {output_path}")
    print(f"  Timesteps: {len(temporal_graphs)}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    """Main pipeline"""
    print("\n" + "="*70)
    print("TEMPORAL GRAPH CONSTRUCTION")
    print("="*70)
    
    # Step 1: Load data
    data, edges = load_data()
    
    # Step 2: Build temporal snapshots
    temporal_graphs = build_temporal_snapshots(data, edges)
    
    # Step 3: Analyze temporal evolution
    metrics = analyze_temporal_evolution(temporal_graphs)
    
    # Step 4: Save temporal graphs
    save_temporal_graphs(temporal_graphs)
    
    # Summary
    print("\n" + "="*70)
    print("TEMPORAL GRAPH CONSTRUCTION COMPLETE")
    print("="*70)
    print(f"Created {len(temporal_graphs)} temporal snapshots")
    print(f"Timestep range: {min(temporal_graphs.keys())} - {max(temporal_graphs.keys())}")
    print(f"\nOutput:")
    print(f"  - data/processed/temporal_graphs.pkl")
    print(f"\nUsage:")
    print(f"  These graphs can be used for:")
    print(f"  - LSTM temporal modeling")
    print(f"  - Temporal pattern analysis")
    print(f"  - Graph evolution studies")


if __name__ == "__main__":
    main()