"""
Script: precompute_merchant_embeddings.py
Description: Precompute merchant embeddings for backward attribution

This script creates initial merchant embeddings by aggregating
features of transactions associated with each merchant.

These embeddings will be updated later with GNN-learned embeddings.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def load_data():
    """Load data and merchants"""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    data = pd.read_pickle('data/processed/data_with_merchants.pkl')
    merchants_df = pd.read_csv('data/merchants/known_merchants.csv')
    
    print(f"✓ Data: {data.shape}")
    print(f"✓ Merchants: {len(merchants_df)}")
    
    return data, merchants_df


def compute_merchant_embeddings(data, merchants_df, embedding_dim=128):
    """
    Compute initial merchant embeddings
    
    Strategy:
    - For each merchant: aggregate features of associated transactions
    - Use mean aggregation
    - For merchants without transactions: random initialization
    
    Args:
        data: Transaction data
        merchants_df: Merchant database
        embedding_dim: Embedding dimension (default: 128)
    
    Returns:
        merchant_embeddings: Dict {merchant_id: embedding}
    """
    print("\n" + "="*60)
    print("COMPUTING MERCHANT EMBEDDINGS")
    print("="*60)
    
    # Feature columns
    feature_cols = [f'feat_{i}' for i in range(1, 166)]
    feature_cols += ['sin_time', 'cos_time', 'degree_centrality', 
                     'clustering_coefficient', 'betweenness_centrality']
    
    print(f"Using {len(feature_cols)} features for embeddings")
    
    merchant_embeddings = {}
    merchants_with_txs = 0
    merchants_without_txs = 0
    
    for _, merchant in tqdm(merchants_df.iterrows(), total=len(merchants_df), desc="Computing embeddings"):
        merchant_id = merchant['merchant_id']
        
        # Get transactions associated with this merchant
        merchant_txs = data[data['merchant_id'] == merchant_id]
        
        if len(merchant_txs) > 0:
            # Aggregate features (mean)
            embedding = merchant_txs[feature_cols].mean().values
            
            # Reduce dimensionality if needed (PCA or simple projection)
            # For now, use first 128 features if more than 128
            if len(embedding) > embedding_dim:
                embedding = embedding[:embedding_dim]
            elif len(embedding) < embedding_dim:
                # Pad with zeros
                embedding = np.pad(embedding, (0, embedding_dim - len(embedding)))
            
            merchant_embeddings[merchant_id] = embedding
            merchants_with_txs += 1
        else:
            # No transactions yet → random initialization
            # Will be updated later with GNN embeddings
            embedding = np.random.randn(embedding_dim) * 0.01
            merchant_embeddings[merchant_id] = embedding
            merchants_without_txs += 1
    
    # Add NO_MERCHANT embedding (all zeros)
    merchant_embeddings['NO_MERCHANT'] = np.zeros(embedding_dim)
    
    print(f"\n✓ Computed embeddings for {len(merchant_embeddings)} entities")
    print(f"  Merchants with transactions: {merchants_with_txs}")
    print(f"  Merchants without transactions: {merchants_without_txs}")
    print(f"  NO_MERCHANT: 1")
    print(f"  Embedding dimension: {embedding_dim}")
    
    return merchant_embeddings


def validate_embeddings(merchant_embeddings, embedding_dim=128):
    """Validate embeddings quality"""
    print("\n" + "="*60)
    print("VALIDATING EMBEDDINGS")
    print("="*60)
    
    embeddings_array = np.array(list(merchant_embeddings.values()))
    
    print(f"Embedding shape: {embeddings_array.shape}")
    print(f"Expected: ({len(merchant_embeddings)}, {embedding_dim})")
    
    # Check for NaN
    nan_count = np.isnan(embeddings_array).sum()
    print(f"NaN values: {nan_count} (should be 0)")
    
    # Statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embeddings_array.mean():.6f}")
    print(f"  Std: {embeddings_array.std():.6f}")
    print(f"  Min: {embeddings_array.min():.6f}")
    print(f"  Max: {embeddings_array.max():.6f}")
    
    # Check NO_MERCHANT
    no_merchant_emb = merchant_embeddings['NO_MERCHANT']
    print(f"\nNO_MERCHANT embedding:")
    print(f"  All zeros: {np.allclose(no_merchant_emb, 0)}")
    
    assert nan_count == 0, "Embeddings should not contain NaN!"
    print("\n✅ Embeddings validation PASSED")


def save_embeddings(merchant_embeddings):
    """Save merchant embeddings"""
    print("\n" + "="*60)
    print("SAVING MERCHANT EMBEDDINGS")
    print("="*60)
    
    os.makedirs('data/processed', exist_ok=True)
    
    output_path = 'data/processed/merchant_embeddings_init.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(merchant_embeddings, f)
    
    print(f"✓ Saved: {output_path}")
    print(f"  Embeddings: {len(merchant_embeddings)}")
    
    # Also save as numpy array for faster loading
    # Create ordered array matching label encoder
    with open('data/processed/merchant_label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    # Create embedding matrix [num_classes, embedding_dim]
    num_classes = len(le.classes_)
    embedding_dim = len(list(merchant_embeddings.values())[0])
    
    embedding_matrix = np.zeros((num_classes, embedding_dim))
    
    for idx, merchant_id in enumerate(le.classes_):
        if merchant_id in merchant_embeddings:
            embedding_matrix[idx] = merchant_embeddings[merchant_id]
    
    matrix_path = 'data/processed/merchant_embeddings_matrix.npy'
    np.save(matrix_path, embedding_matrix)
    print(f"✓ Saved matrix: {matrix_path} ({embedding_matrix.shape})")


def main():
    """Main pipeline"""
    print("\n" + "="*70)
    print("MERCHANT EMBEDDINGS PRECOMPUTATION")
    print("="*70)
    
    # Step 1: Load data
    data, merchants_df = load_data()
    
    # Step 2: Compute embeddings
    merchant_embeddings = compute_merchant_embeddings(data, merchants_df, embedding_dim=128)
    
    # Step 3: Validate embeddings
    validate_embeddings(merchant_embeddings, embedding_dim=128)
    
    # Step 4: Save embeddings
    save_embeddings(merchant_embeddings)
    
    # Summary
    print("\n" + "="*70)
    print("MERCHANT EMBEDDINGS COMPLETE")
    print("="*70)
    print(f"Created embeddings for {len(merchant_embeddings)} entities")
    print(f"Embedding dimension: 128")
    
    print(f"\nNOTE:")
    print(f"  These are INITIAL embeddings based on feature aggregation")
    print(f"  They will be UPDATED with GNN-learned embeddings during training")
    
    print(f"\nUsage:")
    print(f"  - Backward attribution (retrieval)")
    print(f"  - Similarity computation")
    print(f"  - Merchant clustering/visualization")


if __name__ == "__main__":
    main()