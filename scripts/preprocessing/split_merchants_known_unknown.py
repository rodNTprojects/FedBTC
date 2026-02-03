"""
Script: split_merchants_known_unknown.py
Description: Split merchants into known (90%) and unknown (10%) for hybrid evaluation
"""

import pandas as pd
import numpy as np
import pickle
import os

def split_merchants():
    """Split merchants into known/unknown sets"""
    
    print("\n" + "="*60)
    print("SPLITTING MERCHANTS: KNOWN vs UNKNOWN")
    print("="*60)
    
    # Load merchants
    merchants_df = pd.read_csv('data/merchants/known_merchants.csv')
    print(f"Total merchants: {len(merchants_df)}")
    
    # Stratified split by category
    np.random.seed(42)
    
    known_merchants = []
    unknown_merchants = []
    
    for category in merchants_df['category'].unique():
        cat_merchants = merchants_df[merchants_df['category'] == category]
        n_total = len(cat_merchants)
        n_known = int(n_total * 0.9)  # 90% known
        
        indices = np.random.permutation(n_total)
        known_idx = indices[:n_known]
        unknown_idx = indices[n_known:]
        
        known_merchants.extend(cat_merchants.iloc[known_idx]['merchant_id'].tolist())
        unknown_merchants.extend(cat_merchants.iloc[unknown_idx]['merchant_id'].tolist())
    
    print(f"\nSplit:")
    print(f"  Known merchants: {len(known_merchants)} (90%)")
    print(f"  Unknown merchants: {len(unknown_merchants)} (10%)")
    
    # Save split
    split_info = {
        'known_merchants': known_merchants,
        'unknown_merchants': unknown_merchants
    }
    
    os.makedirs('data/merchants', exist_ok=True)
    with open('data/merchants/merchant_split.pkl', 'wb') as f:
        pickle.dump(split_info, f)
    
    print(f"\n✓ Saved: data/merchants/merchant_split.pkl")
    
    # Create mapping merchant_id → category
    merchant_to_category = dict(zip(
        merchants_df['merchant_id'], 
        merchants_df['category']
    ))
    
    with open('data/merchants/merchant_categories.pkl', 'wb') as f:
        pickle.dump(merchant_to_category, f)
    
    print(f"✓ Saved: data/merchants/merchant_categories.pkl")
    
    return split_info

if __name__ == "__main__":
    split_merchants()