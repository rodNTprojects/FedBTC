"""
Script: simulate_merchants.py
Description: Simulate merchant database (500 merchants) and assign to transactions

Creates merchant database with:
- Categories: E-commerce, Gambling, Dark Web, Services, Exchanges
- Metadata: Volume, patterns, geographic, criminal status
- Transaction-Merchant assignments (~30%)
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from faker import Faker

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# Mapping category_name → category_id (pour backward attribution)
CATEGORY_NAME_TO_ID = {
    'E-commerce': 1,
    'Gambling': 2,
    'Services': 3,
    'Retail': 4,
    'Luxury': 5,
}

def generate_merchant_database(num_merchants=500):
    """
    Generate synthetic merchant database
    
    Args:
        num_merchants: Number of merchants to generate (default: 500)
    
    Returns:
        merchants_df: DataFrame with merchant metadata
    """
    print("\n" + "="*60)
    print(f"GENERATING MERCHANT DATABASE (N={num_merchants})")
    print("="*60)
    
    fake = Faker()
    Faker.seed(42)
    np.random.seed(42)
    
    # Category distribution
    # categories = {
    #     'E-commerce': 200,
    #     'Gambling': 100,
    #     'Services': 100,
    #     'Dark Web': 50,
    #     'Exchanges': 50
    # }
    
    categories = {
    'E-commerce': 175,   # 35% - BitPay, Coinbase Commerce
    'Gambling': 125,     # 25% - Stake, BC.Game
    'Services': 100,     # 20% - VPN, Hosting
    'Retail': 75,        # 15% - Physical stores
    'Luxury': 25,        # 5%  - High-value goods
    }

    print(f"Category distribution:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
    
    merchants = []
    merchant_id_counter = 0
    
    for category, count in categories.items():
        for i in range(count):
            # Generate merchant
            # is_criminal = (category == 'Dark Web' and np.random.random() < 0.8)
            is_criminal = False
            
            merchant = {
                'merchant_id': f'MERCH_{merchant_id_counter:03d}',
                'name': f'{category}_{fake.company()}' if category != 'Dark Web' 
                        else f'DarkMarket_{fake.word().capitalize()}_{merchant_id_counter}',
                'category': category,
                'category_id': CATEGORY_NAME_TO_ID[category],
                'num_addresses': np.random.randint(5, 50),
                'avg_transaction_amount': np.random.lognormal(mean=-2, sigma=1),  # BTC
                'transaction_frequency': np.random.choice(['hourly', 'daily', 'weekly'], p=[0.1, 0.6, 0.3]),
                'geographic_region': np.random.choice(['Europe', 'Asia', 'Americas'], p=[0.4, 0.35, 0.25]),
                'known_criminal': is_criminal,
                'registration_year': np.random.randint(2015, 2020) if not is_criminal else np.random.randint(2017, 2020)
            }
            
            merchants.append(merchant)
            merchant_id_counter += 1
    
    merchants_df = pd.DataFrame(merchants)
    
    print(f"\n✓ Generated {len(merchants_df)} merchants")
    print(f"  Criminal entities: {merchants_df['known_criminal'].sum()}")
    print(f"  Legitimate: {(~merchants_df['known_criminal']).sum()}")
    
    return merchants_df


def assign_merchants_to_transactions(data, merchants_df, assignment_ratio=0.30):
    """
    Assign merchants to transactions
    
    Strategy:
    - ~30% of transactions interact with a merchant
    - Bias illicit → dark web merchants
    - Bias licit → e-commerce/services merchants
    
    Args:
        data: Transaction data
        merchants_df: Merchant database
        assignment_ratio: Ratio of transactions with merchant (default: 0.30)
    
    Returns:
        data: Enhanced with merchant_id and merchant_distance columns
    """
    print("\n" + "="*60)
    print(f"ASSIGNING MERCHANTS TO TRANSACTIONS")
    print("="*60)
    
    # Initialize columns
    data['merchant_id'] = None
    data['merchant_distance'] = -1  # -1 = no merchant, 0 = direct
    
    # Defragment DataFrame
    data = data.copy()
    # Select transactions for merchant assignment
    # Focus on unknown transactions (class == -1) to avoid labeling bias
    unknown_txs = data[data['class'] == -1].copy()
    
    # Also include some licit/illicit for realism
    licit_txs = data[data['class'] == 0].sample(frac=0.2, random_state=42)
    illicit_txs = data[data['class'] == 1].sample(frac=0.4, random_state=42)
    
    # Combine
    assignable_txs = pd.concat([unknown_txs, licit_txs, illicit_txs])
    
    # Select subset for assignment
    num_to_assign = int(len(assignable_txs) * assignment_ratio)
    txs_with_merchant = assignable_txs.sample(n=num_to_assign, random_state=42)
    
    print(f"Assigning merchants to {len(txs_with_merchant):,} transactions ({len(txs_with_merchant)/len(data)*100:.1f}%)")
    
    # Assign merchants based on transaction class
    assignments_count = {}
    
    for idx in txs_with_merchant.index:
        tx_class = data.loc[idx, 'class']
        
        # Select merchant category based on class
        if tx_class == 1:  # Illicit
            # Bias toward dark web and gambling
            category = np.random.choice(
                ['Dark Web', 'Gambling', 'E-commerce', 'Services'],
                p=[0.60, 0.25, 0.10, 0.05]
            )
        elif tx_class == 0:  # Licit
            # Bias toward e-commerce and services
            category = np.random.choice(
                ['E-commerce', 'Services', 'Gambling', 'Exchanges'],
                p=[0.60, 0.25, 0.10, 0.05]
            )
        else:  # Unknown
            # Uniform distribution
            category = np.random.choice(merchants_df['category'].unique())
        
        # Select random merchant from category
        category_merchants = merchants_df[merchants_df['category'] == category]
        
        if len(category_merchants) > 0:
            merchant_id = category_merchants.sample(1, random_state=None)['merchant_id'].values[0]
            
            data.at[idx, 'merchant_id'] = merchant_id
            data.at[idx, 'merchant_distance'] = 0  # Direct interaction
            
            # Track assignments
            category_key = f"{category}"
            assignments_count[category_key] = assignments_count.get(category_key, 0) + 1
    
    print(f"\n✓ Merchant assignment complete")
    print(f"\nAssignments by category:")
    for category, count in sorted(assignments_count.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count} ({count/len(txs_with_merchant)*100:.1f}%)")
    
    # Statistics
    has_merchant = data['merchant_id'].notna()
    print(f"\nFinal statistics:")
    print(f"  Transactions with merchant: {has_merchant.sum():,} ({has_merchant.sum()/len(data)*100:.1f}%)")
    print(f"  Transactions without merchant: {(~has_merchant).sum():,} ({(~has_merchant).sum()/len(data)*100:.1f}%)")
    
    return data


# def create_backward_labels(data, merchants_df):
#     """
#     Create backward attribution labels
    
#     Label encoding:
#     - 0: NO_MERCHANT
#     - 1-500: merchant_id
    
#     Args:
#         data: Transaction data with merchant_id
#         merchants_df: Merchant database
    
#     Returns:
#         data: Enhanced with backward_label column
#     """
#     print("\n" + "="*60)
#     print("CREATING BACKWARD ATTRIBUTION LABELS")
#     print("="*60)
    
#     # Fill NaN with 'NO_MERCHANT'
#     data['merchant_id_filled'] = data['merchant_id'].fillna('NO_MERCHANT')
    
#     # Create label encoder
#     from sklearn.preprocessing import LabelEncoder
    
#     # le = LabelEncoder()
#     # data['backward_label'] = le.fit_transform(data['merchant_id_filled'])
    
#     # print(f"✓ Created backward labels")
#     # print(f"  Label 0 (NO_MERCHANT): {(data['backward_label'] == 0).sum():,}")
#     # print(f"  Labels 1-{len(le.classes_)-1} (merchants): {(data['backward_label'] > 0).sum():,}")

#     # Force NO_MERCHANT = 0
#     data['backward_label'] = 0  # Défaut

#     # Map merchants 1-500
#     merchant_mask = data['merchant_id'] != 'NO_MERCHANT'
#     if merchant_mask.sum() > 0:
#         # Encode seulement les merchants
#         le = LabelEncoder()
#         merchant_labels = le.fit_transform(data.loc[merchant_mask, 'merchant_id']) + 1  # +1 pour commencer à 1
#         data.loc[merchant_mask, 'backward_label'] = merchant_labels

#     print(f"✓ Created backward labels")
#     print(f"  Label 0 (NO_MERCHANT): {(data['backward_label'] == 0).sum():,}")
#     print(f"  Labels 1-500 (merchants): {(data['backward_label'] > 0).sum():,}")
    
#     # Save label encoder
#     os.makedirs('data/processed', exist_ok=True)
#     with open('data/processed/merchant_label_encoder.pkl', 'wb') as f:
#         pickle.dump(le, f)
#     print(f"  Saved encoder: data/processed/merchant_label_encoder.pkl")
    
#     return data, le


def create_backward_labels(data, merchants_df):
    """
    Create backward attribution labels
    
    Label encoding:
    - 0: NO_MERCHANT
    - 1-500: merchant_id
    
    Args:
        data: Transaction data with merchant_id
        merchants_df: Merchant database
    
    Returns:
        data: Enhanced with backward_label column
    """
    print("\n" + "="*60)
    print("CREATING BACKWARD ATTRIBUTION LABELS")
    print("="*60)
    
    # Fill NaN with 'NO_MERCHANT'
    data['merchant_id_filled'] = data['merchant_id'].fillna('NO_MERCHANT')
    
    # Create label encoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    # Force NO_MERCHANT = 0
    data['backward_label'] = 0  # Défaut pour tous
    
    # Map merchants 1-500
    merchant_mask = (data['merchant_id'].notna()) & (data['merchant_id'] != 'NO_MERCHANT')
    
    if merchant_mask.sum() > 0:
        # Encode seulement les merchants
        unique_merchants = data.loc[merchant_mask, 'merchant_id'].unique()
        le.fit(unique_merchants)
        
        # Assign labels 1-500 (shifted by +1)
        for idx in data[merchant_mask].index:
            merchant_id = data.loc[idx, 'merchant_id']
            data.loc[idx, 'backward_label'] = le.transform([merchant_id])[0] + 1
    else:
        # Fit encoder with dummy data if no merchants
        le.fit(['NO_MERCHANT'])
    
    print(f"✓ Created backward labels")
    print(f"  Label 0 (NO_MERCHANT): {(data['backward_label'] == 0).sum():,}")
    print(f"  Labels 1-500 (merchants): {(data['backward_label'] > 0).sum():,}")
    
    # Save label encoder
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/merchant_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print(f"  Saved encoder: data/processed/merchant_label_encoder.pkl")
    
    return data, le

def compute_merchant_profiles(data, merchants_df):
    """
    Compute merchant profiles from assigned transactions
    
    Profiles include:
    - Total transaction volume
    - Average amount
    - Peak activity hours (simulated)
    - Geographic distribution
    """
    print("\n" + "="*60)
    print("COMPUTING MERCHANT PROFILES")
    print("="*60)
    
    merchant_profiles = []
    
    for _, merchant in merchants_df.iterrows():
        merchant_id = merchant['merchant_id']
        
        # Transactions associated with this merchant
        merchant_txs = data[data['merchant_id'] == merchant_id]
        
        if len(merchant_txs) == 0:
            # No transactions yet (will be assigned later via k-hop)
            total_volume = 0
            avg_amount = merchant['avg_transaction_amount']
            peak_hours = 'Unknown'
        else:
            total_volume = len(merchant_txs)
            avg_amount = merchant['avg_transaction_amount']  # Use simulated (no amount in Elliptic)
            
            # Peak hours (simulated based on category)
            if merchant['category'] == 'Dark Web':
                peak_hours = '02:00-06:00'  # Nocturnal
            elif merchant['category'] == 'E-commerce':
                peak_hours = '10:00-22:00'  # Business hours
            elif merchant['category'] == 'Gambling':
                peak_hours = '18:00-02:00'  # Evening
            else:
                peak_hours = '09:00-18:00'  # Standard
        
        profile = {
            'merchant_id': merchant_id,
            'name': merchant['name'],
            'category': merchant['category'],
            'total_volume': total_volume,
            'avg_transaction_amount': avg_amount,
            'peak_hours': peak_hours,
            'geographic_region': merchant['geographic_region'],
            'known_criminal': merchant['known_criminal']
        }
        
        merchant_profiles.append(profile)
    
    profiles_df = pd.DataFrame(merchant_profiles)
    
    print(f"✓ Computed profiles for {len(profiles_df)} merchants")
    print(f"\nTop 5 by volume:")
    print(profiles_df.nlargest(5, 'total_volume')[['merchant_id', 'name', 'category', 'total_volume']])
    
    return profiles_df


def save_merchant_data(merchants_df, profiles_df, data):
    """Save merchant database and enhanced transaction data"""
    print("\n" + "="*60)
    print("SAVING MERCHANT DATA")
    print("="*60)
    
    os.makedirs('data/merchants', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Save merchant database
    merchants_path = 'data/merchants/known_merchants.csv'
    merchants_df.to_csv(merchants_path, index=False)
    print(f"✓ Saved merchants: {merchants_path}")
    
    # Save merchant profiles
    profiles_path = 'data/merchants/merchant_profiles.pkl'
    profiles_df.to_pickle(profiles_path)
    print(f"✓ Saved profiles: {profiles_path}")
    
    # Save enhanced transaction data
    data_path = 'data/processed/data_with_merchants.pkl'
    data.to_pickle(data_path)
    print(f"✓ Saved transaction data: {data_path}")
    
    print(f"\nOutput files:")
    print(f"  - data/merchants/known_merchants.csv (500 merchants)")
    print(f"  - data/merchants/merchant_profiles.pkl (profiles)")
    print(f"  - data/processed/data_with_merchants.pkl (transactions + backward_label)")


def main():
    """Main merchant simulation pipeline"""
    print("\n" + "="*70)
    print("MERCHANT DATABASE SIMULATION")
    print("="*70)
    
    # Step 1: Generate merchant database
    merchants_df = generate_merchant_database(num_merchants=500)
    
    # Step 2: Load transaction data
    print("\n" + "="*60)
    print("LOADING TRANSACTION DATA")
    print("="*60)
    
    data_path = 'data/processed/data_with_graph_features.pkl'
    data = pd.read_pickle(data_path)
    print(f"✓ Loaded data: {data.shape}")
    
    # Step 3: Assign merchants to transactions
    data = assign_merchants_to_transactions(data, merchants_df, assignment_ratio=0.30)
    
    # Step 4: Create backward labels
    data, label_encoder = create_backward_labels(data, merchants_df)
    
    # Step 5: Compute merchant profiles
    profiles_df = compute_merchant_profiles(data, merchants_df)
    
    # Step 6: Save merchant data
    save_merchant_data(merchants_df, profiles_df, data)
    
    # Summary
    print("\n" + "="*70)
    print("MERCHANT SIMULATION COMPLETE")
    print("="*70)
    print(f"Merchants created: {len(merchants_df)}")
    print(f"  E-commerce: {(merchants_df['category'] == 'E-commerce').sum()}")
    print(f"  Gambling: {(merchants_df['category'] == 'Gambling').sum()}")
    print(f"  Dark Web: {(merchants_df['category'] == 'Dark Web').sum()}")
    print(f"  Services: {(merchants_df['category'] == 'Services').sum()}")
    print(f"  Exchanges: {(merchants_df['category'] == 'Exchanges').sum()}")
    
    print(f"\nTransaction-Merchant assignments:")
    print(f"  Direct interactions: {(data['merchant_distance'] == 0).sum():,}")
    print(f"  No merchant: {(data['merchant_distance'] == -1).sum():,}")
    
    print(f"\nNext steps:")
    print(f"  1. Run expand_merchants_khop.py to add k-hop merchant interactions")
    print(f"  2. Run create_criminal_db.py to create criminal entities database")
    print(f"  3. Proceed to model training")


if __name__ == "__main__":
    main()