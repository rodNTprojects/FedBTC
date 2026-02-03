"""
Script: create_criminal_db.py
Description: Create criminal entities database
Author: Rodrigue
Date: 2025-01-03

UTILITÉ DE CE FICHIER:
======================
Ce fichier crée une base de données d'entités criminelles CONNUES.

Pourquoi c'est important:
1. Investigation Context: Quand backward attribution identifie un merchant,
   on peut vérifier s'il est dans la base criminelle connue
   
2. Alerte Automatique: Si merchant détecté = criminal → Flag HIGH PRIORITY
   
3. Cross-référencement: Permet de lier transactions suspectes à entités
   criminelles connues (dark web markets, ransomware groups, etc.)
   
4. Validation: Mesurer si backward attribution détecte correctement les
   merchants criminels connus

Exemple workflow:
  Transaction T → Backward attribution → Merchant "DarkMarket_X"
  → Check criminal_entities.csv → MATCH! → Alert law enforcement

C'est comme une "watchlist" de merchants criminels.
"""

import os
import sys
import numpy as np
import pandas as pd
from faker import Faker

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def create_criminal_entities_database():
    """
    Create criminal entities database from known criminal merchants
    
    This database contains:
    - Known dark web markets
    - Ransomware groups
    - Scam operations
    - Money laundering services
    
    Returns:
        criminal_entities_df: DataFrame with criminal entity metadata
    """
    print("\n" + "="*60)
    print("CREATING CRIMINAL ENTITIES DATABASE")
    print("="*60)
    
    # Load merchants
    merchants_df = pd.read_csv('data/merchants/known_merchants.csv')
    
    # Filter criminal merchants (known_criminal = True)
    criminal_merchants = merchants_df[merchants_df['known_criminal'] == True].copy()
    
    print(f"Found {len(criminal_merchants)} criminal merchants")
    
    fake = Faker()
    Faker.seed(42)
    
    # Create criminal entities
    criminal_entities = []
    
    for _, merchant in criminal_merchants.iterrows():
        # Assign criminal type
        criminal_type = np.random.choice([
            'Dark Web Market',
            'Ransomware',
            'Scam',
            'Money Laundering Service',
            'Illicit Exchange'
        ], p=[0.50, 0.20, 0.15, 0.10, 0.05])
        
        # Known since (when discovered)
        known_since = f'20{np.random.randint(15, 24)}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}'
        
        # Source of information
        source = np.random.choice([
            'Interpol',
            'FBI',
            'Europol',
            'Academic Research',
            'Blockchain Analysis',
            'Law Enforcement Collaboration'
        ])
        
        # Create entity
        entity = {
            'entity_id': merchant['merchant_id'],
            'name': merchant['name'],
            'type': criminal_type,
            'category': merchant['category'],
            'geographic_region': merchant['geographic_region'],
            'known_since': known_since,
            'source': source,
            'threat_level': np.random.choice(['HIGH', 'CRITICAL'], p=[0.6, 0.4]),
            'estimated_volume_btc': np.random.lognormal(mean=2, sigma=1.5),  # Estimated criminal volume
            'status': np.random.choice(['Active', 'Shutdown', 'Under Investigation'], p=[0.3, 0.4, 0.3])
        }
        
        criminal_entities.append(entity)
    
    criminal_entities_df = pd.DataFrame(criminal_entities)
    
    print(f"\n✓ Created {len(criminal_entities_df)} criminal entities")
    print(f"\nBy type:")
    print(criminal_entities_df['type'].value_counts())
    
    print(f"\nBy threat level:")
    print(criminal_entities_df['threat_level'].value_counts())
    
    print(f"\nBy status:")
    print(criminal_entities_df['status'].value_counts())
    
    return criminal_entities_df


def save_criminal_database(criminal_entities_df):
    """Save criminal entities database"""
    print("\n" + "="*60)
    print("SAVING CRIMINAL DATABASE")
    print("="*60)
    
    os.makedirs('data/merchants', exist_ok=True)
    
    output_path = 'data/merchants/criminal_entities.csv'
    criminal_entities_df.to_csv(output_path, index=False)
    
    print(f"✓ Saved: {output_path}")
    print(f"  Entities: {len(criminal_entities_df)}")
    
    print(f"\nUSAGE:")
    print(f"  This database is used during investigation:")
    print(f"  1. Backward attribution identifies merchant M")
    print(f"  2. Check if M in criminal_entities.csv")
    print(f"  3. If MATCH → Alert HIGH PRIORITY + provide entity details")
    print(f"  4. Cross-reference with law enforcement databases")


def main():
    """Main pipeline"""
    print("\n" + "="*70)
    print("CRIMINAL ENTITIES DATABASE CREATION")
    print("="*70)
    
    print("\nPURPOSE:")
    print("  Create watchlist of known criminal merchants for:")
    print("  - Automatic flagging during investigation")
    print("  - Cross-referencing with backward attribution")
    print("  - Validation of detection performance")
    print("  - Law enforcement alerts")
    
    # Step 1: Create criminal entities
    criminal_entities_df = create_criminal_entities_database()
    
    # Step 2: Save database
    save_criminal_database(criminal_entities_df)
    
    # Summary
    print("\n" + "="*70)
    print("CRIMINAL DATABASE COMPLETE")
    print("="*70)
    print(f"Created watchlist of {len(criminal_entities_df)} criminal entities")
    
    print(f"\nExample usage in investigation:")
    print(f"  >>> import pandas as pd")
    print(f"  >>> criminals = pd.read_csv('data/merchants/criminal_entities.csv')")
    print(f"  >>> merchant_detected = 'MERCH_042'")
    print(f"  >>> match = criminals[criminals['entity_id'] == merchant_detected]")
    print(f"  >>> if len(match) > 0:")
    print(f"  >>>     print('🚨 CRIMINAL ENTITY DETECTED!')")
    print(f"  >>>     print(match[['name', 'type', 'threat_level']])")


if __name__ == "__main__":
    main()