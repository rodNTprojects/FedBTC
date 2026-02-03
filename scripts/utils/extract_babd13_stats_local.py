#!/usr/bin/env python3
"""
Script: extract_babd13_stats_local.py
Description: Run this script LOCALLY on your machine to extract statistics from BABD-13
             Then upload ONLY the small JSON output (~50KB) instead of the 5.35GB file

Usage:
    1. Place this script in the same folder as babd.zip
    2. Run: python extract_babd13_stats_local.py
    3. Upload the generated 'babd13_extracted_stats.json' to Claude

Author: Rodrigue
Date: 2025-01-19
"""

import os
import sys
import json
import zipfile
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
    from scipy import stats
except ImportError:
    print("❌ Required packages not installed!")
    print("   Run: pip install pandas numpy scipy")
    sys.exit(1)


def find_babd_zip():
    """Find the BABD-13 zip file"""
    search_paths = [
        Path("babd.zip"),
        Path("BABD-13.zip"),
        Path("archive.zip"),
        Path.home() / "Downloads" / "babd.zip",
        Path.home() / "Downloads" / "archive.zip",
    ]
    
    for p in search_paths:
        if p.exists():
            return p
    
    # Search current directory for any zip
    for f in Path(".").glob("*.zip"):
        if "babd" in f.name.lower() or "bitcoin" in f.name.lower():
            return f
    
    return None


def extract_and_load(zip_path):
    """Extract and load the CSV from zip"""
    print(f"\n📦 Opening: {zip_path}")
    print(f"   Size: {zip_path.stat().st_size / (1024**3):.2f} GB")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find main CSV
        csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
        print(f"   Found {len(csv_files)} CSV files")
        
        # Find largest CSV (likely the main dataset)
        main_csv = max(csv_files, key=lambda f: zf.getinfo(f).file_size)
        print(f"   Loading: {main_csv}")
        
        # Load CSV
        with zf.open(main_csv) as f:
            df = pd.read_csv(f)
    
    print(f"   ✅ Loaded: {len(df):,} rows, {len(df.columns)} columns")
    return df


def find_label_column(df):
    """Find the label/category column"""
    candidates = ['label', 'Label', 'category', 'Category', 'type', 'Type', 'class']
    
    for col in candidates:
        if col in df.columns:
            return col
    
    # Find categorical column with reasonable number of unique values
    for col in df.columns:
        if df[col].dtype == 'object' and 5 <= df[col].nunique() <= 20:
            return col
    
    return None


def compute_statistics(df, label_col):
    """Compute statistics for each entity type"""
    print(f"\n📊 Computing statistics by '{label_col}'...")
    
    results = {
        "metadata": {
            "source_file": "BABD-13",
            "total_samples": len(df),
            "n_features": len(df.columns),
            "extracted_at": datetime.now().isoformat()
        },
        "entity_distribution": {},
        "entity_statistics": {},
        "feature_importance": {}
    }
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"   Found {len(numeric_cols)} numeric features")
    
    # Remove label column if numeric
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    
    # Distribution
    for label in df[label_col].unique():
        count = len(df[df[label_col] == label])
        results["entity_distribution"][str(label)] = {
            "count": int(count),
            "percentage": round(100 * count / len(df), 2)
        }
    
    print(f"\n   Entity distribution:")
    for label, info in sorted(results["entity_distribution"].items(), 
                               key=lambda x: x[1]["count"], reverse=True)[:10]:
        print(f"     {label}: {info['count']:,} ({info['percentage']:.1f}%)")
    
    # Statistics by entity
    print(f"\n   Computing per-entity statistics...")
    
    # Focus on most important features (to keep output small)
    # Select top 30 features by variance
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top_features = variances.head(30).index.tolist()
    
    for label in df[label_col].unique():
        entity_df = df[df[label_col] == label]
        
        if len(entity_df) < 10:
            continue
        
        entity_stats = {"n_samples": len(entity_df), "features": {}}
        
        for col in top_features:
            values = entity_df[col].dropna()
            if len(values) > 0:
                entity_stats["features"][col] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75))
                }
        
        results["entity_statistics"][str(label)] = entity_stats
    
    # Feature importance (variance ratio between entities)
    print(f"\n   Computing feature importance...")
    
    for col in top_features[:20]:
        between_var = df.groupby(label_col)[col].mean().var()
        within_var = df.groupby(label_col)[col].var().mean()
        
        if within_var > 0:
            f_ratio = between_var / within_var
            results["feature_importance"][col] = {
                "f_ratio": float(f_ratio),
                "discriminative": f_ratio > 0.1
            }
    
    return results


def create_exchange_profiles(results):
    """Create simplified exchange profiles for direct use"""
    
    # Look for Exchange entity
    exchange_stats = None
    for label, stats in results["entity_statistics"].items():
        if "exchange" in label.lower():
            exchange_stats = stats
            break
    
    if exchange_stats is None:
        print("   ⚠️ No 'Exchange' entity found, using defaults")
        return None
    
    features = exchange_stats["features"]
    
    # Create 3 profiles based on quartiles
    profiles = {}
    
    for feat_name, feat_stats in list(features.items())[:5]:
        q25 = feat_stats["q25"]
        q75 = feat_stats["q75"]
        median = feat_stats["median"]
        std = feat_stats["std"]
        
        profiles[feat_name] = {
            "retail_like": {"mean": q25, "std": std * 0.5},
            "pro_like": {"mean": q75, "std": std * 0.8},
            "mixed": {"mean": median, "std": std}
        }
    
    results["derived_exchange_profiles"] = profiles
    return profiles


def main():
    print("=" * 70)
    print(" BABD-13 LOCAL STATISTICS EXTRACTOR")
    print(" (Generates small JSON for upload instead of 5.35GB file)")
    print("=" * 70)
    
    # Find zip
    zip_path = find_babd_zip()
    
    if zip_path is None:
        print("\n❌ Could not find BABD-13 zip file!")
        print("   Please place babd.zip in the current directory")
        print(f"   Current directory: {Path.cwd()}")
        return False
    
    # Load data
    df = extract_and_load(zip_path)
    
    # Find label column
    label_col = find_label_column(df)
    if label_col is None:
        print("\n❌ Could not find label column!")
        print(f"   Columns: {list(df.columns)[:20]}")
        return False
    
    print(f"\n   Label column: '{label_col}'")
    print(f"   Unique labels: {df[label_col].nunique()}")
    
    # Compute statistics
    results = compute_statistics(df, label_col)
    
    # Create exchange profiles
    create_exchange_profiles(results)
    
    # Save output
    output_path = Path("babd13_extracted_stats.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    file_size = output_path.stat().st_size / 1024
    
    print("\n" + "=" * 70)
    print(" ✅ EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\n   Output: {output_path}")
    print(f"   Size: {file_size:.1f} KB (vs 5.35 GB original!)")
    print(f"\n   📤 Upload this JSON file to Claude for calibration")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
