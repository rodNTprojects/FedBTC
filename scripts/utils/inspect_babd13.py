#!/usr/bin/env python3
"""
Script: inspect_babd13.py
Description: Quick inspection of BABD-13 zip file contents and structure
Usage: python inspect_babd13.py [path_to_babd.zip]
"""

import sys
import zipfile
import pandas as pd
from pathlib import Path
import io


def inspect_zip(zip_path):
    """Inspect contents of BABD-13 zip"""
    print(f"\n📦 Inspecting: {zip_path}")
    print("=" * 60)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # List all files
        files = zf.namelist()
        print(f"\n📁 Contents ({len(files)} files):")
        
        total_size = 0
        csv_files = []
        
        for f in files[:20]:  # Show first 20
            info = zf.getinfo(f)
            size_mb = info.file_size / (1024 * 1024)
            total_size += info.file_size
            print(f"   {f} ({size_mb:.2f} MB)")
            
            if f.endswith('.csv'):
                csv_files.append(f)
        
        if len(files) > 20:
            print(f"   ... and {len(files) - 20} more files")
        
        print(f"\n📊 Total uncompressed size: {total_size / (1024*1024):.1f} MB")
        
        # Inspect main CSV
        if csv_files:
            main_csv = csv_files[0]
            # Find largest CSV
            for f in csv_files:
                if zf.getinfo(f).file_size > zf.getinfo(main_csv).file_size:
                    main_csv = f
            
            print(f"\n📄 Inspecting main CSV: {main_csv}")
            print("-" * 60)
            
            with zf.open(main_csv) as f:
                # Read first few lines to understand structure
                try:
                    df = pd.read_csv(f, nrows=1000)
                except UnicodeDecodeError:
                    f.seek(0)
                    df = pd.read_csv(f, nrows=1000, encoding='latin-1')
                
                print(f"\n   Shape (first 1000 rows): {df.shape}")
                print(f"\n   Columns ({len(df.columns)}):")
                
                # Categorize columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                object_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                print(f"     - Numeric: {len(numeric_cols)}")
                print(f"     - Object/String: {len(object_cols)}")
                
                # Show first 20 columns
                print(f"\n   First 20 columns:")
                for i, col in enumerate(df.columns[:20]):
                    dtype = str(df[col].dtype)
                    sample = str(df[col].iloc[0])[:30]
                    print(f"     {i+1}. {col} ({dtype}): {sample}...")
                
                # Find potential label column
                print(f"\n   🏷️ Potential label columns:")
                for col in df.columns:
                    if df[col].dtype == 'object' and df[col].nunique() < 20:
                        print(f"     - {col}: {df[col].nunique()} unique values")
                        print(f"       Values: {df[col].unique()[:10].tolist()}")
                
                # Show sample data
                print(f"\n   📋 Sample data (first 3 rows):")
                print(df.head(3).to_string())
        
        return True


def main():
    # Find BABD zip
    search_paths = [
        Path(sys.argv[1]) if len(sys.argv) > 1 else None,
        Path("babd.zip"),
        Path("data/external/babd.zip"),
        Path("/mnt/user-data/uploads/babd.zip"),
        Path.home() / "babd.zip"
    ]
    
    zip_path = None
    for p in search_paths:
        if p and p.exists():
            zip_path = p
            break
    
    if zip_path is None:
        print("❌ BABD-13 zip not found!")
        print("\nSearched in:")
        for p in search_paths:
            if p:
                print(f"   - {p}")
        print("\nUsage: python inspect_babd13.py [path_to_babd.zip]")
        return False
    
    return inspect_zip(zip_path)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
