#!/usr/bin/env python3
"""
Script: p1_data_preparation.py
Description: Phase 1 - Complete Data Preparation Pipeline (v2 with BABD-13 calibration)
Author: Rodrigue
Date: 2025-01-19

Pipeline Steps:
  0. calibrate_from_babd13.py    - Extract distributions from BABD-13
  1. preprocess_elliptic.py      - Preprocess Elliptic dataset
  2. build_temporal_graph.py     - Build temporal graphs
  3. simulate_merchants.py       - Simulate merchants (with calibrated params)
  4. expand_merchants_khop.py    - BFS k-hop expansion
  5. precompute_merchant_embeddings.py - Precompute embeddings
  6. create_criminal_db.py       - Create criminal database
  7. split_merchants_known_unknown.py - Split 90/10 known/unknown
  8. partition_federated.py      - Partition for K=3 exchanges
  9. add_hybrid_features_elliptic.py - Add calibrated proxy features
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "preprocessing"
CONFIG_DIR = PROJECT_ROOT / "config"


def print_header(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def print_step(step_num, total, script_name, description):
    """Print step information"""
    print(f"\n{'─' * 70}")
    print(f" STEP {step_num}/{total}: {script_name}")
    print(f" {description}")
    print(f"{'─' * 70}\n")


def run_script(script_path, description, step_num, total):
    """Run a Python script with proper error handling"""
    print_step(step_num, total, script_path.name, description)
    
    if not script_path.exists():
        print(f"⚠️  Script not found: {script_path}")
        print(f"   Creating placeholder...")
        create_placeholder_script(script_path, description)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        elapsed = time.time() - start_time
        print(f"\n✅ SUCCESS - Completed in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ FAILED after {elapsed:.1f}s (exit code: {e.returncode})")
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def create_placeholder_script(script_path, description):
    """Create a placeholder script if it doesn't exist"""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    content = f'''#!/usr/bin/env python3
"""
Script: {script_path.name}
Description: {description}
Status: PLACEHOLDER - To be implemented

TODO: Implement this script
"""

import sys

def main():
    print(f"⚠️  PLACEHOLDER: {{__file__}}")
    print(f"   Description: {description}")
    print(f"   Status: Not yet implemented")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"   Created: {script_path}")


def check_prerequisites():
    """Check if prerequisites are met"""
    print("\n🔍 Checking prerequisites...")
    
    issues = []
    
    # Check for BABD-13 zip file
    babd_locations = [
        PROJECT_ROOT / "data" / "external" / "babd.zip",
        PROJECT_ROOT / "babd.zip",
        Path("/mnt/user-data/uploads/babd.zip")
    ]
    
    babd_found = any(p.exists() for p in babd_locations)
    if not babd_found:
        issues.append(f"BABD-13 zip not found. Expected at:\n" + 
                     "\n".join(f"     - {p}" for p in babd_locations))
    else:
        print("   ✅ BABD-13 zip found")
    
    # Check for Elliptic dataset
    elliptic_locations = [
        PROJECT_ROOT / "data" / "raw" / "elliptic_bitcoin_dataset",
        PROJECT_ROOT / "data" / "elliptic_bitcoin_dataset",
        Path.home() / "elliptic_bitcoin_dataset"
    ]
    
    elliptic_found = any(p.exists() for p in elliptic_locations)
    if not elliptic_found:
        issues.append(f"Elliptic dataset not found. Expected at:\n" +
                     "\n".join(f"     - {p}" for p in elliptic_locations))
    else:
        print("   ✅ Elliptic dataset found")
    
    # Check Python packages
    required_packages = ['pandas', 'numpy', 'scipy', 'torch', 'torch_geometric']
    missing_packages = []
    
    for pkg in required_packages:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing_packages.append(pkg)
    
    if missing_packages:
        issues.append(f"Missing Python packages: {', '.join(missing_packages)}")
    else:
        print("   ✅ Required Python packages installed")
    
    if issues:
        print("\n⚠️  Prerequisites issues found:")
        for issue in issues:
            print(f"\n   {issue}")
        return False
    
    return True


def main():
    """Main orchestration"""
    print_header("PHASE 1: DATA PREPARATION (v2 - BABD-13 Calibrated)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")
    
    # Define pipeline steps
    # Format: (script_name, description, required)
    steps = [
        # Step 0: Calibration from Literature (No 5.35GB download needed!)
        ('calibration_from_literature.py', 
         'Generate calibration from published research (IEEE TIFS, CoinMarketCap)',
         True),
        
        # Step 1: Preprocess Elliptic
        ('preprocess_elliptic.py',
         'Load and preprocess Elliptic dataset (203K transactions)',
         True),
        
        # Step 2: Build temporal graphs
        ('build_temporal_graph.py',
         'Build temporal transaction graphs (49 timesteps)',
         True),
        
        # Step 3: Simulate merchants
        ('simulate_merchants.py',
         'Simulate 500 merchant entities with calibrated patterns',
         True),
        
        # Step 4: K-hop expansion
        ('expand_merchants_khop.py',
         'BFS k-hop expansion from merchant seeds',
         True),
        
        # Step 5: Precompute embeddings
        ('precompute_merchant_embeddings.py',
         'Precompute graph embeddings for merchants',
         True),
        
        # Step 6: Create criminal database
        ('create_criminal_db.py',
         'Create criminal transaction database',
         True),
        
        # Step 7: Split known/unknown merchants
        ('split_merchants_known_unknown.py',
         'Split merchants into 90% known, 10% unknown',
         True),
        
        # Step 8: Partition for federated learning
        ('partition_federated.py',
         'Partition data for K=3 exchanges (AFTER merchant split)',
         True),
        
        # Step 9: Add hybrid features
        ('add_hybrid_features_elliptic.py',
         'Add BABD-13 calibrated proxy features',
         True),
    ]
    
    total_steps = len(steps)
    
    # Check prerequisites
    prereq_ok = check_prerequisites()
    if not prereq_ok:
        print("\n" + "=" * 70)
        print(" ⚠️  PREREQUISITES NOT MET - Some steps may fail")
        print("=" * 70)
        response = input("\nContinue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("\nAborted.")
            return
    
    # Execute pipeline
    print("\n" + "=" * 70)
    print(" STARTING PIPELINE")
    print("=" * 70)
    
    results = []
    
    for i, (script_name, description, required) in enumerate(steps):
        script_path = SCRIPTS_DIR / script_name
        
        success = run_script(script_path, description, i, total_steps - 1)
        results.append((script_name, success))
        
        if not success and required:
            print(f"\n{'=' * 70}")
            print(f" ❌ PIPELINE FAILED at Step {i}: {script_name}")
            print(f"{'=' * 70}")
            
            # Show summary
            print("\n📊 Execution Summary:")
            for name, status in results:
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {name}")
            
            print(f"\nNext: Fix {script_name} and re-run pipeline")
            return
    
    # Success
    print_header("✅ PHASE 1 COMPLETE")
    
    print("📊 Execution Summary:")
    for name, status in results:
        status_icon = "✅" if status else "⚠️"
        print(f"   {status_icon} {name}")
    
    print(f"\n📁 Outputs created:")
    print(f"   - config/calibration_params.json (BABD-13 calibration)")
    print(f"   - config/calibration_report.md")
    print(f"   - data/processed/... (Elliptic features)")
    print(f"   - data/federated/exchange_{{0,1,2}}_enriched.pkl")
    
    print(f"\n🚀 Next: python p2_baseline_training.py")


if __name__ == "__main__":
    main()
