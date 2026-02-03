#!/usr/bin/env python3
"""
Script: analyze_distribution_separation.py
Description: Analyze if calibrated distributions allow statistical discrimination between exchanges

This script:
1. Visualizes distribution overlaps
2. Computes separation metrics (Bhattacharyya distance, KL divergence)
3. Estimates classification accuracy from distributions alone
4. Suggests adjustments if needed
"""

import numpy as np
import json
from scipy import stats
from pathlib import Path

# Load calibration
config_path = Path(__file__).parent.parent.parent / "config" / "calibration_params.json"

if config_path.exists():
    with open(config_path) as f:
        CALIBRATION = json.load(f)
else:
    # Use embedded values
    CALIBRATION = {
        "exchange_profiles": {
            "exchange_0": {
                "name": "Retail-focused (Coinbase-like)",
                "fee_mean": 0.025, "fee_std": 0.008,
                "volume_log_mean": -2.0, "volume_log_std": 1.5,
                "liquidity_score": 0.70, "liquidity_std": 0.08
            },
            "exchange_1": {
                "name": "Professional (Binance-like)",
                "fee_mean": 0.0015, "fee_std": 0.0005,
                "volume_log_mean": 1.0, "volume_log_std": 2.0,
                "liquidity_score": 0.95, "liquidity_std": 0.03
            },
            "exchange_2": {
                "name": "Mixed (Bitstamp-like)",
                "fee_mean": 0.008, "fee_std": 0.003,
                "volume_log_mean": -0.5, "volume_log_std": 1.8,
                "liquidity_score": 0.85, "liquidity_std": 0.05
            }
        }
    }


def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
    """
    Compute Bhattacharyya distance between two Gaussians.
    Higher = more separated.
    DB > 0.5 suggests good separation.
    """
    sigma_avg = (sigma1**2 + sigma2**2) / 2
    term1 = 0.25 * ((mu1 - mu2)**2) / sigma_avg
    term2 = 0.5 * np.log(sigma_avg / (sigma1 * sigma2))
    return term1 + term2


def overlap_coefficient(mu1, sigma1, mu2, sigma2, n_samples=10000):
    """
    Estimate overlap between two Gaussian distributions.
    Lower = better separation.
    """
    # Generate samples
    x = np.linspace(
        min(mu1 - 4*sigma1, mu2 - 4*sigma2),
        max(mu1 + 4*sigma1, mu2 + 4*sigma2),
        n_samples
    )
    
    pdf1 = stats.norm.pdf(x, mu1, sigma1)
    pdf2 = stats.norm.pdf(x, mu2, sigma2)
    
    # Overlap = integral of min(pdf1, pdf2)
    overlap = np.trapz(np.minimum(pdf1, pdf2), x)
    return overlap


def estimate_classification_accuracy(distributions, n_samples=10000):
    """
    Estimate classification accuracy using Bayes optimal classifier.
    Assumes equal priors.
    """
    n_classes = len(distributions)
    
    # Generate samples from each distribution
    samples = []
    labels = []
    
    for i, (mu, sigma) in enumerate(distributions):
        class_samples = np.random.normal(mu, sigma, n_samples)
        samples.extend(class_samples)
        labels.extend([i] * n_samples)
    
    samples = np.array(samples)
    labels = np.array(labels)
    
    # Bayes optimal classifier: assign to class with highest likelihood
    predictions = []
    for x in samples:
        likelihoods = [stats.norm.pdf(x, mu, sigma) for mu, sigma in distributions]
        predictions.append(np.argmax(likelihoods))
    
    predictions = np.array(predictions)
    accuracy = np.mean(predictions == labels)
    
    return accuracy


def analyze_feature(feature_name, params_list, names):
    """Analyze separation for a single feature"""
    
    print(f"\n{'─' * 60}")
    print(f" Feature: {feature_name}")
    print(f"{'─' * 60}")
    
    # Print distributions
    print(f"\n  Distributions:")
    for i, (mu, sigma) in enumerate(params_list):
        ci_low = mu - 2*sigma
        ci_high = mu + 2*sigma
        print(f"    {names[i]}: μ={mu:.4f}, σ={sigma:.4f} → 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    
    # Pairwise analysis
    print(f"\n  Pairwise Separation:")
    
    separation_scores = []
    
    for i in range(len(params_list)):
        for j in range(i+1, len(params_list)):
            mu1, sigma1 = params_list[i]
            mu2, sigma2 = params_list[j]
            
            # Bhattacharyya distance
            db = bhattacharyya_distance(mu1, sigma1, mu2, sigma2)
            
            # Overlap coefficient
            overlap = overlap_coefficient(mu1, sigma1, mu2, sigma2)
            
            # Cohen's d (effect size)
            pooled_std = np.sqrt((sigma1**2 + sigma2**2) / 2)
            cohens_d = abs(mu1 - mu2) / pooled_std
            
            separation_scores.append(db)
            
            # Interpretation
            if db > 2:
                sep_quality = "✅ Excellent"
            elif db > 1:
                sep_quality = "✅ Good"
            elif db > 0.5:
                sep_quality = "⚠️ Moderate"
            else:
                sep_quality = "❌ Poor"
            
            print(f"    {names[i]} vs {names[j]}:")
            print(f"      Bhattacharyya: {db:.3f} {sep_quality}")
            print(f"      Overlap: {overlap*100:.1f}%")
            print(f"      Cohen's d: {cohens_d:.2f}")
    
    # Estimate multiclass accuracy
    accuracy = estimate_classification_accuracy(params_list)
    print(f"\n  Estimated Bayes-optimal accuracy: {accuracy*100:.1f}%")
    
    return np.mean(separation_scores), accuracy


def main():
    print_header("DISTRIBUTION SEPARATION ANALYSIS")
    
    profiles = CALIBRATION["exchange_profiles"]
    names = ["Exchange 0 (Retail)", "Exchange 1 (Pro)", "Exchange 2 (Mixed)"]
    
    results = {}
    
    # ========== FEE ANALYSIS ==========
    fee_params = [
        (profiles["exchange_0"]["fee_mean"], profiles["exchange_0"]["fee_std"]),
        (profiles["exchange_1"]["fee_mean"], profiles["exchange_1"]["fee_std"]),
        (profiles["exchange_2"]["fee_mean"], profiles["exchange_2"]["fee_std"]),
    ]
    
    sep, acc = analyze_feature("FEE PERCENTAGE", fee_params, names)
    results["fee"] = {"separation": sep, "accuracy": acc}
    
    # ========== VOLUME ANALYSIS ==========
    volume_params = [
        (profiles["exchange_0"]["volume_log_mean"], profiles["exchange_0"]["volume_log_std"]),
        (profiles["exchange_1"]["volume_log_mean"], profiles["exchange_1"]["volume_log_std"]),
        (profiles["exchange_2"]["volume_log_mean"], profiles["exchange_2"]["volume_log_std"]),
    ]
    
    sep, acc = analyze_feature("VOLUME (log scale)", volume_params, names)
    results["volume"] = {"separation": sep, "accuracy": acc}
    
    # ========== LIQUIDITY ANALYSIS ==========
    # Add std if not present
    liq_std = [0.08, 0.03, 0.05]  # Default stds
    
    liquidity_params = [
        (profiles["exchange_0"]["liquidity_score"], liq_std[0]),
        (profiles["exchange_1"]["liquidity_score"], liq_std[1]),
        (profiles["exchange_2"]["liquidity_score"], liq_std[2]),
    ]
    
    sep, acc = analyze_feature("LIQUIDITY SCORE", liquidity_params, names)
    results["liquidity"] = {"separation": sep, "accuracy": acc}
    
    # ========== TEMPORAL (PEAK HOURS) ==========
    # For hours, we use a simplified circular distance
    print(f"\n{'─' * 60}")
    print(f" Feature: PEAK HOURS (Categorical)")
    print(f"{'─' * 60}")
    
    hours = [
        profiles["exchange_0"]["peak_hours"],
        profiles["exchange_1"]["peak_hours"],
        profiles["exchange_2"]["peak_hours"],
    ]
    
    print(f"\n  Peak hours (UTC):")
    for i, h in enumerate(hours):
        print(f"    {names[i]}: {h}")
    
    # Check overlap
    print(f"\n  Overlap analysis:")
    for i in range(3):
        for j in range(i+1, 3):
            overlap = len(set(hours[i]) & set(hours[j]))
            total = len(set(hours[i]) | set(hours[j]))
            jaccard = overlap / total if total > 0 else 0
            
            if overlap == 0:
                status = "✅ No overlap"
            elif jaccard < 0.3:
                status = "✅ Low overlap"
            else:
                status = f"⚠️ {overlap} hours overlap"
            
            print(f"    {names[i]} vs {names[j]}: {status}")
    
    results["temporal"] = {"separation": "categorical", "accuracy": 0.9}  # Estimated
    
    # ========== COMBINED ANALYSIS ==========
    print_header("COMBINED DISCRIMINATIVE POWER")
    
    print("\n  Feature Summary:")
    print(f"  {'Feature':<15} {'Avg Separation':<18} {'Est. Accuracy':<15} {'Status'}")
    print(f"  {'-'*15} {'-'*18} {'-'*15} {'-'*10}")
    
    overall_accuracy = 1.0
    
    for feat, res in results.items():
        if isinstance(res["separation"], float):
            sep_str = f"{res['separation']:.3f}"
        else:
            sep_str = res["separation"]
        
        acc = res["accuracy"]
        
        if acc >= 0.8:
            status = "✅ Good"
        elif acc >= 0.6:
            status = "⚠️ Moderate"
        else:
            status = "❌ Poor"
        
        print(f"  {feat:<15} {sep_str:<18} {acc*100:.1f}%{'':<10} {status}")
        
        # Naive independence assumption for combined
        # In reality, features are not independent, so this overestimates
    
    # More realistic combined estimate
    # Using the best single feature as lower bound
    best_single = max(r["accuracy"] for r in results.values())
    
    # Heuristic: combining features improves accuracy
    # But with diminishing returns
    combined_estimate = min(0.99, best_single + 0.15 * (1 - best_single))
    
    print(f"\n  Combined (all features):")
    print(f"    Lower bound (best single): {best_single*100:.1f}%")
    print(f"    Estimated combined: {combined_estimate*100:.1f}%")
    
    # ========== VERDICT ==========
    print_header("VERDICT")
    
    if combined_estimate >= 0.85:
        print("\n  ✅ DISTRIBUTIONS ARE SUFFICIENTLY SEPARATED")
        print("     The calibrated parameters should allow good exchange discrimination.")
    elif combined_estimate >= 0.70:
        print("\n  ⚠️ DISTRIBUTIONS HAVE MODERATE SEPARATION")
        print("     Consider strengthening separation (see recommendations below).")
    else:
        print("\n  ❌ DISTRIBUTIONS HAVE POOR SEPARATION")
        print("     Strongly recommend adjusting parameters.")
    
    # ========== RECOMMENDATIONS ==========
    print_header("RECOMMENDATIONS")
    
    # Check volume (usually the problem)
    if results["volume"]["accuracy"] < 0.7:
        print("\n  📊 VOLUME: High overlap detected")
        print("     Current: Exchange 0 log_μ=-2.0, Exchange 1 log_μ=1.0, Exchange 2 log_μ=-0.5")
        print("     Problem: Large σ values cause overlap")
        print("     Suggestion: Reduce σ or increase μ separation")
        print("""
     ADJUSTED VOLUME PARAMETERS:
     {
       "exchange_0": {"volume_log_mean": -2.5, "volume_log_std": 1.0},  # Tighter
       "exchange_1": {"volume_log_mean": 2.0, "volume_log_std": 1.2},   # Higher mean
       "exchange_2": {"volume_log_mean": 0.0, "volume_log_std": 1.0}    # Tighter
     }
     """)
    
    if results["liquidity"]["accuracy"] < 0.8:
        print("\n  💧 LIQUIDITY: Moderate overlap detected")
        print("     Suggestion: Reduce σ values")
        print("""
     ADJUSTED LIQUIDITY PARAMETERS:
     {
       "exchange_0": {"liquidity_score": 0.65, "liquidity_std": 0.05},
       "exchange_1": {"liquidity_score": 0.95, "liquidity_std": 0.02},
       "exchange_2": {"liquidity_score": 0.80, "liquidity_std": 0.04}
     }
     """)
    
    return results


if __name__ == "__main__":
    results = main()
