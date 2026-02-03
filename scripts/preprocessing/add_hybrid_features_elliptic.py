"""
================================================================================
Script: add_hybrid_features_elliptic.py (v4 - Proxy Features Only)
Description: Génération des 20 Features Proxy UNIQUEMENT (calibrées depuis littérature)
Author: Rodrigue
Date: Janvier 2026

MODIFICATIONS v4:
- SUPPRESSION des features dérivées (polynomiales, temporelles, graph)
- Conservation UNIQUEMENT des 20 features proxy calibrées
- Cohérent avec l'article Table 2 (166 + 20 = 186 features)

FEATURES PROXY (20 total, calibrées depuis littérature):
1. Fee Proxy (5):      fee_percentage, fee_tier, fee_volatility, fee_log, fee_normalized
2. Volume Proxy (5):   volume_btc, volume_class, volume_percentile, volume_log, volume_sqrt
3. Temporal Proxy (6): synthetic_hour, peak_activity_prob, timezone_proxy, hour_sin, hour_cos, is_business_hours
4. Liquidity Proxy (4): liquidity_score, processing_speed, reliability_index, exchange_quality

CALIBRATION SOURCES:
- Fee: CoinMarketCap 2024-2025 [4]
- Volume: BABD-13 (Xiang et al., IEEE TIFS 2024) [3], Zhou et al. PRE 2023 [8]
- Temporal: Juhász et al. 2018 [10], Coin Metrics 2024 [5]
- Liquidity: Kaiko Exchange Benchmark 2024 [12]

OUTPUT: 166 original + 20 proxy = 186 features (as per article Table 2)

USAGE:
    python add_hybrid_features_elliptic.py --input_dir data/federated --output_dir data/federated_enriched
    python add_hybrid_features_elliptic.py --config config/calibration_params.json
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, pearsonr, truncnorm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Tuple, Optional
import warnings
import json
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION - LOADED FROM CALIBRATION FILE
# ============================================================================

def load_calibration_config(config_path: str = None) -> dict:
    """
    Load calibration parameters from JSON file.
    
    Args:
        config_path: Path to calibration_params.json
        
    Returns:
        Calibration configuration dictionary
    """
    search_paths = [
        config_path,
        'config/calibration_params.json',
        '../config/calibration_params.json',
        '../../config/calibration_params.json',
        Path(__file__).parent.parent / 'config' / 'calibration_params.json',
    ]
    
    for path in search_paths:
        if path and os.path.exists(path):
            print(f"✓ Loading calibration from: {path}")
            with open(path, 'r') as f:
                return json.load(f)
    
    print("⚠ No calibration file found, using default parameters")
    return None


class Config:
    """
    Configuration for proxy feature engineering.
    Loads parameters from calibration_params.json for literature-calibrated distributions.
    """
    
    # Feature names (constant) - exactly 20 proxy features
    PROXY_FEE_FEATURES = ['fee_percentage', 'fee_tier', 'fee_volatility', 'fee_log', 'fee_normalized']
    PROXY_VOLUME_FEATURES = ['volume_btc', 'volume_class', 'volume_percentile', 'volume_log', 'volume_sqrt']
    PROXY_TEMPORAL_FEATURES = ['synthetic_hour', 'peak_activity_prob', 'timezone_proxy', 'hour_sin', 'hour_cos', 'is_business_hours']
    PROXY_LIQUIDITY_FEATURES = ['liquidity_score', 'processing_speed', 'reliability_index', 'exchange_quality']
    
    def __init__(self, calibration_path: str = None):
        """
        Initialize configuration, optionally loading from calibration file.
        """
        self.calibration = load_calibration_config(calibration_path)
        self.EXCHANGE_PROFILES = self._build_exchange_profiles()
        
        # Validation thresholds
        self.CORRELATION_THRESHOLD = 0.3
        self.ANOVA_PVALUE_THRESHOLD = 0.05
        self.RANDOM_SEED = 42
        
        # Feature groups - PROXY ONLY (no derived features)
        self.FEATURE_GROUPS = {
            'original': [],
            'proxy_fee': [],
            'proxy_volume': [],
            'proxy_temporal': [],
            'proxy_liquidity': []
        }
        
        self._print_calibration_summary()
    
    def _build_exchange_profiles(self) -> Dict:
        """Build exchange profiles from calibration file or use defaults."""
        if self.calibration and 'exchange_profiles' in self.calibration:
            profiles = {}
            proxy_features = self.calibration.get('proxy_features', {})
            
            for i in range(3):
                key = f'exchange_{i}'
                cal = self.calibration['exchange_profiles'].get(key, {})
                
                fee_cal = proxy_features.get('fee', {}).get(f'exchange_{i}_{"retail" if i == 0 else "pro" if i == 1 else "mixed"}', {})
                vol_cal = proxy_features.get('volume', {}).get(f'exchange_{i}_{"retail" if i == 0 else "pro" if i == 1 else "mixed"}', {})
                temp_cal = proxy_features.get('temporal', {}).get(f'exchange_{i}_{"retail" if i == 0 else "pro" if i == 1 else "mixed"}', {})
                liq_cal = proxy_features.get('liquidity', {}).get(f'exchange_{i}_{"retail" if i == 0 else "pro" if i == 1 else "mixed"}', {})
                
                profiles[i] = {
                    'name': cal.get('name', f'Exchange {i}'),
                    'fee_mean': cal.get('fee_mean', fee_cal.get('mean', [0.025, 0.0015, 0.008][i])),
                    'fee_std': cal.get('fee_std', fee_cal.get('std', [0.008, 0.0005, 0.003][i])),
                    'fee_min': fee_cal.get('min', [0.01, 0.0003, 0.003][i]),
                    'fee_max': fee_cal.get('max', [0.045, 0.003, 0.015][i]),
                    'volume_log_mean': cal.get('volume_log_mean', vol_cal.get('log_mean', [-3.0, 2.0, 0.0][i])),
                    'volume_log_std': cal.get('volume_log_std', vol_cal.get('log_std', [0.8, 1.0, 0.9][i])),
                    'peak_hours': cal.get('peak_hours', temp_cal.get('peak_hours', [[14,15,16,17,18,19,20,21], [0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15]][i])),
                    'activity_concentration': temp_cal.get('activity_concentration', [0.7, 0.6, 0.65][i]),
                    'timezone_label': temp_cal.get('timezone_label', ['US', 'Asia', 'Europe'][i]),
                    'liquidity_score': cal.get('liquidity_score', liq_cal.get('score', [0.65, 0.95, 0.80][i])),
                    'liquidity_std': liq_cal.get('std', [0.05, 0.02, 0.04][i]),
                    'processing_speed': cal.get('processing_speed', liq_cal.get('processing_speed', [0.5, 1.75, 1.0][i])),
                    'user_type': ['retail', 'professional', 'mixed'][i],
                }
            return profiles
        else:
            # Default profiles (fallback)
            return {
                0: {
                    'name': 'Retail Exchange (Coinbase-like)',
                    'fee_mean': 0.025, 'fee_std': 0.008, 'fee_min': 0.01, 'fee_max': 0.045,
                    'volume_log_mean': -3.0, 'volume_log_std': 0.8,
                    'peak_hours': [14, 15, 16, 17, 18, 19, 20, 21],
                    'activity_concentration': 0.7, 'timezone_label': 'US',
                    'liquidity_score': 0.65, 'liquidity_std': 0.05, 'processing_speed': 0.5,
                    'user_type': 'retail'
                },
                1: {
                    'name': 'Pro Exchange (Binance-like)',
                    'fee_mean': 0.0015, 'fee_std': 0.0005, 'fee_min': 0.0003, 'fee_max': 0.003,
                    'volume_log_mean': 2.0, 'volume_log_std': 1.0,
                    'peak_hours': [0, 1, 2, 3, 4, 5, 6, 7],
                    'activity_concentration': 0.6, 'timezone_label': 'Asia',
                    'liquidity_score': 0.95, 'liquidity_std': 0.02, 'processing_speed': 1.75,
                    'user_type': 'professional'
                },
                2: {
                    'name': 'Mixed Exchange (Bitstamp-like)',
                    'fee_mean': 0.008, 'fee_std': 0.003, 'fee_min': 0.003, 'fee_max': 0.015,
                    'volume_log_mean': 0.0, 'volume_log_std': 0.9,
                    'peak_hours': [8, 9, 10, 11, 12, 13, 14, 15],
                    'activity_concentration': 0.65, 'timezone_label': 'Europe',
                    'liquidity_score': 0.80, 'liquidity_std': 0.04, 'processing_speed': 1.0,
                    'user_type': 'mixed'
                }
            }
    
    def _print_calibration_summary(self):
        """Print summary of calibration parameters"""
        print("\n" + "="*70)
        print("CALIBRATION SUMMARY (PROXY FEATURES ONLY)")
        print("="*70)
        
        if self.calibration:
            meta = self.calibration.get('metadata', {})
            print(f"Source: {meta.get('source', 'Unknown')}")
            print(f"Version: {meta.get('version', 'Unknown')}")
        else:
            print("Source: Default parameters (no calibration file)")
        
        print("\nFeatures to generate: 20 proxy features")
        print(f"  - Fee:       {len(self.PROXY_FEE_FEATURES)} features")
        print(f"  - Volume:    {len(self.PROXY_VOLUME_FEATURES)} features")
        print(f"  - Temporal:  {len(self.PROXY_TEMPORAL_FEATURES)} features")
        print(f"  - Liquidity: {len(self.PROXY_LIQUIDITY_FEATURES)} features")
        
        print("\nExchange Profiles:")
        for i, profile in self.EXCHANGE_PROFILES.items():
            print(f"\n  Exchange {i}: {profile['name']}")
            print(f"    Fee:      μ={profile['fee_mean']:.4f}, σ={profile['fee_std']:.4f}")
            print(f"    Volume:   log_μ={profile['volume_log_mean']:.1f}, log_σ={profile['volume_log_std']:.1f} → median={np.exp(profile['volume_log_mean']):.2f} BTC")
            print(f"    Peak:     {profile['peak_hours'][:3]}...{profile['peak_hours'][-1]} UTC ({profile['timezone_label']})")
            print(f"    Liquidity: {profile['liquidity_score']:.2f}")


# ============================================================================
# PROXY FEATURES GENERATOR (Calibrated from Literature)
# ============================================================================

class ProxyFeatureGenerator:
    """
    Generate proxy features based on CALIBRATED exchange characteristics.
    
    CALIBRATION SOURCES:
    - Fee: CoinMarketCap 2024-2025 [4]
    - Volume: BABD-13 (Xiang et al., IEEE TIFS 2024) [3], Zhou et al. PRE 2023 [8]
    - Temporal: Juhász et al. 2018 [10], Coin Metrics 2024 [5]
    - Liquidity: Kaiko Exchange Benchmark 2024 [12]
    """
    
    def __init__(self, config: Config):
        self.config = config
        np.random.seed(config.RANDOM_SEED)
        
    def _get_exchange_column(self, df: pd.DataFrame, exchange_col: str) -> str:
        """Find the actual exchange/forward label column"""
        candidates = [exchange_col, 'forward_label', 'exchange_partition', 
                     'exchange_id', 'partition', 'exchange']
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _generate_truncated_normal(self, mean: float, std: float, 
                                   low: float, high: float, size: int) -> np.ndarray:
        """Generate samples from truncated normal distribution."""
        if std <= 0:
            return np.full(size, mean)
        a = (low - mean) / std
        b = (high - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
        
    def generate_fee_proxy_features(self, df: pd.DataFrame, 
                                    exchange_col: str = 'forward_label') -> pd.DataFrame:
        """
        Generate fee structure proxy features (5 features).
        Source: CoinMarketCap 2024-2025 [4]
        """
        print("\n  [1/4] Generating Fee Proxy Features (5 features)...")
        
        actual_col = self._get_exchange_column(df, exchange_col)
        if actual_col is None:
            df['_temp_exchange'] = np.random.randint(0, 3, len(df))
            actual_col = '_temp_exchange'
        else:
            print(f"       Using column: {actual_col}")
        
        df['fee_percentage'] = 0.0
        df['fee_tier'] = 0
        df['fee_volatility'] = 0.0
        
        for exchange_id, profile in self.config.EXCHANGE_PROFILES.items():
            mask = df[actual_col] == exchange_id
            n_samples = mask.sum()
            if n_samples == 0:
                continue
            
            fees = self._generate_truncated_normal(
                mean=profile['fee_mean'],
                std=profile['fee_std'],
                low=profile['fee_min'],
                high=profile['fee_max'],
                size=n_samples
            )
            df.loc[mask, 'fee_percentage'] = fees
            
            if profile['fee_mean'] > 0.01:
                df.loc[mask, 'fee_tier'] = 2
            elif profile['fee_mean'] > 0.002:
                df.loc[mask, 'fee_tier'] = 1
            else:
                df.loc[mask, 'fee_tier'] = 0
            
            df.loc[mask, 'fee_volatility'] = profile['fee_std'] / max(profile['fee_mean'], 1e-10)
        
        df['fee_log'] = np.log1p(df['fee_percentage'] * 1000)
        df['fee_normalized'] = (df['fee_percentage'] - df['fee_percentage'].mean()) / \
                               (df['fee_percentage'].std() + 1e-10)
        
        if '_temp_exchange' in df.columns:
            df.drop('_temp_exchange', axis=1, inplace=True)
        
        self.config.FEATURE_GROUPS['proxy_fee'] = list(Config.PROXY_FEE_FEATURES)
        print(f"       ✓ Generated: {Config.PROXY_FEE_FEATURES}")
        return df
    
    def generate_volume_proxy_features(self, df: pd.DataFrame,
                                       exchange_col: str = 'forward_label') -> pd.DataFrame:
        """
        Generate volume pattern proxy features (5 features).
        Source: BABD-13 [3], Zhou et al. PRE 2023 [8]
        """
        print("\n  [2/4] Generating Volume Proxy Features (5 features)...")
        
        actual_col = self._get_exchange_column(df, exchange_col)
        if actual_col is None:
            df['_temp_exchange'] = np.random.randint(0, 3, len(df))
            actual_col = '_temp_exchange'
        
        df['volume_btc'] = 1.0
        df['volume_class'] = 0
        df['volume_percentile'] = 0.5
        
        for exchange_id, profile in self.config.EXCHANGE_PROFILES.items():
            mask = df[actual_col] == exchange_id
            n_samples = mask.sum()
            if n_samples == 0:
                continue
            
            log_mean = profile['volume_log_mean']
            log_std = profile['volume_log_std']
            volume_btc = np.random.lognormal(mean=log_mean, sigma=log_std, size=n_samples)
            volume_btc = np.clip(volume_btc, 0.0001, 1000.0)
            df.loc[mask, 'volume_btc'] = volume_btc
            
            if profile['user_type'] == 'retail':
                df.loc[mask, 'volume_class'] = 0
            elif profile['user_type'] == 'professional':
                df.loc[mask, 'volume_class'] = 2
            else:
                df.loc[mask, 'volume_class'] = 1
        
        df['volume_percentile'] = df.groupby(actual_col)['volume_btc'].rank(pct=True)
        df['volume_log'] = np.log1p(df['volume_btc'])
        df['volume_sqrt'] = np.sqrt(df['volume_btc'])
        
        if '_temp_exchange' in df.columns:
            df.drop('_temp_exchange', axis=1, inplace=True)
        
        self.config.FEATURE_GROUPS['proxy_volume'] = list(Config.PROXY_VOLUME_FEATURES)
        print(f"       ✓ Generated: {Config.PROXY_VOLUME_FEATURES}")
        return df
    
    def generate_temporal_proxy_features(self, df: pd.DataFrame,
                                         exchange_col: str = 'forward_label') -> pd.DataFrame:
        """
        Generate temporal activity proxy features (6 features).
        Source: Juhász et al. 2018 [10], Coin Metrics 2024 [5]
        """
        print("\n  [3/4] Generating Temporal Proxy Features (6 features)...")
        
        actual_col = self._get_exchange_column(df, exchange_col)
        if actual_col is None:
            df['_temp_exchange'] = np.random.randint(0, 3, len(df))
            actual_col = '_temp_exchange'
        
        df['synthetic_hour'] = 12
        df['peak_activity_prob'] = 0.5
        df['timezone_proxy'] = 0
        
        for exchange_id, profile in self.config.EXCHANGE_PROFILES.items():
            mask = df[actual_col] == exchange_id
            n_samples = mask.sum()
            if n_samples == 0:
                continue
            
            peak_hours = profile['peak_hours']
            concentration = profile.get('activity_concentration', 0.7)
            
            n_peak = int(concentration * n_samples)
            n_random = n_samples - n_peak
            
            peak_samples = np.random.choice(peak_hours, n_peak)
            random_samples = np.random.randint(0, 24, n_random)
            hours = np.concatenate([peak_samples, random_samples])
            np.random.shuffle(hours)
            
            df.loc[mask, 'synthetic_hour'] = hours
            df.loc[mask, 'peak_activity_prob'] = df.loc[mask, 'synthetic_hour'].apply(
                lambda h: 0.85 if h in peak_hours else 0.25
            )
            
            tz_map = {'Asia': 0, 'Europe': 1, 'US': 2}
            df.loc[mask, 'timezone_proxy'] = tz_map.get(profile['timezone_label'], 1)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['synthetic_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['synthetic_hour'] / 24)
        df['is_business_hours'] = ((df['synthetic_hour'] >= 9) & 
                                   (df['synthetic_hour'] <= 17)).astype(int)
        
        if '_temp_exchange' in df.columns:
            df.drop('_temp_exchange', axis=1, inplace=True)
        
        self.config.FEATURE_GROUPS['proxy_temporal'] = list(Config.PROXY_TEMPORAL_FEATURES)
        print(f"       ✓ Generated: {Config.PROXY_TEMPORAL_FEATURES}")
        return df
    
    def generate_liquidity_proxy_features(self, df: pd.DataFrame,
                                          exchange_col: str = 'forward_label') -> pd.DataFrame:
        """
        Generate liquidity proxy features (4 features).
        Source: Kaiko Exchange Benchmark 2024 [12]
        """
        print("\n  [4/4] Generating Liquidity Proxy Features (4 features)...")
        
        actual_col = self._get_exchange_column(df, exchange_col)
        if actual_col is None:
            df['_temp_exchange'] = np.random.randint(0, 3, len(df))
            actual_col = '_temp_exchange'
        
        df['liquidity_score'] = 0.8
        df['processing_speed'] = 1.0
        df['reliability_index'] = 0.9
        
        for exchange_id, profile in self.config.EXCHANGE_PROFILES.items():
            mask = df[actual_col] == exchange_id
            n_samples = mask.sum()
            if n_samples == 0:
                continue
            
            liquidity = np.random.normal(
                profile['liquidity_score'],
                profile.get('liquidity_std', 0.05),
                n_samples
            )
            liquidity = np.clip(liquidity, 0.5, 1.0)
            df.loc[mask, 'liquidity_score'] = liquidity
            
            base_speed = profile['processing_speed']
            speed = np.random.normal(base_speed, base_speed * 0.1, n_samples)
            speed = np.clip(speed, 0.1, 3.0)
            df.loc[mask, 'processing_speed'] = speed
            
            reliability = liquidity * 0.9 + np.random.normal(0, 0.02, n_samples)
            reliability = np.clip(reliability, 0.6, 1.0)
            df.loc[mask, 'reliability_index'] = reliability
        
        df['exchange_quality'] = (df['liquidity_score'] * 0.4 + 
                                  df['processing_speed'] / 2 * 0.3 + 
                                  df['reliability_index'] * 0.3)
        
        if '_temp_exchange' in df.columns:
            df.drop('_temp_exchange', axis=1, inplace=True)
        
        self.config.FEATURE_GROUPS['proxy_liquidity'] = list(Config.PROXY_LIQUIDITY_FEATURES)
        print(f"       ✓ Generated: {Config.PROXY_LIQUIDITY_FEATURES}")
        return df


# ============================================================================
# STATISTICAL VALIDATION
# ============================================================================

class FeatureValidator:
    """Validate generated proxy features for academic rigor."""
    
    def __init__(self, config: Config):
        self.config = config
        self.validation_results = {}
        
    def check_label_leakage(self, df: pd.DataFrame, 
                           feature_cols: List[str],
                           label_col: str = 'forward_label') -> Dict:
        """Check for label leakage via correlation analysis."""
        print("\n  [Validation 1/3] Checking Label Leakage...")
        
        results = {'passed': [], 'failed': [], 'correlations': {}}
        
        actual_label_col = None
        for col in [label_col, 'forward_label', 'exchange_partition', 'exchange_id']:
            if col in df.columns:
                actual_label_col = col
                break
        
        if actual_label_col is None:
            print(f"       WARNING: No label column found")
            return results
        
        y = df[actual_label_col].values
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            try:
                x = df[col].values.astype(float)
                mask = ~(np.isnan(x) | np.isnan(y.astype(float)))
                if mask.sum() < 100:
                    continue
                corr, _ = pearsonr(x[mask], y.astype(float)[mask])
                corr = abs(corr)
                results['correlations'][col] = corr
                if corr > self.config.CORRELATION_THRESHOLD:
                    results['failed'].append(col)
                else:
                    results['passed'].append(col)
            except Exception:
                pass
        
        print(f"       ✓ Passed: {len(results['passed'])} | ✗ Failed: {len(results['failed'])}")
        self.validation_results['label_leakage'] = results
        return results
    
    def perform_anova_test(self, df: pd.DataFrame,
                          feature_cols: List[str],
                          group_col: str = 'forward_label') -> Dict:
        """Perform ANOVA test for feature discrimination."""
        print("\n  [Validation 2/3] ANOVA Test (Feature Discrimination)...")
        
        results = {'significant': [], 'not_significant': [], 'statistics': {}}
        
        actual_group_col = None
        for col in [group_col, 'forward_label', 'exchange_partition', 'exchange_id']:
            if col in df.columns:
                actual_group_col = col
                break
        
        if actual_group_col is None:
            print(f"       WARNING: No group column found")
            return results
        
        groups = df[actual_group_col].unique()
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            try:
                group_values = [df[df[actual_group_col] == g][col].dropna().values for g in groups]
                group_values = [gv for gv in group_values if len(gv) >= 10]
                if len(group_values) < 2:
                    continue
                f_stat, p_value = f_oneway(*group_values)
                results['statistics'][col] = {'f_statistic': float(f_stat), 'p_value': float(p_value)}
                if p_value < self.config.ANOVA_PVALUE_THRESHOLD:
                    results['significant'].append(col)
                else:
                    results['not_significant'].append(col)
            except Exception:
                pass
        
        print(f"       ✓ Significant (p<0.05): {len(results['significant'])} | Not significant: {len(results['not_significant'])}")
        self.validation_results['anova'] = results
        return results
    
    def run_ablation_study(self, df: pd.DataFrame,
                          label_col: str = 'forward_label',
                          test_ratio: float = 0.3) -> Dict:
        """Run ablation study for proxy feature groups."""
        print("\n  [Validation 3/3] Ablation Study...")
        
        results = {'groups': {}, 'cumulative': {}}
        
        actual_label_col = None
        for col in [label_col, 'forward_label', 'exchange_partition', 'exchange_id']:
            if col in df.columns:
                actual_label_col = col
                break
        
        if actual_label_col is None:
            print(f"       WARNING: No label column found")
            return results
        
        y = df[actual_label_col].values
        original_features = [c for c in df.columns if c.startswith('feat_')]
        self.config.FEATURE_GROUPS['original'] = original_features
        
        n = len(df)
        train_indices = np.random.choice(n, int(n * (1 - test_ratio)), replace=False)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[train_indices] = True
        test_mask = ~train_mask
        
        feature_groups = ['original', 'proxy_fee', 'proxy_volume', 'proxy_temporal', 'proxy_liquidity']
        cumulative_features = []
        
        for group_name in feature_groups:
            group_features = self.config.FEATURE_GROUPS.get(group_name, [])
            existing = [f for f in group_features if f in df.columns]
            if not existing:
                continue
            
            cumulative_features.extend(existing)
            X = df[cumulative_features].fillna(0).values
            
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                        random_state=self.config.RANDOM_SEED, n_jobs=-1)
            rf.fit(X[train_mask], y[train_mask])
            
            test_acc = accuracy_score(y[test_mask], rf.predict(X[test_mask]))
            test_f1 = f1_score(y[test_mask], rf.predict(X[test_mask]), average='weighted')
            
            results['cumulative'][group_name] = {
                'n_features': len(cumulative_features),
                'test_accuracy': float(test_acc),
                'test_f1': float(test_f1)
            }
            print(f"       + {group_name}: {len(existing)} features → Acc={test_acc:.4f}, F1={test_f1:.4f}")
        
        self.validation_results['ablation'] = results
        return results
    
    def generate_validation_report(self) -> str:
        """Generate markdown validation report"""
        report = f"""
# Feature Engineering Validation Report (v4 - Proxy Only)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Feature Summary (as per Article Table 2)

| Category | # Features | Source |
|----------|------------|--------|
| Original Elliptic | 166 | Weber et al. (2019) |
| Fee Proxy | 5 | CoinMarketCap [4] |
| Volume Proxy | 5 | BABD-13 [3] |
| Temporal Proxy | 6 | Coin Metrics [5] |
| Liquidity Proxy | 4 | Kaiko [12] |
| **TOTAL** | **186** | |

## 1. Label Leakage Check
"""
        if 'label_leakage' in self.validation_results:
            ll = self.validation_results['label_leakage']
            report += f"- Passed: {len(ll.get('passed', []))} features (r < {self.config.CORRELATION_THRESHOLD})\n"
            report += f"- Failed: {len(ll.get('failed', []))} features\n"
        
        report += "\n## 2. ANOVA Significance Test\n"
        if 'anova' in self.validation_results:
            anova = self.validation_results['anova']
            report += f"- Significant: {len(anova.get('significant', []))} features (p < 0.05)\n"
            report += f"- Not Significant: {len(anova.get('not_significant', []))} features\n"
        
        report += "\n## 3. Ablation Study Results\n\n| Feature Group | # Features | Test Accuracy | Test F1 |\n|---------------|------------|---------------|---------|"
        if 'ablation' in self.validation_results:
            for group, metrics in self.validation_results['ablation'].get('cumulative', {}).items():
                report += f"\n| {group} | {metrics['n_features']} | {metrics['test_accuracy']:.4f} | {metrics['test_f1']:.4f} |"
        
        return report


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class ProxyFeatureEngineeringPipeline:
    """Main pipeline for proxy feature engineering (20 features only)."""
    
    def __init__(self, config: Config = None, calibration_path: str = None):
        self.config = config or Config(calibration_path)
        self.proxy_generator = ProxyFeatureGenerator(self.config)
        self.validator = FeatureValidator(self.config)
        
    def process_partition(self, df: pd.DataFrame, 
                         exchange_col: str = 'forward_label') -> pd.DataFrame:
        """Process a single partition with proxy features only."""
        
        print("\n" + "="*70)
        print("GENERATING PROXY FEATURES (20 total)")
        print("="*70)
        
        df = self.proxy_generator.generate_fee_proxy_features(df, exchange_col)
        df = self.proxy_generator.generate_volume_proxy_features(df, exchange_col)
        df = self.proxy_generator.generate_temporal_proxy_features(df, exchange_col)
        df = self.proxy_generator.generate_liquidity_proxy_features(df, exchange_col)
        
        return df
    
    def validate_features(self, df: pd.DataFrame, label_col: str = 'forward_label') -> Dict:
        """Run validation suite."""
        print("\n" + "="*70)
        print("STATISTICAL VALIDATION")
        print("="*70)
        
        proxy_features = (Config.PROXY_FEE_FEATURES + Config.PROXY_VOLUME_FEATURES + 
                         Config.PROXY_TEMPORAL_FEATURES + Config.PROXY_LIQUIDITY_FEATURES)
        
        self.validator.check_label_leakage(df, proxy_features, label_col)
        self.validator.perform_anova_test(df, proxy_features, label_col)
        self.validator.run_ablation_study(df, label_col)
        
        return self.validator.validation_results
    
    def _detect_file_format(self, input_dir: str) -> Tuple[List[str], str]:
        """Detect file format and return list of partitions."""
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        if csv_files:
            return csv_files, 'csv'
        
        exchange_dirs = [d for d in os.listdir(input_dir) 
                         if os.path.isdir(os.path.join(input_dir, d)) 
                         and d.startswith('exchange_')]
        if exchange_dirs:
            valid_dirs = [d for d in sorted(exchange_dirs) 
                         if os.path.exists(os.path.join(input_dir, d, 'data.pkl'))]
            if valid_dirs:
                return valid_dirs, 'pkl_dir'
        
        pkl_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
        if pkl_files:
            return pkl_files, 'pkl'
        
        return [], None
    
    def _load_partition(self, input_dir: str, partition: str, file_format: str) -> pd.DataFrame:
        """Load a single partition."""
        if file_format == 'csv':
            return pd.read_csv(os.path.join(input_dir, partition))
        elif file_format == 'pkl_dir':
            return pd.read_pickle(os.path.join(input_dir, partition, 'data.pkl'))
        elif file_format == 'pkl':
            return pd.read_pickle(os.path.join(input_dir, partition))
        raise ValueError(f"Unknown file format: {file_format}")
    
    def _save_partition(self, df: pd.DataFrame, output_dir: str, 
                       partition: str, file_format: str) -> str:
        """Save enriched partition."""
        if file_format == 'csv':
            output_name = partition.replace('.csv', '_enriched.pkl')
        elif file_format == 'pkl_dir':
            output_name = f"{partition}_enriched.pkl"
        else:
            output_name = partition.replace('.pkl', '_enriched.pkl')
        
        output_path = os.path.join(output_dir, output_name)
        df.to_pickle(output_path)
        print(f"\n  Saved: {output_path}")
        return output_path
    
    def run(self, input_dir: str, output_dir: str, 
            exchange_col: str = 'forward_label',
            validate: bool = True) -> None:
        """Run the complete pipeline."""
        
        print("\n" + "="*70)
        print("PROXY FEATURE ENGINEERING PIPELINE (v4 - Proxy Only)")
        print("="*70)
        print(f"\nInput: {input_dir}")
        print(f"Output: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        partition_files, file_format = self._detect_file_format(input_dir)
        
        if not partition_files:
            print(f"\nERROR: No data files found in {input_dir}")
            return
        
        print(f"\nFound {len(partition_files)} partitions (format: {file_format})")
        
        all_data = []
        
        for i, partition in enumerate(partition_files):
            print(f"\n{'='*70}")
            print(f"Processing: {partition} ({i+1}/{len(partition_files)})")
            print('='*70)
            
            df = self._load_partition(input_dir, partition, file_format)
            print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            
            df_enriched = self.process_partition(df, exchange_col)
            self._save_partition(df_enriched, output_dir, partition, file_format)
            all_data.append(df_enriched)
        
        if validate and all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            print(f"\nCombined: {len(df_all)} rows, {len(df_all.columns)} columns")
            
            self.validate_features(df_all, exchange_col)
            
            # Save combined data
            combined_path = os.path.join(output_dir, 'combined_enriched.pkl')
            df_all.to_pickle(combined_path)
            
            # Save report
            report = self.validator.generate_validation_report()
            report_path = os.path.join(output_dir, 'validation_report.md')
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"\nSaved validation report: {report_path}")
        
        # Final summary
        print("\n" + "="*70)
        print("✅ PROXY FEATURE ENGINEERING COMPLETE")
        print("="*70)
        print(f"""
FEATURE SUMMARY (as per Article Table 2):
  Original Elliptic:  166 features
  + Fee Proxy:          5 features [CoinMarketCap]
  + Volume Proxy:       5 features [BABD-13]
  + Temporal Proxy:     6 features [Coin Metrics]
  + Liquidity Proxy:    4 features [Kaiko]
  ─────────────────────────────────
  TOTAL:              186 features

Output directory: {output_dir}
""")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Proxy Feature Engineering for Elliptic Dataset (v4 - 20 features only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python add_hybrid_features_elliptic.py --input_dir data/federated --output_dir data/federated_enriched
    python add_hybrid_features_elliptic.py --config config/calibration_params.json
    python add_hybrid_features_elliptic.py --no-validate  # Skip validation (faster)

Output: 166 original + 20 proxy = 186 features (matches Article Table 2)
        """
    )
    
    parser.add_argument('--input_dir', type=str, default='data/federated',
                       help='Input directory with partitioned data')
    parser.add_argument('--output_dir', type=str, default='data/federated_enriched',
                       help='Output directory for enriched data')
    parser.add_argument('--config', type=str, default='config/calibration_params.json',
                       help='Path to calibration parameters JSON file')
    parser.add_argument('--exchange_col', type=str, default='forward_label',
                       help='Column containing exchange partition')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation suite')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    config = Config(calibration_path=args.config)
    config.RANDOM_SEED = args.seed
    np.random.seed(args.seed)
    
    pipeline = ProxyFeatureEngineeringPipeline(config)
    pipeline.run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        exchange_col=args.exchange_col,
        validate=not args.no_validate
    )


if __name__ == '__main__':
    main()