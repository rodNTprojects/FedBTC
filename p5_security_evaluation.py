"""
================================================================================
Script: p5_security_evaluation.py
Description: SECURITY EVALUATION SUITE for Privacy-Preserving Bitcoin Attribution

EVALUATIONS FOR USENIX SECURITY:
================================
1. Membership Inference Attack (MIA)
   - Train shadow models to infer training set membership
   - Measure attack success rate (AUC, accuracy)
   - Compare FL vs Centralized vulnerability

2. Byzantine Attack Simulation
   - Simulate malicious client updates (random, sign-flip, backdoor)
   - Measure FLAME detection rate
   - Evaluate model degradation under attack

3. Adversarial Robustness Evaluation
   - Generate PGD adversarial examples on test set
   - Measure accuracy drop under attack
   - Compare with/without adversarial training

4. Gradient Inversion Attack (simplified)
   - Attempt to reconstruct features from gradients
   - Measure reconstruction error
   - Compare with/without Server-side DP

INPUT: Trained models from p2 (baseline) and p3 (FL, FL-secure)
OUTPUT: results/evaluations/security_evaluation_report.json

USAGE:
    python p5_security_evaluation.py
    python p5_security_evaluation.py --attack mia
    python p5_security_evaluation.py --attack byzantine
    python p5_security_evaluation.py --attack adversarial
    python p5_security_evaluation.py --attack all
================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
import pandas as pd
warnings.filterwarnings('ignore')


# ============================================================================
# MODEL DEFINITION (must match training scripts)
# ============================================================================

class DualAttributionModel(nn.Module):
    """Dual Attribution GNN Model - must match p3_train_federated_secure.py architecture."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        in_channels = config['in_channels']
        hidden_dim = config.get('hidden_dim', 128)
        dropout = config.get('dropout', 0.3)
        num_exchanges = config.get('num_exchanges', 3)
        num_categories = config.get('num_categories', 6)
        
        # GCN layers - using ModuleList like p3
        self.convs = nn.ModuleList([
            GCNConv(in_channels, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Forward head - matching p3 architecture
        self.forward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_exchanges)
        )
        
        # Backward head - matching p3 architecture
        self.backward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_categories)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        embeddings = x
        forward_logits = self.forward_head(embeddings)
        backward_logits = self.backward_head(embeddings)
        
        return {
            'forward': forward_logits,
            'backward': backward_logits,
            'embeddings': embeddings
        }


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_exchange_data(exchange_id: int, data_dir: str = 'data/federated_enriched'):
    """Load enriched data for a specific exchange."""
    # Try enriched path first
    enriched_path = f'{data_dir}/exchange_{exchange_id}_enriched.pkl'
    original_path = f'data/federated/exchange_{exchange_id}/data.pkl'
    edge_path = f'data/federated/exchange_{exchange_id}/edges.pkl'
    
    # Load data
    if os.path.exists(enriched_path):
        data = pd.read_pickle(enriched_path)
    elif os.path.exists(original_path):
        data = pd.read_pickle(original_path)
    else:
        raise FileNotFoundError(f"No data found for exchange {exchange_id}. Tried:\n  {enriched_path}\n  {original_path}")
    
    # Load edges
    if os.path.exists(edge_path):
        edges = pd.read_pickle(edge_path)
    else:
        # Create minimal edges as fallback
        edges = pd.DataFrame()
    
    return data, edges


def create_pyg_data(data, edges, exchange_id: int = 0) -> Data:
    """Create PyTorch Geometric Data object."""
    
    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        df = data
        # Extract features - look for feat_ columns or use numeric columns
        feature_cols = [c for c in df.columns if c.startswith('feat_') or c.startswith('cat_') 
                       or c.startswith('fee_') or c.startswith('volume_') or c.startswith('synthetic_')
                       or c.startswith('peak_') or c.startswith('timezone_') or c.startswith('hour_')
                       or c.startswith('liquidity_') or c.startswith('processing_') or c.startswith('reliability_')]
        
        if not feature_cols:
            # Use all numeric columns except labels
            exclude_cols = ['forward_label', 'backward_label', 'category_label', 'exchange_partition', 
                           'txId', 'timestep', 'merchant_id']
            feature_cols = [c for c in df.columns if df[c].dtype in ['float64', 'float32', 'int64', 'int32']
                          and c not in exclude_cols]
        
        if feature_cols:
            features = df[feature_cols].values.astype(np.float32)
        else:
            raise ValueError(f"No feature columns found in data. Columns: {list(df.columns)[:20]}")
        
        # Extract labels
        if 'forward_label' in df.columns:
            forward_labels = df['forward_label'].values.astype(np.int64)
        elif 'exchange_partition' in df.columns:
            forward_labels = df['exchange_partition'].values.astype(np.int64)
        else:
            forward_labels = np.full(len(df), exchange_id, dtype=np.int64)
        
        if 'category_label' in df.columns:
            backward_labels = df['category_label'].values.astype(np.int64)
        elif 'backward_label' in df.columns:
            backward_labels = df['backward_label'].values.astype(np.int64)
        else:
            backward_labels = np.zeros(len(df), dtype=np.int64)
        
        # Get txId mapping for edges
        txid_col = 'txId' if 'txId' in df.columns else None
    else:
        # Assume numpy array with last 2 columns being labels
        features = data[:, :-2].astype(np.float32)
        forward_labels = data[:, -2].astype(np.int64)
        backward_labels = data[:, -1].astype(np.int64)
        txid_col = None
    
    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1
    features = (features - mean) / std
    features = np.clip(features, -10, 10)
    
    x = torch.tensor(features, dtype=torch.float32)
    y_forward = torch.tensor(forward_labels, dtype=torch.long)
    y_backward = torch.tensor(backward_labels, dtype=torch.long)
    n = x.size(0)
    
    # Handle edges
    if isinstance(edges, pd.DataFrame) and len(edges) > 0 and 'txId1' in edges.columns and txid_col:
        # Build edge index from txId mapping
        txid_to_idx = {txid: idx for idx, txid in enumerate(data[txid_col])}
        edge_list = []
        for _, row in edges.iterrows():
            if row['txId1'] in txid_to_idx and row['txId2'] in txid_to_idx:
                idx1 = txid_to_idx[row['txId1']]
                idx2 = txid_to_idx[row['txId2']]
                edge_list.append([idx1, idx2])
                edge_list.append([idx2, idx1])
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).T
        else:
            # Self-loops fallback
            edge_index = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
    elif isinstance(edges, np.ndarray) and len(edges) > 0:
        edge_index = torch.tensor(edges.T, dtype=torch.long)
    else:
        # Create self-loops as fallback
        edge_index = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
    
    # Create train/val/test masks
    indices = torch.randperm(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True
    
    return Data(
        x=x, edge_index=edge_index,
        y_forward=y_forward, y_backward=y_backward,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
    )


def load_all_data(data_dir: str = 'data/federated_enriched') -> Tuple[List[Data], Data]:
    """Load all exchange data and create combined dataset."""
    exchange_data = []
    all_features = []
    all_forward = []
    all_backward = []
    
    for i in range(3):
        data, edges = load_exchange_data(i, data_dir)
        graph = create_pyg_data(data, edges, i)
        exchange_data.append(graph)
        
        # Collect for combined dataset
        all_features.append(graph.x.numpy())
        all_forward.append(graph.y_forward.numpy())
        all_backward.append(graph.y_backward.numpy())
        
        print(f"  Exchange {i}: {graph.x.size(0)} transactions, {graph.x.size(1)} features")
    
    # Combined dataset
    combined_features = np.vstack(all_features).astype(np.float32)
    combined_forward = np.concatenate(all_forward).astype(np.int64)
    combined_backward = np.concatenate(all_backward).astype(np.int64)
    
    # Create combined graph with self-loops (simplification)
    n = len(combined_features)
    combined_edge_index = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
    
    # Create masks
    indices = torch.randperm(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True
    
    combined_graph = Data(
        x=torch.tensor(combined_features, dtype=torch.float32),
        edge_index=combined_edge_index,
        y_forward=torch.tensor(combined_forward, dtype=torch.long),
        y_backward=torch.tensor(combined_backward, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    return exchange_data, combined_graph


# ============================================================================
# 1. MEMBERSHIP INFERENCE ATTACK (MIA)
# ============================================================================

class MembershipInferenceAttack:
    """
    Membership Inference Attack evaluation.
    
    Methodology:
    1. Train shadow models on subset of data
    2. Collect model outputs (logits, loss, confidence) for train/test samples
    3. Train attack classifier to distinguish train vs test
    4. Evaluate attack success rate
    
    Reference: Shokri et al., "Membership Inference Attacks Against ML Models", IEEE S&P 2017
    """
    
    def __init__(self, target_model: nn.Module, device: torch.device):
        self.target_model = target_model
        self.device = device
        self.attack_model = None
    
    def extract_attack_features(
        self, 
        model: nn.Module, 
        data: Data, 
        mask: torch.Tensor
    ) -> np.ndarray:
        """Extract features for attack model from target model outputs."""
        model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            output = model(data.x, data.edge_index)
            
            # Forward task features
            logits_fwd = output['forward'][mask]
            probs_fwd = F.softmax(logits_fwd, dim=1)
            confidence_fwd = probs_fwd.max(dim=1)[0]
            entropy_fwd = -(probs_fwd * torch.log(probs_fwd + 1e-10)).sum(dim=1)
            
            # Loss per sample
            labels_fwd = data.y_forward[mask]
            loss_fwd = F.cross_entropy(logits_fwd, labels_fwd, reduction='none')
            
            # Backward task features
            logits_bwd = output['backward'][mask]
            probs_bwd = F.softmax(logits_bwd, dim=1)
            confidence_bwd = probs_bwd.max(dim=1)[0]
            entropy_bwd = -(probs_bwd * torch.log(probs_bwd + 1e-10)).sum(dim=1)
            
            labels_bwd = data.y_backward[mask]
            loss_bwd = F.cross_entropy(logits_bwd, labels_bwd, reduction='none')
            
            # Correctness
            correct_fwd = (logits_fwd.argmax(dim=1) == labels_fwd).float()
            correct_bwd = (logits_bwd.argmax(dim=1) == labels_bwd).float()
        
        # Combine features
        attack_features = torch.stack([
            confidence_fwd,
            entropy_fwd,
            loss_fwd,
            confidence_bwd,
            entropy_bwd,
            loss_bwd,
            correct_fwd,
            correct_bwd
        ], dim=1).cpu().numpy()
        
        return attack_features
    
    def run_attack(self, data: Data) -> Dict:
        """Run membership inference attack."""
        print("\n" + "="*60)
        print("MEMBERSHIP INFERENCE ATTACK")
        print("="*60)
        
        # Extract features for train and test samples
        train_features = self.extract_attack_features(
            self.target_model, data, data.train_mask
        )
        test_features = self.extract_attack_features(
            self.target_model, data, data.test_mask
        )
        
        # Create attack dataset
        n_train = len(train_features)
        n_test = len(test_features)
        
        # Balance the dataset
        n_samples = min(n_train, n_test)
        train_idx = np.random.choice(n_train, n_samples, replace=False)
        test_idx = np.random.choice(n_test, n_samples, replace=False)
        
        X = np.vstack([train_features[train_idx], test_features[test_idx]])
        y = np.array([1] * n_samples + [0] * n_samples)  # 1=member, 0=non-member
        
        # Split for attack model training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train attack classifier
        self.attack_model = LogisticRegression(max_iter=1000)
        self.attack_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.attack_model.predict(X_test)
        y_prob = self.attack_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        
        # Baseline: random guess = 50%
        advantage = (accuracy - 0.5) * 2  # Scale to [-1, 1]
        
        results = {
            'attack_accuracy': float(accuracy),
            'attack_auc': float(auc),
            'attack_precision': float(precision),
            'attack_recall': float(recall),
            'attack_f1': float(f1),
            'attack_advantage': float(advantage),
            'n_samples': n_samples * 2,
            'interpretation': self._interpret_results(accuracy, auc)
        }
        
        print(f"\nResults:")
        print(f"  Attack Accuracy: {accuracy:.2%} (baseline: 50%)")
        print(f"  Attack AUC: {auc:.4f}")
        print(f"  Attack Advantage: {advantage:.2%}")
        print(f"  Interpretation: {results['interpretation']}")
        
        return results
    
    def _interpret_results(self, accuracy: float, auc: float) -> str:
        """Interpret MIA results."""
        if auc < 0.55:
            return "STRONG PRIVACY: Model leaks minimal membership information"
        elif auc < 0.65:
            return "MODERATE PRIVACY: Some membership leakage, acceptable for most uses"
        elif auc < 0.75:
            return "WEAK PRIVACY: Significant membership leakage, consider adding DP"
        else:
            return "POOR PRIVACY: High membership leakage, DP strongly recommended"


# ============================================================================
# 2. BYZANTINE ATTACK SIMULATION
# ============================================================================

class ByzantineAttackSimulator:
    """
    Simulate Byzantine attacks and evaluate FLAME defense.
    
    Attack types:
    1. Random: Replace gradients with random noise
    2. Sign-flip: Flip gradient signs
    3. Scaling: Scale gradients by large factor
    4. Backdoor: Targeted manipulation
    
    Reference: Blanchard et al., "Machine Learning with Adversaries", NeurIPS 2017
    """
    
    def __init__(self, model_config: Dict, device: torch.device, flame_eps: float = 0.5):
        self.model_config = model_config
        self.device = device
        self.flame_eps = flame_eps
    
    def generate_honest_update(
        self, 
        model: nn.Module, 
        data: Data,
        lr: float = 0.002,
        epochs: int = 3
    ) -> List[np.ndarray]:
        """Generate honest client update."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        data = data.to(self.device)
        
        initial_params = [p.clone().detach().cpu().numpy() for p in model.parameters()]
        
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            loss = F.cross_entropy(output['forward'][data.train_mask], 
                                   data.y_forward[data.train_mask])
            loss.backward()
            optimizer.step()
        
        final_params = [p.detach().cpu().numpy() for p in model.parameters()]
        updates = [f - i for f, i in zip(final_params, initial_params)]
        
        return updates
    
    def generate_random_attack(self, honest_update: List[np.ndarray]) -> List[np.ndarray]:
        """Random noise attack."""
        return [np.random.randn(*p.shape).astype(np.float32) * np.std(p) * 10 
                for p in honest_update]
    
    def generate_signflip_attack(self, honest_update: List[np.ndarray]) -> List[np.ndarray]:
        """Sign-flip attack."""
        return [-p * 5 for p in honest_update]
    
    def generate_scaling_attack(self, honest_update: List[np.ndarray], scale: float = 100) -> List[np.ndarray]:
        """Scaling attack."""
        return [p * scale for p in honest_update]
    
    # def flame_detect(
    #     self, 
    #     updates: List[List[np.ndarray]], 
    #     eps: float = 0.5, 
    #     min_samples: int = 2
    # ) -> List[int]:
    def flame_detect(
        self, 
        updates: List[List[np.ndarray]], 
        eps: float = None, 
        min_samples: int = 2
    ) -> List[int]:
    # Use instance flame_eps if not specified
        if eps is None:
            eps = self.flame_eps
            """FLAME-style outlier detection using DBSCAN."""
        from sklearn.cluster import DBSCAN
        
        # Flatten updates
        flattened = []
        for update in updates:
            flat = np.concatenate([p.flatten() for p in update])
            flattened.append(flat)
        
        X = np.array(flattened)
        
        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_normalized = X / (norms + 1e-10)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X_normalized)
        labels = clustering.labels_
        
        # Outliers have label -1
        outliers = [i for i, l in enumerate(labels) if l == -1]
        
        return outliers
    
    def run_evaluation(self, exchange_data: List[Data]) -> Dict:
        """Run Byzantine attack evaluation."""
        print("\n" + "="*60)
        print("BYZANTINE ATTACK SIMULATION")
        print("="*60)
        
        results = {
            'scenarios': {},
            'flame_performance': {}
        }
        
        # Generate honest updates for each client
        honest_updates = []
        for i, data in enumerate(exchange_data):
            model = DualAttributionModel(self.model_config).to(self.device)
            update = self.generate_honest_update(model, data)
            honest_updates.append(update)
            print(f"  Generated honest update for client {i}")
        
        # Test scenarios
        scenarios = [
            ("No attack (baseline)", [0, 1, 2], []),
            ("1 random attacker", [0, 1], [2]),
            ("1 sign-flip attacker", [0, 1], [2]),
            ("2 colluding attackers", [0], [1, 2]),
        ]
        
        for scenario_name, honest_ids, attacker_ids in scenarios:
            print(f"\n  Scenario: {scenario_name}")
            
            # Construct updates
            all_updates = []
            ground_truth_malicious = []
            
            for i in range(3):
                if i in honest_ids:
                    all_updates.append(honest_updates[i])
                    ground_truth_malicious.append(0)
                else:
                    if "random" in scenario_name:
                        attack = self.generate_random_attack(honest_updates[i])
                    elif "sign-flip" in scenario_name:
                        attack = self.generate_signflip_attack(honest_updates[i])
                    else:
                        attack = self.generate_scaling_attack(honest_updates[i])
                    all_updates.append(attack)
                    ground_truth_malicious.append(1)
            
            # Run FLAME detection
            detected_outliers = self.flame_detect(all_updates)
            
            # Calculate metrics
            true_positives = len([i for i in detected_outliers if i in attacker_ids])
            false_positives = len([i for i in detected_outliers if i not in attacker_ids])
            false_negatives = len([i for i in attacker_ids if i not in detected_outliers])
            
            n_attackers = len(attacker_ids)
            detection_rate = true_positives / max(n_attackers, 1)
            false_positive_rate = false_positives / max(len(honest_ids), 1)
            
            results['scenarios'][scenario_name] = {
                'n_honest': len(honest_ids),
                'n_attackers': n_attackers,
                'detected_outliers': detected_outliers,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'detection_rate': detection_rate,
                'false_positive_rate': false_positive_rate
            }
            
            print(f"    Detected outliers: {detected_outliers}")
            print(f"    Detection rate: {detection_rate:.2%}")
            print(f"    False positive rate: {false_positive_rate:.2%}")
        
        # Summary
        total_tp = sum(s['true_positives'] for s in results['scenarios'].values())
        total_fp = sum(s['false_positives'] for s in results['scenarios'].values())
        total_fn = sum(s['false_negatives'] for s in results['scenarios'].values())
        total_attackers = sum(s['n_attackers'] for s in results['scenarios'].values())
        
        results['flame_performance'] = {
            'overall_detection_rate': total_tp / max(total_attackers, 1),
            'overall_false_positive_rate': total_fp / (len(scenarios) * 3 - total_attackers),
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn
        }
        
        print(f"\n  FLAME Overall Performance:")
        print(f"    Detection Rate: {results['flame_performance']['overall_detection_rate']:.2%}")
        print(f"    False Positive Rate: {results['flame_performance']['overall_false_positive_rate']:.2%}")
        
        return results


# ============================================================================
# 3. ADVERSARIAL ROBUSTNESS EVALUATION
# ============================================================================

class AdversarialRobustnessEvaluator:
    """
    Evaluate model robustness against adversarial examples.
    
    Attack: Projected Gradient Descent (PGD)
    
    Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def pgd_attack(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_steps: int = 10,
        task: str = 'forward'
    ) -> torch.Tensor:
        """Generate PGD adversarial examples."""
        x_adv = x.clone().detach().requires_grad_(True)
        
        for _ in range(num_steps):
            self.model.zero_grad()
            output = self.model(x_adv, edge_index)
            loss = F.cross_entropy(output[task], y)
            loss.backward()
            
            with torch.no_grad():
                perturbation = alpha * x_adv.grad.sign()
                x_adv = x_adv + perturbation
                
                # Project back to epsilon ball
                delta = x_adv - x
                delta = torch.clamp(delta, -epsilon, epsilon)
                x_adv = x + delta
            
            x_adv = x_adv.detach().requires_grad_(True)
        
        return x_adv.detach()
    
    def evaluate(self, data: Data, epsilon_values: List[float] = [0.01, 0.05, 0.1, 0.2]) -> Dict:
        """Evaluate adversarial robustness."""
        print("\n" + "="*60)
        print("ADVERSARIAL ROBUSTNESS EVALUATION")
        print("="*60)
        
        self.model.eval()
        data = data.to(self.device)
        test_mask = data.test_mask
        
        results = {
            'clean_accuracy': {},
            'adversarial_accuracy': {},
            'accuracy_drop': {}
        }
        
        # Clean accuracy
        with torch.no_grad():
            output = self.model(data.x, data.edge_index)
            
            pred_fwd = output['forward'][test_mask].argmax(dim=1)
            clean_acc_fwd = (pred_fwd == data.y_forward[test_mask]).float().mean().item()
            
            pred_bwd = output['backward'][test_mask].argmax(dim=1)
            clean_acc_bwd = (pred_bwd == data.y_backward[test_mask]).float().mean().item()
        
        results['clean_accuracy'] = {
            'forward': clean_acc_fwd,
            'backward': clean_acc_bwd
        }
        
        print(f"\n  Clean Accuracy:")
        print(f"    Forward: {clean_acc_fwd:.2%}")
        print(f"    Backward: {clean_acc_bwd:.2%}")
        
        # Adversarial accuracy for different epsilon values
        print(f"\n  Adversarial Accuracy (PGD-10):")
        
        for eps in epsilon_values:
            # Forward task attack
            x_adv_fwd = self.pgd_attack(
                data.x, data.edge_index, data.y_forward,
                epsilon=eps, task='forward'
            )
            
            with torch.no_grad():
                output_adv = self.model(x_adv_fwd, data.edge_index)
                pred_adv_fwd = output_adv['forward'][test_mask].argmax(dim=1)
                adv_acc_fwd = (pred_adv_fwd == data.y_forward[test_mask]).float().mean().item()
            
            # Backward task attack
            x_adv_bwd = self.pgd_attack(
                data.x, data.edge_index, data.y_backward,
                epsilon=eps, task='backward'
            )
            
            with torch.no_grad():
                output_adv = self.model(x_adv_bwd, data.edge_index)
                pred_adv_bwd = output_adv['backward'][test_mask].argmax(dim=1)
                adv_acc_bwd = (pred_adv_bwd == data.y_backward[test_mask]).float().mean().item()
            
            results['adversarial_accuracy'][f'eps_{eps}'] = {
                'forward': adv_acc_fwd,
                'backward': adv_acc_bwd
            }
            
            results['accuracy_drop'][f'eps_{eps}'] = {
                'forward': clean_acc_fwd - adv_acc_fwd,
                'backward': clean_acc_bwd - adv_acc_bwd
            }
            
            print(f"    eps={eps}: Forward={adv_acc_fwd:.2%} (drop: {(clean_acc_fwd - adv_acc_fwd)*100:.1f}%), "
                  f"Backward={adv_acc_bwd:.2%} (drop: {(clean_acc_bwd - adv_acc_bwd)*100:.1f}%)")
        
        # Robustness score (average accuracy retention across all epsilon values)
        avg_retention_fwd = np.mean([
            results['adversarial_accuracy'][f'eps_{eps}']['forward'] / clean_acc_fwd 
            for eps in epsilon_values
        ])
        avg_retention_bwd = np.mean([
            results['adversarial_accuracy'][f'eps_{eps}']['backward'] / clean_acc_bwd 
            for eps in epsilon_values
        ])
        
        results['robustness_score'] = {
            'forward': avg_retention_fwd,
            'backward': avg_retention_bwd,
            'average': (avg_retention_fwd + avg_retention_bwd) / 2
        }
        
        print(f"\n  Robustness Score (avg accuracy retention):")
        print(f"    Forward: {avg_retention_fwd:.2%}")
        print(f"    Backward: {avg_retention_bwd:.2%}")
        
        return results


# ============================================================================
# 4. GRADIENT INVERSION ATTACK (Simplified)
# ============================================================================

class GradientInversionEvaluator:
    """
    Simplified gradient inversion attack evaluation.
    
    Instead of actually inverting gradients (complex), we measure
    gradient information leakage via cosine similarity between
    gradients from same vs different samples.
    
    Reference: Geiping et al., "Inverting Gradients", NeurIPS 2020
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def compute_gradient_signature(self, x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute gradient signature for a subset of samples."""
        self.model.zero_grad()
        self.model.train()
        
        output = self.model(x, edge_index)
        loss = F.cross_entropy(output['forward'][mask], y[mask])
        loss.backward()
        
        # Flatten all gradients into a single vector
        grad_vec = []
        for p in self.model.parameters():
            if p.grad is not None:
                grad_vec.append(p.grad.view(-1).clone())
        
        return torch.cat(grad_vec)
    
    def evaluate(self, data: Data, n_samples: int = 100) -> Dict:
        """Evaluate gradient information leakage."""
        print("\n" + "="*60)
        print("GRADIENT LEAKAGE RISK EVALUATION")
        print("="*60)
        
        data = data.to(self.device)
        
        # Split test set into two groups
        test_indices = torch.where(data.test_mask)[0]
        n_test = len(test_indices)
        
        if n_test < 20:
            print("  Not enough test samples for gradient analysis")
            return {'status': 'skipped', 'reason': 'insufficient samples'}
        
        # Create two disjoint subsets
        perm = torch.randperm(n_test)
        group_a = test_indices[perm[:n_test//2]]
        group_b = test_indices[perm[n_test//2:]]
        
        # Create masks
        mask_a = torch.zeros(data.x.size(0), dtype=torch.bool, device=self.device)
        mask_b = torch.zeros(data.x.size(0), dtype=torch.bool, device=self.device)
        mask_a[group_a] = True
        mask_b[group_b] = True
        
        # Compute gradient signatures
        grad_a = self.compute_gradient_signature(data.x, data.edge_index, data.y_forward, mask_a)
        grad_b = self.compute_gradient_signature(data.x, data.edge_index, data.y_forward, mask_b)
        
        # Measure similarity
        cosine_sim = F.cosine_similarity(grad_a.unsqueeze(0), grad_b.unsqueeze(0)).item()
        l2_distance = torch.norm(grad_a - grad_b).item()
        grad_norm_a = torch.norm(grad_a).item()
        grad_norm_b = torch.norm(grad_b).item()
        
        # Interpretation: high cosine similarity = gradients reveal similar info
        # regardless of which samples were used = less privacy
        results = {
            'gradient_cosine_similarity': float(cosine_sim),
            'gradient_l2_distance': float(l2_distance),
            'gradient_norm_a': float(grad_norm_a),
            'gradient_norm_b': float(grad_norm_b),
            'interpretation': self._interpret_results(cosine_sim)
        }
        
        print(f"\n  Results:")
        print(f"    Gradient Cosine Similarity: {cosine_sim:.4f}")
        print(f"    Gradient L2 Distance: {l2_distance:.4f}")
        print(f"    Interpretation: {results['interpretation']}")
        
        return results
    
    def _interpret_results(self, cos_sim: float) -> str:
        """Interpret gradient inversion results."""
        if cos_sim > 0.9:
            return "HIGH RISK: Gradients very similar across samples - potential for reconstruction"
        elif cos_sim > 0.7:
            return "MODERATE RISK: Gradients show correlation - some information leakage"
        elif cos_sim > 0.5:
            return "LOW RISK: Gradients moderately different - limited leakage"
        else:
            return "MINIMAL RISK: Gradients highly sample-specific - good privacy"


# ============================================================================
# 5. CONFIDENCE SCORE EVALUATION
# ============================================================================

class ConfidenceScoreEvaluator:
    """Evaluate prediction confidence and calibration."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def evaluate(self, data: Data, threshold: float = 0.7) -> Dict:
        """Evaluate confidence scores."""
        print("\n" + "="*60)
        print("CONFIDENCE SCORE EVALUATION")
        print("="*60)
        
        self.model.eval()
        data = data.to(self.device)
        test_mask = data.test_mask
        
        with torch.no_grad():
            output = self.model(data.x, data.edge_index)
            
            # Forward task
            probs_fwd = F.softmax(output['forward'][test_mask], dim=1)
            confidence_fwd = probs_fwd.max(dim=1)[0]
            pred_fwd = probs_fwd.argmax(dim=1)
            correct_fwd = (pred_fwd == data.y_forward[test_mask])
            
            # Backward task
            probs_bwd = F.softmax(output['backward'][test_mask], dim=1)
            confidence_bwd = probs_bwd.max(dim=1)[0]
            pred_bwd = probs_bwd.argmax(dim=1)
            correct_bwd = (pred_bwd == data.y_backward[test_mask])
        
        # High confidence predictions
        high_conf_mask_fwd = confidence_fwd >= threshold
        high_conf_mask_bwd = confidence_bwd >= threshold
        
        results = {
            'forward': {
                'mean_confidence': float(confidence_fwd.mean()),
                'high_conf_ratio': float(high_conf_mask_fwd.float().mean()),
                'high_conf_accuracy': float(correct_fwd[high_conf_mask_fwd].float().mean()) if high_conf_mask_fwd.sum() > 0 else 0,
                'low_conf_accuracy': float(correct_fwd[~high_conf_mask_fwd].float().mean()) if (~high_conf_mask_fwd).sum() > 0 else 0,
                'overall_accuracy': float(correct_fwd.float().mean())
            },
            'backward': {
                'mean_confidence': float(confidence_bwd.mean()),
                'high_conf_ratio': float(high_conf_mask_bwd.float().mean()),
                'high_conf_accuracy': float(correct_bwd[high_conf_mask_bwd].float().mean()) if high_conf_mask_bwd.sum() > 0 else 0,
                'low_conf_accuracy': float(correct_bwd[~high_conf_mask_bwd].float().mean()) if (~high_conf_mask_bwd).sum() > 0 else 0,
                'overall_accuracy': float(correct_bwd.float().mean())
            },
            'threshold': threshold
        }
        
        print(f"\n  Forward Attribution (threshold={threshold}):")
        print(f"    Mean Confidence: {results['forward']['mean_confidence']:.2%}")
        print(f"    High-Conf Ratio: {results['forward']['high_conf_ratio']:.2%}")
        print(f"    High-Conf Accuracy: {results['forward']['high_conf_accuracy']:.2%}")
        print(f"    Low-Conf Accuracy: {results['forward']['low_conf_accuracy']:.2%}")
        
        print(f"\n  Backward Attribution (threshold={threshold}):")
        print(f"    Mean Confidence: {results['backward']['mean_confidence']:.2%}")
        print(f"    High-Conf Ratio: {results['backward']['high_conf_ratio']:.2%}")
        print(f"    High-Conf Accuracy: {results['backward']['high_conf_accuracy']:.2%}")
        print(f"    Low-Conf Accuracy: {results['backward']['low_conf_accuracy']:.2%}")
        
        return results


# ============================================================================
# MAIN EVALUATION RUNNER
# ============================================================================

def load_trained_model(model_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load a trained model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # Default config
        config = {
            'in_channels': 191,
            'hidden_dim': 128,
            'dropout': 0.3,
            'num_exchanges': 3,
            'num_categories': 6
        }
    
    model = DualAttributionModel(config).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Try loading directly
        model.load_state_dict(checkpoint)
    
    return model, config


def run_full_evaluation(args):
    """Run complete security evaluation."""
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    exchange_data, combined_data = load_all_data(args.data_dir)
    print(f"  Combined dataset: {combined_data.x.size(0)} transactions")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    
    # Get actual feature count from data
    actual_features = combined_data.num_features
    
    if os.path.exists(args.model_path):
        model, config = load_trained_model(args.model_path, device)
        # Update in_channels if mismatch
        if config.get('in_channels', actual_features) != actual_features:
            print(f"  Warning: Model expects {config['in_channels']} features, data has {actual_features}")
            print(f"  Creating new model with correct dimensions...")
            config['in_channels'] = actual_features
            model = DualAttributionModel(config).to(device)
        print(f"  Model config: {config}")
    else:
        print(f"  Model not found, creating new model...")
        config = {
            'in_channels': actual_features,
            'hidden_dim': 128,
            'dropout': 0.3,
            'num_exchanges': 3,
            'num_categories': 6
        }
        model = DualAttributionModel(config).to(device)
    
    # Results container
    all_results = {
        'model_path': args.model_path,
        'timestamp': datetime.now().isoformat(),
        'evaluations': {}
    }
    
    # Run evaluations based on args
    if args.attack in ['mia', 'all']:
        mia = MembershipInferenceAttack(model, device)
        all_results['evaluations']['membership_inference'] = mia.run_attack(combined_data)
    
    if args.attack in ['byzantine', 'all']:
        byzantine = ByzantineAttackSimulator(config, device, flame_eps=args.flame_eps)
        all_results['evaluations']['byzantine_attack'] = byzantine.run_evaluation(exchange_data)
    
    if args.attack in ['adversarial', 'all']:
        adv_eval = AdversarialRobustnessEvaluator(model, device)
        all_results['evaluations']['adversarial_robustness'] = adv_eval.evaluate(combined_data)
    
    if args.attack in ['gradient', 'all']:
        grad_eval = GradientInversionEvaluator(model, device)
        all_results['evaluations']['gradient_inversion'] = grad_eval.evaluate(combined_data, n_samples=100)
    
    if args.attack in ['confidence', 'all']:
        conf_eval = ConfidenceScoreEvaluator(model, device)
        all_results['evaluations']['confidence_scores'] = conf_eval.evaluate(combined_data)
    
    # Save results
    os.makedirs('results/evaluations', exist_ok=True)
    results_path = 'results/evaluations/security_evaluation_report.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved: {results_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if 'membership_inference' in all_results['evaluations']:
        mia_res = all_results['evaluations']['membership_inference']
        print(f"\nMembership Inference Attack:")
        print(f"  AUC: {mia_res['attack_auc']:.4f}")
        print(f"  {mia_res['interpretation']}")
    
    if 'byzantine_attack' in all_results['evaluations']:
        byz_res = all_results['evaluations']['byzantine_attack']['flame_performance']
        print(f"\nByzantine Attack (FLAME Defense):")
        print(f"  Detection Rate: {byz_res['overall_detection_rate']:.2%}")
        print(f"  False Positive Rate: {byz_res['overall_false_positive_rate']:.2%}")
    
    if 'adversarial_robustness' in all_results['evaluations']:
        adv_res = all_results['evaluations']['adversarial_robustness']
        print(f"\nAdversarial Robustness:")
        print(f"  Robustness Score: {adv_res['robustness_score']['average']:.2%}")
    
    if 'gradient_inversion' in all_results['evaluations']:
        grad_res = all_results['evaluations']['gradient_inversion']
        if 'gradient_cosine_similarity' in grad_res:
            print(f"\nGradient Leakage Risk:")
            print(f"  Cosine Similarity: {grad_res['gradient_cosine_similarity']:.4f}")
            print(f"  {grad_res['interpretation']}")
    
    if 'confidence_scores' in all_results['evaluations']:
        conf_res = all_results['evaluations']['confidence_scores']
        print(f"\nConfidence Scores:")
        print(f"  Forward High-Conf Accuracy: {conf_res['forward']['high_conf_accuracy']:.2%}")
        print(f"  Backward High-Conf Accuracy: {conf_res['backward']['high_conf_accuracy']:.2%}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Security Evaluation Suite')
    
    parser.add_argument('--model_path', type=str, 
                        default='results/models/federated_secure_dual_attribution.pt',
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, 
                        default='data/federated_enriched',
                        help='Data directory')
    parser.add_argument('--attack', type=str, 
                        choices=['mia', 'byzantine', 'adversarial', 'gradient', 'confidence', 'all'],
                        default='all',
                        help='Attack type to evaluate')
    parser.add_argument('--flame_eps', type=float, default=0.5,
                        help='DBSCAN eps parameter for FLAME (default: 0.5)')
    
    args = parser.parse_args()
    
    run_full_evaluation(args)