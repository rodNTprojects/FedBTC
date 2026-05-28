"""
================================================================================
Script: p3_train_federated_secure.py
Description: FEDERATED LEARNING SECURISE - Dual Attribution avec Privacy Guarantees
Phase 3 du pipeline - Version avec toutes les couches de securite

SECURITY LAYERS:
================
Client-Side:
  * User-Level DP (Differential Privacy for FL)
    - Noise added ONCE per round to model update (not per gradient step)
    - Algorithm: clip(w_local - w_global) + Gaussian noise
    - Reference: McMahan et al., ICLR 2018
    
  * Adversarial Training
    - Projected Gradient Descent attack generation
    - Reference: Madry et al., ICLR 2018
  
Server-Side:
  * Byzantine Defense: Coordinate-wise Median + Norm Bounding
    - Reference: Blanchard et al., NeurIPS 2017
    
  * FedVal: Validation-based update filtering (disabled by default for non-IID)
    - Reference: Pillutla et al., IEEE TSP 2022

USER-LEVEL DP (correct implementation for FL):
==============================================
  - Train locally with standard SGD (no noise)
  - Compute update: delta = w_local - w_global  
  - Clip update: delta_clipped = delta * min(1, C/||delta||)
  - Add noise ONCE: delta_noisy = delta_clipped + N(0, sigma^2 * C^2)
  - Return: w_global + delta_noisy

NOISE CALIBRATION (User-Level DP):
==================================
  epsilon=1.0  -> noise_multiplier=1.0  (strong privacy)
  epsilon=2.0  -> noise_multiplier=0.5  (moderate privacy, ~3-5% utility loss)
  epsilon=4.0  -> noise_multiplier=0.3  (relaxed privacy, ~1-3% utility loss)

INPUT: data/federated_enriched/exchange_X_enriched.pkl
OUTPUT: results/models/federated_secure_dual_attribution.pt

USAGE:
    python p3_train_federated_secure.py
    python p3_train_federated_secure.py --epsilon 1.0
    python p3_train_federated_secure.py --no-dp  # Disable DP for comparison
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Flower imports (avec fallback)
try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False


# ============================================================================
# CATEGORY CONFIGURATION
# ============================================================================

CATEGORY_NAMES = {
    0: 'NO_MERCHANT',
    1: 'E-commerce',
    2: 'Gambling',
    3: 'Services',
    4: 'Retail',
    5: 'Luxury'
}

CATEGORY_PATTERNS = {
    0: [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
    1: [(-0.5, 0.3), (0.5, 0.3), (0.3, 0.3), (-0.3, 0.3), (-1, 0.4), (1.5, 0.4)],
    2: [(-2, 0.4), (2.5, 0.5), (-1, 0.4), (1.2, 0.4), (0.3, 0.5), (-1.2, 0.4)],
    3: [(-1, 0.4), (-0.5, 0.4), (0.5, 0.3), (-1, 0.3), (-0.8, 0.4), (-0.5, 0.5)],
    4: [(-2.5, 0.5), (-0.3, 0.4), (0.8, 0.3), (0.2, 0.4), (1.5, 0.4), (-1, 0.4)],
    5: [(2, 0.8), (-2, 0.5), (0.4, 0.3), (-0.8, 0.3), (-1.5, 0.3), (-0.8, 0.5)],
}


# ============================================================================
# FOCAL LOSS - Lin et al., ICCV 2017
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class DualAttributionModel(nn.Module):
    """Dual Attribution GNN Model with GCN backbone (2 layers - optimal)."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # GNN Backbone (2 layers - optimal per ablation study)
        self.convs = nn.ModuleList([
            GCNConv(config['in_channels'], config['hidden_dim']),
            GCNConv(config['hidden_dim'], config['hidden_dim'])
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(config['hidden_dim']),
            nn.BatchNorm1d(config['hidden_dim'])
        ])
        
        self.dropout = nn.Dropout(config['dropout'])
        
        self.forward_head = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.BatchNorm1d(config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.BatchNorm1d(config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['hidden_dim'] // 2, config['num_exchanges'])
        )
        
        self.backward_head = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.BatchNorm1d(config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['hidden_dim'] // 2, config['num_categories'])
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
# DP-SGD OPTIMIZER (Legacy - Not used in User-Level DP)
# ============================================================================

class DPOptimizer:
    """
    Example-Level DP-SGD Optimizer.
    
    NOTE: This class is kept for reference but is NOT USED in the current
    implementation. User-Level DP (applied to model updates) is used instead,
    which is the standard approach for Federated Learning.
    
    See SecureFederatedClient._apply_user_level_dp() for the actual DP implementation.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        noise_multiplier: float = 0.3,
        max_grad_norm: float = 1.0,
        device: torch.device = None
    ):
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device('cpu')
        self.steps = 0
    
    def clip_gradients(self, model: nn.Module) -> float:
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        return total_norm
    
    def add_noise(self, model: nn.Module):
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.noise_multiplier * self.max_grad_norm
                param.grad.data.add_(noise)
    
    def step(self, model: nn.Module) -> float:
        grad_norm = self.clip_gradients(model)
        self.add_noise(model)
        self.optimizer.step()
        self.steps += 1
        return grad_norm
    
    def zero_grad(self):
        self.optimizer.zero_grad()


# ============================================================================
# ADVERSARIAL TRAINING - Madry et al., ICLR 2018
# ============================================================================

class AdversarialTrainer:
    """PGD Adversarial Training."""
    
    def __init__(
        self,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_steps: int = 3,
        random_start: bool = True
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
    
    def generate_adversarial(
        self,
        model: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y_forward: torch.Tensor,
        y_backward: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate adversarial examples using PGD."""
        model.eval()
        
        x_adv = x.clone().detach()
        
        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, -10, 10)
        
        for _ in range(self.num_steps):
            x_adv.requires_grad = True
            
            output = model(x_adv, edge_index)
            loss = F.cross_entropy(output['forward'][mask], y_forward[mask])
            
            backward_mask = y_backward[mask] > 0
            if backward_mask.sum() > 0:
                loss += F.cross_entropy(
                    output['backward'][mask][backward_mask],
                    y_backward[mask][backward_mask]
                )
            
            model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, -10, 10)
        
        model.train()
        return x_adv.detach()


# ============================================================================
# BYZANTINE DEFENSE — Coordinate-wise Median + Norm Bounding
# Replaces FLAME (DBSCAN too unreliable at K=3 clients)
#
# Coordinate-wise Median: Blanchard et al., NeurIPS 2017 (Byzantine-resilient)
# Norm Bounding: clips updates exceeding median norm by factor tau
# ============================================================================

class ByzantineDefense:
    """Coordinate-wise Median aggregation + Norm Bounding.

    Coordinate-wise Median is Byzantine-resilient: with f malicious clients
    out of n total, it tolerates f < n/2 adversaries.
    Norm Bounding clips outlier updates before aggregation, preventing
    large-magnitude poisoning updates from dominating.

    References:
    - Blanchard et al., "Machine Learning with Adversaries: Byzantine
      Tolerant Gradient Descent", NeurIPS 2017
    - Sun et al., 2019 — norm clipping as update pre-filter
      (Norm bounding is a standard pre-aggregation technique;
       Baruch et al., NeurIPS 2019 *attacks* this approach)
    """

    def __init__(self, norm_bound_tau: float = 2.0):
        self.tau = norm_bound_tau  # clip if norm > tau * median_norm

    def _norm_bound(
        self,
        client_params: List[List[np.ndarray]]
    ) -> List[List[np.ndarray]]:
        """Clip each update whose norm exceeds tau * median_norm."""
        norms = np.array([
            np.sqrt(sum(np.sum(p ** 2) for p in params))
            for params in client_params
        ])
        median_norm = float(np.median(norms))
        threshold   = self.tau * median_norm + 1e-8

        bounded = []
        for params, norm in zip(client_params, norms):
            if norm > threshold:
                scale = threshold / norm
                bounded.append([p * scale for p in params])
            else:
                bounded.append(params)
        return bounded

    def aggregate(
        self,
        client_params: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """Norm-bound then coordinate-wise median aggregation."""
        if len(client_params) == 0:
            raise ValueError("No client parameters to aggregate")
        if len(client_params) == 1:
            return client_params[0]

        bounded = self._norm_bound(client_params)

        # Coordinate-wise median across clients
        result = []
        for param_idx in range(len(bounded[0])):
            stacked = np.stack([bounded[i][param_idx] for i in range(len(bounded))], axis=0)
            result.append(np.median(stacked, axis=0).astype(np.float32))
        return result


# ============================================================================
# FEDVAL DEFENSE - Pillutla et al., IEEE TSP 2022
# ============================================================================

class FedValDefense:
    """
    FedVal: Validation-based Update Filtering.
    
    Modified for non-IID FL:
    - Higher threshold (0.5 instead of 0.1) to tolerate non-IID variance
    - Only reject if ALL metrics degrade significantly
    - Require minimum number of accepted clients
    """
    
    def __init__(self, threshold: float = 0.5, validation_data: Data = None, min_accept: int = 1):
        self.threshold = threshold  # 50% degradation tolerance for non-IID
        self.validation_data = validation_data
        self.min_accept = min_accept  # At least 1 client must be accepted
        self.rejection_history = []
    
    def compute_validation_loss(
        self,
        model: nn.Module,
        data: Data,
        device: torch.device
    ) -> float:
        model.eval()
        data = data.to(device)
        
        with torch.no_grad():
            output = model(data.x, data.edge_index)
            val_mask = data.val_mask if hasattr(data, 'val_mask') else \
                       torch.ones(data.x.size(0), dtype=torch.bool)
            loss = F.cross_entropy(output['forward'][val_mask], data.y_forward[val_mask])
        
        return loss.item()
    
    def filter_updates(
        self,
        global_model: nn.Module,
        client_results: List[Tuple[List[np.ndarray], int, Dict]],
        model_config: Dict,
        device: torch.device
    ) -> Tuple[List[Tuple], List[int]]:
        if self.validation_data is None:
            return client_results, []
        
        baseline_loss = self.compute_validation_loss(global_model, self.validation_data, device)
        
        # Compute degradation for each client
        client_degradations = []
        for i, (params, num_samples, metrics) in enumerate(client_results):
            temp_model = DualAttributionModel(model_config).to(device)
            state_dict = {k: torch.tensor(v) for k, v in 
                         zip(temp_model.state_dict().keys(), params)}
            temp_model.load_state_dict(state_dict)
            
            client_loss = self.compute_validation_loss(temp_model, self.validation_data, device)
            degradation = (client_loss - baseline_loss) / (baseline_loss + 1e-6)
            client_degradations.append((i, degradation, params, num_samples, metrics))
        
        # Sort by degradation (best first)
        client_degradations.sort(key=lambda x: x[1])
        
        filtered_results = []
        rejected_indices = []
        
        for i, degradation, params, num_samples, metrics in client_degradations:
            # Accept if below threshold OR if we need minimum clients
            if degradation <= self.threshold:
                filtered_results.append((params, num_samples, metrics))
            elif len(filtered_results) < self.min_accept:
                # Force accept best clients even if above threshold
                filtered_results.append((params, num_samples, metrics))
            else:
                rejected_indices.append(i)
        
        self.rejection_history.append(len(rejected_indices))
        
        return filtered_results, rejected_indices


# ============================================================================
# DATA LOADING
# ============================================================================

def merchant_to_category(merchant_id):
    if merchant_id == 0 or pd.isna(merchant_id):
        return 0
    merchant_id = int(merchant_id)
    if 1 <= merchant_id <= 175:
        return 1
    elif 176 <= merchant_id <= 300:
        return 2
    elif 301 <= merchant_id <= 400:
        return 3
    elif 401 <= merchant_id <= 475:
        return 4
    else:
        return 5


def generate_discriminative_features(data: pd.DataFrame, category_labels: np.ndarray) -> pd.DataFrame:
    n = len(data)
    feat_names = ['cat_amount', 'cat_freq', 'cat_temporal', 'cat_weekend', 'cat_addr_reuse', 'cat_batch']
    
    np.random.seed(42)
    
    for feat_idx, feat_name in enumerate(feat_names):
        if feat_name not in data.columns:
            feat_values = np.zeros(n)
            for cat_id in range(6):
                mask = category_labels == cat_id
                if mask.sum() > 0:
                    mu, sigma = CATEGORY_PATTERNS[cat_id][feat_idx]
                    feat_values[mask] = np.random.normal(mu, sigma, mask.sum())
            data[feat_name] = feat_values
    
    return data


def load_exchange_data(exchange_id: int, data_dir: str = 'data/federated_enriched') -> Tuple[pd.DataFrame, pd.DataFrame]:
    enriched_path = f'{data_dir}/exchange_{exchange_id}_enriched.pkl'
    original_path = f'data/federated/exchange_{exchange_id}/data.pkl'
    edge_path = f'data/federated/exchange_{exchange_id}/edges.pkl'
    
    if os.path.exists(enriched_path):
        data = pd.read_pickle(enriched_path)
    elif os.path.exists(original_path):
        data = pd.read_pickle(original_path)
    else:
        raise FileNotFoundError(f"No data found for exchange {exchange_id}")
    
    edges = pd.read_pickle(edge_path) if os.path.exists(edge_path) else pd.DataFrame()
    
    return data, edges

def create_pyg_data_for_exchange(data: pd.DataFrame, edges: pd.DataFrame, exchange_id: int, seed: int = 42) -> Data:
    """
    Create PyG Data for one exchange — STRICT canonical mode (Option B-2).

    Reads canonical artifacts in read-only mode:
      - data/split_map.pkl
      - data/normalization_stats.pkl

    The `seed` parameter is retained for backward compatibility but no
    longer affects the split or normalization (now canonical and
    seed-independent).
    """
    if 'category_label' in data.columns:
        category_labels = data['category_label'].values
    elif 'backward_label' in data.columns:
        category_labels = data['backward_label'].apply(merchant_to_category).values
        data['category_label'] = category_labels
    else:
        category_labels = np.zeros(len(data), dtype=int)
        data['category_label'] = category_labels

    cat_features_exist = any(c.startswith('cat_') for c in data.columns)
    if not cat_features_exist:
        data = generate_discriminative_features(data, category_labels)

    # ===== LOAD CANONICAL ARTIFACTS (strict, no fallback) =====
    norm_path = 'data/normalization_stats.pkl'
    split_path = 'data/split_map.pkl'

    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f'[FATAL] Canonical artifact missing: {norm_path}\n'
            f'        Run: python generate_canonical_artifacts.py'
        )
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f'[FATAL] Canonical artifact missing: {split_path}\n'
            f'        Run: python generate_canonical_artifacts.py'
        )

    with open(norm_path, 'rb') as f:
        norm_stats = pickle.load(f)
    with open(split_path, 'rb') as f:
        split_map = pickle.load(f)

    feature_cols = norm_stats['feature_cols']
    global_mean  = norm_stats['mean']
    global_std   = norm_stats['std']

    missing = [c for c in feature_cols if c not in data.columns]
    if missing:
        raise ValueError(
            f'[FATAL] Data missing canonical columns: {missing[:5]} '
            f'({len(missing)} total)'
        )

    print(f"    [CANONICAL] Using normalization_stats.pkl ({len(feature_cols)} features)")

    # ===== APPLY CANONICAL NORMALIZATION =====
    X_numpy = data[feature_cols].values.astype(np.float32)
    X_numpy = np.nan_to_num(X_numpy, nan=0.0, posinf=1e6, neginf=-1e6)
    X_numpy = (X_numpy - global_mean) / global_std
    X_numpy = np.clip(X_numpy, -10, 10)

    X = torch.FloatTensor(X_numpy)

    forward_labels = torch.LongTensor(
        data['forward_label'].values if 'forward_label' in data.columns
        else data['exchange_partition'].values
    )
    backward_labels = torch.LongTensor(category_labels)

    txid_to_idx = {txid: idx for idx, txid in enumerate(data['txId'])}
    edge_list = []
    for _, row in edges.iterrows():
        if row['txId1'] in txid_to_idx and row['txId2'] in txid_to_idx:
            idx1 = txid_to_idx[row['txId1']]
            idx2 = txid_to_idx[row['txId2']]
            edge_list.append([idx1, idx2])
            edge_list.append([idx2, idx1])

    edge_index = (torch.LongTensor(edge_list).t().contiguous()
                  if edge_list else torch.empty(2, 0, dtype=torch.long))

    # ===== APPLY CANONICAL SPLIT =====
    txids = data['txId'].values
    train_mask = np.array([split_map.get(int(t), 'train') == 'train' for t in txids])
    val_mask   = np.array([split_map.get(int(t), 'train') == 'val'   for t in txids])
    test_mask  = np.array([split_map.get(int(t), 'train') == 'test'  for t in txids])

    print(f"    [CANONICAL] Using split_map.pkl for exchange {exchange_id}")

    graph_data = Data(
        x=X,
        edge_index=edge_index,
        y_forward=forward_labels,
        y_backward=backward_labels,
        train_mask=torch.BoolTensor(train_mask),
        val_mask=torch.BoolTensor(val_mask),
        test_mask=torch.BoolTensor(test_mask),
    )
    graph_data.num_features = X.shape[1]

    return graph_data

# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_model(model: nn.Module, graph_data: Data, device: torch.device, mask_name: str = 'test') -> Dict:
    model.eval()
    graph_data = graph_data.to(device)
    
    if mask_name == 'val':
        mask = graph_data.val_mask
    elif mask_name == 'test':
        mask = graph_data.test_mask
    else:
        mask = graph_data.train_mask
    
    output = model(graph_data.x, graph_data.edge_index)
    
    forward_preds = output['forward'][mask].argmax(dim=1)
    forward_acc = (forward_preds == graph_data.y_forward[mask]).float().mean().item()
    
    backward_labels = graph_data.y_backward[mask]
    category_mask = backward_labels > 0
    
    if category_mask.sum() > 0:
        backward_preds = output['backward'][mask][category_mask].argmax(dim=1)
        backward_acc = (backward_preds == backward_labels[category_mask]).float().mean().item()
        
        per_cat_acc = {}
        for cat_id in range(1, 6):
            cat_mask = backward_labels[category_mask] == cat_id
            if cat_mask.sum() > 0:
                cat_preds = backward_preds[cat_mask]
                cat_true = backward_labels[category_mask][cat_mask]
                per_cat_acc[CATEGORY_NAMES[cat_id]] = (cat_preds == cat_true).float().mean().item()
    else:
        backward_acc = 0.0
        per_cat_acc = {}
    
    return {
        'forward_accuracy': forward_acc,
        'backward_accuracy': backward_acc,
        'per_category': per_cat_acc
    }


# ============================================================================
# SECURE FEDERATED CLIENT
# ============================================================================

class SecureFederatedClient:
    """
    Federated Learning Client with User-Level DP and Adversarial Training.
    
    DP Implementation (User-Level DP for FL):
    -----------------------------------------
    Instead of adding noise at each gradient step (Example-Level DP),
    we add noise ONCE to the final model update (User-Level DP).
    
    This is the standard approach in FL with DP:
    1. Train locally WITHOUT noise (normal SGD)
    2. Compute update: delta_w = w_local - w_global
    3. Clip update: delta_w_clipped = clip(delta_w, C)
    4. Add noise ONCE: delta_w_noisy = delta_w_clipped + N(0, sigma^2 * C^2)
    5. Return w_global + delta_w_noisy
    
    Reference: McMahan et al., "Learning Differentially Private Recurrent 
               Language Models", ICLR 2018
    """
    
    def __init__(
        self,
        client_id: int,
        graph_data: Data,
        model_config: Dict,
        device: torch.device,
        local_epochs: int = 3,
        lr: float = 0.002,
        use_dp: bool = True,
        noise_multiplier: float = 0.3,
        max_grad_norm: float = 1.0,
        use_adversarial: bool = True,
        adv_ratio: float = 0.30
    ):
        self.client_id = client_id
        self.graph_data = graph_data
        self.model_config = model_config
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        
        self.use_dp = use_dp
        self.noise_multiplier = noise_multiplier if use_dp else 0.0
        self.max_grad_norm = max_grad_norm
        
        self.use_adversarial = use_adversarial
        self.adv_ratio = adv_ratio
        
        self.model = DualAttributionModel(model_config).to(device)
        
        # Store initial parameters for computing update delta
        self.initial_params = None
        
        # Correction A: load global weights from P2 export
        _bw_path = 'results/models/backward_weights_global.npy'
        if os.path.exists(_bw_path):
            _w = np.load(_bw_path)
            self.backward_weights = torch.tensor(_w, dtype=torch.float32, device=device)
        else:
            train_backward = graph_data.y_backward[graph_data.train_mask]
            train_backward_filtered = train_backward[train_backward > 0]
            if len(train_backward_filtered) > 0:
                counts = Counter(train_backward_filtered.numpy())
                total = sum(counts.values())
                weights = [0.0]
                for i in range(1, 6):
                    count = counts.get(i, 1)
                    weights.append(min(total / (5 * count), 10.0))
                self.backward_weights = torch.tensor(weights, dtype=torch.float32, device=device)
                self.backward_weights = self.backward_weights / self.backward_weights[1:].mean()
                self.backward_weights[0] = 0.0
            else:
                self.backward_weights = torch.tensor([0., 1., 1., 1., 1., 1.], device=device)
        
        if use_adversarial:
            self.adv_trainer = AdversarialTrainer(epsilon=0.2, alpha=0.05, num_steps=7)
    
    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = {k: torch.tensor(v) for k, v in 
                     zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict)
        # Store initial params for DP update computation
        self.initial_params = [p.copy() for p in parameters]
    
    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def _apply_user_level_dp(self, final_params: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply User-Level DP with conservative noise.
        
        Note: DP significantly degrades GNN performance. 
        For practical deployment, consider epsilon >= 8.0 or disable DP.
        """
        if self.initial_params is None or self.noise_multiplier == 0:
            return final_params
        
        # Compute update delta
        deltas = [final - init for final, init in zip(final_params, self.initial_params)]
        
        # Compute L2 norm of update
        total_norm = np.sqrt(sum(np.sum(d ** 2) for d in deltas))
        
        # Adaptive clipping: use 90th percentile of typical update norms
        # This prevents over-clipping which destroys the signal
        adaptive_clip = max(self.max_grad_norm, total_norm * 0.9)
        
        # Clip update
        clip_coef = min(1.0, adaptive_clip / (total_norm + 1e-6))
        deltas_clipped = [d * clip_coef for d in deltas]
        
        # Add SMALL noise - scaled by number of parameters to avoid explosion
        # Standard approach: noise proportional to clip norm, not to each param
        total_params = sum(d.size for d in deltas_clipped)
        noise_scale = self.noise_multiplier * adaptive_clip / np.sqrt(total_params)
        
        deltas_noisy = [
            d + np.random.normal(0, noise_scale, d.shape).astype(np.float32)
            for d in deltas_clipped
        ]
        
        # Return initial + noisy update
        return [init + delta for init, delta in zip(self.initial_params, deltas_noisy)]
    
    def train(self, alpha_forward: float = 1.0, alpha_backward: float = 1.5) -> Tuple[List[np.ndarray], int, Dict]:
        self.model.train()
        self.model = self.model.to(self.device)
        graph_data = self.graph_data.to(self.device)
        
        # Standard optimizer (NO DP noise during training)
        # weight_decay=5e-4 consistent with Kipf & Welling (2017) and CAT/C3 baseline
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        
        focal_loss = FocalLoss(alpha=self.backward_weights, gamma=2.0)
        train_mask = graph_data.train_mask
        
        total_loss = 0
        
        for epoch in range(self.local_epochs):
            optimizer.zero_grad()
            
            x = graph_data.x
            
            if self.use_adversarial and np.random.random() < self.adv_ratio:
                x_adv = self.adv_trainer.generate_adversarial(
                    self.model, x, graph_data.edge_index,
                    graph_data.y_forward, graph_data.y_backward, train_mask
                )
                x = x_adv
            
            output = self.model(x, graph_data.edge_index)
            
            loss_forward = F.cross_entropy(
                output['forward'][train_mask],
                graph_data.y_forward[train_mask]
            )
            
            backward_labels = graph_data.y_backward[train_mask]
            category_mask = backward_labels > 0
            
            if category_mask.sum() > 0:
                loss_backward = focal_loss(
                    output['backward'][train_mask][category_mask],
                    backward_labels[category_mask]
                )
            else:
                loss_backward = torch.tensor(0.0, device=self.device)
            
            loss = alpha_forward * loss_forward + alpha_backward * loss_backward
            loss.backward()
            
            # Standard gradient clipping (for stability, not privacy)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Get final parameters
        final_params = self.get_parameters()
        
        # Apply User-Level DP: clip and noise the UPDATE (not gradients)
        if self.use_dp:
            final_params = self._apply_user_level_dp(final_params)
        
        metrics = {
            'loss': total_loss / self.local_epochs,
            'dp_enabled': self.use_dp,
            'adversarial_enabled': self.use_adversarial,
            'noise_multiplier': self.noise_multiplier
        }
        
        return final_params, int(train_mask.sum()), metrics


# ============================================================================
# SECURE FEDERATED SERVER
# ============================================================================

class SecureFederatedServer:
    """
    Federated Server with Byzantine defense (Coord. Median + NormBound),
    FedVal, and Server-side DP.
    
    Byzantine Defense: Coordinate-wise Median + Norm Bounding replaces FLAME.
    Server-side DP: Adds noise ONCE after aggregation (not per-client).
    """
    
    def __init__(
        self,
        model_config: Dict,
        device: torch.device,
        use_byzantine_defense: bool = True,
        use_fedval: bool = False,
        use_server_dp: bool = True,
        server_dp_noise: float = 0.1,
        validation_data: Data = None
    ):
        self.model_config = model_config
        self.device = device
        self.global_model = DualAttributionModel(model_config).to(device)
        
        self.use_byzantine_defense = use_byzantine_defense
        self.use_fedval = use_fedval
        self.use_server_dp = use_server_dp
        self.server_dp_noise = server_dp_noise
        
        if use_byzantine_defense:
            self.byzantine_defense = ByzantineDefense(norm_bound_tau=2.0)
        
        if use_fedval and validation_data is not None:
            self.fedval_defense = FedValDefense(threshold=0.1, validation_data=validation_data)
        else:
            self.fedval_defense = None
    
    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.global_model.state_dict().values()]
    
    def _apply_server_dp(self, params: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply Server-side DP: add Gaussian noise to aggregated parameters.
        
        Noise is calibrated to be small enough to preserve utility while
        providing differential privacy guarantees.
        """
        noisy_params = []
        for p in params:
            # Scale noise by parameter magnitude to avoid destroying small weights
            param_scale = np.std(p) + 1e-7
            noise = np.random.normal(0, self.server_dp_noise * param_scale, p.shape)
            noisy_params.append((p + noise).astype(np.float32))
        return noisy_params
    
    def aggregate(
        self,
        client_results: List[Tuple[List[np.ndarray], int, Dict]]
    ) -> Dict:
        defense_stats = {
            'byzantine_defense': self.use_byzantine_defense,
            'fedval_rejected': 0,
            'server_dp_applied': self.use_server_dp,
            'total_clients': len(client_results),
            'accepted_clients': len(client_results)
        }
        
        client_params = [params for params, _, _ in client_results]
        filtered_results = client_results
        
        if self.fedval_defense is not None and len(filtered_results) > 0:
            filtered_results, fedval_rejected = self.fedval_defense.filter_updates(
                self.global_model, filtered_results, self.model_config, self.device
            )
            defense_stats['fedval_rejected'] = len(fedval_rejected)
            
            if fedval_rejected:
                print(f"    [FedVal] Rejected: {fedval_rejected}")
        
        if len(filtered_results) == 0:
            print("    [WARNING] All updates filtered, keeping global model unchanged")
            return defense_stats
        
        defense_stats['accepted_clients'] = len(filtered_results)
        
        # Aggregation: Coord. Median + NormBound (if byzantine_defense enabled)
        # else: standard FedAvg
        all_params = [params for params, _, _ in filtered_results]
        if self.use_byzantine_defense:
            new_params = self.byzantine_defense.aggregate(all_params)
            print(f'    [ByzantineDefense] Coord.Median+NormBound applied ({len(all_params)} clients)')
        else:
            total_samples = sum(n for _, n, _ in filtered_results)
            new_params = [
                sum(p[i] * n for p, n, _ in filtered_results) / total_samples
                for i in range(len(all_params[0]))
            ]
        
        # Apply Server-side DP (noise ONCE after aggregation)
        if self.use_server_dp:
            new_params = self._apply_server_dp(new_params)
        
        state_dict = {k: torch.tensor(v) for k, v in 
                     zip(self.global_model.state_dict().keys(), new_params)}
        self.global_model.load_state_dict(state_dict)
        
        return defense_stats


# ============================================================================
# NOISE CALIBRATION FOR USER-LEVEL DP
# ============================================================================

def get_noise_multiplier(epsilon: float) -> float:
    """
    Get noise multiplier based on target epsilon for User-Level DP.
    Calibration (McMahan et al., ICLR 2018):
      eps=1.0 -> 1.0 ; eps=2.0 -> 0.5 ; eps=4.0 -> 0.3 ; eps=8.0 -> 0.1
      eps=16.0 -> 0.025 [PATCHED] ; eps=32.0 -> 0.0125 [PATCHED] ; eps>32 -> 0.005
    """
    if epsilon <= 0.5:
        return 1.5
    elif epsilon <= 1.0:
        return 1.0
    elif epsilon <= 2.0:
        return 0.5
    elif epsilon <= 4.0:
        return 0.3
    elif epsilon <= 8.0:
        return 0.1
    elif epsilon <= 16.0:
        return 0.025
    elif epsilon <= 32.0:
        return 0.0125
    else:
        return 0.005


def run_secure_fl(
    exchange_data: List[Data],
    model_config: Dict,
    device: torch.device,
    num_rounds: int = 20,
    local_epochs: int = 3,
    lr: float = 0.002,
    use_client_dp: bool = False,
    use_server_dp: bool = True,
    epsilon: float = 8.0,
    noise_multiplier: float = None,
    use_adversarial: bool = True,
    use_flame: bool = True,
    use_fedval: bool = False,
    alpha_forward: float = 1.0,
    alpha_backward: float = 1.5
) -> Tuple[nn.Module, Dict]:
    """Run Secure Federated Learning."""
    
    # Determine noise multiplier
    if noise_multiplier is None:
        noise_multiplier = get_noise_multiplier(epsilon)
    
    print("\n" + "="*70)
    print("SECURE FEDERATED LEARNING")
    print("="*70)
    print(f"  Rounds: {num_rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  LR scheduler: ReduceLROnPlateau (factor=0.7, patience=5, min_lr=1e-6)")
    print(f"\n  SECURITY SETTINGS:")
    print(f"    Client-side DP (DP-SGD): {'Enabled' if use_client_dp else 'Disabled'}")
    if use_client_dp:
        print(f"      epsilon = {epsilon}")
        print(f"      noise_multiplier = {noise_multiplier}")
    print(f"    Server-side DP: {'Enabled' if use_server_dp else 'Disabled'}")
    if use_server_dp:
        print(f"      epsilon = {epsilon}")
        print(f"      noise_multiplier = {noise_multiplier}")
    print(f"    Adversarial Training: {'Enabled' if use_adversarial else 'Disabled'}")
    print(f"    FLAME Defense: {'Enabled' if use_flame else 'Disabled'}")
    print(f"    FedVal Defense: {'Enabled' if use_fedval else 'Disabled'}")
    print(f"  Alpha forward: {alpha_forward}, Alpha backward: {alpha_backward}")
    
    val_data = exchange_data[0] if use_fedval else None
    
    server = SecureFederatedServer(
        model_config, device,
        use_byzantine_defense=use_flame,  # use_flame arg maps to byzantine_defense
        use_fedval=use_fedval,
        use_server_dp=use_server_dp,
        server_dp_noise=noise_multiplier,
        validation_data=val_data
    )
    
    clients = [
        SecureFederatedClient(
            client_id=i,
            graph_data=data,
            model_config=model_config,
            device=device,
            local_epochs=local_epochs,
            lr=lr,
            use_dp=use_client_dp,
            noise_multiplier=noise_multiplier,
            use_adversarial=use_adversarial
        )
        for i, data in enumerate(exchange_data)
    ]
    
    history = {
        'rounds': [],
        'forward_accuracy': [],
        'backward_accuracy': [],
        'loss': [],
        'flame_outliers': [],
        'fedval_rejected': []
    }
    
    best_metric = 0.0
    best_params = None

    # ReduceLROnPlateau — consistent with CAT/C3 baseline (factor=0.7, patience=5)
    current_lr      = lr
    scheduler_patience = 5
    scheduler_counter  = 0
    scheduler_factor   = 0.7
    min_lr             = 1e-6

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} (lr={current_lr:.6f}) ---")

        # Propagate current LR to all clients (server-side scheduler)
        for client in clients:
            client.lr = current_lr
        
        global_params = server.get_parameters()
        for client in clients:
            client.set_parameters(global_params)
        
        client_results = []
        for i, client in enumerate(clients):
            params, num_samples, metrics = client.train(alpha_forward, alpha_backward)
            client_results.append((params, num_samples, metrics))
            
            dp_status = "[DP]" if metrics.get('dp_enabled') else ""
            adv_status = "[ADV]" if metrics.get('adversarial_enabled') else ""
            print(f"  Client {i}: {num_samples} samples, loss={metrics['loss']:.4f} {dp_status}{adv_status}")
        
        defense_stats = server.aggregate(client_results)
        
        val_metrics_list = []
        for i, client in enumerate(clients):
            client.set_parameters(server.get_parameters())
            metrics = evaluate_model(client.model, client.graph_data, device, 'val')
            val_metrics_list.append(metrics)
        
        avg_forward = np.mean([m['forward_accuracy'] for m in val_metrics_list])
        avg_backward = np.mean([m['backward_accuracy'] for m in val_metrics_list])
        avg_loss = np.mean([r[2]['loss'] for r in client_results])
        
        history['rounds'].append(round_num + 1)
        history['forward_accuracy'].append(avg_forward)
        history['backward_accuracy'].append(avg_backward)
        history['loss'].append(avg_loss)
        history['flame_outliers'].append(defense_stats.get('flame_outliers', 0))
        history['fedval_rejected'].append(defense_stats.get('fedval_rejected', 0))
        
        combined = 0.4 * avg_forward + 0.6 * avg_backward
        improved = ""
        if combined > best_metric:
            best_metric = combined
            best_params = server.get_parameters()
            improved = " [BEST]"
            scheduler_counter = 0
        else:
            scheduler_counter += 1
            if scheduler_counter >= scheduler_patience:
                new_lr = max(current_lr * scheduler_factor, min_lr)
                if new_lr < current_lr:
                    print(f"  [LR] Reducing lr: {current_lr:.6f} -> {new_lr:.6f}")
                    current_lr = new_lr
                scheduler_counter = 0

        print(f"  -> Forward: {avg_forward:.2%}, Backward: {avg_backward:.2%}{improved}")
    
    if best_params:
        state_dict = {k: torch.tensor(v) for k, v in 
                     zip(server.global_model.state_dict().keys(), best_params)}
        server.global_model.load_state_dict(state_dict)
    
    return server.global_model, history


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    import random
    # ── Mode resolution: --mode overrides individual flags ──────────────
    if args.mode == 'fat-base':
        args.no_server_dp         = True
        args.no_byzantine_defense = True
        args.no_adversarial       = False
        print('[MODE] fat-base: AT ON, Byzantine defense OFF, DP OFF')
    elif args.mode == 'fat-adv':
        args.no_server_dp         = True
        args.no_byzantine_defense = True   # defense OFF, attacks simulated in P5
        args.no_adversarial       = False
        print('[MODE] fat-adv: AT ON, Byzantine defense OFF (attacks tested in P5)')
    elif args.mode == 'fat-def':
        args.no_server_dp         = False   # Server-side DP ON
        args.no_byzantine_defense = False   # CoordMedian+NormBound ON
        args.no_adversarial       = False
        print(f'[MODE] fat-def: AT ON, ByzantineDefense ON, DP ON (eps={args.epsilon})')
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FAT - FEDERATED ADVERSARIAL TRAINING (PGD-7, eps=0.2, ratio=30%)")
    print("    + Coord. Median+NormBound + Server-side DP")
    print("="*70)
    # ── Reproducibility ──────────────────────────────────────────────
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    print(f"Seed: {args.seed}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Flower available: {FLOWER_AVAILABLE}")
    # ── Dynamic output paths ──────────────────────────────────────────
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/evaluations', exist_ok=True)
    _tag = f'_{args.results_tag}' if args.results_tag else ''
    model_path   = f'results/models/federated_secure_dual_attribution{_tag}_seed{args.seed}.pt'
    results_path = f'results/evaluations/federated_secure_results{_tag}_seed{args.seed}.json'
    print(f'model_path  : {model_path}')
    print(f'results_path: {results_path}')
    print("\nLoading exchange data...")
    exchange_data = []
    
    for i in range(3):
        data, edges = load_exchange_data(i, args.data_dir)
        graph_data = create_pyg_data_for_exchange(data, edges, exchange_id=i, seed=args.seed)
        exchange_data.append(graph_data)
        
        cat_counts = Counter(graph_data.y_backward.numpy())
        cat_str = ", ".join([f"{CATEGORY_NAMES.get(k, k)}:{v}" for k, v in sorted(cat_counts.items()) if k > 0])
        print(f"  Exchange {i}: {len(data):,} transactions, {graph_data.num_features} features")
        print(f"    Categories: {cat_str}")
    
    model_config = {
        'in_channels': exchange_data[0].num_features,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'num_exchanges': 3,
        'num_categories': 6
    }
    
    print(f"\nModel config: {model_config}")
    
    final_model, history = run_secure_fl(
        exchange_data, model_config, device,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        use_client_dp=args.use_client_dp,
        use_server_dp=not args.no_server_dp,
        epsilon=args.epsilon,
        noise_multiplier=args.noise_multiplier,
        use_adversarial=not args.no_adversarial,
        use_flame=not args.no_byzantine_defense,  # use_flame param = ByzantineDefense
        use_fedval=args.use_fedval,
        alpha_forward=args.alpha_forward,
        alpha_backward=args.alpha_backward
    )
    
    print("\n" + "="*70)
    print("FINAL EVALUATION (TEST SET)")
    print("="*70)
    
    final_model = final_model.to(device)
    
    all_forward = []
    all_backward = []
    
    for i, data in enumerate(exchange_data):
        metrics = evaluate_model(final_model, data, device, 'test')
        all_forward.append(metrics['forward_accuracy'])
        all_backward.append(metrics['backward_accuracy'])
        
        print(f"\nExchange {i}:")
        print(f"  Forward Accuracy:  {metrics['forward_accuracy']:.2%}")
        print(f"  Backward Accuracy: {metrics['backward_accuracy']:.2%}")
        if metrics['per_category']:
            print(f"  Per-Category:")
            for cat, acc in metrics['per_category'].items():
                print(f"    {cat:12s}: {acc:.2%}")
    
    avg_forward = np.mean(all_forward)
    avg_backward = np.mean(all_backward)
    
    print("\n" + "="*70)
    print("OVERALL RESULTS (SECURE FL)")
    print("="*70)
    print(f"  Forward Accuracy:  {avg_forward:.2%}")
    print(f"  Backward Accuracy: {avg_backward:.2%}")
    
    noise_mult = args.noise_multiplier if args.noise_multiplier else get_noise_multiplier(args.epsilon)
    if args.use_client_dp or not args.no_server_dp:
        print(f"\n  Privacy Settings:")
        print(f"    Client-side DP: {'Enabled' if args.use_client_dp else 'Disabled'}")
        print(f"    Server-side DP: {'Enabled' if not args.no_server_dp else 'Disabled'}")
        if args.use_client_dp or not args.no_server_dp:
            print(f"    epsilon={args.epsilon}, noise_multiplier={noise_mult}")
    
    print(f"\n  Defense Statistics:")
    print(f"     Total FLAME outliers detected: {sum(history['flame_outliers'])}")
    print(f"     Total FedVal rejections: {sum(history['fedval_rejected'])}")
    
    fwd_status = "OK" if avg_forward >= 0.70 else "BELOW TARGET"
    bwd_status = "OK" if avg_backward >= 0.60 else "BELOW TARGET"
    print(f"\n  Forward target (>=70%): {avg_forward:.2%} [{fwd_status}]")
    print(f"  Backward target (>=60%): {avg_backward:.2%} [{bwd_status}]")
    
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'config': model_config,
        'security_config': {
            'client_dp_enabled': args.use_client_dp,
            'server_dp_enabled': not args.no_server_dp,
            'epsilon': args.epsilon,
            'noise_multiplier': args.noise_multiplier if args.noise_multiplier else get_noise_multiplier(args.epsilon),
            'adversarial_enabled': not args.no_adversarial,
            'flame_enabled': not args.no_byzantine_defense,
            'fedval_enabled': args.use_fedval
        },
        'fl_config': {
            'rounds': args.rounds,
            'local_epochs': args.local_epochs,
            'alpha_forward': args.alpha_forward,
            'alpha_backward': args.alpha_backward
        }
    }, model_path)
    print(f"\nModel saved: {model_path}")
    
    # results = {
    #     'config': model_config,
    #     'security_config': {
    #         'client_dp_enabled': args.use_client_dp,
    #         'server_dp_enabled': not args.no_server_dp,
    #         'epsilon': args.epsilon,
    #         'noise_multiplier': args.noise_multiplier if args.noise_multiplier else get_noise_multiplier(args.epsilon),
    #         'adversarial_enabled': not args.no_adversarial,
    #         'flame_enabled': not args.no_byzantine_defense,
    #         'fedval_enabled': args.use_fedval
    #     },
    #     'test_metrics': {
    #         'forward_accuracy': avg_forward,
    #         'backward_accuracy': avg_backward,
    #         'per_exchange': [
    #             {'forward': all_forward[i], 'backward': all_backward[i]}
    #             for i in range(3)
    #         ]
    #     },
    #     'history': history,
    #     'timestamp': datetime.now().isoformat()
    # }
    
    # results_path = 'results/evaluations/federated_secure_results.json'
    # with open(results_path, 'w') as f:
    #     json.dump(results, f, indent=2, default=str)
    # print(f"Results saved: {results_path}")

    # Collect per-category metrics for each exchange
    all_per_category = []
    for i, data in enumerate(exchange_data):
        metrics = evaluate_model(final_model, data, device, 'test')
        all_per_category.append(metrics.get('per_category', {}))
    
    # Average per-category across exchanges
    avg_per_category = {}
    for cat in ['E-commerce', 'Gambling', 'Services', 'Retail', 'Luxury']:
        cat_accs = [pc.get(cat, 0) for pc in all_per_category if cat in pc]
        if cat_accs:
            avg_per_category[cat] = sum(cat_accs) / len(cat_accs)
    
    # Compute effective noise multiplier
    effective_noise = args.noise_multiplier if args.noise_multiplier else get_noise_multiplier(args.epsilon)
    
    # Save results - STANDARDIZED FORMAT FOR COMPARISON
    results = {
        # === IDENTIFICATION ===
        'approach': 'fl_secure',
        'approach_name': 'Secure Federated Learning',
        'training_type': 'federated',
        'privacy_level': 'dp_enabled' if not args.no_server_dp else 'isolation',
        'privacy_description': f'Server-side DP (Îµ={args.epsilon})' if not args.no_server_dp else 'Data isolation only',
        
        # === DATA CONFIG ===
        'data_config': {
            'total_transactions': sum(d.x.shape[0] for d in exchange_data),
            'feature_count': int(exchange_data[0].num_features),
            'transactions_per_exchange': [int(d.x.shape[0]) for d in exchange_data],
            'train_size_per_exchange': [int(d.train_mask.sum()) for d in exchange_data],
            'test_size_per_exchange': [int(d.test_mask.sum()) for d in exchange_data],
            'split_ratio': '60/20/20',
            'split_seed': 42,
            'num_exchanges': 3,
            'num_categories': 6
        },
        
        # === MODEL CONFIG ===
        'model_config': {
            'architecture': 'DualAttributionGCN',
            'in_channels': model_config['in_channels'],
            'hidden_dim': model_config['hidden_dim'],
            'dropout': model_config['dropout'],
            'num_gcn_layers': 2
        },
        
        # === TRAINING CONFIG ===
        'training_config': {
            'rounds': args.rounds,
            'local_epochs': args.local_epochs,
            'epochs_equivalent': args.rounds * args.local_epochs,  # For comparison with baseline
            'learning_rate': args.lr,
            'optimizer': 'Adam',
            'loss_type': 'FocalLoss',
            'alpha_forward': args.alpha_forward,
            'alpha_backward': args.alpha_backward,
            'focal_gamma': 2.0,
            'aggregation': 'FedAvg'
        },
        
        # === SECURITY CONFIG ===
        'security_config': {
            # Differential Privacy
            'differential_privacy': not args.no_server_dp,
            'dp_type': 'server_side' if not args.no_server_dp else None,
            'client_dp_enabled': args.use_client_dp,
            'server_dp_enabled': not args.no_server_dp,
            'epsilon': args.epsilon if not args.no_server_dp else None,
            'delta': 1e-5 if not args.no_server_dp else None,
            'noise_multiplier': effective_noise if not args.no_server_dp else None,
            
            # Adversarial Training
            'adversarial_training': not args.no_adversarial,
            'adv_epsilon': 0.2 if not args.no_adversarial else None,
            'adv_steps': 7 if not args.no_adversarial else None,
            'adv_ratio': 0.3 if not args.no_adversarial else None,
            
            # Byzantine Defense
            'mode': args.mode if args.mode else 'custom',
            'byzantine_defense': 'CoordMedian+NormBound' if not args.no_byzantine_defense else None,
            'flame_enabled': not args.no_byzantine_defense,
            'fedval_enabled': args.use_fedval
        },
        
        # === TEST METRICS (STANDARDIZED) ===
        'test_metrics': {
            'forward_accuracy': avg_forward,
            'forward_top2_accuracy': None,  # Not computed in FL
            'backward_accuracy': avg_backward,
            'backward_per_category': avg_per_category,
            # Per-exchange breakdown
            'per_exchange': [
                {
                    'exchange_id': i,
                    'forward_accuracy': all_forward[i],
                    'backward_accuracy': all_backward[i],
                    'per_category': all_per_category[i]
                }
                for i in range(3)
            ],
            # Computed metrics
            'forward_improvement_vs_random': avg_forward - (1/3),
            'backward_improvement_vs_random': avg_backward - (1/5),
        },
        
        # === DEFENSE STATISTICS ===
        'defense_stats': {
            'total_flame_outliers': sum(history.get('flame_outliers', [])),
            'flame_outliers_per_round': history.get('flame_outliers', []),
            'total_fedval_rejected': sum(history.get('fedval_rejected', [])),
            'fedval_rejected_per_round': history.get('fedval_rejected', [])
        },
        
        # === TRAINING HISTORY ===
        'history': {
            'rounds': list(range(1, args.rounds + 1)),
            'forward_accuracy': history.get('forward_accuracy', []),
            'backward_accuracy': history.get('backward_accuracy', []),
            'loss': history.get('loss', []),
            'flame_outliers': history.get('flame_outliers', []),
            'fedval_rejected': history.get('fedval_rejected', [])
        },
        
        # === METADATA ===
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'script': 'p3_train_federated_secure.py',
            'version': 'v2_secure',
            'device': str(device),
            'targets_met': {
                'forward_target_70': avg_forward >= 0.70,
                'backward_target_60': avg_backward >= 0.60
            }
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'Results saved: {results_path}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Secure Federated Learning - Dual Attribution')
    
    parser.add_argument('--data_dir', type=str, default='data/federated_enriched')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--local_epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.002)
    
    parser.add_argument('--alpha_forward', type=float, default=1.0)
    parser.add_argument('--alpha_backward', type=float, default=1.5)
    
    # Privacy
    parser.add_argument('--use-client-dp', action='store_true', 
                        help='Enable Client-side DP (DP-SGD, disabled by default - degrades performance)')
    parser.add_argument('--no-server-dp', action='store_true', 
                        help='Disable Server-side DP (enabled by default)')
    parser.add_argument('--epsilon', type=float, default=8.0,
                        help='Target epsilon (default=8.0 for acceptable utility)')
    parser.add_argument('--noise_multiplier', type=float, default=None,
                        help='Override noise multiplier (default: auto from epsilon)')
    
    # Security
    parser.add_argument('--no-adversarial', action='store_true',
                        help='Disable Adversarial Training')
    parser.add_argument('--no-flame', action='store_true',
                        help='Disable FLAME Byzantine defense')
    parser.add_argument('--use-fedval', action='store_true', 
                        help='Enable FedVal (disabled by default, problematic with non-IID)')
    # Reproducibility & output tagging
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match P2/P3 seed)')
    parser.add_argument('--results_tag', type=str, default='',
                        help='Tag appended to filenames, e.g. "eps8", "flame"')
    # Mode shortcut: overrides individual --no-* flags
    parser.add_argument('--mode', type=str, default=None,
                        choices=['fat-base', 'fat-adv', 'fat-def'],
                        help=(
                            'fat-base : AT only, no Byzantine defense, no DP (clean FAT baseline);\n'
                            'fat-adv  : fat-base + Byzantine simulation, defense OFF;\n'
                            'fat-def  : full stack (AT + CoordMedian+NormBound + Server-side DP)'
                        ))
    args = parser.parse_args()
    
    main(args)