"""
================================================================================
Script: p3_train_federated_flexible.py
Description: Version flexible de p3_train_federated_secure.py

BASÉ SUR: p3_train_federated_secure.py (structure identique)
AJOUTS:
  --num_layers : Nombre de couches GCN (1-5, défaut: 3)
  --experiment_name : Nom de l'expérience pour le fichier de sortie

USAGE:
    python p3_train_federated_flexible.py --experiment_name fl_full
    python p3_train_federated_flexible.py --num_layers 2 --experiment_name ablation_layers_2
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter
import copy
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CATEGORY CONFIGURATION (identique à l'original)
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
# FOCAL LOSS (identique à l'original)
# ============================================================================

class FocalLoss(nn.Module):
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
# MODEL ARCHITECTURE (avec num_layers flexible)
# ============================================================================

class DualAttributionModel(nn.Module):
    """Dual Attribution GNN Model with configurable number of GCN layers."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        in_channels = config['in_channels']
        hidden_dim = config.get('hidden_dim', 128)
        dropout = config.get('dropout', 0.3)
        num_exchanges = config.get('num_exchanges', 3)
        num_categories = config.get('num_categories', 6)
        num_layers = config.get('num_layers', 3)  # FLEXIBLE
        
        # GCN layers - nombre variable
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Première couche
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Couches intermédiaires
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Forward head (identique à l'original)
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
        
        # Backward head (identique à l'original)
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
# ADVERSARIAL TRAINER (identique à l'original)
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
# FLAME DEFENSE (identique à l'original)
# ============================================================================

class FLAMEDefense:
    """FLAME: Filtering Aggregation using Model Embedding."""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 2):
        self.eps = eps
        self.min_samples = min_samples
        self.outlier_history = []
    
    def flatten_update(self, params: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([p.flatten() for p in params])
    
    def filter_updates(
        self,
        client_updates: List[List[np.ndarray]]
    ) -> Tuple[List[List[np.ndarray]], List[int]]:
        """Filter malicious updates using DBSCAN clustering."""
        from sklearn.cluster import DBSCAN
        
        if len(client_updates) < 3:
            return client_updates, []
        
        update_vectors = np.array([self.flatten_update(u) for u in client_updates])
        
        # Handle NaN/Inf values - replace with 0
        update_vectors = np.nan_to_num(update_vectors, nan=0.0, posinf=1e6, neginf=-1e6)

        norms = np.linalg.norm(update_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        update_vectors_norm = update_vectors / norms

        # Final safety check for NaN
        if np.any(np.isnan(update_vectors_norm)):
            print("    [FLAME] Warning: NaN detected after normalization, skipping filtering")
            self.outlier_history.append(0)
            return client_updates, []
        
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(update_vectors_norm)
        labels = clustering.labels_
        
        if len(set(labels)) <= 1:
            return client_updates, []
        
        label_counts = Counter(labels)
        main_cluster = max((l for l in label_counts if l != -1), key=lambda x: label_counts[x], default=-1)
        
        outlier_indices = [i for i, l in enumerate(labels) if l != main_cluster]
        filtered_updates = [u for i, u in enumerate(client_updates) if i not in outlier_indices]
        
        self.outlier_history.append(len(outlier_indices))
        
        return filtered_updates, outlier_indices


# ============================================================================
# DATA LOADING (identique à l'original)
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


def create_pyg_data_for_exchange(data: pd.DataFrame, edges: pd.DataFrame, exchange_id: int) -> Data:
    """Create PyG Data object with standard 60/20/20 split."""
    
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
    
    feature_cols = [f'feat_{i}' for i in range(1, 166)]
    feature_cols += [c for c in data.columns if c.startswith('cat_')]
    
    proxy_prefixes = ['fee_', 'volume_', 'synthetic_hour', 'peak_activity', 'timezone_', 
                      'hour_sin', 'hour_cos', 'is_business', 'liquidity_', 'processing_', 
                      'reliability_', 'exchange_quality']
    for prefix in proxy_prefixes:
        feature_cols += [c for c in data.columns if c.startswith(prefix)]
    
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [c for c in feature_cols if c in data.columns]
    
    X_numpy = data[feature_cols].values.astype(np.float32)
    X_numpy = np.nan_to_num(X_numpy, nan=0.0, posinf=1e6, neginf=-1e6)
    
    mean = X_numpy.mean(axis=0)
    std = X_numpy.std(axis=0)
    std[std == 0] = 1
    X_numpy = (X_numpy - mean) / std
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
    
    edge_index = torch.LongTensor(edge_list).t().contiguous() if edge_list else torch.empty(2, 0, dtype=torch.long)
    
    # Standard 60/20/20 split (identique à l'original)
    n = len(data)
    np.random.seed(42)
    indices = np.random.permutation(n)
    
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True
    
    graph_data = Data(
        x=X,
        edge_index=edge_index,
        y_forward=forward_labels,
        y_backward=backward_labels,
        train_mask=torch.BoolTensor(train_mask),
        val_mask=torch.BoolTensor(val_mask),
        test_mask=torch.BoolTensor(test_mask)
    )
    graph_data.num_features = X.shape[1]
    
    return graph_data


# ============================================================================
# EVALUATION (identique à l'original)
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
# SECURE FEDERATED CLIENT (identique à l'original)
# ============================================================================

class SecureFederatedClient:
    """Federated Learning Client with User-Level DP and Adversarial Training."""
    
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
        adv_ratio: float = 0.15
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
        
        # Compute backward class weights ONCE (critical!)
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
            self.adv_trainer = AdversarialTrainer(epsilon=0.1, alpha=0.01, num_steps=3)
    
    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = {k: torch.tensor(v) for k, v in 
                     zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict)
        # Store initial params for DP update computation
        self.initial_params = [p.copy() for p in parameters]
    
    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def _apply_user_level_dp(self, final_params: List[np.ndarray]) -> List[np.ndarray]:
        """Apply User-Level DP with conservative noise."""
        if self.initial_params is None or self.noise_multiplier == 0:
            return final_params
        
        # Compute update delta
        deltas = [final - init for final, init in zip(final_params, self.initial_params)]
        
        # Compute L2 norm of update
        total_norm = np.sqrt(sum(np.sum(d ** 2) for d in deltas))
        
        # Adaptive clipping
        adaptive_clip = max(self.max_grad_norm, total_norm * 0.9)
        
        # Clip update
        clip_coef = min(1.0, adaptive_clip / (total_norm + 1e-6))
        deltas_clipped = [d * clip_coef for d in deltas]
        
        # Add noise
        total_params = sum(d.size for d in deltas_clipped)
        noise_scale = self.noise_multiplier * adaptive_clip / np.sqrt(total_params)
        
        deltas_noisy = [
            d + np.random.normal(0, noise_scale, d.shape).astype(np.float32)
            for d in deltas_clipped
        ]
        
        return [init + delta for init, delta in zip(self.initial_params, deltas_noisy)]
    
    def train(self, alpha_forward: float = 1.0, alpha_backward: float = 1.5) -> Tuple[List[np.ndarray], int, Dict]:
        self.model.train()
        self.model = self.model.to(self.device)
        graph_data = self.graph_data.to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        focal_loss = FocalLoss(alpha=self.backward_weights, gamma=2.0)
        train_mask = graph_data.train_mask
        
        total_loss = 0
        
        for epoch in range(self.local_epochs):
            optimizer.zero_grad()
            
            x = graph_data.x
            
            # Adversarial training with probability adv_ratio
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
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        final_params = self.get_parameters()
        
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
# SECURE FEDERATED SERVER (identique à l'original)
# ============================================================================

class SecureFederatedServer:
    """Federated Server with FLAME and Server-side DP defenses."""
    
    def __init__(
        self,
        model_config: Dict,
        device: torch.device,
        use_flame: bool = True,
        use_server_dp: bool = True,
        server_dp_noise: float = 0.1
    ):
        self.model_config = model_config
        self.device = device
        self.global_model = DualAttributionModel(model_config).to(device)
        
        self.use_flame = use_flame
        self.use_server_dp = use_server_dp
        self.server_dp_noise = server_dp_noise
        
        if use_flame:
            self.flame_defense = FLAMEDefense(eps=0.5, min_samples=2)
    
    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.global_model.state_dict().values()]
    
    def _apply_server_dp(self, params: List[np.ndarray]) -> List[np.ndarray]:
        """Apply Server-side DP: add Gaussian noise to aggregated parameters."""
        noisy_params = []
        for p in params:
            param_scale = np.std(p) + 1e-7
            noise = np.random.normal(0, self.server_dp_noise * param_scale, p.shape)
            noisy_params.append((p + noise).astype(np.float32))
        return noisy_params
    
    def aggregate(
        self,
        client_results: List[Tuple[List[np.ndarray], int, Dict]]
    ) -> Dict:
        defense_stats = {
            'flame_outliers': 0,
            'server_dp_applied': self.use_server_dp,
            'total_clients': len(client_results),
            'accepted_clients': len(client_results)
        }
        
        client_params = [params for params, _, _ in client_results]
        
        if self.use_flame and len(client_params) >= 3:
            filtered_params, flame_outliers = self.flame_defense.filter_updates(client_params)
            defense_stats['flame_outliers'] = len(flame_outliers)
            
            if flame_outliers:
                print(f"    [FLAME] Detected outliers: {flame_outliers}")
            
            filtered_results = [
                (p, client_results[i][1], client_results[i][2])
                for i, p in enumerate(client_params) if i not in flame_outliers
            ]
        else:
            filtered_results = client_results
        
        if len(filtered_results) == 0:
            print("    [WARNING] All updates filtered, keeping global model unchanged")
            return defense_stats
        
        defense_stats['accepted_clients'] = len(filtered_results)
        
        # FedAvg aggregation
        total_samples = sum([num_samples for _, num_samples, _ in filtered_results])
        
        new_params = []
        for param_idx in range(len(filtered_results[0][0])):
            weighted_sum = sum([
                params[param_idx] * num_samples 
                for params, num_samples, _ in filtered_results
            ])
            new_params.append(weighted_sum / total_samples)
        
        # Apply Server-side DP
        if self.use_server_dp:
            new_params = self._apply_server_dp(new_params)
        
        state_dict = {k: torch.tensor(v) for k, v in 
                     zip(self.global_model.state_dict().keys(), new_params)}
        self.global_model.load_state_dict(state_dict)
        
        return defense_stats


# ============================================================================
# NOISE CALIBRATION (identique à l'original)
# ============================================================================

def get_noise_multiplier(epsilon: float) -> float:
    """Get noise multiplier based on target epsilon."""
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
    else:
        return 0.05


# ============================================================================
# MAIN FL TRAINING
# ============================================================================

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
    alpha_forward: float = 1.0,
    alpha_backward: float = 1.5
) -> Tuple[nn.Module, Dict]:
    """Run Secure Federated Learning."""
    
    if noise_multiplier is None:
        noise_multiplier = get_noise_multiplier(epsilon)
    
    print("\n" + "="*70)
    print("SECURE FEDERATED LEARNING")
    print("="*70)
    print(f"  Rounds: {num_rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"\n  SECURITY SETTINGS:")
    print(f"    Client-side DP: {'Enabled' if use_client_dp else 'Disabled'}")
    print(f"    Server-side DP: {'Enabled' if use_server_dp else 'Disabled'}")
    if use_server_dp:
        print(f"      epsilon = {epsilon}")
        print(f"      noise_multiplier = {noise_multiplier}")
    print(f"    Adversarial Training: {'Enabled' if use_adversarial else 'Disabled'}")
    print(f"    FLAME Defense: {'Enabled' if use_flame else 'Disabled'}")
    print(f"  Alpha forward: {alpha_forward}, Alpha backward: {alpha_backward}")
    
    server = SecureFederatedServer(
        model_config, device,
        use_flame=use_flame,
        use_server_dp=use_server_dp,
        server_dp_noise=noise_multiplier
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
        'flame_outliers': []
    }
    
    best_metric = 0.0
    best_params = None
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
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
        
        # Evaluate on validation set
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
        
        combined = 0.4 * avg_forward + 0.6 * avg_backward
        improved = ""
        if combined > best_metric:
            best_metric = combined
            best_params = server.get_parameters()
            improved = " [BEST]"
        
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
    print("="*70)
    print("FLEXIBLE FEDERATED LEARNING - DUAL ATTRIBUTION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print(f"\nConfiguration:")
    print(f"  Rounds: {args.rounds}")
    print(f"  Local epochs: {args.local_epochs}")
    print(f"  Num GCN layers: {args.num_layers}")
    print(f"  Split: 60/20/20 (train/val/test)")
    print(f"  Epsilon: {args.epsilon if not args.no_server_dp else 'inf (no DP)'}")
    print(f"  FLAME: {'Enabled' if not args.no_flame else 'Disabled'}")
    print(f"  Adversarial: {'Enabled' if not args.no_adversarial else 'Disabled'}")
    
    # Load data
    print("\nLoading exchange data...")
    exchange_data = []
    for i in range(3):
        data, edges = load_exchange_data(i, args.data_dir)
        graph_data = create_pyg_data_for_exchange(data, edges, i)
        exchange_data.append(graph_data)
        
        cat_counts = Counter(graph_data.y_backward.numpy())
        print(f"  Exchange {i}: {graph_data.x.shape[0]:,} transactions, {graph_data.num_features} features")
        print(f"    Categories: " + ", ".join([f"{CATEGORY_NAMES[k]}:{v}" for k, v in sorted(cat_counts.items()) if k > 0]))
    
    # Model config
    model_config = {
        'in_channels': exchange_data[0].num_features,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'num_exchanges': 3,
        'num_categories': 6,
        'num_layers': args.num_layers
    }
    print(f"\nModel config: {model_config}")
    
    # Determine noise multiplier
    noise_multiplier = get_noise_multiplier(args.epsilon) if not args.no_server_dp else 0.0
    
    # Run FL
    model, history = run_secure_fl(
        exchange_data=exchange_data,
        model_config=model_config,
        device=device,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        use_client_dp=args.use_client_dp,
        use_server_dp=not args.no_server_dp,
        epsilon=args.epsilon,
        noise_multiplier=noise_multiplier,
        use_adversarial=not args.no_adversarial,
        use_flame=not args.no_flame,
        alpha_forward=args.alpha_forward,
        alpha_backward=args.alpha_backward
    )
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION (TEST SET)")
    print("="*70)
    
    all_forward = []
    all_backward = []
    all_per_category = []
    per_exchange_results = []
    
    for i, data in enumerate(exchange_data):
        metrics = evaluate_model(model, data, device, 'test')
        all_forward.append(metrics['forward_accuracy'])
        all_backward.append(metrics['backward_accuracy'])
        all_per_category.append(metrics['per_category'])
        
        per_exchange_results.append({
            'exchange_id': i,
            'forward_accuracy': metrics['forward_accuracy'],
            'backward_accuracy': metrics['backward_accuracy'],
            'per_category': metrics['per_category']
        })
        
        print(f"\nExchange {i}:")
        print(f"  Forward Accuracy:  {metrics['forward_accuracy']:.2%}")
        print(f"  Backward Accuracy: {metrics['backward_accuracy']:.2%}")
        if metrics['per_category']:
            print(f"  Per-Category:")
            for cat, acc in metrics['per_category'].items():
                print(f"    {cat:12s}: {acc:.2%}")
    
    avg_forward = np.mean(all_forward)
    avg_backward = np.mean(all_backward)
    
    # Average per-category
    avg_per_category = {}
    for cat in ['E-commerce', 'Gambling', 'Services', 'Retail', 'Luxury']:
        cat_accs = [pc.get(cat, 0) for pc in all_per_category if cat in pc]
        if cat_accs:
            avg_per_category[cat] = np.mean(cat_accs)
    
    print(f"\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print(f"  Forward Accuracy:  {avg_forward:.2%}")
    print(f"  Backward Accuracy: {avg_backward:.2%}")
    
    f1_score = 2 * avg_forward * avg_backward / (avg_forward + avg_backward) if (avg_forward + avg_backward) > 0 else 0
    
    # Save results
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/experiments', exist_ok=True)
    
    output_name = args.experiment_name if args.experiment_name else f"fl_secure_eps{args.epsilon if not args.no_server_dp else 'inf'}"
    
    results = {
        'approach': 'fl_secure',
        'experiment_name': output_name,
        'timestamp': datetime.now().isoformat(),
        
        'model_config': model_config,
        'training_config': {
            'rounds': args.rounds,
            'local_epochs': args.local_epochs,
            'lr': args.lr,
            'alpha_forward': args.alpha_forward,
            'alpha_backward': args.alpha_backward,
            'split_ratio': '60/20/20'
        },
        'security_config': {
            'client_dp_enabled': args.use_client_dp,
            'server_dp_enabled': not args.no_server_dp,
            'epsilon': args.epsilon if not args.no_server_dp else None,
            'noise_multiplier': noise_multiplier,
            'flame_enabled': not args.no_flame,
            'adversarial_enabled': not args.no_adversarial
        },
        'data_config': {
            'split_seed': 42,
            'split_ratio': '60/20/20',
            'feature_count': model_config['in_channels'],
            'total_transactions': sum(d.x.shape[0] for d in exchange_data),
            'transactions_per_exchange': [int(d.x.shape[0]) for d in exchange_data],
            'train_size_per_exchange': [int(d.train_mask.sum()) for d in exchange_data],
            'test_size_per_exchange': [int(d.test_mask.sum()) for d in exchange_data]
        },
        
        'test_metrics': {
            'forward_accuracy': avg_forward,
            'backward_accuracy': avg_backward,
            'f1_score': f1_score,
            'per_category': avg_per_category,
            'per_exchange': per_exchange_results
        },
        
        'history': history,
        
        'defense_stats': {
            'total_flame_outliers': sum(history['flame_outliers'])
        }
    }
    
    results_path = f'results/experiments/{output_name}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved: {results_path}")
    
    model_path = f'results/models/{output_name}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'results': results
    }, model_path)
    print(f"✓ Model saved: {model_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flexible Federated Learning - Dual Attribution')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/federated_enriched')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GCN layers (1-5, default: 3)')
    
    # Training
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--local_epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--alpha_forward', type=float, default=1.0)
    parser.add_argument('--alpha_backward', type=float, default=1.5)
    
    # Privacy
    parser.add_argument('--use-client-dp', action='store_true',
                        help='Enable Client-side DP (disabled by default)')
    parser.add_argument('--no-server-dp', action='store_true',
                        help='Disable Server-side DP (enabled by default)')
    parser.add_argument('--epsilon', type=float, default=8.0,
                        help='Target epsilon (default=8.0)')
    
    # Security
    parser.add_argument('--no-adversarial', action='store_true',
                        help='Disable Adversarial Training')
    parser.add_argument('--no-flame', action='store_true',
                        help='Disable FLAME Byzantine defense')
    
    # Output
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for output files')
    
    args = parser.parse_args()
    main(args)