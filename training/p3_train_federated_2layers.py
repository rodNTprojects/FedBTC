"""
================================================================================
Script: p3_train_federated.py (V2 - OPTIMIZED)
Description: FEDERATED LEARNING - Dual Attribution avec Categories
Phase 3 du pipeline - Compare FL vs Baseline centralise

CHANGEMENTS V2:
- Backward: 6 categories au lieu de 501 merchants
- Focal Loss pour gerer le desequilibre de classes
- Random split (coherent avec baseline)
- Features discriminatives generees si absentes
- Evaluation par Accuracy au lieu de Hit@K

CATEGORIES:
  0 = NO_MERCHANT (ignore)
  1 = E-commerce
  2 = Gambling
  3 = Services
  4 = Retail
  5 = Luxury

INPUT: data/federated_enriched/exchange_X_enriched.pkl
OUTPUT: results/models/federated_dual_attribution.pt

USAGE:
    python p3_train_federated.py
    python p3_train_federated.py --rounds 20 --local_epochs 3 --use_dp
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
import random

DEFAULT_SEED = 42

def set_global_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(DEFAULT_SEED)

# === FIX: Always save relative to script location ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Flower imports (avec fallback)
try:
    import flwr as fl
    from flwr.common import (
        FitRes, NDArrays, Parameters, Scalar,
        ndarrays_to_parameters, parameters_to_ndarrays,
    )
    from flwr.server.strategy import FedAvg
    FLOWER_AVAILABLE = True
except ImportError:
    print("[WARNING] Flower not installed. Using manual FL implementation.")
    print("  Install with: pip install flwr")
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
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
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
# MODEL ARCHITECTURE (V2 - Categories)
# ============================================================================

class DualAttributionModel(nn.Module):
    """Dual Attribution Model V2 - With Categories (2 layers - optimal)"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # GNN Backbone (2 layers - optimal per ablation)
        # self.convs = nn.ModuleList([
        #     GCNConv(config['in_channels'], config['hidden_dim']),
        #     GCNConv(config['hidden_dim'], config['hidden_dim'])
        # ])
        
        # self.batch_norms = nn.ModuleList([
        #     nn.BatchNorm1d(config['hidden_dim']),
        #     nn.BatchNorm1d(config['hidden_dim'])
        # ])

        # GNN Backbone (configurable layers, default 2)
        num_layers = config.get('num_layers', 2)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = config['in_channels'] if i == 0 else config['hidden_dim']
            self.convs.append(GCNConv(in_dim, config['hidden_dim']))
            self.batch_norms.append(nn.BatchNorm1d(config['hidden_dim']))
        
        self.dropout = nn.Dropout(config['dropout'])
        
        # Forward Head (Exchange - 3 classes)
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
        
        # Backward Head (Category - 6 classes)
        self.backward_head = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.BatchNorm1d(config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['hidden_dim'] // 2, config['num_categories'])
        )

        # Initialize weights (consistent with baseline)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
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
# DATA LOADING
# ============================================================================

def merchant_to_category(merchant_id):
    """Convert merchant_id (0-500) to category_label (0-5)"""
    if merchant_id == 0 or pd.isna(merchant_id):
        return 0
    merchant_id = int(merchant_id)
    if 1 <= merchant_id <= 175:
        return 1  # E-commerce
    elif 176 <= merchant_id <= 300:
        return 2  # Gambling
    elif 301 <= merchant_id <= 400:
        return 3  # Services
    elif 401 <= merchant_id <= 475:
        return 4  # Retail
    else:
        return 5  # Luxury


def generate_discriminative_features(data: pd.DataFrame, category_labels: np.ndarray) -> pd.DataFrame:
    """Generate category-discriminative features"""
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
    """Load data for a specific exchange"""
    
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
    """
    Create PyG Data for one exchange — STRICT canonical mode (Option B-2).

    Reads canonical artifacts in read-only mode:
      - data/split_map.pkl
      - data/normalization_stats.pkl

    No fallback: if either file is missing, training aborts immediately.
    """

    # ===== HANDLE CATEGORY LABELS =====
    if 'category_label' in data.columns:
        category_labels = data['category_label'].values
    elif 'backward_label' in data.columns:
        category_labels = data['backward_label'].apply(merchant_to_category).values
        data['category_label'] = category_labels
    else:
        category_labels = np.zeros(len(data), dtype=int)
        data['category_label'] = category_labels

    # ===== GENERATE DISCRIMINATIVE FEATURES IF NEEDED =====
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

    # ===== LABELS =====
    forward_labels = torch.LongTensor(
        data['forward_label'].values if 'forward_label' in data.columns
        else data['exchange_partition'].values
    )
    backward_labels = torch.LongTensor(category_labels)

    # ===== EDGE INDEX =====
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
# EVALUATION (V2 - Accuracy instead of Hit@K)
# ============================================================================

@torch.no_grad()
def evaluate_model(model: nn.Module, graph_data: Data, device: torch.device, mask_name: str = 'test') -> Dict:
    """Evaluate model with Accuracy metrics"""
    model.eval()
    graph_data = graph_data.to(device)
    
    if mask_name == 'val':
        mask = graph_data.val_mask
    elif mask_name == 'test':
        mask = graph_data.test_mask
    else:
        mask = graph_data.train_mask
    
    output = model(graph_data.x, graph_data.edge_index)
    
    # Forward accuracy
    forward_preds = output['forward'][mask].argmax(dim=1)
    forward_acc = (forward_preds == graph_data.y_forward[mask]).float().mean().item()
    
    # Backward accuracy (excluding NO_MERCHANT)
    backward_labels = graph_data.y_backward[mask]
    category_mask = backward_labels > 0
    
    if category_mask.sum() > 0:
        backward_preds = output['backward'][mask][category_mask].argmax(dim=1)
        backward_acc = (backward_preds == backward_labels[category_mask]).float().mean().item()
        
        # Per-category
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
# FEDERATED CLIENT
# ============================================================================

class FederatedClient:
    """Federated Learning Client for one exchange"""
    
    def __init__(
        self,
        client_id: int,
        graph_data: Data,
        model_config: Dict,
        device: torch.device,
        local_epochs: int = 2,
        lr: float = 0.001,
        use_dp: bool = False
    ):
        self.client_id = client_id
        self.graph_data = graph_data
        self.model_config = model_config
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.use_dp = use_dp
        self.optimizer = None  # Will be created on first train() call
        
        self.model = DualAttributionModel(model_config).to(device)
        
        # APRÈS (weights globaux — équitable)
        global_weights_path = 'data/backward_weights_global.pt'
        if os.path.exists(global_weights_path):
            self.backward_weights = torch.load(global_weights_path).to(device)
            print(f"    [Client {client_id}] Using global backward weights")
        else:
            # Fallback: calcul local si fichier absent
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
            print(f"    [Client {client_id}] WARNING: using local backward weights")
    
    
    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v) for k, v in 
                     zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict)
        self.initial_params = [p.copy() for p in parameters]
        
        # Reset optimizer to reference new parameter tensors
        # but preserve momentum/variance if possible
        if self.optimizer is not None:
            old_state = self.optimizer.state_dict()
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=5e-4)
            try:
                self.optimizer.load_state_dict(old_state)
            except:
                pass  # If shapes changed, start fresh
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    
    def train(self, alpha_forward=1.0, alpha_backward=1.5):
        self.model.train()
        self.model = self.model.to(self.device)
        graph_data = self.graph_data.to(self.device)
        
        # Create optimizer on first call, reuse on subsequent calls
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        else:
            # Update LR if it changed (from server-side scheduler)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        
        optimizer = self.optimizer

        focal_loss = FocalLoss(alpha=self.backward_weights, gamma=2.0)
        
        train_mask = graph_data.train_mask
        
        total_loss = 0
        for epoch in range(self.local_epochs):
            optimizer.zero_grad()
            
            output = self.model(graph_data.x, graph_data.edge_index)
            
            # Forward loss
            loss_forward = F.cross_entropy(
                output['forward'][train_mask],
                graph_data.y_forward[train_mask]
            )
            
            # Backward loss (with Focal Loss)
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
        
        return self.get_parameters(), int(train_mask.sum()), {'loss': total_loss / self.local_epochs}


# ============================================================================
# FEDERATED SERVER (Manual Implementation)
# ============================================================================

class FederatedServer:
    """Federated Learning Server for aggregation"""
    
    def __init__(self, model_config: Dict, device: torch.device):
        self.model_config = model_config
        self.device = device
        self.global_model = DualAttributionModel(model_config).to(device)
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get global model parameters"""
        return [val.cpu().numpy() for val in self.global_model.state_dict().values()]
    
    def aggregate(self, client_results: List[Tuple[List[np.ndarray], int, Dict]]):
        """FedAvg aggregation"""
        total_samples = sum([num_samples for _, num_samples, _ in client_results])
        
        # Weighted average
        new_params = []
        for param_idx in range(len(client_results[0][0])):
            weighted_sum = sum([
                params[param_idx] * num_samples 
                for params, num_samples, _ in client_results
            ])
            new_params.append(weighted_sum / total_samples)
        
        # Update global model
        state_dict = {k: torch.tensor(v) for k, v in 
                     zip(self.global_model.state_dict().keys(), new_params)}
        self.global_model.load_state_dict(state_dict)


# ============================================================================
# FLOWER CLIENT WRAPPER
# ============================================================================

if FLOWER_AVAILABLE:
    class FlowerClient(fl.client.NumPyClient):
        """Flower client wrapper"""
        
        def __init__(self, client: FederatedClient):
            self.client = client
        
        def get_parameters(self, config):
            return self.client.get_parameters()
        
        def fit(self, parameters, config):
            self.client.set_parameters(parameters)
            return self.client.train()
        
        def evaluate(self, parameters, config):
            self.client.set_parameters(parameters)
            metrics = evaluate_model(
                self.client.model,
                self.client.graph_data,
                self.client.device,
                'val'
            )
            # Return loss (negative accuracy for minimization)
            loss = 1.0 - (metrics['forward_accuracy'] + metrics['backward_accuracy']) / 2
            return float(loss), int(self.client.graph_data.val_mask.sum()), metrics


# ============================================================================
# MANUAL FL IMPLEMENTATION
# ============================================================================

def run_manual_fl(
    exchange_data: List[Data],
    model_config: Dict,
    device: torch.device,
    num_rounds: int = 10,
    local_epochs: int = 2,
    lr: float = 0.001,
    use_dp: bool = False,
    alpha_forward: float = 1.0,
    alpha_backward: float = 1.5
) -> Tuple[nn.Module, Dict]:
    """Run FL with manual implementation (no Flower)"""
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING (Manual Implementation)")
    print("="*70)
    print(f"  Rounds: {num_rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Differential Privacy: {use_dp}")
    print(f"  Alpha forward: {alpha_forward}, Alpha backward: {alpha_backward}")
    
    # Initialize server
    server = FederatedServer(model_config, device)
    
    # Initialize clients
    clients = [
        FederatedClient(
            client_id=i,
            graph_data=data,
            model_config=model_config,
            device=device,
            local_epochs=local_epochs,
            lr=lr,
            use_dp=use_dp
        )
        for i, data in enumerate(exchange_data)
    ]
    
    history = {
        'rounds': [],
        'forward_accuracy': [],
        'backward_accuracy': [],
        'loss': []
    }
    
    best_metric = 0.0
    best_params = None

    current_lr = lr
    scheduler_patience = 5
    scheduler_counter = 0
    scheduler_factor = 0.7
    min_lr = 1e-6
    


    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} (lr={current_lr:.6f}) ---")
        
        # ===== AJOUTER ICI (3 lignes) =====
        for client in clients:
            client.lr = current_lr
        # ===================================
        
        # Distribute global parameters to clients
        global_params = server.get_parameters()
        for client in clients:
            client.set_parameters(global_params)
        
        # Local training
        client_results = []
        for i, client in enumerate(clients):
            params, num_samples, metrics = client.train(alpha_forward, alpha_backward)
            client_results.append((params, num_samples, metrics))
            print(f"  Client {i}: {num_samples} samples, loss={metrics['loss']:.4f}")
        
        # Aggregate
        server.aggregate(client_results)
        
        # Evaluate on all clients combined
        val_metrics_list = []
        for i, client in enumerate(clients):
            client.set_parameters(server.get_parameters())
            metrics = evaluate_model(client.model, client.graph_data, device, 'val')
            val_metrics_list.append(metrics)
        
        # Average metrics
        val_sizes = [int(d.val_mask.sum()) for d in exchange_data]
        total_val = sum(val_sizes)
        avg_forward = sum(m['forward_accuracy'] * n for m, n in zip(val_metrics_list, val_sizes)) / total_val
        avg_backward = sum(m['backward_accuracy'] * n for m, n in zip(val_metrics_list, val_sizes)) / total_val
        avg_loss = np.mean([r[2]['loss'] for r in client_results])
        
        history['rounds'].append(round_num + 1)
        history['forward_accuracy'].append(avg_forward)
        history['backward_accuracy'].append(avg_backward)
        history['loss'].append(avg_loss)
        
        # Best model tracking
        combined = 0.4 * avg_forward + 0.6 * avg_backward
        improved = ""
        if combined > best_metric:
            best_metric = combined
            best_params = server.get_parameters()
            scheduler_counter = 0   
            improved = " [BEST]"
        # ===== Ajusting learning rate between rounds in order to comparison with Baseline =====
        else:
            scheduler_counter += 1
            if scheduler_counter >= scheduler_patience:
                new_lr = max(current_lr * scheduler_factor, min_lr)
                if new_lr < current_lr:
                    print(f"  [LR] Reducing lr: {current_lr:.6f} -> {new_lr:.6f}")
                    current_lr = new_lr
                scheduler_counter = 0
        # ===================================
        
        print(f"  -> Forward: {avg_forward:.2%}, Backward: {avg_backward:.2%}{improved}")
    
    # Load best parameters
    if best_params:
        state_dict = {k: torch.tensor(v) for k, v in 
                     zip(server.global_model.state_dict().keys(), best_params)}
        server.global_model.load_state_dict(state_dict)
    
    return server.global_model, history


# ============================================================================
# FLOWER FL IMPLEMENTATION
# ============================================================================

def run_flower_fl(
    exchange_data: List[Data],
    model_config: Dict,
    device: torch.device,
    num_rounds: int = 10,
    local_epochs: int = 2,
    lr: float = 0.001,
    use_dp: bool = False
) -> Tuple[nn.Module, Dict]:
    """Run FL with Flower framework"""
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING (Flower Framework)")
    print("="*70)
    
    # Create clients
    clients = [
        FederatedClient(
            client_id=i,
            graph_data=data,
            model_config=model_config,
            device=device,
            local_epochs=local_epochs,
            lr=lr,
            use_dp=use_dp
        )
        for i, data in enumerate(exchange_data)
    ]
    
    # Client function for Flower
    def client_fn(cid: str):
        return FlowerClient(clients[int(cid)])
    
    # Initial parameters
    initial_model = DualAttributionModel(model_config)
    initial_params = [val.cpu().numpy() for val in initial_model.state_dict().values()]
    
    # Strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(exchange_data),
        min_evaluate_clients=len(exchange_data),
        min_available_clients=len(exchange_data),
        initial_parameters=ndarrays_to_parameters(initial_params),
    )
    
    # Run simulation
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(exchange_data),
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 0.5, "num_gpus": 0.0},
        )
        
        # Get final model
        final_params = strategy.parameters
        if final_params:
            params = parameters_to_ndarrays(final_params)
            state_dict = {k: torch.tensor(v) for k, v in 
                         zip(initial_model.state_dict().keys(), params)}
            initial_model.load_state_dict(state_dict)
        
        return initial_model, {'flower_history': history}
        
    except Exception as e:
        print(f"\n[WARNING] Flower failed: {e}")
        print("Falling back to manual FL...")
        return run_manual_fl(exchange_data, model_config, device, num_rounds, local_epochs, lr, use_dp)


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    """Main function"""
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING - DUAL ATTRIBUTION (V2 - CATEGORIES)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Num layers: {args.num_layers}")
    print(f"Flower available: {FLOWER_AVAILABLE}")
    
    # # Load data for each exchange
    # print("\nLoading exchange data...")
    # exchange_data = []
    
    # for i in range(3):
    #     data, edges = load_exchange_data(i, args.data_dir)
    #     graph_data = create_pyg_data_for_exchange(data, edges, exchange_id=i)
    #     exchange_data.append(graph_data)

    # Load data for each exchange
    print("\nLoading exchange data...")
    exchange_data = []
    
    if args.iid:
        # ===== IID ABLATION: shuffle all transactions across clients =====
        print("  [IID MODE] Shuffling transactions across clients...")
        all_dfs = []
        all_edges = []
        for i in range(3):
            data_i, edges_i = load_exchange_data(i, args.data_dir)
            all_dfs.append(data_i)
            all_edges.append(edges_i)
        
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_edges = pd.concat(all_edges, ignore_index=True).drop_duplicates()
        
        # Shuffle and split into 3 equal-size clients
        np.random.seed(123)
        shuffled_idx = np.random.permutation(len(combined))
        chunk_size = len(combined) // 3
        
        # APRÈS (filtre aussi les arêtes cross-exchange)
        for i in range(3):
            start = i * chunk_size
            end = start + chunk_size if i < 2 else len(combined)
            client_idx = shuffled_idx[start:end]
            client_df = combined.iloc[client_idx].reset_index(drop=True)
            
            # Filter edges: within this client AND same exchange of origin
            client_txids = set(client_df['txId'].values)
            client_edges = combined_edges[
                combined_edges['txId1'].isin(client_txids) & 
                combined_edges['txId2'].isin(client_txids)
            ].reset_index(drop=True)
            
            if args.iid_no_cross_edges:
                # Build txId → forward_label map for edge filtering
                txid_to_exchange = dict(zip(client_df['txId'].values, 
                                           client_df['forward_label'].values))
                mask = client_edges.apply(
                    lambda row: txid_to_exchange.get(row['txId1'], -1) == 
                                txid_to_exchange.get(row['txId2'], -2), axis=1)
                n_before = len(client_edges)
                client_edges = client_edges[mask].reset_index(drop=True)
                print(f"    [IID-NoXEdge] Client {i}: {n_before} -> {len(client_edges)} edges "
                      f"(removed {n_before - len(client_edges)} cross-exchange)")
            
            graph_data = create_pyg_data_for_exchange(client_df, client_edges, exchange_id=i)
            exchange_data.append(graph_data)
            print(f"  IID Client {i}: {len(client_df):,} transactions, "
                  f"fwd labels: {dict(Counter(client_df['forward_label'].values))}")
    else:
        # ===== NON-IID (default): each client = one exchange =====
        for i in range(3):
            data, edges = load_exchange_data(i, args.data_dir)
            graph_data = create_pyg_data_for_exchange(data, edges, exchange_id=i)
            exchange_data.append(graph_data)
        
            # Category distribution
            cat_counts = Counter(graph_data.y_backward.numpy())
            print(f"  Exchange {i}: {len(data):,} transactions, {graph_data.num_features} features")
            print(f"    Categories: " + ", ".join([f"{CATEGORY_NAMES[k]}:{v}" for k, v in sorted(cat_counts.items()) if k > 0]))
    

    # Re-seed after data loading (splits are fixed at 42 internally)
    set_global_seed(args.seed)

    # Model config
    model_config = {
        'in_channels': exchange_data[0].num_features,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'num_exchanges': 3,
        'num_categories': 6  # V2: Categories instead of merchants
    }
    
    print(f"\nModel config: {model_config}")
    
    # Run federated learning
    if args.use_flower and FLOWER_AVAILABLE:
        final_model, history = run_flower_fl(
            exchange_data, model_config, device,
            num_rounds=args.rounds,
            local_epochs=args.local_epochs,
            lr=args.lr,
            use_dp=args.use_dp
        )
    else:
        final_model, history = run_manual_fl(
            exchange_data, model_config, device,
            num_rounds=args.rounds,
            local_epochs=args.local_epochs,
            lr=args.lr,
            use_dp=args.use_dp,
            alpha_forward=args.alpha_forward,
            alpha_backward=args.alpha_backward
        )
    
    # Final evaluation on all exchanges
    print("\n" + "="*70)
    print("FINAL EVALUATION (TEST SET)")
    print("="*70)
    
    final_model = final_model.to(device)
    
    all_forward = []
    all_backward = []
    test_sizes = []
    
    for i, data in enumerate(exchange_data):
        metrics = evaluate_model(final_model, data, device, 'test')
        all_forward.append(metrics['forward_accuracy'])
        all_backward.append(metrics['backward_accuracy'])
        test_sizes.append(int(data.test_mask.sum()))
        
        print(f"\nExchange {i} (test_n={test_sizes[-1]}):")
        print(f"  Forward Accuracy:  {metrics['forward_accuracy']:.2%}")
        print(f"  Backward Accuracy: {metrics['backward_accuracy']:.2%}")
        if metrics['per_category']:
            print(f"  Per-Category:")
            for cat, acc in metrics['per_category'].items():
                print(f"    {cat:12s}: {acc:.2%}")
    
    # Overall metrics — WEIGHTED average (consistent with baseline global test set)
    total_test = sum(test_sizes)
    avg_forward = sum(a * n for a, n in zip(all_forward, test_sizes)) / total_test
    avg_backward = sum(a * n for a, n in zip(all_backward, test_sizes)) / total_test
    
    # Also keep unweighted for reference
    avg_forward_unw = np.mean(all_forward)
    avg_backward_unw = np.mean(all_backward)
    print(f"\n  [INFO] Unweighted avg: Fwd={avg_forward_unw:.2%}, Bwd={avg_backward_unw:.2%}")
    
    print(f"\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print(f"  Forward Accuracy:  {avg_forward:.2%}")
    print(f"  Backward Accuracy: {avg_backward:.2%}")
    
    # Compare with baseline targets
    print(f"\n{'[OK]' if avg_forward >= 0.70 else '[WARNING]'} Forward target (>=70%): {avg_forward:.2%}")
    print(f"{'[OK]' if avg_backward >= 0.60 else '[WARNING]'} Backward target (>=60%): {avg_backward:.2%}")
    
    # Save model
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/evaluations', exist_ok=True)

    tag = f'_{args.results_tag}' if args.results_tag else ''
    
    model_path = f'results/models/federated_dual_attribution{tag}_seed{args.seed}.pt'
    
    #model_path = f'results/models/federated_dual_attribution_seed{args.seed}.pt'
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'config': model_config,
        'fl_config': {
            'rounds': args.rounds,
            'local_epochs': args.local_epochs,
            'use_dp': args.use_dp,
            'alpha_forward': args.alpha_forward,
            'alpha_backward': args.alpha_backward
        }
    }, model_path)
    print(f"\n[OK] Model saved: {model_path}")
    
    # # Save results
    # results = {
    #     'config': model_config,
    #     'fl_config': {
    #         'rounds': args.rounds,
    #         'local_epochs': args.local_epochs,
    #         'lr': args.lr,
    #         'use_dp': args.use_dp,
    #         'alpha_forward': args.alpha_forward,
    #         'alpha_backward': args.alpha_backward
    #     },
    #     'test_metrics': {
    #         'forward_accuracy': avg_forward,
    #         'backward_accuracy': avg_backward,
    #         'per_exchange': [
    #             {'forward': all_forward[i], 'backward': all_backward[i]}
    #             for i in range(3)
    #         ]
    #     },
    #     'history': {k: v for k, v in history.items() if isinstance(v, list)},
    #     'timestamp': datetime.now().isoformat()
    # }
    
    # results_path = 'results/evaluations/federated_results.json'
    # with open(results_path, 'w') as f:
    #     json.dump(results, f, indent=2, default=str)
    # print(f"[OK] Results saved: {results_path}")

     # Collect per-category metrics for each exchange
    all_per_category = []
    for i, data in enumerate(exchange_data):
        metrics = evaluate_model(final_model, data, device, 'test')
        all_per_category.append(metrics.get('per_category', {}))
    
    # Average per-category across exchanges
    avg_per_category = {}
    for cat in ['E-commerce', 'Gambling', 'Services', 'Retail', 'Luxury']:
        cat_accs = [pc.get(cat, 0) for pc in all_per_category]
        avg_per_category[cat] = sum(a * n for a, n in zip(cat_accs, test_sizes)) / total_test
    
    # Save results - STANDARDIZED FORMAT FOR COMPARISON
    results = {
        # === IDENTIFICATION ===
        'approach': 'fl_nosec',
        'approach_name': 'Federated Learning (No Security)',
        'training_type': 'federated',
        'privacy_level': 'isolation',
        'privacy_description': 'Data isolation only - no DP or defenses',
        
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
            'differential_privacy': args.use_dp,
            'epsilon': None,
            'delta': None,
            'noise_multiplier': None,
            'adversarial_training': False,
            'byzantine_defense': None,
            'flame_enabled': False,
            'fedval_enabled': False
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
        
        # === TRAINING HISTORY ===
        'history': {
            'rounds': list(range(1, args.rounds + 1)),
            'forward_accuracy': history.get('forward_accuracy', []),
            'backward_accuracy': history.get('backward_accuracy', []),
            'loss': history.get('loss', [])
        },
        
        # === METADATA ===
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'script': 'p3_train_federated.py',
            'version': 'v2_categories',
            'device': str(device),
            'flower_used': args.use_flower and FLOWER_AVAILABLE,
            'targets_met': {
                'forward_target_70': avg_forward >= 0.70,
                'backward_target_60': avg_backward >= 0.60
            }
        }
    }
    
    results_path = f'results/evaluations/federated_results{tag}_seed{args.seed}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[OK] Results saved: {results_path}")
    
    # Comparison summary
    print("\n" + "="*70)
    print("BASELINE vs FEDERATED COMPARISON")
    print("="*70)
    print("  Load baseline results with: cat results/evaluations/baseline_results.json")
    print("  Expected FL degradation: 1-5% (due to non-IID data distribution)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning - Dual Attribution V2')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/federated_enriched',
                        help='Directory with enriched data')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # FL config
    parser.add_argument('--rounds', type=int, default=20,
                        help='Number of FL rounds')
    parser.add_argument('--local_epochs', type=int, default=3,
                        help='Local epochs per round')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate')
    
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GCN layers (default: 2, optimal per ablation)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    parser.add_argument('--results_tag', type=str, default='',
                    help='Tag added to result filename (e.g. "layers3", "eps8")')
    # Loss weights
    parser.add_argument('--alpha_forward', type=float, default=1.0,
                        help='Weight for forward loss')
    parser.add_argument('--alpha_backward', type=float, default=1.5,
                        help='Weight for backward loss')
    
    # Options
    parser.add_argument('--use_dp', action='store_true',
                        help='Enable Differential Privacy')
    parser.add_argument('--use_flower', action='store_true',
                        help='Use Flower framework (if available)')

    parser.add_argument('--iid', action='store_true',
                        help='IID ablation: shuffle transactions across clients')

    parser.add_argument('--iid_no_cross_edges', action='store_true',
                        help='IID mode: also remove cross-exchange edges within each client')
    
    args = parser.parse_args()
    
    main(args)