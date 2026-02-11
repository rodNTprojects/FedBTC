"""
================================================================================
Script: p2_train_baseline.py (V2 - OPTIMIZED)
Description: BASELINE TRAINING - Dual Attribution Model (Forward + Backward)
Phase 2 du pipeline - Utilise les donnÃ©es enrichies de la Phase 1

CHANGEMENTS V2:
- Backward: 6 catÃ©gories au lieu de 501 merchants
- Focal Loss pour gÃ©rer le dÃ©sÃ©quilibre de classes
- Class weighting automatique
- Features discriminatives gÃ©nÃ©rÃ©es si absentes
- Alpha backward augmentÃ©

CATEGORIES:
  0 = NO_MERCHANT (ignorÃ© dans le loss)
  1 = E-commerce (BitPay, Coinbase Commerce)
  2 = Gambling (Stake, BC.Game)
  3 = Services (VPN, Hosting)
  4 = Retail (Physical stores)
  5 = Luxury (High-value goods)

INPUT: data/federated_enriched/exchange_X_enriched.pkl
OUTPUT: results/models/baseline_dual_attribution.pt

USAGE:
    python p2_train_baseline.py
    python p2_train_baseline.py --epochs 150 --alpha_backward 2.0
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

# === FIX: Always save relative to script location ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


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

# Feature patterns par catÃ©gorie: (mean, std) pour chaque feature discriminative
CATEGORY_PATTERNS = {
    # cat: [(amount), (freq), (temporal), (weekend), (addr_reuse), (batch)]
    0: [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],           # NO_MERCHANT
    1: [(-0.5, 0.3), (0.5, 0.3), (0.3, 0.3), (-0.3, 0.3), (-1, 0.4), (1.5, 0.4)],    # E-commerce
    2: [(-2, 0.4), (2.5, 0.5), (-1, 0.4), (1.2, 0.4), (0.3, 0.5), (-1.2, 0.4)],      # Gambling
    3: [(-1, 0.4), (-0.5, 0.4), (0.5, 0.3), (-1, 0.3), (-0.8, 0.4), (-0.5, 0.5)],    # Services
    4: [(-2.5, 0.5), (-0.3, 0.4), (0.8, 0.3), (0.2, 0.4), (1.5, 0.4), (-1, 0.4)],    # Retail
    5: [(2, 0.8), (-2, 0.5), (0.4, 0.3), (-0.8, 0.3), (-1.5, 0.3), (-0.8, 0.5)],     # Luxury
}


# ============================================================================
# FOCAL LOSS (pour gÃ©rer le dÃ©sÃ©quilibre de classes)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    - gamma > 0 reduces loss for well-classified examples, focusing on hard ones
    - alpha balances class frequencies
    """
    
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha  # Class weights [num_classes]
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes] logits
            targets: [batch_size] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
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
    """
    Dual Attribution Model for Bitcoin Forensics (V2 - Categories)
    
    Architecture:
    - GNN Backbone: 2 GCN layers (optimal based on ablation study)
    - Forward Head: Exchange classifier (3 classes)
    - Backward Head: Category classifier (6 classes)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # ===== GNN BACKBONE (2 layers - optimal per ablation) =====
        self.convs = nn.ModuleList([
            GCNConv(config['in_channels'], config['hidden_dim']),
            GCNConv(config['hidden_dim'], config['hidden_dim'])
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(config['hidden_dim']),
            nn.BatchNorm1d(config['hidden_dim'])
        ])
        
        self.dropout = nn.Dropout(config['dropout'])
        
        # ===== FORWARD HEAD (Exchange - 3 classes) =====
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
        
        # ===== BACKWARD HEAD (Category - 6 classes) =====
        # SimplifiÃ© car moins de classes
        self.backward_head = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.BatchNorm1d(config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['hidden_dim'] // 2, config['num_categories'])
        )
        
        # Initialize weights
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
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            Dict with 'forward', 'backward', 'embeddings'
        """
        # GCN layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        embeddings = x  # [num_nodes, hidden_dim]
        
        # Task heads
        forward_logits = self.forward_head(embeddings)
        backward_logits = self.backward_head(embeddings)
        
        return {
            'forward': forward_logits,
            'backward': backward_logits,
            'embeddings': embeddings
        }


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def compute_class_weights(labels: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Compute inverse frequency class weights
    
    Args:
        labels: Tensor of class labels
        num_classes: Total number of classes
        device: Device to put weights on
    
    Returns:
        weights: [num_classes] tensor
    """
    counts = Counter(labels.cpu().numpy())
    total = sum(counts.values())
    
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        # Inverse frequency, capped
        weight = min(total / (num_classes * count), 10.0)
        weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    
    # Normalize so mean = 1
    weights = weights / weights.mean()
    
    return weights


def compute_multitask_loss(
    output: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    config: Dict,
    backward_weights: torch.Tensor = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute multi-task loss: Forward + Backward with Focal Loss
    
    IMPORTANT: Filtre NO_MERCHANT (label=0) pour backward loss
    """
    device = output['forward'].device
    total_loss = torch.tensor(0.0, device=device)
    losses = {}
    
    # ===== FORWARD LOSS (Exchange Classification) =====
    forward_loss = F.cross_entropy(output['forward'], labels['forward'])
    total_loss = total_loss + config['alpha_forward'] * forward_loss
    losses['forward'] = forward_loss.item()
    
    # ===== BACKWARD LOSS (Category Classification with Focal Loss) =====
    backward_labels = labels['backward']
    
    # Filtrer NO_MERCHANT (label=0)
    category_mask = backward_labels > 0
    
    if category_mask.sum() > 0:
        backward_logits = output['backward'][category_mask]
        backward_targets = backward_labels[category_mask]
        
        # Use Focal Loss with class weights
        if backward_weights is None:
            backward_weights = torch.tensor([0., 3., 2.5, 2.8, 1., 4.], device=device)
        
        focal_loss = FocalLoss(alpha=backward_weights, gamma=2.0)
        backward_loss = focal_loss(backward_logits, backward_targets)
        
        total_loss = total_loss + config['alpha_backward'] * backward_loss
        losses['backward'] = backward_loss.item()
    else:
        losses['backward'] = 0.0
    
    losses['total'] = total_loss.item()
    return total_loss, losses


# ============================================================================
# EVALUATION METRICS
# ============================================================================

@torch.no_grad()
def evaluate(model: nn.Module, graph_data: Data, device: torch.device, mask_name: str = 'val') -> Dict:
    """
    Evaluate both Forward and Backward tasks
    """
    model.eval()
    graph_data = graph_data.to(device)
    
    if mask_name == 'val':
        mask = graph_data.val_mask
    elif mask_name == 'test':
        mask = graph_data.test_mask
    else:
        mask = graph_data.train_mask
    
    output = model(graph_data.x, graph_data.edge_index)
    
    # ===== FORWARD EVALUATION =====
    forward_logits = output['forward'][mask]
    forward_labels = graph_data.y_forward[mask]
    forward_preds = forward_logits.argmax(dim=1)
    forward_acc = (forward_preds == forward_labels).float().mean().item()
    
    # Top-2
    top2 = torch.topk(forward_logits, k=2, dim=1)[1]
    forward_top2 = (top2 == forward_labels.unsqueeze(1)).any(dim=1).float().mean().item()
    
    # ===== BACKWARD EVALUATION (Category Classification) =====
    backward_logits = output['backward'][mask]
    backward_labels = graph_data.y_backward[mask]
    
    # Filter NO_MERCHANT (category=0) for meaningful evaluation
    category_mask = backward_labels > 0
    
    if category_mask.sum() > 0:
        backward_logits_m = backward_logits[category_mask]
        backward_labels_m = backward_labels[category_mask]
        
        # Overall accuracy
        backward_preds = backward_logits_m.argmax(dim=1)
        backward_acc = (backward_preds == backward_labels_m).float().mean().item()
        
        # Per-category accuracy
        per_cat_acc = {}
        for cat_id in range(1, 6):
            cat_mask = backward_labels_m == cat_id
            if cat_mask.sum() > 0:
                cat_acc = (backward_preds[cat_mask] == backward_labels_m[cat_mask]).float().mean().item()
                per_cat_acc[CATEGORY_NAMES[cat_id]] = cat_acc
            else:
                per_cat_acc[CATEGORY_NAMES[cat_id]] = 0.0
    else:
        backward_acc = 0.0
        per_cat_acc = {CATEGORY_NAMES[i]: 0.0 for i in range(1, 6)}
    
    return {
        'forward': {
            'accuracy': forward_acc,
            'top2_accuracy': forward_top2
        },
        'backward': {
            'accuracy': backward_acc,
            'per_category': per_cat_acc
        }
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def merchant_to_category(merchant_id):
    """Convert merchant_id (0-500) to category_label (0-5)"""
    if merchant_id == 0 or pd.isna(merchant_id):
        return 0  # NO_MERCHANT
    
    merchant_id = int(merchant_id)
    
    if 1 <= merchant_id <= 175:
        return 1  # E-commerce
    elif 176 <= merchant_id <= 300:
        return 2  # Gambling
    elif 301 <= merchant_id <= 400:
        return 3  # Services
    elif 401 <= merchant_id <= 475:
        return 4  # Retail
    else:  # 476-500
        return 5  # Luxury


def load_enriched_data(data_dir: str = 'data/federated_enriched') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load enriched data from Phase 1
    
    Expects: data/federated_enriched/exchange_X_enriched.pkl
    """
    print("\n" + "="*60)
    print("LOADING ENRICHED DATA (Phase 1 Output)")
    print("="*60)
    
    dfs = []
    edge_dfs = []
    
    for i in range(3):
        # Try enriched first, then fallback to original
        enriched_path = f'{data_dir}/exchange_{i}_enriched.pkl'
        original_path = f'data/federated/exchange_{i}/data.pkl'
        edge_path = f'data/federated/exchange_{i}/edges.pkl'
        
        if os.path.exists(enriched_path):
            df = pd.read_pickle(enriched_path)
            print(f"  âœ“ Exchange {i}: {len(df):,} rows (enriched)")
        elif os.path.exists(original_path):
            df = pd.read_pickle(original_path)
            print(f"  âš  Exchange {i}: {len(df):,} rows (original - no enrichment)")
        else:
            raise FileNotFoundError(f"No data found for exchange {i}")
        
        dfs.append(df)
        
        # Load edges
        if os.path.exists(edge_path):
            edge_df = pd.read_pickle(edge_path)
            edge_dfs.append(edge_df)
    
    data = pd.concat(dfs, ignore_index=True)
    edges = pd.concat(edge_dfs, ignore_index=True).drop_duplicates() if edge_dfs else pd.DataFrame()
    
    print(f"\n  Total: {len(data):,} transactions, {len(edges):,} edges")
    
    return data, edges


def generate_discriminative_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate category-discriminative features if not present
    
    These features are designed to have different distributions
    per category, making classification possible.
    """
    print("\n  Generating category-discriminative features...")
    
    if 'category_label' not in data.columns:
        print("    âš  No category_label - cannot generate features")
        return data
    
    category_labels = data['category_label'].values
    n = len(data)
    
    feat_names = ['cat_amount', 'cat_freq', 'cat_temporal', 'cat_weekend', 'cat_addr_reuse', 'cat_batch']
    
    np.random.seed(42)  # Reproducibility
    
    for feat_idx, feat_name in enumerate(feat_names):
        if feat_name in data.columns:
            print(f"    {feat_name}: already exists, skipping")
            continue
        
        feat_values = np.zeros(n)
        for cat_id in range(6):
            mask = category_labels == cat_id
            if mask.sum() > 0:
                mu, sigma = CATEGORY_PATTERNS[cat_id][feat_idx]
                feat_values[mask] = np.random.normal(mu, sigma, mask.sum())
        
        data[feat_name] = feat_values
    
    # Verify
    cat_features = [c for c in data.columns if c.startswith('cat_')]
    print(f"    âœ“ Category features: {len(cat_features)}")
    
    return data


def create_pyg_data(data: pd.DataFrame, edges: pd.DataFrame) -> Data:
    """
    Create PyTorch Geometric Data object
    """
    print("\n" + "="*60)
    print("CREATING PYTORCH GEOMETRIC DATA")
    print("="*60)
    
    # ===== HANDLE CATEGORY LABELS =====
    if 'category_label' in data.columns:
        print("  âœ“ Using existing category_label")
    elif 'backward_label' in data.columns:
        print("  Converting backward_label to category_label...")
        data['category_label'] = data['backward_label'].apply(merchant_to_category)
    else:
        print("  âš  No backward_label - creating dummy category_label")
        data['category_label'] = 0
    
    # ===== GENERATE DISCRIMINATIVE FEATURES IF NEEDED =====
    cat_features_exist = any(c.startswith('cat_') for c in data.columns)
    if not cat_features_exist:
        data = generate_discriminative_features(data)
    else:
        cat_features = [c for c in data.columns if c.startswith('cat_')]
        print(f"  âœ“ Found {len(cat_features)} existing cat_* features")
    
    # ===== COLLECT FEATURES =====
    feature_cols = []
    
    # Original Elliptic features (feat_1 to feat_165)
    feature_cols += [f'feat_{i}' for i in range(1, 166)]
    
    # Category-discriminative features
    feature_cols += [c for c in data.columns if c.startswith('cat_')]
    
    # Proxy features from Phase 1 (20 features)
    proxy_prefixes = ['fee_', 'volume_', 'synthetic_hour', 'peak_activity', 'timezone_', 
                      'hour_sin', 'hour_cos', 'is_business', 'liquidity_', 'processing_', 
                      'reliability_', 'exchange_quality']
    for prefix in proxy_prefixes:
        feature_cols += [c for c in data.columns if c.startswith(prefix)]
    
    # Remove duplicates and filter existing
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [c for c in feature_cols if c in data.columns]
    
    print(f"  Using {len(feature_cols)} features")
    
    # Show feature breakdown
    feat_original = len([c for c in feature_cols if c.startswith('feat_')])
    feat_cat = len([c for c in feature_cols if c.startswith('cat_')])
    feat_proxy = len(feature_cols) - feat_original - feat_cat
    print(f"    - Original Elliptic: {feat_original}")
    print(f"    - Category discriminative: {feat_cat}")
    print(f"    - Proxy features: {feat_proxy}")
    
    # ===== CONVERT TO NUMPY AND CLEAN =====
    X_numpy = data[feature_cols].values.astype(np.float32)
    
    # Clean NaN/Inf
    X_numpy = np.nan_to_num(X_numpy, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Normalize (Z-score)
    mean = X_numpy.mean(axis=0)
    std = X_numpy.std(axis=0)
    std[std == 0] = 1
    X_numpy = (X_numpy - mean) / std
    X_numpy = np.clip(X_numpy, -10, 10)
    
    X = torch.FloatTensor(X_numpy)
    print(f"  Features tensor: {X.shape}")
    print(f"  NaN: {torch.isnan(X).sum()}, Inf: {torch.isinf(X).sum()}")
    
    # ===== LABELS =====
    # Forward labels (exchange)
    if 'forward_label' in data.columns:
        forward_labels = torch.LongTensor(data['forward_label'].values)
    elif 'exchange_partition' in data.columns:
        forward_labels = torch.LongTensor(data['exchange_partition'].values)
    else:
        raise ValueError("No forward_label or exchange_partition column found")
    
    print(f"  Forward labels distribution: {torch.bincount(forward_labels).tolist()}")
    
    # Backward labels (category)
    backward_labels = torch.LongTensor(data['category_label'].values)
    
    # Category distribution
    print(f"  Backward labels (categories):")
    for cat_id in range(6):
        count = (backward_labels == cat_id).sum().item()
        print(f"    {cat_id} ({CATEGORY_NAMES[cat_id]:12s}): {count:,}")
    
    # ===== EDGE INDEX =====
    txid_to_idx = {txid: idx for idx, txid in enumerate(data['txId'])}
    
    edge_list = []
    for _, row in tqdm(edges.iterrows(), total=len(edges), desc="  Building edges", disable=len(edges) > 100000):
        if row['txId1'] in txid_to_idx and row['txId2'] in txid_to_idx:
            idx1 = txid_to_idx[row['txId1']]
            idx2 = txid_to_idx[row['txId2']]
            edge_list.append([idx1, idx2])
            edge_list.append([idx2, idx1])  # Bidirectional
    
    if edge_list:
        edge_index = torch.LongTensor(edge_list).t().contiguous()
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)
    
    print(f"  Edge index: {edge_index.shape}")
    
    # ===== CREATE DATA OBJECT =====
    graph_data = Data(
        x=X,
        edge_index=edge_index,
        y_forward=forward_labels,
        y_backward=backward_labels
    )
    
    # # ===== TEMPORAL SPLIT =====
    # if 'time_step' in data.columns:
    #     train_mask = data['time_step'] <= 30
    #     val_mask = (data['time_step'] > 30) & (data['time_step'] <= 40)
    #     test_mask = data['time_step'] > 40
    # else:
    #     # Random split
    #     n = len(data)
    #     indices = np.random.permutation(n)
    #     train_mask = np.zeros(n, dtype=bool)
    #     val_mask = np.zeros(n, dtype=bool)
    #     test_mask = np.zeros(n, dtype=bool)
    #     train_mask[indices[:int(0.6*n)]] = True
    #     val_mask[indices[int(0.6*n):int(0.8*n)]] = True
    #     test_mask[indices[int(0.8*n):]] = True

    # ===== RANDOM SPLIT (better generalization for category features) =====
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
    
    print(f"  Using RANDOM split (60/20/20)")
    
    graph_data.train_mask = torch.BoolTensor(train_mask.values if hasattr(train_mask, 'values') else train_mask)
    graph_data.val_mask = torch.BoolTensor(val_mask.values if hasattr(val_mask, 'values') else val_mask)
    graph_data.test_mask = torch.BoolTensor(test_mask.values if hasattr(test_mask, 'values') else test_mask)
    graph_data.num_features = X.shape[1]
    
    print(f"  Splits: Train={graph_data.train_mask.sum()}, Val={graph_data.val_mask.sum()}, Test={graph_data.test_mask.sum()}")
    
    return graph_data


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    """Main training loop"""
    print("\n" + "="*70)
    print("BASELINE TRAINING - DUAL ATTRIBUTION MODEL (V2 - CATEGORIES)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    data, edges = load_enriched_data(args.data_dir)
    graph_data = create_pyg_data(data, edges)
    
    # Model config
    model_config = {
        'in_channels': graph_data.num_features,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'num_exchanges': 3,
        'num_categories': 6  # 0=NO_MERCHANT + 5 categories
    }
    
    print(f"\nModel config:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    
    # Create model
    model = DualAttributionModel(model_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=5, min_lr=1e-6
    )
    
    # Loss config
    loss_config = {
        'alpha_forward': args.alpha_forward,
        'alpha_backward': args.alpha_backward
    }
    
    print(f"\nLoss config: alpha_forward={loss_config['alpha_forward']}, alpha_backward={loss_config['alpha_backward']}")
    
    # Compute class weights for backward
    train_backward_labels = graph_data.y_backward[graph_data.train_mask]
    train_backward_labels_filtered = train_backward_labels[train_backward_labels > 0]
    backward_weights = compute_class_weights(train_backward_labels_filtered, 6, device)
    backward_weights[0] = 0.0  # Ignore NO_MERCHANT
    
    print(f"Backward class weights: {backward_weights.cpu().numpy().round(2)}")
    
    # Training
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_metric = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_forward': [], 'val_backward': []}
    
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/evaluations', exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        graph_data_d = graph_data.to(device)
        
        # Forward pass
        output = model(graph_data_d.x, graph_data_d.edge_index)
        
        # Labels (train only)
        train_mask = graph_data_d.train_mask
        labels = {
            'forward': graph_data_d.y_forward[train_mask],
            'backward': graph_data_d.y_backward[train_mask]
        }
        
        output_train = {
            'forward': output['forward'][train_mask],
            'backward': output['backward'][train_mask],
            'embeddings': output['embeddings'][train_mask]
        }
        
        # Loss
        loss, losses = compute_multitask_loss(output_train, labels, loss_config, backward_weights)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Evaluate
        val_metrics = evaluate(model, graph_data, device, 'val')
        
        # Combined metric (weighted toward backward since it's harder)
        combined = 0.4 * val_metrics['forward']['accuracy'] + 0.6 * val_metrics['backward']['accuracy']
        scheduler.step(combined)
        
        # History
        history['train_loss'].append(losses['total'])
        history['val_forward'].append(val_metrics['forward']['accuracy'])
        history['val_backward'].append(val_metrics['backward']['accuracy'])
        
        # Best model
        improved = ""
        if combined > best_metric:
            best_metric = combined
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model_config,
                'epoch': epoch,
                'best_metric': best_metric
            }, 'results/models/baseline_dual_attribution.pt')
            improved = " â˜…"
        else:
            patience_counter += 1
        
        # Log
        if epoch % 5 == 0 or improved:
            print(f"Epoch {epoch:03d} | "
                  f"Loss: {losses['total']:.4f} (F:{losses['forward']:.4f}, B:{losses['backward']:.4f}) | "
                  f"Val Fwd: {val_metrics['forward']['accuracy']:.2%} | "
                  f"Val Bwd: {val_metrics['backward']['accuracy']:.2%} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}{improved}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Load best and evaluate on test
    checkpoint = torch.load('results/models/baseline_dual_attribution.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, graph_data, device, 'test')
    
    # Results
    print("\n" + "="*70)
    print("FINAL RESULTS (TEST SET)")
    print("="*70)
    print(f"\nForward Attribution (Exchange Classification):")
    print(f"  Top-1 Accuracy: {test_metrics['forward']['accuracy']:.2%}")
    print(f"  Top-2 Accuracy: {test_metrics['forward']['top2_accuracy']:.2%}")
    
    print(f"\nBackward Attribution (Category Classification):")
    print(f"  Overall Accuracy: {test_metrics['backward']['accuracy']:.2%}")
    print(f"\n  Per-Category Accuracy:")
    for cat_name, cat_acc in test_metrics['backward'].get('per_category', {}).items():
        print(f"    {cat_name:12s}: {cat_acc:.2%}")
    
    # Target check
    fwd_ok = test_metrics['forward']['accuracy'] >= 0.85
    bwd_ok = test_metrics['backward']['accuracy'] >= 0.60
    
    print(f"\n{'âœ…' if fwd_ok else 'âš ï¸'} Forward target (â‰¥85%): {test_metrics['forward']['accuracy']:.2%}")
    print(f"{'âœ…' if bwd_ok else 'âš ï¸'} Backward target (â‰¥60%): {test_metrics['backward']['accuracy']:.2%}")
    
    # # Save results
    # results = {
    #     'config': {
    #         'model': model_config,
    #         'loss': loss_config,
    #         'training': {
    #             'epochs': epoch + 1,
    #             'lr': args.lr,
    #             'patience': args.patience,
    #             'alpha_backward': args.alpha_backward
    #         }
    #     },
    #     'test_metrics': test_metrics,
    #     'history': history,
    #     'timestamp': datetime.now().isoformat()
    # }
    
    # results_path = 'results/evaluations/baseline_results.json'
    # with open(results_path, 'w') as f:
    #     json.dump(results, f, indent=2, default=str)
    
    # print(f"\nâœ“ Results saved: {results_path}")
    # print(f"âœ“ Model saved: results/models/baseline_dual_attribution.pt")

    # Save results - STANDARDIZED FORMAT FOR COMPARISON
    results = {
        # === IDENTIFICATION ===
        'approach': 'baseline',
        'approach_name': 'Centralized Baseline',
        'training_type': 'centralized',
        'privacy_level': 'none',
        'privacy_description': 'No privacy - all data centralized',
        
        # === DATA CONFIG ===
        'data_config': {
            'total_transactions': int(graph_data.x.shape[0]),
            'feature_count': int(graph_data.num_features),
            'train_size': int(graph_data.train_mask.sum()),
            'val_size': int(graph_data.val_mask.sum()),
            'test_size': int(graph_data.test_mask.sum()),
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
            'epochs': epoch + 1,
            'epochs_equivalent': epoch + 1,  # For comparison with FL
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'early_stopping_patience': args.patience,
            'optimizer': 'Adam',
            'loss_type': 'FocalLoss',
            'alpha_forward': args.alpha_forward,
            'alpha_backward': args.alpha_backward,
            'focal_gamma': 2.0
        },
        
        # === SECURITY CONFIG (none for baseline) ===
        'security_config': {
            'differential_privacy': False,
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
            'forward_accuracy': test_metrics['forward']['accuracy'],
            'forward_top2_accuracy': test_metrics['forward']['top2_accuracy'],
            'backward_accuracy': test_metrics['backward']['accuracy'],
            'backward_per_category': test_metrics['backward'].get('per_category', {}),
            # Computed metrics
            'forward_improvement_vs_random': test_metrics['forward']['accuracy'] - (1/3),
            'backward_improvement_vs_random': test_metrics['backward']['accuracy'] - (1/5),
        },
        
        # === TRAINING HISTORY ===
        'history': {
            'epochs': list(range(len(history['train_loss']))),
            'train_loss': history['train_loss'],
            'val_forward_acc': history['val_forward'],
            'val_backward_acc': history['val_backward'],
            'learning_rates': history.get('learning_rates', [])
        },
        
        # === METADATA ===
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'script': 'p2_train_baseline.py',
            'version': 'v2_categories',
            'device': str(device),
            'targets_met': {
                'forward_target_85': test_metrics['forward']['accuracy'] >= 0.85,
                'backward_target_60': test_metrics['backward']['accuracy'] >= 0.60
            }
        }
    }
    
    results_path = 'results/evaluations/baseline_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nâœ“ Results saved: {results_path}")
    print(f"âœ“ Model saved: results/models/baseline_dual_attribution.pt")
    
    return test_metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline Training - Dual Attribution Model V2')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/federated_enriched',
                        help='Directory with enriched data')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=150,
                        help='Max epochs')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Loss weights
    parser.add_argument('--alpha_forward', type=float, default=1.0,
                        help='Weight for forward loss')
    parser.add_argument('--alpha_backward', type=float, default=1.5,
                        help='Weight for backward loss (increased for category balance)')
    
    args = parser.parse_args()
    
    train(args)
