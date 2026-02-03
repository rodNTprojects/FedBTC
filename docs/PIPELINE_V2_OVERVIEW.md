# 🔄 Pipeline de Préparation des Données v2

## Vue d'ensemble

Ce document décrit le pipeline de préparation des données avec calibration BABD-13.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE DU PIPELINE v2                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUTS:                                                                │
│  ├── babd.zip (BABD-13 Dataset - 544K labeled addresses)               │
│  └── elliptic_bitcoin_dataset/ (203K transactions)                     │
│                                                                         │
│  STEP 0: calibrate_from_babd13.py                                      │
│  ├── Extraire distributions réelles                                    │
│  └── OUTPUT: config/calibration_params.json                            │
│                                                                         │
│  STEPS 1-9: Pipeline principal                                          │
│  └── OUTPUT: data/federated/exchange_{0,1,2}_enriched.pkl              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Étapes Détaillées

### Step 0: Calibration BABD-13 (NOUVEAU)

**Script**: `scripts/preprocessing/calibrate_from_babd13.py`

**Fonction**:
- Charge le dataset BABD-13 (544,462 adresses labellisées)
- Extrait les statistiques par type d'entité (Exchange, Gambling, Mining, etc.)
- Génère des profils d'exchange calibrés
- Crée des profils marchands basés sur patterns réels

**Outputs**:
- `config/calibration_params.json` - Paramètres calibrés
- `config/calibration_report.md` - Rapport de comparaison

**Justification scientifique**:
- Distributions basées sur données réelles (pas inventées)
- Reference: Xiang et al., IEEE TIFS 2024

---

### Step 1: Preprocess Elliptic

**Script**: `scripts/preprocessing/preprocess_elliptic.py`

**Fonction**:
- Charge le dataset Elliptic (203K transactions)
- Nettoie et normalise les features
- Crée les labels illicit/licit

---

### Step 2: Build Temporal Graph

**Script**: `scripts/preprocessing/build_temporal_graph.py`

**Fonction**:
- Construit le graphe temporel (49 timesteps)
- Crée les edges entre transactions

---

### Step 3: Simulate Merchants

**Script**: `scripts/preprocessing/simulate_merchants.py`

**Fonction**:
- Simule 500 entités marchandes
- **UTILISE** `calibration_params.json` pour patterns réalistes
- 5 catégories: e-commerce, retail, gaming, services, luxury

**Justification**:
Les patterns sont basés sur:
- BitPay Annual Report 2023
- Chainalysis Crypto Crime Report 2024
- BTCPay Server documentation

---

### Step 4: K-hop Expansion

**Script**: `scripts/preprocessing/expand_merchants_khop.py`

**Fonction**:
- BFS depuis les seeds marchands
- k=2 ou k=3 hops
- Capture les patterns de transaction

---

### Step 5: Precompute Embeddings

**Script**: `scripts/preprocessing/precompute_merchant_embeddings.py`

**Fonction**:
- Calcule les embeddings GNN pour chaque marchand
- Cache pour efficacité

---

### Step 6: Create Criminal DB

**Script**: `scripts/preprocessing/create_criminal_db.py`

**Fonction**:
- Crée la base de données des transactions criminelles
- Utilise les labels illicit d'Elliptic

---

### Step 7: Split Known/Unknown Merchants

**Script**: `scripts/preprocessing/split_merchants_known_unknown.py`

**Fonction**:
- Split 90% known / 10% unknown
- Known: utilisés pour training
- Unknown: utilisés pour évaluation de généralisation

**IMPORTANT**: Cette étape DOIT être AVANT partition_federated.py

---

### Step 8: Partition Federated

**Script**: `scripts/preprocessing/partition_federated.py`

**Fonction**:
- Partition les données pour K=3 exchanges
- Répartition équilibrée des transactions

---

### Step 9: Add Hybrid Features

**Script**: `scripts/preprocessing/add_hybrid_features_elliptic.py`

**Fonction**:
- Ajoute les features proxy discriminantes
- **UTILISE** `calibration_params.json` pour distributions calibrées

**Features ajoutées**:
| Catégorie | Features | Source calibration |
|-----------|----------|-------------------|
| Fee proxy | fee_percentage, fee_tier, etc. | BABD-13 + CoinMarketCap |
| Volume proxy | volume_scale, volume_class, etc. | BABD-13 |
| Hour proxy | synthetic_hour, timezone_proxy, etc. | Juhász 2018 |
| Liquidity proxy | liquidity_score, processing_speed, etc. | BABD-13 |

---

## Structure des Fichiers

```
bitcoin_fl_project/
├── babd.zip                          # À placer ici
├── p1_data_preparation.py            # Script principal
├── config/
│   ├── calibration_params.json       # Généré par Step 0
│   └── calibration_report.md         # Rapport
├── data/
│   ├── external/
│   │   └── babd13/                   # Extrait de babd.zip
│   ├── raw/
│   │   └── elliptic_bitcoin_dataset/ # Dataset Elliptic
│   ├── processed/                    # Données intermédiaires
│   └── federated/
│       ├── exchange_0_enriched.pkl   # Output final
│       ├── exchange_1_enriched.pkl
│       └── exchange_2_enriched.pkl
└── scripts/
    ├── preprocessing/
    │   ├── calibrate_from_babd13.py      # Step 0
    │   ├── preprocess_elliptic.py        # Step 1
    │   ├── build_temporal_graph.py       # Step 2
    │   ├── simulate_merchants.py         # Step 3
    │   ├── expand_merchants_khop.py      # Step 4
    │   ├── precompute_merchant_embeddings.py  # Step 5
    │   ├── create_criminal_db.py         # Step 6
    │   ├── split_merchants_known_unknown.py   # Step 7
    │   ├── partition_federated.py        # Step 8
    │   └── add_hybrid_features_elliptic.py    # Step 9
    └── utils/
        └── inspect_babd13.py             # Utilitaire
```

---

## Prochaines Étapes

1. **Upload babd.zip** à la racine du projet

2. **Exécuter l'inspection**:
   ```bash
   python scripts/utils/inspect_babd13.py
   ```

3. **Exécuter le pipeline**:
   ```bash
   python p1_data_preparation.py
   ```

4. **Vérifier les outputs**:
   - `config/calibration_params.json`
   - `data/federated/exchange_*_enriched.pkl`

---

## Validation Scientifique

| Composant | Avant (v1) | Après (v2) |
|-----------|------------|------------|
| Exchange features | Inventées | Calibrées BABD-13 |
| Merchant features | Inventées | Basées sur API docs |
| Distributions | Normal(guess) | Empiriques |
| Justification | Faible | Forte (citations) |
