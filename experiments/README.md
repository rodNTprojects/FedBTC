# Experiments Framework for USENIX Security Paper

Ce répertoire contient les scripts pour exécuter toutes les expériences et générer les tableaux LaTeX pour l'article.

## Structure

```
experiments/
├── config.py                        # Configuration partagée
├── run_all_experiments.py           # Orchestrateur principal
├── generate_latex_tables.py         # Génération des tableaux LaTeX
├── scripts/
│   └── p3_train_federated_flexible.py  # Script FL modifié avec paramètres flexibles
└── README.md                        # Ce fichier
```

## Prérequis

Les scripts originaux doivent être présents dans le répertoire parent:
- `p2_train_baseline.py`
- `p5_security_evaluation.py`

Et les données dans:
- `data/federated_enriched/exchange_X_enriched.pkl`

## Utilisation

### Exécuter toutes les expériences

```bash
# Tout exécuter (~3-4 heures)
python run_all_experiments.py

# Voir ce qui sera exécuté (sans exécuter)
python run_all_experiments.py --dry-run

# Lister toutes les expériences disponibles
python run_all_experiments.py --list
```

### Exécuter des expériences spécifiques

```bash
# Seulement la comparaison principale (Table 1 & 2)
python run_all_experiments.py --only main_comparison

# Seulement privacy trade-off (Table 3)
python run_all_experiments.py --only privacy_tradeoff

# Plusieurs expériences
python run_all_experiments.py --only main_comparison ablation_layers

# Tout sauf byzantine (car utilise p5_security_evaluation.py)
python run_all_experiments.py --skip byzantine_attack
```

### Exécuter un seul run manuellement

```bash
# FL avec epsilon spécifique
python scripts/p3_train_federated_flexible.py --epsilon 4.0 --experiment_name privacy_eps_4

# FL avec nombre de couches différent
python scripts/p3_train_federated_flexible.py --num_layers 2 --experiment_name ablation_layers_2

# FL avec distribution de labels différente
python scripts/p3_train_federated_flexible.py --label_dist 80-10-10 --experiment_name ablation_dist_80_10_10

# FL sans DP (epsilon = infini)
python scripts/p3_train_federated_flexible.py --no-server-dp --experiment_name fl_nosec
```

### Générer les tableaux LaTeX

```bash
# Générer tous les tableaux depuis les résultats existants
python generate_latex_tables.py

# Générer dans un fichier spécifique
python generate_latex_tables.py --output my_tables.tex

# Générer seulement certains tableaux
python generate_latex_tables.py --tables 1 2 3
```

## Paramètres du script FL flexible

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--num_layers` | 3 | Nombre de couches GCN (1-5) |
| `--label_dist` | 60-20-20 | Distribution own-other1-other2 |
| `--epsilon` | 8.0 | Epsilon pour DP |
| `--local_epochs` | 3 | Epochs locaux par round |
| `--rounds` | 20 | Nombre de rounds FL |
| `--no-server-dp` | - | Désactiver DP serveur |
| `--no-flame` | - | Désactiver FLAME |
| `--no-adversarial` | - | Désactiver adversarial training |
| `--experiment_name` | - | Nom du fichier de sortie |

## Expériences et tableaux

| Expérience | Tableau | Runs | Temps estimé |
|------------|---------|------|--------------|
| `main_comparison` | Table 1, 2 | 5 | ~42 min |
| `privacy_tradeoff` | Table 3 | 6 | ~48 min |
| `byzantine_attack` | Table 4 | 1 | ~15 min |
| `ablation_layers` | Table 5 | 5 | ~40 min |
| `ablation_epochs` | Table 6 | 4 | ~30 min |
| ~~`ablation_distribution`~~ | ~~Table 7~~ | - | *Désactivée* |
| **Total** | | **21** | **~3 heures** |

> **Note:** L'expérience `ablation_distribution` (Table 7) est temporairement désactivée car elle nécessite une implémentation différente. La distribution de labels affecte la façon dont les données sont partitionnées entre les exchanges, pas le split train/val/test.

## Résultats

Les résultats sont sauvegardés dans:
- `results/experiments/` - Fichiers JSON des expériences
- `results/experiments/evaluation_tables.tex` - Tableaux LaTeX générés

## Dépannage

### Erreur "No data found"
Vérifiez que les données sont dans `data/federated_enriched/`

### Erreur de mémoire
Réduisez `--hidden_dim` à 64 ou exécutez les expériences une par une

### Comparaison invalide
Vérifiez que toutes les expériences utilisent `--split_seed 42` (par défaut)
