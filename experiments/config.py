"""
================================================================================
Configuration partagée pour les expériences
================================================================================
"""

import os
from datetime import datetime

# Répertoires (à ajuster selon votre structure)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'federated_enriched')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'experiments')

# Créer les répertoires si nécessaires
os.makedirs(RESULTS_DIR, exist_ok=True)

# Catégories de marchands
CATEGORY_NAMES = {
    0: 'NO_MERCHANT',
    1: 'E-commerce',
    2: 'Gambling',
    3: 'Services',
    4: 'Retail',
    5: 'Luxury'
}

# Mapping epsilon -> noise_multiplier (approximatif)
EPSILON_TO_NOISE = {
    16.0: 0.05,
    8.0: 0.1,
    4.0: 0.2,
    2.0: 0.4,
    1.0: 0.8
}


def get_experiment_dir(experiment_name: str) -> str:
    """Get directory for experiment results."""
    exp_dir = os.path.join(RESULTS_DIR, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')
