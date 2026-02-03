"""
Script pour calculer et mettre à jour rounds_to_converge dans les fichiers JSON existants.
"""

import json
import os
import glob

def calculate_convergence(history, threshold_forward=0.80, threshold_backward=0.80):
    """
    Calcule le round où le modèle a convergé.
    Convergence = premier round où forward > threshold ET backward > threshold
    """
    rounds = history.get('rounds', [])
    forward = history.get('forward_accuracy', [])
    backward = history.get('backward_accuracy', [])
    
    for i, r in enumerate(rounds):
        if i < len(forward) and i < len(backward):
            if forward[i] > threshold_forward and backward[i] > threshold_backward:
                return r
    
    return None  # N'a pas convergé

def update_json_files(results_dir='results/experiments'):
    """Met à jour tous les fichiers JSON avec rounds_to_converge."""
    
    json_files = glob.glob(f'{results_dir}/*.json')
    
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if 'history' in data:
                convergence = calculate_convergence(data['history'])
                
                if 'defense_stats' not in data:
                    data['defense_stats'] = {}
                
                data['defense_stats']['rounds_to_converge'] = convergence
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"✓ {os.path.basename(filepath)}: convergence = {convergence}")
            else:
                print(f"⚠ {os.path.basename(filepath)}: pas d'historique")
                
        except Exception as e:
            print(f"✗ {os.path.basename(filepath)}: {e}")

if __name__ == '__main__':
    update_json_files()
    print("\nTerminé! Relancez generate_latex_tables.py pour voir les valeurs mises à jour.")
