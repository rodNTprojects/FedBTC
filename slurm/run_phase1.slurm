#!/bin/bash
#SBATCH --job-name=fedbtc_p1
#SBATCH --output=logs/phase1_%A_%a.out
#SBATCH --error=logs/phase1_%A_%a.err
#SBATCH --account=def-umaivodj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-4

# ── Environment ──
#module load python/3.11.4             # adapter au module disponible
source $HOME/venvs/fedbtc-test/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

cd ~/links/scratch/fedbtc

# ── Seed mapping ──
SEEDS=(0 45 123 654 789)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "=== SEED=${SEED} | Node=$(hostname) | Start=$(date) ==="

# ────────────────────────────────────────
# CENTRALIZED
# ────────────────────────────────────────

# C1: Centralized 100 epochs (fair comparison vs F2)
echo "[$(date +%H:%M)] C1: Centralized 100ep"
python p2_train_baseline_2layers.py \
    --epochs 100 --seed ${SEED} --results_tag "cent_100ep"

# C2: Centralized 200 epochs (convergence proof)
echo "[$(date +%H:%M)] C2: Centralized 200ep"
python p2_train_baseline_2layers.py \
    --epochs 200 --seed ${SEED} --results_tag "cent_200ep"

# C3: Centralized NoXEdge 100 epochs
echo "[$(date +%H:%M)] C3: Centralized NoXEdge 100ep"
python p2_train_baseline_2layers_NoXEdge.py \
    --epochs 100 --seed ${SEED} --results_tag "cent_noxedge_100ep"

# ────────────────────────────────────────
# FEDERATED
# ────────────────────────────────────────

# F1: FL non-IID R20×E3 (config opérationnelle)
echo "[$(date +%H:%M)] F1: FL R20xE3 non-IID"
python p3_train_federated_2layers.py \
    --rounds 20 --local_epochs 3 \
    --seed ${SEED} --results_tag "fl_r20e3"

# F2: FL non-IID R100×E1 (fair comparison vs C1)
echo "[$(date +%H:%M)] F2: FL R100xE1 non-IID"
python p3_train_federated_2layers.py \
    --rounds 100 --local_epochs 1 \
    --seed ${SEED} --results_tag "fl_r100e1"

# F3: FL IID NoXEdge R20×E3
echo "[$(date +%H:%M)] F3: FL IID NoXEdge R20xE3"
python p3_train_federated_2layers.py \
    --iid --iid_no_cross_edges \
    --rounds 20 --local_epochs 3 \
    --seed ${SEED} --results_tag "fl_iid_noxedge_r20e3"

# F4: FL IID NoXEdge R100×E1 (fair comparison vs C3)
echo "[$(date +%H:%M)] F4: FL IID NoXEdge R100xE1"
python p3_train_federated_2layers.py \
    --iid --iid_no_cross_edges \
    --rounds 100 --local_epochs 1 \
    --seed ${SEED} --results_tag "fl_iid_noxedge_r100e1"

# ────────────────────────────────────────
# ABLATION MULTI-TASK (optionnel)
# ────────────────────────────────────────

# A1: Single-task forward only
echo "[$(date +%H:%M)] A1: Single-task forward"
python p2_train_baseline_2layers.py \
    --alpha_backward 0.0 --epochs 100 \
    --seed ${SEED} --results_tag "single_fwd"

# A2: Single-task backward only
echo "[$(date +%H:%M)] A2: Single-task backward"
python p2_train_baseline_2layers.py \
    --alpha_forward 0.0 --epochs 100 \
    --seed ${SEED} --results_tag "single_bwd"

# ────────────────────────────────────────
# TOPOLOGY EXPERIMENT
# ────────────────────────────────────────

echo "[$(date +%H:%M)] Topology eval"
python eval_centralized_on_subgraphs.py \
    --model_path results/models/*cent_noxedge_100ep*seed${SEED}*.pt

echo "=== DONE SEED=${SEED} at $(date) ==="
