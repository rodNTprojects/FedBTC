#!/bin/bash
#SBATCH --job-name=fedbtc_cat
#SBATCH --output=logs/cat_%a.out
#SBATCH --error=logs/cat_%a.err
#SBATCH --account=def-umaivodj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --array=0-4

source $HOME/venvs/fedbtc-test/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd ~/links/scratch/fedbtc

SEEDS=(0 45 123 654 789)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================================"
echo "  FedBTC CAT — NoXEdge + Adversarial Training"
echo "  SEED=${SEED} | Node=$(hostname) | $(date)"
echo "======================================================================"

mkdir -p logs results/models results/evaluations

# Outputs (no results_tag to avoid double naming):
#   results/models/baseline_dual_attribution_noxedge_cat_seed${SEED}.pt
#   results/evaluations/baseline_results_noxedge_cat_seed${SEED}.json

python p2_train_baseline_2layers_NoXEdge_CAT.py \
    --seed        ${SEED} \
    --enable_at \
    --at_epsilon  0.2 \
    --at_steps    7 \
    --at_ratio    0.30 \
    --epochs      100 \
    --lr          0.002 \
    --patience    60 \
    --hidden_dim  128 \
    --dropout     0.3 \
    --results_tag "" \
    && echo "[OK] CAT seed=${SEED}" \
    || echo "[FAILED] CAT seed=${SEED}"

echo "======================================================================"
echo "  Done: SEED=${SEED} | $(date)"
echo "======================================================================"
