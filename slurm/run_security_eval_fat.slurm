#!/bin/bash
#SBATCH --job-name=fedbtc_sec_eval_fat
#SBATCH --output=logs/sec_eval_fat_%A_%a.out
#SBATCH --error=logs/sec_eval_fat_%A_%a.err
#SBATCH --account=def-umaivodj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-4

# ============================================================================
# Security evaluation for FAT-base and FAT-def eps=8.0 only.
# Extracted from run_cluster_phase6_v2.slurm (Phase 7 block, lines 103-136).
#
# Re-training blocks (Phases 6a/6b/6c of run_cluster_phase6_v2.slurm) are
# omitted here: the 130 Option B-2 models are already trained.
#
# Output: results/evaluations/security_eval_{tag}.json
# (produced by run_security_evaluation.py for each --output_tag)
# ============================================================================

source $HOME/venvs/fedbtc-test/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4
cd ~/links/scratch/fedbtc

SEEDS=(0 45 123 654 789)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================================"
echo "  Security evaluation FAT | SEED=${SEED} | Array=${SLURM_ARRAY_TASK_ID}"
echo "  Job=${SLURM_JOB_ID}  Node=$(hostname)  $(date)"
echo "======================================================================"

mkdir -p logs results/evaluations

# ── FAT-base ────────────────────────────────────────────────────────────────
MODEL_PATH="results/models/federated_secure_dual_attribution_fat_base_seed${SEED}.pt"
if [ -f "${MODEL_PATH}" ]; then
    echo ""
    echo "[$(date +%H:%M)]   Evaluating: fat_base seed=${SEED}"
    python run_security_evaluation.py \
        --model_path  "${MODEL_PATH}" \
        --eval_mode   both \
        --attacks     all \
        --seed        ${SEED} \
        --output_tag  "fat_base_seed${SEED}" \
        && echo "[$(date +%H:%M)] [OK] Security eval fat_base" \
        || echo "[$(date +%H:%M)] [FAILED] Security eval fat_base"
else
    echo "[$(date +%H:%M)]   SKIP (not found): ${MODEL_PATH}"
fi

# ── FAT-def eps=8.0 ─────────────────────────────────────────────────────────
MODEL_PATH="results/models/federated_secure_dual_attribution_fat_def_eps8.0_seed${SEED}.pt"
if [ -f "${MODEL_PATH}" ]; then
    echo ""
    echo "[$(date +%H:%M)]   Evaluating: fat_def_eps8.0 seed=${SEED}"
    python run_security_evaluation.py \
        --model_path  "${MODEL_PATH}" \
        --eval_mode   both \
        --attacks     all \
        --seed        ${SEED} \
        --output_tag  "fat_def_eps8.0_seed${SEED}" \
        && echo "[$(date +%H:%M)] [OK] Security eval fat_def_eps8.0" \
        || echo "[$(date +%H:%M)] [FAILED] Security eval fat_def_eps8.0"
else
    echo "[$(date +%H:%M)]   SKIP (not found): ${MODEL_PATH}"
fi

echo ""
echo "=== DONE SEED=${SEED} at $(date) ==="
