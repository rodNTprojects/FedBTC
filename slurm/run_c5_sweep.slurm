#!/bin/bash
#SBATCH --job-name=fedbtc_c5_sweep
#SBATCH --output=logs/c5_sweep_%A_%a.out
#SBATCH --error=logs/c5_sweep_%A_%a.err
#SBATCH --account=def-umaivodj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-4

source $HOME/venvs/fedbtc-test/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $MPLCONFIGDIR
cd ~/links/scratch/fedbtc

SEEDS=(0 45 123 654 789)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================================"
echo "  FedBTC C5 DP Sweep | SEED=${SEED} | $(date)"
echo "  ε=8 (σ=0.10) → ε=16 (σ=0.05) → ε=32 (σ=0.025)"
echo "======================================================================"

mkdir -p logs results/models results/interaction_c5

# ── Train C5 at ε=16 (σ=0.05) ─────────────────────────────────────────────
MODEL_EPS16="results/models/federated_secure_dual_attribution_fat_def_eps16.0_seed${SEED}.pt"
if [ ! -f "$MODEL_EPS16" ]; then
    echo "[$(date +%H:%M)] Training C5 ε=16 (σ=0.05)..."
    python p3_train_federated_secure_2layers_FAT.py \
        --mode fat-def \
        --epsilon 16.0 \
        --rounds 100 \
        --local_epochs 1 \
        --seed ${SEED} \
        --results_tag "fat_def_eps16.0" \
        && echo "[OK] C5 ε=16 trained" \
        || echo "[FAILED] C5 ε=16"
else
    echo "[SKIP] C5 ε=16 already exists: $(basename $MODEL_EPS16)"
fi

# ── Train C5 at ε=32 (σ=0.025) ────────────────────────────────────────────
# get_noise_multiplier caps at 0.05 for eps>8 — we pass sigma explicitly
MODEL_EPS32="results/models/federated_secure_dual_attribution_fat_def_eps32.0_seed${SEED}.pt"
if [ ! -f "$MODEL_EPS32" ]; then
    echo "[$(date +%H:%M)] Training C5 ε=32 (σ=0.025)..."
    python p3_train_federated_secure_2layers_FAT.py \
        --mode fat-def \
        --epsilon 32.0 \
        --noise_multiplier 0.025 \
        --rounds 100 \
        --local_epochs 1 \
        --seed ${SEED} \
        --results_tag "fat_def_eps32.0" \
        && echo "[OK] C5 ε=32 trained" \
        || echo "[FAILED] C5 ε=32"
else
    echo "[SKIP] C5 ε=32 already exists: $(basename $MODEL_EPS32)"
fi

# ── Evaluate all three C5 variants + C0 ──────────────────────────────────
echo ""
echo "[$(date +%H:%M)] Evaluating C5 epsilon sweep..."

python p5_evaluate_c5_sweep.py \
    --seed ${SEED} \
    --data_dir data/federated_enriched \
    --output_dir results/interaction_c5 \
    --rounds 100 \
    && echo "[OK] Evaluation saved: results/interaction_c5/c5_sweep_seed${SEED}.json" \
    || echo "[FAILED] Evaluation"

# ── Aggregate and plot (last seed only) ──────────────────────────────────
if [ "${SLURM_ARRAY_TASK_ID}" = "4" ]; then
    echo ""
    echo "[$(date +%H:%M)] Aggregating 5 seeds and generating comparison plot..."
    sleep 60
    python p5_plot_c5_sweep.py \
        --results_dir results/interaction_c5 \
        --output_dir  results/interaction_c5 \
        && echo "[OK] Plot: results/interaction_c5/c5_sweep_comparison.png" \
        || echo "[FAILED] Plot"
fi

echo ""
echo "======================================================================"
echo "  Done: SEED=${SEED} | $(date)"
echo "======================================================================"
