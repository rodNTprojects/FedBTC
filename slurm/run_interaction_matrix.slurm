#!/bin/bash
#SBATCH --job-name=fedbtc_p5_eps16
#SBATCH --output=logs/p5_eps16_%A_%a.out
#SBATCH --error=logs/p5_eps16_%A_%a.err
#SBATCH --account=def-umaivodj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-4

# =============================================================================
# FedBTC Phase 5 — Interaction Matrix at epsilon = 16
# =============================================================================
# Uses COPIES of the original scripts patched with --models_dir:
#   - p5_train_defense_configs_eps16.py  (copy + --models_dir flag)
#   - p5_interaction_matrix_eps16.py     (copy + --models_dir flag)
#
# Output layout (entirely isolated, original results/models/ untouched):
#   results/models_eps16/        ← .pt files for C2/C3/C4/C6/C7 at eps=16
#   results/interaction_eps16/   ← per-seed JSONs + aggregated + plots
#   logs/p5_eps16_*.{out,err}
#
# C0 (No defense), C1 (AT only), C5 (Full stack eps=16) are not retrained here:
#   - C0/C1 do not depend on DP, already trained.
#   - C5 corresponds to fat_def_eps16.0 produced by retrain_fat.slurm.
# They are symlinked at Step 0 to results/models_eps16/ so that the
# interaction-matrix evaluation can find them via --models_dir.
# =============================================================================

source $HOME/venvs/fedbtc-test/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd ~/links/scratch/fedbtc
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $MPLCONFIGDIR

SEEDS=(0 45 123 654 789)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================================"
echo "  FedBTC Phase 5 — Interaction Matrix @ epsilon = 16"
echo "  SEED=${SEED} | Node=$(hostname) | $(date)"
echo "======================================================================"

mkdir -p logs results/interaction_eps16 results/models_eps16

# ── Step 0: Symlink models that do NOT depend on DP (C0, C1) and C5 ───────────
echo ""
echo "[$(date +%H:%M)] Step 0: Symlink existing models to results/models_eps16/"
for SRC_NAME in \
    "federated_dual_attribution_fl_r100e1_seed${SEED}.pt" \
    "federated_secure_dual_attribution_fat_base_seed${SEED}.pt" \
    "federated_secure_dual_attribution_fat_def_eps16.0_seed${SEED}.pt"; do
    SRC="results/models/${SRC_NAME}"
    DST="results/models_eps16/${SRC_NAME}"
    if [ -f "$SRC" ] && [ ! -e "$DST" ]; then
        ln -s "$(readlink -f $SRC)" "$DST"
        echo "  LINK ${SRC_NAME}"
    elif [ -e "$DST" ]; then
        echo "  EXISTS ${SRC_NAME} (skipped)"
    else
        echo "  MISS source missing: ${SRC_NAME}"
    fi
done

# ── Step 1: Train C2 / C3 / C4 / C6 / C7 at epsilon = 16 ──────────────────────
echo ""
echo "[$(date +%H:%M)] Step 1: Training C2 / C3 / C4 / C6 / C7 at epsilon=16.0"

python p5_train_defense_configs_eps16.py \
    --seed ${SEED} \
    --configs C2,C3,C4,C6,C7 \
    --rounds 100 \
    --local_epochs 1 \
    --epsilon 16.0 \
    --models_dir results/models_eps16 \
    && echo "[OK] Training done" \
    || echo "[WARN] Some configs may have failed — check logs"

# ── Step 2: Pre-flight model check ────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M)] Step 2: Model availability"
declare -A MODEL_PATHS
MODEL_PATHS[C0]="results/models_eps16/federated_dual_attribution_fl_r100e1_seed${SEED}.pt"
MODEL_PATHS[C1]="results/models_eps16/federated_secure_dual_attribution_fat_base_seed${SEED}.pt"
MODEL_PATHS[C2]="results/models_eps16/federated_secure_dual_attribution_coord_only_seed${SEED}.pt"
MODEL_PATHS[C3]="results/models_eps16/federated_secure_dual_attribution_dp_only_seed${SEED}.pt"
MODEL_PATHS[C4]="results/models_eps16/federated_secure_dual_attribution_at_coord_seed${SEED}.pt"
MODEL_PATHS[C5]="results/models_eps16/federated_secure_dual_attribution_fat_def_eps16.0_seed${SEED}.pt"
MODEL_PATHS[C6]="results/models_eps16/federated_secure_dual_attribution_coord_dp_seed${SEED}.pt"
MODEL_PATHS[C7]="results/models_eps16/federated_secure_dual_attribution_at_dp_seed${SEED}.pt"

AVAILABLE_CONFIGS=""
for CFG in C0 C1 C2 C3 C4 C5 C6 C7; do
    if [ -e "${MODEL_PATHS[$CFG]}" ]; then
        echo "  OK   ${CFG}: $(basename ${MODEL_PATHS[$CFG]})"
        AVAILABLE_CONFIGS="${AVAILABLE_CONFIGS},${CFG}"
    else
        echo "  MISS ${CFG}: $(basename ${MODEL_PATHS[$CFG]})"
    fi
done
AVAILABLE_CONFIGS="${AVAILABLE_CONFIGS#,}"

if [ -z "$AVAILABLE_CONFIGS" ]; then
    echo "[ERROR] No models found. Aborting."
    exit 1
fi
echo "  Running with: ${AVAILABLE_CONFIGS}"

# ── Step 3: Evaluate interaction matrix at eps = 16 ──────────────────────────
echo ""
echo "[$(date +%H:%M)] Step 3: Interaction matrix evaluation @ eps=16"

python p5_interaction_matrix_eps16.py \
    --seed ${SEED} \
    --data_dir data/federated_enriched \
    --models_dir results/models_eps16 \
    --output_dir results/interaction_eps16 \
    --configs ${AVAILABLE_CONFIGS} \
    --rounds 100 \
    && echo "[OK] Interaction matrix: results/interaction_eps16/interaction_matrix_seed${SEED}.json" \
    || echo "[FAILED] Interaction matrix"

# ── Step 4: Aggregate and plots (last seed only) ──────────────────────────────
if [ "${SLURM_ARRAY_TASK_ID}" = "4" ]; then
    echo ""
    echo "[$(date +%H:%M)] Step 4: Aggregation and visualizations"
    sleep 60
    python p5_plot_interaction_matrix.py \
        --results_dir results/interaction_eps16 \
        --output_dir  results/interaction_eps16 \
        && echo "[OK] Plots in results/interaction_eps16/" \
        || echo "[FAILED] Plots"
fi

echo ""
echo "======================================================================"
echo "  Done: SEED=${SEED} | epsilon=16 | $(date)"
echo "======================================================================"
