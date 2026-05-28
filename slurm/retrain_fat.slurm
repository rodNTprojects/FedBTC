#!/bin/bash
#SBATCH --account=def-umaivodj
#SBATCH --job-name=retrain_fat
#SBATCH --array=0-34
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/retrain_fat_%A_%a.out
#SBATCH --error=logs/retrain_fat_%A_%a.err

# ============================================================================
# Re-training of FAT (Federated Adversarial Training) configurations (v2)
# under canonical artifacts
# ----------------------------------------------------------------------------
# 7 configs × 5 seeds = 35 jobs (array indices 0..34)
#
# CORRECTIONS vs v1:
#   - Mode names use HYPHEN: 'fat-base', 'fat-def' (NOT 'fat_base', 'fat_def')
#     verified at line 1174,1184 of p3_train_federated_secure_2layers_FAT.py
#   - Privacy flag is --epsilon (NOT --dp_epsilon)
#   - Tags include '.0' suffix to match analyze_results.py PHASE6_CONFIGS
#
# Configs:
#   0: fat_base                  FAT-base (AT, no DP)        --mode fat-base
#   1: fat_def_eps1.0            FAT-def (eps=1.0)           --mode fat-def --epsilon 1.0
#   2: fat_def_eps2.0            FAT-def (eps=2.0)           --mode fat-def --epsilon 2.0
#   3: fat_def_eps4.0            FAT-def (eps=4.0)           --mode fat-def --epsilon 4.0
#   4: fat_def_eps8.0            FAT-def (eps=8.0)           --mode fat-def --epsilon 8.0
#   5: fat_def_eps16.0           FAT-def (eps=16.0)          --mode fat-def --epsilon 16.0
#   6: fat_def_eps32.0           FAT-def (eps=32.0)          --mode fat-def --epsilon 32.0
#
# Layout: index = (config_idx * 5) + seed_idx
# Seeds:  0, 45, 123, 654, 789
# ============================================================================

source $HOME/venvs/fedbtc-test/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4
cd ~/links/scratch/fedbtc

mkdir -p logs results/models results/evaluations

SEEDS=(0 45 123 654 789)
IDX=$SLURM_ARRAY_TASK_ID
CFG_IDX=$((IDX / 5))
SEED_IDX=$((IDX % 5))
SEED=${SEEDS[$SEED_IDX]}

case $CFG_IDX in
    0) TAG="fat_base";        EXTRA="--mode fat-base" ;;
    1) TAG="fat_def_eps1.0";  EXTRA="--mode fat-def --epsilon 1.0" ;;
    2) TAG="fat_def_eps2.0";  EXTRA="--mode fat-def --epsilon 2.0" ;;
    3) TAG="fat_def_eps4.0";  EXTRA="--mode fat-def --epsilon 4.0" ;;
    4) TAG="fat_def_eps8.0";  EXTRA="--mode fat-def --epsilon 8.0" ;;
    5) TAG="fat_def_eps16.0"; EXTRA="--mode fat-def --epsilon 16.0" ;;
    6) TAG="fat_def_eps32.0"; EXTRA="--mode fat-def --epsilon 32.0" ;;
    *) echo "[FATAL] Invalid CFG_IDX: $CFG_IDX"; exit 1 ;;
esac

EXPECTED_MODEL="results/models/federated_secure_dual_attribution_${TAG}_seed${SEED}.pt"
EXPECTED_RESULT="results/evaluations/federated_secure_results_${TAG}_seed${SEED}.json"
MARKER_FILE="${EXPECTED_MODEL}.canonical"

echo "======================================================================"
echo "  Re-train FAT | Array=${IDX} CFG=${CFG_IDX} SEED=${SEED}"
echo "  Tag=${TAG}  Extra=${EXTRA}"
echo "  Job=${SLURM_JOB_ID}  Node=$(hostname)  $(date)"
echo "======================================================================"

# ── Idempotence + cleanup of stale (non-canonical) models ──────────────────
# A model is considered "canonical" if both the .pt and the .canonical marker
# file exist. The marker is written ONLY by this SLURM, never by the training
# script itself. Three cases:
#   (a) Both .pt and marker present  → skip (already done under canonical)
#   (b) .pt present, marker absent   → stale model, delete and re-train
#   (c) .pt absent                   → fresh training
if [ -f "$EXPECTED_MODEL" ] && [ -f "$MARKER_FILE" ]; then
    echo "[SKIP] Already trained under canonical: $(basename $EXPECTED_MODEL)"
    exit 0
fi
if [ -f "$EXPECTED_MODEL" ] && [ ! -f "$MARKER_FILE" ]; then
    echo "[CLEAN] Removing stale model (no canonical marker): $(basename $EXPECTED_MODEL)"
    rm -f "$EXPECTED_MODEL"
    if [ -f "$EXPECTED_RESULT" ]; then
        echo "[CLEAN] Removing stale result: $(basename $EXPECTED_RESULT)"
        rm -f "$EXPECTED_RESULT"
    fi
fi

# ── Sanity ──────────────────────────────────────────────────────────────────
for f in data/split_map.pkl data/normalization_stats.pkl; do
    if [ ! -f "$f" ]; then
        echo "[FATAL] Canonical artifact missing: $f"
        exit 1
    fi
done

# ── Train ───────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M)] python p3_train_federated_secure_2layers_FAT.py ${EXTRA} --seed ${SEED} --results_tag ${TAG}"
python p3_train_federated_secure_2layers_FAT.py ${EXTRA} --rounds 100 --local_epochs 1 --seed ${SEED} --results_tag "${TAG}"
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[FAILED] Training exited with code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

# ── Validate ────────────────────────────────────────────────────────────────
echo
echo "[$(date +%H:%M)] Validating outputs..."
[ ! -f "$EXPECTED_MODEL" ]  && { echo "[FATAL] Missing: $EXPECTED_MODEL"; exit 1; }
SIZE=$(stat -c%s "$EXPECTED_MODEL")
[ "$SIZE" -lt 100000 ]      && { echo "[FATAL] Model too small: $SIZE bytes"; exit 1; }
echo "[OK] Model: $(basename $EXPECTED_MODEL) ($SIZE bytes)"

[ ! -f "$EXPECTED_RESULT" ] && { echo "[FATAL] Missing: $EXPECTED_RESULT"; exit 1; }
python << PYEOF
import json, sys
try:
    d = json.load(open('$EXPECTED_RESULT'))
    fwd = d.get('test_metrics', {}).get('forward_accuracy')
    bwd = d.get('test_metrics', {}).get('backward_accuracy')
    if fwd is None: sys.exit('Missing test_metrics.forward_accuracy')
    if not (0.20 < fwd < 0.99): sys.exit(f'Suspicious fwd_acc: {fwd}')
    print(f'[OK] forward_acc={fwd:.4f}  backward_acc={bwd:.4f}')
except Exception as e:
    sys.exit(f'JSON validation failed: {e}')
PYEOF
[ $? -ne 0 ] && exit 1

touch "$MARKER_FILE"
echo "[OK] Canonical marker written"
echo "======================================================================"
echo "  [$(date +%H:%M)] OK ${TAG} seed=${SEED}"
echo "======================================================================"
exit 0
