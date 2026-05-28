#!/bin/bash
#SBATCH --account=def-umaivodj
#SBATCH --job-name=retrain_fl
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/retrain_fl_%A_%a.out
#SBATCH --error=logs/retrain_fl_%A_%a.err

# ============================================================================
# Re-training of FL standard configurations under canonical artifacts (v2)
# ----------------------------------------------------------------------------
# 6 configs × 5 seeds = 30 jobs (array indices 0..29)
#
# Configs (verified against analyze_results.py PHASE1+ABLATION):
#   0: fl_r20e3                  F1 — FL non-IID, 20 rounds × 3 epochs
#   1: fl_r100e1                 F2 — FL non-IID, 100 rounds × 1 epoch
#   2: fl_iid_noxedge_r20e3      F3 — FL IID NoXEdge, 20×3
#   3: fl_iid_noxedge_r100e1     F4 — FL IID NoXEdge, 100×1
#   4: fl_r100e1_single_fwd      A5 — FL R100×E1 single-task fwd
#   5: fl_r100e1_single_bwd      A6 — FL R100×E1 single-task bwd
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
    0) TAG="fl_r20e3"
       EXTRA="--rounds 20 --local_epochs 3" ;;

    1) TAG="fl_r100e1"
       EXTRA="--rounds 100 --local_epochs 1" ;;

    2) TAG="fl_iid_noxedge_r20e3"
       EXTRA="--iid --iid_no_cross_edges --rounds 20 --local_epochs 3" ;;

    3) TAG="fl_iid_noxedge_r100e1"
       EXTRA="--iid --iid_no_cross_edges --rounds 100 --local_epochs 1" ;;

    4) TAG="fl_r100e1_single_fwd"
       EXTRA="--rounds 100 --local_epochs 1 --alpha_backward 0.0" ;;

    5) TAG="fl_r100e1_single_bwd"
       EXTRA="--rounds 100 --local_epochs 1 --alpha_forward 0.0" ;;

    *) echo "[FATAL] Invalid CFG_IDX: $CFG_IDX"; exit 1 ;;
esac

EXPECTED_MODEL="results/models/federated_dual_attribution_${TAG}_seed${SEED}.pt"
EXPECTED_RESULT="results/evaluations/federated_results_${TAG}_seed${SEED}.json"
MARKER_FILE="${EXPECTED_MODEL}.canonical"

echo "======================================================================"
echo "  Re-train FL standard | Array=${IDX} CFG=${CFG_IDX} SEED=${SEED}"
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
echo "[$(date +%H:%M)] python p3_train_federated_2layers.py ${EXTRA} --seed ${SEED} --results_tag ${TAG}"
python p3_train_federated_2layers.py ${EXTRA} --seed ${SEED} --results_tag "${TAG}"
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[FAILED] Training exited with code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

# ── Validate outputs ────────────────────────────────────────────────────────
echo
echo "[$(date +%H:%M)] Validating outputs..."
[ ! -f "$EXPECTED_MODEL" ]   && { echo "[FATAL] Missing: $EXPECTED_MODEL"; exit 1; }
SIZE=$(stat -c%s "$EXPECTED_MODEL")
[ "$SIZE" -lt 100000 ]       && { echo "[FATAL] Model too small: $SIZE bytes"; exit 1; }
echo "[OK] Model: $(basename $EXPECTED_MODEL) ($SIZE bytes)"

[ ! -f "$EXPECTED_RESULT" ]  && { echo "[FATAL] Missing: $EXPECTED_RESULT"; exit 1; }
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
