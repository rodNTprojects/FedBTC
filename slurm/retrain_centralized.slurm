#!/bin/bash
#SBATCH --account=def-umaivodj
#SBATCH --job-name=retrain_cent
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/retrain_cent_%A_%a.out
#SBATCH --error=logs/retrain_cent_%A_%a.err

# ============================================================================
# Re-training of centralized configurations under canonical artifacts (v2)
# ----------------------------------------------------------------------------
# 8 configs × 5 seeds = 40 jobs (array indices 0..39)
#
# Configs (verified against analyze_results.py PHASE1+ABLATION+PHASE6):
#   0: cent_100ep                C1 — Centralized 100ep
#   1: cent_200ep                C2 — Centralized 200ep
#   2: cent_noxedge_100ep        C3 — Centralized NoXEdge 100ep
#   3: noxedge_cat               CAT (Phase 6+, NoXEdge + AT)
#   4: single_fwd                A1 — Single-task Fwd (Centralized)
#   5: single_bwd                A2 — Single-task Bwd (Centralized)
#   6: noxedge_single_fwd        A3 — Single-task Fwd (NoXEdge)
#   7: noxedge_single_bwd        A4 — Single-task Bwd (NoXEdge)
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

# ── Decode (config, seed) from array index ──────────────────────────────────
IDX=$SLURM_ARRAY_TASK_ID
CFG_IDX=$((IDX / 5))
SEED_IDX=$((IDX % 5))
SEED=${SEEDS[$SEED_IDX]}

# Map config_idx -> (script, results_tag, extra_args, expected_model)
# Note: CAT uses --results_tag "" because the CAT script hardcodes
# "noxedge_cat" in the output path; passing the tag would duplicate it.
case $CFG_IDX in
    0) SCRIPT="p2_train_baseline_2layers.py"
       TAG="cent_100ep"
       RESULTS_TAG="cent_100ep"
       EXTRA="--epochs 100"
       MODEL_BASENAME="baseline_dual_attribution_cent_100ep"
       RESULT_BASENAME="baseline_results_cent_100ep" ;;

    1) SCRIPT="p2_train_baseline_2layers.py"
       TAG="cent_200ep"
       RESULTS_TAG="cent_200ep"
       EXTRA="--epochs 200"
       MODEL_BASENAME="baseline_dual_attribution_cent_200ep"
       RESULT_BASENAME="baseline_results_cent_200ep" ;;

    2) SCRIPT="p2_train_baseline_2layers_NoXEdge.py"
       TAG="cent_noxedge_100ep"
       RESULTS_TAG="cent_noxedge_100ep"
       EXTRA="--epochs 100"
       MODEL_BASENAME="baseline_dual_attribution_cent_noxedge_100ep"
       RESULT_BASENAME="baseline_results_cent_noxedge_100ep" ;;

    3) SCRIPT="p2_train_baseline_2layers_NoXEdge_CAT.py"
       TAG="noxedge_cat"
       RESULTS_TAG=""
       EXTRA="--epochs 100 --enable_at"
       MODEL_BASENAME="baseline_dual_attribution_noxedge_cat"
       RESULT_BASENAME="baseline_results_noxedge_cat" ;;

    4) SCRIPT="p2_train_baseline_2layers.py"
       TAG="single_fwd"
       RESULTS_TAG="single_fwd"
       EXTRA="--epochs 100 --alpha_backward 0.0"
       MODEL_BASENAME="baseline_dual_attribution_single_fwd"
       RESULT_BASENAME="baseline_results_single_fwd" ;;

    5) SCRIPT="p2_train_baseline_2layers.py"
       TAG="single_bwd"
       RESULTS_TAG="single_bwd"
       EXTRA="--epochs 100 --alpha_forward 0.0"
       MODEL_BASENAME="baseline_dual_attribution_single_bwd"
       RESULT_BASENAME="baseline_results_single_bwd" ;;

    6) SCRIPT="p2_train_baseline_2layers_NoXEdge.py"
       TAG="noxedge_single_fwd"
       RESULTS_TAG="noxedge_single_fwd"
       EXTRA="--epochs 100 --alpha_backward 0.0"
       MODEL_BASENAME="baseline_dual_attribution_noxedge_single_fwd"
       RESULT_BASENAME="baseline_results_noxedge_single_fwd" ;;

    7) SCRIPT="p2_train_baseline_2layers_NoXEdge.py"
       TAG="noxedge_single_bwd"
       RESULTS_TAG="noxedge_single_bwd"
       EXTRA="--epochs 100 --alpha_forward 0.0"
       MODEL_BASENAME="baseline_dual_attribution_noxedge_single_bwd"
       RESULT_BASENAME="baseline_results_noxedge_single_bwd" ;;

    *) echo "[FATAL] Invalid CFG_IDX: $CFG_IDX"; exit 1 ;;
esac

EXPECTED_MODEL="results/models/${MODEL_BASENAME}_seed${SEED}.pt"
EXPECTED_RESULT="results/evaluations/${RESULT_BASENAME}_seed${SEED}.json"
MARKER_FILE="${EXPECTED_MODEL}.canonical"

echo "======================================================================"
echo "  Re-train centralized | Array=${IDX}  CFG=${CFG_IDX} SEED=${SEED}"
echo "  Script=${SCRIPT}  Tag=${TAG}"
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
    echo "[SKIP] Already trained under canonical artifacts: $(basename $EXPECTED_MODEL)"
    exit 0
fi
if [ -f "$EXPECTED_MODEL" ] && [ ! -f "$MARKER_FILE" ]; then
    echo "[CLEAN] Removing stale model (no canonical marker): $(basename $EXPECTED_MODEL)"
    rm -f "$EXPECTED_MODEL"
    # Also remove the corresponding evaluation result if any (it would be stale too)
    if [ -f "$EXPECTED_RESULT" ]; then
        echo "[CLEAN] Removing stale result: $(basename $EXPECTED_RESULT)"
        rm -f "$EXPECTED_RESULT"
    fi
fi

# ── Sanity: canonical artifacts must exist ──────────────────────────────────
for f in data/split_map.pkl data/normalization_stats.pkl; do
    if [ ! -f "$f" ]; then
        echo "[FATAL] Canonical artifact missing: $f"
        echo "        Run: python generate_canonical_artifacts.py"
        exit 1
    fi
done

# ── Run training ────────────────────────────────────────────────────────────
if [ -z "$RESULTS_TAG" ]; then
    echo "[$(date +%H:%M)] Training: python ${SCRIPT} ${EXTRA} --seed ${SEED}"
    python ${SCRIPT} ${EXTRA} --seed ${SEED}
else
    echo "[$(date +%H:%M)] Training: python ${SCRIPT} ${EXTRA} --seed ${SEED} --results_tag ${RESULTS_TAG}"
    python ${SCRIPT} ${EXTRA} --seed ${SEED} --results_tag "${RESULTS_TAG}"
fi
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[FAILED] Training exited with code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

# ── Health checks ───────────────────────────────────────────────────────────
echo
echo "[$(date +%H:%M)] Validating outputs..."

if [ ! -f "$EXPECTED_MODEL" ]; then
    echo "[FATAL] Expected model file missing: $EXPECTED_MODEL"
    exit 1
fi
SIZE=$(stat -c%s "$EXPECTED_MODEL")
if [ "$SIZE" -lt 100000 ]; then
    echo "[FATAL] Model file suspiciously small: $SIZE bytes"
    exit 1
fi
echo "[OK] Model file: $(basename $EXPECTED_MODEL) ($SIZE bytes)"

if [ ! -f "$EXPECTED_RESULT" ]; then
    echo "[FATAL] Expected result file missing: $EXPECTED_RESULT"
    exit 1
fi

python << PYEOF
import json, sys
try:
    d = json.load(open('$EXPECTED_RESULT'))
    fwd = d.get('test_metrics', {}).get('forward_accuracy')
    bwd = d.get('test_metrics', {}).get('backward_accuracy')
    if fwd is None:
        sys.exit('Missing test_metrics.forward_accuracy')
    if not (0.20 < fwd < 0.99):
        sys.exit(f'Suspicious forward_accuracy: {fwd}')
    print(f'[OK] forward_acc={fwd:.4f}  backward_acc={bwd:.4f}')
except Exception as e:
    sys.exit(f'JSON validation failed: {e}')
PYEOF
JSON_EXIT=$?
if [ $JSON_EXIT -ne 0 ]; then
    echo "[FATAL] Result JSON validation failed"
    exit 1
fi

# Mark model as canonical (used by idempotence check on re-run)
touch "$MARKER_FILE"
echo "[OK] Canonical marker written: $(basename $MARKER_FILE)"

echo
echo "======================================================================"
echo "  [$(date +%H:%M)] OK ${TAG} seed=${SEED}"
echo "======================================================================"
exit 0
