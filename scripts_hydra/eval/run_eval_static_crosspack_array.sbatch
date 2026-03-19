#!/bin/bash
#SBATCH --job-name=eval_static_cross_%a
#SBATCH --output=logs/eval_static_cross_%A_%a.out
#SBATCH --error=logs/eval_static_cross_%A_%a.err
#SBATCH --partition=cpu-2d
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --array=0-14
#SBATCH --nodelist=head065,head066,head067,head068,head069,head070,head071

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-}"
if [ -z "$ROOT_DIR" ] || [ ! -d "$ROOT_DIR" ]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT_DIR"

mkdir -p logs
source .venv/bin/activate

TEST_LISTS_DIR="${TEST_LISTS_DIR:-$ROOT_DIR/sokoban_core/levels/splits/letslogic/clusters_transferability_driven/test}"
MODELS_DIR="${MODELS_DIR:-$ROOT_DIR/artifacts/letslogic/transferability_driven}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/evaluate_transferability_features_final_test/static_crosspack/raw}"
MODE="${MODE:-speed}"
TIME_LIMIT="${TIME_LIMIT:-600}"
NODE_LIMIT="${NODE_LIMIT:-2000000}"
GNN_JOBS="${GNN_JOBS:-1}"
HUNGARIAN_JOBS="${HUNGARIAN_JOBS:-1}"
GENERAL_MODEL="${GENERAL_MODEL:-$ROOT_DIR/artifacts/letslogic/single_model/letslogic_single.pt}"
INCLUDE_SELF="${INCLUDE_SELF:-1}"  # 0 -> evaluate only other packs, 1 -> include own model
SKIP_EXISTING="${SKIP_EXISTING:-1}"

TARGET_ID="${SLURM_ARRAY_TASK_ID:-0}"
TARGET_GROUP="group_$(printf "%03d" "$TARGET_ID")"
TARGET_LIST="$TEST_LISTS_DIR/${TARGET_GROUP}.list"

if [ ! -f "$TARGET_LIST" ]; then
  echo "ERROR: target list not found: $TARGET_LIST"
  exit 1
fi

TARGET_OUT_DIR="$OUT_DIR/target_${TARGET_GROUP}"
OUT_BASE_DIR="$(dirname "$OUT_DIR")"
GENERAL_GNN_DIR="$OUT_BASE_DIR/general/gnn"
GENERAL_HUNGARIAN_DIR="$OUT_BASE_DIR/general/hungarian"
mkdir -p "$TARGET_OUT_DIR"
mkdir -p "$GENERAL_GNN_DIR" "$GENERAL_HUNGARIAN_DIR"

echo "[INFO] target pack: $TARGET_GROUP"
echo "[INFO] list: $TARGET_LIST"
echo "[INFO] models dir: $MODELS_DIR"
echo "[INFO] out dir: $TARGET_OUT_DIR"
echo "[INFO] general model: $GENERAL_MODEL"

# Avoid nested over-parallelism: multiprocessing handles parallel workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

GENERAL_GNN_OUT="$GENERAL_GNN_DIR/${TARGET_GROUP}.csv"
if [ ! -f "$GENERAL_MODEL" ]; then
  echo "ERROR: general checkpoint not found: $GENERAL_MODEL"
  exit 1
fi
if [ "$SKIP_EXISTING" = "1" ] && [ -f "$GENERAL_GNN_OUT" ]; then
  echo "[SKIP] general gnn already exists: $GENERAL_GNN_OUT"
else
  echo "[RUN] target=$TARGET_GROUP model=general -> $GENERAL_GNN_OUT"
  python3 -m scripts.solve.run_batch_gnn \
    --list "$TARGET_LIST" \
    --ckpt "$GENERAL_MODEL" \
    --mode "$MODE" \
    --out "$GENERAL_GNN_OUT" \
    --time_limit "$TIME_LIMIT" \
    --node_limit "$NODE_LIMIT" \
    --jobs "$GNN_JOBS"
fi

GENERAL_HUNGARIAN_OUT="$GENERAL_HUNGARIAN_DIR/${TARGET_GROUP}.csv"
if [ "$SKIP_EXISTING" = "1" ] && [ -f "$GENERAL_HUNGARIAN_OUT" ]; then
  echo "[SKIP] hungarian already exists: $GENERAL_HUNGARIAN_OUT"
else
  echo "[RUN] target=$TARGET_GROUP model=hungarian+dl -> $GENERAL_HUNGARIAN_OUT"
  python3 -m scripts.solve.run_batch \
    --list "$TARGET_LIST" \
    --h hungarian \
    --use_dl \
    --out "$GENERAL_HUNGARIAN_OUT" \
    --time_limit "$TIME_LIMIT" \
    --node_limit "$NODE_LIMIT" \
    --jobs "$HUNGARIAN_JOBS"
fi

shopt -s nullglob
for CKPT_PATH in "$MODELS_DIR"/group_*.pt; do
  MODEL_GROUP="$(basename "$CKPT_PATH" .pt)"
  if [ "$INCLUDE_SELF" = "0" ] && [ "$MODEL_GROUP" = "$TARGET_GROUP" ]; then
    continue
  fi

  OUT_FILE="$TARGET_OUT_DIR/model_${MODEL_GROUP}.csv"

  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$OUT_FILE" ]; then
    echo "[SKIP] already exists: $OUT_FILE"
    continue
  fi

  echo "[RUN] target=$TARGET_GROUP model=$MODEL_GROUP -> $OUT_FILE"
  python3 -m scripts.solve.run_batch_gnn \
    --list "$TARGET_LIST" \
    --ckpt "$CKPT_PATH" \
    --mode "$MODE" \
    --out "$OUT_FILE" \
    --time_limit "$TIME_LIMIT" \
    --node_limit "$NODE_LIMIT" \
    --jobs "$GNN_JOBS"
done
shopt -u nullglob

echo "[DONE] target=$TARGET_GROUP"
