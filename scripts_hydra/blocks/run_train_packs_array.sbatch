#!/bin/bash
#SBATCH --job-name=soko_train_packs
#SBATCH --output=logs/train_packs_%A_%a.out
#SBATCH --error=logs/train_packs_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --partition=gpu-5h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --constraint="80gb|40gb|h100|6000"
#SBATCH --array=0-34

# Train one small GNN per PACK.
#
# Expects labels produced by run_generate_labels_packs_array.sbatch:
#   data/packs_labels_festival/<pack>.jsonl
#
# Required env:
#   LABELS_DIR: directory containing *.jsonl (default: data/packs_labels_festival)
# Optional env:
#   OUT_DIR: output dir for models (default: artifacts/packs_models)
#   EPOCHS, BATCH, HIDDEN, LAYERS, LR, DROPOUT

set -euo pipefail
mkdir -p logs
mkdir -p artifacts

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo ".venv not found in project root"
  exit 1
fi

LABELS_DIR="${LABELS_DIR:-data/group_labels_festival}"
OUT_DIR="${OUT_DIR:-artifacts/packs_models}"
NUM_SHARDS="${NUM_SHARDS:-35}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID}"

EPOCHS="${EPOCHS:-9}"
BATCH="${BATCH:-512}"
HIDDEN="${HIDDEN:-64}"
LAYERS="${LAYERS:-3}"
LR="${LR:-1e-3}"
DROPOUT="${DROPOUT:-0.05}"
AMP="${AMP:-1}"
WORKERS="${WORKERS:-4}"
PREFETCH="${PREFETCH:-4}"
TORCH_THREADS="${TORCH_THREADS:-1}"
USE_TMP="${USE_TMP:-0}"

FILES=($(ls -1 "${LABELS_DIR}"/*.jsonl | sort))
echo "Total label files: ${#FILES[@]}"
echo "Shard: ${SHARD_IDX}/${NUM_SHARDS}"
mkdir -p "$OUT_DIR"

count=0
for i in "${!FILES[@]}"; do
  if [ $(( i % NUM_SHARDS )) -ne "$SHARD_IDX" ]; then
    continue
  fi

  TRAIN_PATH="${FILES[$i]}"
  BASE="$(basename "$TRAIN_PATH" .jsonl)"

  OUT_PATH="${OUT_DIR}/${BASE}_best.pt"

  echo "================================================"
  echo "Pack idx: $i"
  echo "Train   : $TRAIN_PATH"
  echo "Out     : $OUT_PATH"
  echo "================================================"

  python3 -m scripts.train_gnn \
    --train "$TRAIN_PATH" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --hidden "$HIDDEN" \
    --layers "$LAYERS" \
    --lr "$LR" \
    --dropout "$DROPOUT" \
    --workers "$WORKERS" \
    --prefetch_factor "$PREFETCH" \
    --pin_memory \
    --persistent_workers \
    --torch_threads "$TORCH_THREADS" \
    $( [ "$AMP" = "1" ] && echo "--amp" ) \
    --out "$OUT_PATH"

  count=$((count + 1))
done

echo "Done shard ${SHARD_IDX}: trained ${count} packs"


