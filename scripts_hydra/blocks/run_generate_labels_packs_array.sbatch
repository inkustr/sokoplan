#!/bin/bash
#SBATCH --job-name=soko_lbl_packs
#SBATCH --output=logs/labels_packs_%A_%a.out
#SBATCH --error=logs/labels_packs_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu-2d
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-49

set -euo pipefail
mkdir -p logs

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo ".venv not found in project root"
  exit 1
fi

PACKS_DIR="${PACKS_DIR:-sokoban_core/levels/packs_filtered}"
LISTS_DIR="${PACKS_DIR}/lists"
OUT_DIR="${OUT_DIR:-data/packs_labels_festival}"
TIME_LIMIT="${TIME_LIMIT:-3600}"
JOBS="${JOBS:-8}"
SAMPLE_EVERY="${SAMPLE_EVERY:-3}"
NUM_SHARDS="${NUM_SHARDS:-50}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID}"

if [ ! -f "./festival/festival" ]; then
  echo "ERROR: Festival binary not found at ./festival/festival"
  exit 1
fi

LIST_FILES=($(ls -1 "${LISTS_DIR}"/*.list | sort))

echo "Total pack lists: ${#LIST_FILES[@]}"
echo "Shard: ${SHARD_IDX}/${NUM_SHARDS}"
mkdir -p "$OUT_DIR"

count=0
for i in "${!LIST_FILES[@]}"; do
  if [ $(( i % NUM_SHARDS )) -ne "$SHARD_IDX" ]; then
    continue
  fi

  LIST_PATH="${LIST_FILES[$i]}"
  BASE="$(basename "$LIST_PATH" .list)"
  OUT_PATH="${OUT_DIR}/${BASE}.jsonl"

  echo "================================================"
  echo "Pack idx: $i"
  echo "List   : $LIST_PATH"
  echo "Out    : $OUT_PATH"
  echo "================================================"

  python3 -m scripts.generate_labels_festival \
    --list "$LIST_PATH" \
    --out "$OUT_PATH" \
    --time_limit "$TIME_LIMIT" \
    --jobs "$JOBS" \
    --sample_every "$SAMPLE_EVERY"

  count=$((count + 1))
done

echo "Done shard ${SHARD_IDX}: processed ${count} packs"


