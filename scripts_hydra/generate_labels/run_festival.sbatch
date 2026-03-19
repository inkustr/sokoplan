#!/bin/bash
#SBATCH --job-name=soko_festival
#SBATCH --output=logs/festival_%A_%a.out
#SBATCH --error=logs/festival_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu-2d
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-49

set -euo pipefail

mkdir -p logs

echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Started: $(date)"
echo "================================================"

NUM_SHARDS="${NUM_SHARDS:-50}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID}"
if [ "$SHARD_IDX" -ge "$NUM_SHARDS" ]; then
    echo "ERROR: SHARD_IDX=${SHARD_IDX} is >= NUM_SHARDS=${NUM_SHARDS} (check --array size)"
    exit 1
fi

LIST="${LIST:-sokoban_core/levels/splits/train.txt}"
OUT_DIR="${OUT_DIR:-data/generated_labels}"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-labels_festival_part}"
TIME_LIMIT="${TIME_LIMIT:-28800}"
SAMPLE_EVERY="${SAMPLE_EVERY:-2}"
JOBS="${JOBS:-$SLURM_CPUS_PER_TASK}"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo ".venv not found in project root"
    exit 1
fi

if [ ! -f "./festival/festival" ]; then
    echo "ERROR: Festival binary not found at ./festival/festival"
    exit 1
fi

if [ ! -f "$LIST" ]; then
    echo "Input list not found: $LIST"
    exit 1
fi

mkdir -p "$OUT_DIR"
OUTPUT_FILE="${OUT_DIR}/${OUTPUT_BASENAME}_${SHARD_IDX}.jsonl"

python3 -m scripts.labels.generate_labels_festival \
    --list "$LIST" \
    --out "$OUTPUT_FILE" \
    --time_limit "$TIME_LIMIT" \
    --jobs "$JOBS" \
    --shard_idx "$SHARD_IDX" \
    --num_shards "$NUM_SHARDS" \
    --sample_every "$SAMPLE_EVERY"

