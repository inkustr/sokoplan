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

set -e

echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Started: $(date)"
echo "================================================"

NUM_SHARDS=50
SHARD_IDX=$SLURM_ARRAY_TASK_ID

source .venv/bin/activate

if [ ! -f "./festival/festival" ]; then
    echo "ERROR: Festival binary not found at ./festival/festival"
    exit 1
fi

mkdir -p data/generated_labels

OUTPUT_FILE="data/generated_labels/labels_festival_part_${SHARD_IDX}.jsonl"

python3 -m scripts.labels.generate_labels_festival \
    --list "sokoban_core/levels/splits/train.txt" \
    --out "$OUTPUT_FILE" \
    --time_limit 28800 \
    --jobs 8 \
    --shard_idx $SHARD_IDX \
    --num_shards $NUM_SHARDS \
    --sample_every 2

