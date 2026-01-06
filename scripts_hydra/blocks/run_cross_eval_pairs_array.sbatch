#!/bin/bash
#SBATCH --job-name=soko_crosseval
#SBATCH --output=logs/crosseval_%A_%a.out
#SBATCH --error=logs/crosseval_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --partition=cpu-5h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-34

# Requires:
#   results/merge_pairs.csv (build via scripts.blocks.build_merge_pairs)
#
# Outputs:
#   results/cross_eval/cross_eval_shard_<k>.csv

set -euo pipefail
mkdir -p logs
mkdir -p results/cross_eval

source .venv/bin/activate

PAIRS="${PAIRS:-results/merge_pairs.csv}"
LABELS_DIR="${LABELS_DIR:-data/group_labels_festival}"
MODELS_DIR="${MODELS_DIR:-artifacts/packs_models}"
OUT_DIR="${OUT_DIR:-results/cross_eval}"
NUM_SHARDS="${NUM_SHARDS:-35}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID}"
LIMIT_RECORDS="${LIMIT_RECORDS:-7000}"

out="${OUT_DIR}/cross_eval_shard_${SHARD_IDX}.csv"

python3 -m scripts.blocks.merge.run_cross_eval \
  --pairs "$PAIRS" \
  --labels_dir "$LABELS_DIR" \
  --models_dir "$MODELS_DIR" \
  --out "$out" \
  --limit_records "$LIMIT_RECORDS" \
  --num_shards "$NUM_SHARDS" \
  --shard_idx "$SHARD_IDX"


