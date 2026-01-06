#!/bin/bash
#SBATCH --job-name=soko_boxoban_ab
#SBATCH --output=logs/boxoban_ab_%A_%a.out
#SBATCH --error=logs/boxoban_ab_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu-2d
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --array=0-49

set -euo pipefail
mkdir -p logs

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

LIST="${LIST:-sokoban_core/levels/splits/train.txt}"
OUT_DIR="${OUT_DIR:-data/boxoban_labels_ab}"

NUM_SHARDS="${NUM_SHARDS:-50}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID}"

POLICY="${POLICY:-hungarian}"
SAMPLE_PER_LEVEL="${SAMPLE_PER_LEVEL:-15}"
FRONTIER_PER_LEVEL="${FRONTIER_PER_LEVEL:-10}"
MIN_G="${MIN_G:-3}"
SEARCH_NODE_LIMIT="${SEARCH_NODE_LIMIT:-20000}"
SEARCH_TIME_LIMIT="${SEARCH_TIME_LIMIT:-10.0}"

FESTIVAL_TIMEOUT="${FESTIVAL_TIMEOUT:-150}"
KEEP_ONLY_START="${KEEP_ONLY_START:-0}"
SAMPLE_EVERY="${SAMPLE_EVERY:-2}"
START_KEEP_ONLY_START="${START_KEEP_ONLY_START:-0}"
START_SAMPLE_EVERY="${START_SAMPLE_EVERY:-1}"

JOBS="${JOBS:-$SLURM_CPUS_PER_TASK}"

mkdir -p "$OUT_DIR"
OUT_PATH="${OUT_DIR}/boxoban_ab_shard_${SHARD_IDX}.jsonl"

echo "================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Array : ${SLURM_ARRAY_TASK_ID:-}"
echo "List  : $LIST"
echo "Out   : $OUT_PATH"
echo "Shards: ${SHARD_IDX}/${NUM_SHARDS}"
echo "Policy: $POLICY"
echo "================================================"

KEEP_ONLY_START_FLAG=""
if [ "${KEEP_ONLY_START}" = "1" ]; then
  KEEP_ONLY_START_FLAG="--keep_only_start"
fi

START_KEEP_ONLY_START_FLAG=""
if [ "${START_KEEP_ONLY_START}" = "1" ]; then
  START_KEEP_ONLY_START_FLAG="--start_keep_only_start"
fi

python3 -m scripts.labels.collect_offpolicy_labels \
  --list "$LIST" \
  --out "$OUT_PATH" \
  --policy "$POLICY" \
  --sample_per_level "$SAMPLE_PER_LEVEL" \
  --frontier_per_level "$FRONTIER_PER_LEVEL" \
  --min_g "$MIN_G" \
  --search_node_limit "$SEARCH_NODE_LIMIT" \
  --search_time_limit "$SEARCH_TIME_LIMIT" \
  --festival_timeout "$FESTIVAL_TIMEOUT" \
  --include_start_solution \
  $KEEP_ONLY_START_FLAG \
  --sample_every "$SAMPLE_EVERY" \
  $START_KEEP_ONLY_START_FLAG \
  --start_sample_every "$START_SAMPLE_EVERY" \
  --num_shards "$NUM_SHARDS" \
  --shard_idx "$SHARD_IDX" \
  --jobs "$JOBS"

echo "Done shard ${SHARD_IDX}/${NUM_SHARDS}"


