#!/bin/bash
#SBATCH --job-name=soko_gen
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu-2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=0-49

set -euo pipefail

# Create logs/data directories if they don't exist
mkdir -p logs
mkdir -p data

# Activate Virtual Environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found! Run scripts_hydra/setup_venv.sh first."
    exit 1
fi

# Inputs/outputs
LIST="${LIST:-sokoban_core/levels/splits/train.txt}"
OUT_DIR="${OUT_DIR:-data/generated_labels}"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-labels_part}"

# Sharding
NUM_SHARDS="${NUM_SHARDS:-50}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID}"
if [ "$SHARD_IDX" -ge "$NUM_SHARDS" ]; then
    echo "ERROR: SHARD_IDX=${SHARD_IDX} is >= NUM_SHARDS=${NUM_SHARDS} (check --array size)"
    exit 1
fi

# Solver params
HEURISTIC="${HEURISTIC:-hungarian}"
USE_DL="${USE_DL:-1}"
TIME_LIMIT="${TIME_LIMIT:-300}"
NODE_LIMIT="${NODE_LIMIT:-3000000}"
JOBS="${JOBS:-$SLURM_CPUS_PER_TASK}"

if [ ! -f "$LIST" ]; then
    echo "Input list not found: $LIST"
    exit 1
fi

mkdir -p "$OUT_DIR"
OUTPUT_FILE="${OUT_DIR}/${OUTPUT_BASENAME}_${SHARD_IDX}.jsonl"

echo "Starting task $SHARD_IDX / $NUM_SHARDS"
echo "Running on node: $(hostname)"
echo "Python: $(which python3)"
echo "List: $LIST"
echo "Out:  $OUTPUT_FILE"

CMD=(
    python3 -m scripts.labels.generate_labels
    --list "$LIST"
    --out "$OUTPUT_FILE"
    --h "$HEURISTIC"
    --time_limit "$TIME_LIMIT"
    --node_limit "$NODE_LIMIT"
    --jobs "$JOBS"
    --shard_idx "$SHARD_IDX"
    --num_shards "$NUM_SHARDS"
)

if [ "$USE_DL" = "1" ]; then
    CMD+=(--use_dl)
fi

"${CMD[@]}"

echo "Task $SHARD_IDX finished."