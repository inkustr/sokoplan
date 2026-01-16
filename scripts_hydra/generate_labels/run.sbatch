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

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p data

# Activate Virtual Environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found! Run scripts_hydra/setup_venv.sh first."
    exit 1
fi

# Define input and output
INPUT_LIST="sokoban_core/levels/splits/train.txt"
OUTPUT_DIR="data/generated_labels"
mkdir -p $OUTPUT_DIR

# The output file for this shard
OUTPUT_FILE="${OUTPUT_DIR}/labels_part_${SLURM_ARRAY_TASK_ID}.jsonl"

# Number of parallel jobs (shards)
# Must match the --array range size (0-49 = 50 jobs)
NUM_SHARDS=50

echo "Starting task $SLURM_ARRAY_TASK_ID / $NUM_SHARDS"
echo "Running on node: $(hostname)"
echo "Python: $(which python3)"

# Run the command
python3 -m scripts.labels.generate_labels \
    --list "$INPUT_LIST" \
    --out "$OUTPUT_FILE" \
    --h hungarian \
    --use_dl \
    --time_limit 300 \
    --node_limit 3000000 \
    --jobs $SLURM_CPUS_PER_TASK \
    --shard_idx $SLURM_ARRAY_TASK_ID \
    --num_shards $NUM_SHARDS

echo "Task $SLURM_ARRAY_TASK_ID finished."