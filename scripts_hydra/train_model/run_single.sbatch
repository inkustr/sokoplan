#!/bin/bash
#SBATCH --job-name=soko_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu-2d
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --constraint="80gb"

mkdir -p logs
mkdir -p artifacts

# Activate Virtual Environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found!"
    exit 1
fi

echo "Starting training on node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python3)"

# Training parameters
TRAIN_DATA="data/labels.jsonl"
OUTPUT_MODEL="artifacts/gnn_best_${SLURM_JOB_ID}.pt"
CHECKPOINT="artifacts/gnn_checkpoint_${SLURM_JOB_ID}.pt"
EPOCHS=4
BATCH=512 
HIDDEN=128
LAYERS=4
LR=1e-3

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Training data not found: $TRAIN_DATA"
    echo "Attempting to merge generated labels..."
    
    if [ -d "data/generated_labels" ]; then
        cat data/generated_labels/labels_part_*.jsonl > "$TRAIN_DATA"
        echo "Merged labels into $TRAIN_DATA"
    else
        echo "No generated labels found!"
        exit 1
    fi
fi

# Run training with checkpoint support
python3 -m scripts.train_gnn \
    --train "$TRAIN_DATA" \
    --epochs $EPOCHS \
    --batch $BATCH \
    --hidden $HIDDEN \
    --layers $LAYERS \
    --lr $LR \
    --out "$OUTPUT_MODEL" \
    --checkpoint "$CHECKPOINT" \
    --resume

echo "Training finished. Model saved to $OUTPUT_MODEL"
echo "To resume if interrupted, resubmit this job - it will continue from the last checkpoint."

