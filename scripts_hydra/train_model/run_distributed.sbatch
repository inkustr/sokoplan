#!/bin/bash
#SBATCH --job-name=soko_ddp_small
#SBATCH --output=logs/train_ddp_%j.out
#SBATCH --error=logs/train_ddp_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu-2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=4
#SBATCH --constraint="40gb|80gb"

mkdir -p logs
mkdir -p artifacts

# Activate Virtual Environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found!"
    exit 1
fi

echo "Starting distributed training (single node, 4 GPUs)"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_PER_NODE"

# Setup distributed environment variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# Training parameters
TRAIN_DATA="data/labels.jsonl"
OUTPUT_MODEL="artifacts/gnn_best.pt"
EPOCHS=10
BATCH=128
HIDDEN=256
LAYERS=6
LR=1e-3

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Training data not found: $TRAIN_DATA"
    if [ -d "data/generated_labels" ]; then
        cat data/generated_labels/labels_part_*.jsonl > "$TRAIN_DATA"
        echo "Merged labels into $TRAIN_DATA"
    else
        echo "No generated labels found!"
        exit 1
    fi
fi

# Run distributed training
srun python3 -m scripts.train_gnn_distributed \
    --train "$TRAIN_DATA" \
    --epochs $EPOCHS \
    --batch $BATCH \
    --hidden $HIDDEN \
    --layers $LAYERS \
    --lr $LR \
    --out "$OUTPUT_MODEL" \
    --checkpoint "artifacts/gnn_checkpoint_ddp.pt" \
    --resume \
    --distributed

echo "Distributed training finished!"


