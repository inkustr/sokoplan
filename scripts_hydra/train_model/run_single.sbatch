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
#SBATCH --constraint="80gb|40gb|h100|mig40"

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
TRAIN_DATA="${TRAIN_DATA:-data/group_002/train.jsonl}"
VAL_DATA="${VAL_DATA:-data/group_002/val.jsonl}"
INIT_MODEL="${INIT_MODEL:-}"
# INIT_MODEL="${INIT_MODEL:-artifacts/gnn_best_5e.pt}"
OUTPUT_MODEL="${OUTPUT_MODEL:-artifacts/group_002/group_002_${SLURM_JOB_ID}.pt}"
CHECKPOINT="${CHECKPOINT:-artifacts/group_002/group_002_checkpoint_${SLURM_JOB_ID}.pt}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-2048}"
HIDDEN="${HIDDEN:-128}"
LAYERS="${LAYERS:-4}"
LR="${LR:-1e-4}"
DROPOUT="${DROPOUT:-0.1}"
WORKERS="${WORKERS:-12}"
PREFETCH="${PREFETCH:-4}"
TORCH_THREADS="${TORCH_THREADS:-1}"
AMP="${AMP:-1}"
TEMP_FOLDER="${TEMP_FOLDER:-artifacts/group_002/temp_${SLURM_JOB_ID}}"

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

# -----------------------------------------------------------------------------
# Hydra performance: /home is slow for repeated reads. Copy dataset to local SSD (/tmp)
# for the duration of the job.
# -----------------------------------------------------------------------------
TMP_BASE="${SLURM_TMPDIR:-/tmp}"
TMP_DIR="${TMP_BASE}/soko_${SLURM_JOB_ID}"
mkdir -p "$TMP_DIR"

LOCAL_TRAIN="${TMP_DIR}/train.jsonl"
LOCAL_VAL="${TMP_DIR}/val.jsonl"

echo "Copying datasets to local storage: $TMP_DIR"
cp -f "$TRAIN_DATA" "$LOCAL_TRAIN"
if [ -f "$VAL_DATA" ]; then
  cp -f "$VAL_DATA" "$LOCAL_VAL"
  echo "Using explicit val file: $VAL_DATA"
else
  echo "Val file not found at $VAL_DATA; train_gnn will split train 90/10"
  LOCAL_VAL=""
fi

# Cap CPU threading to avoid oversubscription with DataLoader workers
export OMP_NUM_THREADS="$TORCH_THREADS"
export MKL_NUM_THREADS="$TORCH_THREADS"

# Optional init flag (safe: no backslash/comment pitfalls)
INIT_FLAG=""
if [ -n "${INIT_MODEL}" ]; then
  if [ -f "${INIT_MODEL}" ]; then
    echo "Initializing from INIT_MODEL: ${INIT_MODEL}"
    INIT_FLAG="--init_model ${INIT_MODEL}"
  else
    echo "WARNING: INIT_MODEL is set but file not found: ${INIT_MODEL} (ignored)"
  fi
fi

TEMP_FLAG=""
if [ -n "${TEMP_FOLDER}" ]; then
  TEMP_FLAG="--temp_folder ${TEMP_FOLDER}"
  mkdir -p "$TEMP_FOLDER"
fi

# Run training with checkpoint support
python3 -m scripts.train_gnn \
    --train "$LOCAL_TRAIN" \
    ${LOCAL_VAL:+--val "$LOCAL_VAL"} \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --hidden "$HIDDEN" \
    --layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    $INIT_FLAG \
    --workers "$WORKERS" \
    --prefetch_factor "$PREFETCH" \
    --pin_memory \
    --persistent_workers \
    --torch_threads "$TORCH_THREADS" \
    $( [ "$AMP" = "1" ] && echo "--amp" ) \
    --out "$OUTPUT_MODEL" \
    --checkpoint "$CHECKPOINT" \
    --resume
    $TEMP_FLAG

echo "Training finished. Model saved to $OUTPUT_MODEL"
echo "To resume if interrupted, resubmit this job - it will continue from the last checkpoint."

