#!/bin/bash
#SBATCH --job-name=soko_selfeval
#SBATCH --output=logs/selfeval_%A_%a.out
#SBATCH --error=logs/selfeval_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-2h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --constraint="80gb|40gb|h100|6000"
#SBATCH --array=0-20

# Evaluate each pack model on its OWN pack labels and write metrics JSON.

set -euo pipefail
mkdir -p logs
mkdir -p results/packs_self_eval

source .venv/bin/activate

LABELS_DIR="${LABELS_DIR:-data/packs_labels_festival}"
MODELS_DIR="${MODELS_DIR:-artifacts/packs_models}"
OUT_DIR="${OUT_DIR:-results/packs_self_eval}"
NUM_SHARDS="${NUM_SHARDS:-21}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID}"
LIMIT_RECORDS="${LIMIT_RECORDS:-0}"

FILES=($(ls -1 "${LABELS_DIR}"/*.jsonl | sort))
count=0
for i in "${!FILES[@]}"; do
  if [ $(( i % NUM_SHARDS )) -ne "$SHARD_IDX" ]; then
    continue
  fi
  labels="${FILES[$i]}"
  pack="$(basename "$labels" .jsonl)"
  ckpt="${MODELS_DIR}/${pack}_best.pt"
  if [ ! -f "$ckpt" ]; then
    continue
  fi
  out="${OUT_DIR}/${pack}.json"
  python3 -m scripts.blocks.split.eval_pack_model --ckpt "$ckpt" --labels "$labels" --out "$out" ${LIMIT_RECORDS:+--limit_records "$LIMIT_RECORDS"}
  count=$((count + 1))
done
echo "shard ${SHARD_IDX}: evaluated ${count} packs"


