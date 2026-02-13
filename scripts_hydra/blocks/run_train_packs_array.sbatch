#!/bin/bash
#SBATCH --job-name=soko_train_packs
#SBATCH --output=logs/train_packs_%A_%a.out
#SBATCH --error=logs/train_packs_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --partition=gpu-5h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --constraint="80gb|40gb|h100|6000"
#SBATCH --array=0-34

# Train one small GNN per PACK.
#
# Expects labels produced by run_generate_labels_packs_array.sbatch:
#   data/packs_labels_festival/<pack>.jsonl
#
# Required env:
#   LABELS_DIR: directory containing *.jsonl (default: data/packs_labels_festival)
# Optional env:
#   OUT_DIR: output dir for models (default: artifacts/packs_models)
#   EPOCHS, BATCH, HIDDEN, LAYERS, LR, DROPOUT

set -euo pipefail
mkdir -p logs
mkdir -p artifacts

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo ".venv not found in project root"
  exit 1
fi

LABELS_DIR="${LABELS_DIR:-data/group_labels_festival}"
OUT_DIR="${OUT_DIR:-artifacts/packs_models}"
NUM_SHARDS="${NUM_SHARDS:-35}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID}"
SHARD_STRATEGY="${SHARD_STRATEGY:-weighted}"  # weighted|mod

EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-512}"
HIDDEN="${HIDDEN:-128}"
LAYERS="${LAYERS:-4}"
LR="${LR:-1e-3}"
DROPOUT="${DROPOUT:-0.05}"
AMP="${AMP:-1}"
WORKERS="${WORKERS:-4}"
PREFETCH="${PREFETCH:-4}"
TORCH_THREADS="${TORCH_THREADS:-1}"
USE_TMP="${USE_TMP:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OVERWRITE="${OVERWRITE:-0}"

FILES=($(ls -1 "${LABELS_DIR}"/*.jsonl | sort))
echo "Total label files: ${#FILES[@]}"
echo "Shard: ${SHARD_IDX}/${NUM_SHARDS} (strategy=${SHARD_STRATEGY})"
mkdir -p "$OUT_DIR"

count=0
ASSIGNED_FILES=()

if [ "${SHARD_STRATEGY}" = "weighted" ]; then
  # -----------------------------------------------------------------------------
  # Weighted sharding to balance uneven label file sizes:
  # We assign jsonl files to shards greedily by number of lines (wc -l).
  # -----------------------------------------------------------------------------
  export LABELS_DIR NUM_SHARDS SHARD_IDX
  ASSIGNED_FILES=($(
python3 - <<'PY'
import os, sys, subprocess

labels_dir = os.environ["LABELS_DIR"]
num_shards = int(os.environ.get("NUM_SHARDS", "35"))
shard_idx = int(os.environ.get("SHARD_IDX", "0"))

if shard_idx < 0 or shard_idx >= num_shards:
    sys.exit(0)

files = sorted(
    os.path.join(labels_dir, f)
    for f in os.listdir(labels_dir)
    if f.endswith(".jsonl")
)
if not files:
    sys.exit(0)

def count_lines(p: str) -> int:
    # Use wc -l (fast) and fallback to python
    try:
        out = subprocess.check_output(["wc", "-l", p], text=True).strip().split()
        return int(out[0])
    except Exception:
        n = 0
        with open(p, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    n += 1
        return n

weights = [(count_lines(p), p) for p in files]
weights.sort(reverse=True)  # largest first

bins = [(0, []) for _ in range(num_shards)]  # (total_weight, [paths])
for w, p in weights:
    i = min(range(num_shards), key=lambda k: bins[k][0])
    bins[i][1].append(p)
    bins[i] = (bins[i][0] + w, bins[i][1])

for p in bins[shard_idx][1]:
    print(p)
PY
  ))
elif [ "${SHARD_STRATEGY}" = "mod" ]; then
  for i in "${!FILES[@]}"; do
    if [ $(( i % NUM_SHARDS )) -ne "$SHARD_IDX" ]; then
      continue
    fi
    ASSIGNED_FILES+=("${FILES[$i]}")
  done
else
  echo "ERROR: unknown SHARD_STRATEGY=${SHARD_STRATEGY} (expected weighted|mod)"
  exit 1
fi

echo "Assigned label files to this shard: ${#ASSIGNED_FILES[@]}"

for TRAIN_PATH in "${ASSIGNED_FILES[@]}"; do
  if [ ! -f "$TRAIN_PATH" ]; then
    echo "WARN: missing train file (skipping): $TRAIN_PATH"
    continue
  fi

  BASE="$(basename "$TRAIN_PATH" .jsonl)"

  OUT_PATH="${OUT_DIR}/${BASE}_best.pt"

  echo "================================================"
  echo "Pack    : $BASE"
  echo "Train   : $TRAIN_PATH"
  echo "Out     : $OUT_PATH"
  echo "Skip existing: $SKIP_EXISTING  Overwrite: $OVERWRITE"
  echo "================================================"

  if [ -s "$OUT_PATH" ]; then
    if [ "$OVERWRITE" = "1" ]; then
      echo "[resume] removing existing artifact: $OUT_PATH"
      rm -f "$OUT_PATH"
    elif [ "$SKIP_EXISTING" = "1" ]; then
      echo "[resume] artifact exists, skipping: $OUT_PATH"
      continue
    else
      echo "[warn] artifact exists and SKIP_EXISTING=0; will overwrite: $OUT_PATH"
    fi
  fi

  python3 -m scripts.train.train_gnn \
    --train "$TRAIN_PATH" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --hidden "$HIDDEN" \
    --layers "$LAYERS" \
    --lr "$LR" \
    --dropout "$DROPOUT" \
    --workers "$WORKERS" \
    --prefetch_factor "$PREFETCH" \
    --pin_memory \
    --persistent_workers \
    --torch_threads "$TORCH_THREADS" \
    $( [ "$AMP" = "1" ] && echo "--amp" ) \
    --out "$OUT_PATH"

  count=$((count + 1))
done

echo "Done shard ${SHARD_IDX}: trained ${count} packs"


