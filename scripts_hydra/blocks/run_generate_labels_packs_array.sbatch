#!/bin/bash
#SBATCH --job-name=soko_lbl_packs
#SBATCH --output=logs/labels_packs_%A_%a.out
#SBATCH --error=logs/labels_packs_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu-2d
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
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

PACKS_DIR="${PACKS_DIR:-sokoban_core/levels/packs_filtered}"
LISTS_DIR="${PACKS_DIR}/lists"
OUT_DIR="${OUT_DIR:-data/packs_labels_festival}"
TIME_LIMIT="${TIME_LIMIT:-3600}"
JOBS="${JOBS:-$SLURM_CPUS_PER_TASK}"
SAMPLE_EVERY="${SAMPLE_EVERY:-3}"
NUM_SHARDS="${NUM_SHARDS:-50}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID}"

if [ ! -f "./festival/festival" ]; then
  echo "ERROR: Festival binary not found at ./festival/festival"
  exit 1
fi

LIST_FILES=($(ls -1 "${LISTS_DIR}"/*.list | sort))

echo "Total pack lists: ${#LIST_FILES[@]}"
echo "Shard: ${SHARD_IDX}/${NUM_SHARDS}"
mkdir -p "$OUT_DIR"

count=0
export LISTS_DIR
export NUM_SHARDS
export SHARD_IDX

ASSIGNED_LISTS=($(
python3 - <<'PY'
import os, sys, subprocess

lists_dir = os.environ["LISTS_DIR"]
num_shards = int(os.environ.get("NUM_SHARDS", "50"))
shard_idx = int(os.environ.get("SHARD_IDX", "0"))
if shard_idx < 0 or shard_idx >= num_shards:
    sys.exit(0)

files = sorted([os.path.join(lists_dir, f) for f in os.listdir(lists_dir) if f.endswith(".list")])
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
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    n += 1
        return n

weights = [(count_lines(p), p) for p in files]
weights.sort(reverse=True)  # largest first

bins = [(0, []) for _ in range(num_shards)]  # (total_weight, [paths])
for w, p in weights:
    i = min(range(num_shards), key=lambda k: bins[k][0])
    bins[i][1].append(p)
    bins[i] = (bins[i][0] + w, bins[i][1])

assigned = bins[shard_idx][1]
for p in assigned:
    print(p)
PY
))

echo "Assigned pack lists to this shard: ${#ASSIGNED_LISTS[@]}"

for LIST_PATH in "${ASSIGNED_LISTS[@]}"; do
  BASE="$(basename "$LIST_PATH" .list)"
  OUT_PATH="${OUT_DIR}/${BASE}.jsonl"

  idx=$((count + 1))
  echo "================================================"
  echo "Pack   : ${idx}/${#ASSIGNED_LISTS[@]}"
  echo "List   : $LIST_PATH"
  echo "Out    : $OUT_PATH"
  echo "================================================"

  python3 -m scripts.labels.generate_labels_festival \
    --list "$LIST_PATH" \
    --out "$OUT_PATH" \
    --time_limit "$TIME_LIMIT" \
    --jobs "$JOBS" \
    --sample_every "$SAMPLE_EVERY"

  count=$((count + 1))
done

echo "Done shard ${SHARD_IDX}: processed ${count} packs"


