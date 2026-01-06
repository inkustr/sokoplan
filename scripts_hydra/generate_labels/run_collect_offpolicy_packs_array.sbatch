#!/bin/bash
#SBATCH --job-name=soko_offpol
#SBATCH --output=logs/offpolicy_%A_%a.out
#SBATCH --error=logs/offpolicy_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu-2d
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --array=0-44

set -euo pipefail
mkdir -p logs

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo ".venv not found in project root"
  exit 1
fi

# Must have a `lists/` subdir.
PACKS_DIR="${PACKS_DIR:-sokoban_core/levels/packs_solved}"
LISTS_DIR="${LISTS_DIR:-${PACKS_DIR}/lists}"

OUT_DIR="${OUT_DIR:-data/packs_offpolicy_labels_festival}"

NUM_SHARDS="${NUM_SHARDS:-90}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
SHARD_IDX="$((SLURM_ARRAY_TASK_ID + SHARD_OFFSET))"
if [ "$SHARD_IDX" -ge "$NUM_SHARDS" ]; then
  echo "ERROR: SHARD_IDX=${SHARD_IDX} is >= NUM_SHARDS=${NUM_SHARDS} (check --array and SHARD_OFFSET)"
  exit 1
fi

# Export for the embedded Python weighted-sharding snippet (it reads os.environ).
export PACKS_DIR
export LISTS_DIR
export NUM_SHARDS
export SHARD_IDX

POLICY="${POLICY:-hungarian}"
GNN_CKPT="${GNN_CKPT:-}"
GNN_MODE="${GNN_MODE:-speed}"

SAMPLE_PER_LEVEL="${SAMPLE_PER_LEVEL:-7}"
FRONTIER_PER_LEVEL="${FRONTIER_PER_LEVEL:-3}"
MIN_G="${MIN_G:-3}"
SEARCH_NODE_LIMIT="${SEARCH_NODE_LIMIT:-20000}"
SEARCH_TIME_LIMIT="${SEARCH_TIME_LIMIT:-20.0}"

FESTIVAL_TIMEOUT="${FESTIVAL_TIMEOUT:-1200}"
SAMPLE_EVERY="${SAMPLE_EVERY:-4}"

JOBS="${JOBS:-$SLURM_CPUS_PER_TASK}"

if [ ! -f "./festival/festival" ]; then
  echo "ERROR: Festival binary not found at ./festival/festival"
  exit 1
fi

mkdir -p "$OUT_DIR"

LIST_FILES=($(ls -1 "${LISTS_DIR}"/*.list | sort))
echo "Total pack lists: ${#LIST_FILES[@]}"
echo "Shard: ${SHARD_IDX}/${NUM_SHARDS} (offset=${SHARD_OFFSET}, local_task_id=${SLURM_ARRAY_TASK_ID})"
echo "PACKS_DIR: $PACKS_DIR"
echo "LISTS_DIR: $LISTS_DIR"
echo "OUT_DIR  : $OUT_DIR"

# -----------------------------------------------------------------------------
# Weighted sharding to balance uneven pack sizes:
# We assign pack lists to shards greedily by number of levels in each list.
# -----------------------------------------------------------------------------
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

count=0
for LIST_PATH in "${ASSIGNED_LISTS[@]}"; do
  BASE="$(basename "$LIST_PATH" .list)"
  OUT_PATH="${OUT_DIR}/${BASE}.jsonl"

  echo "================================================"
  echo "List : $LIST_PATH"
  echo "Out  : $OUT_PATH"
  echo "================================================"

  python3 -m scripts.labels.collect_offpolicy_labels \
    --list "$LIST_PATH" \
    --out "$OUT_PATH" \
    --policy "hungarian" \
    --sample_per_level "$SAMPLE_PER_LEVEL" \
    --frontier_per_level "$FRONTIER_PER_LEVEL" \
    --min_g "$MIN_G" \
    --search_node_limit "$SEARCH_NODE_LIMIT" \
    --search_time_limit "$SEARCH_TIME_LIMIT" \
    --festival_timeout "$FESTIVAL_TIMEOUT" \
    --sample_every "$SAMPLE_EVERY" \
    --jobs "$JOBS"

  count=$((count + 1))
done

echo "Done shard ${SHARD_IDX}: processed ${count} pack lists"


