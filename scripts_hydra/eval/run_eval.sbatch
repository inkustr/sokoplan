#!/bin/bash
#SBATCH --job-name=soko_eval_10k
#SBATCH --output=logs/eval_10k_%j.out
#SBATCH --error=logs/eval_10k_%j.err
#SBATCH --partition=cpu-2d
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --nodelist=head065,head066,head067,head068,head069,head070,head071

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-}"
if [ -z "$ROOT_DIR" ] || [ ! -d "$ROOT_DIR" ]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT_DIR"

mkdir -p logs results/eval/boxoban

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "ERROR: .venv not found in project root ($ROOT_DIR)"
  exit 1
fi

LIST="${LIST:-sokoban_core/levels/splits/splits_boxoban_2/test.txt}"
CKPT="${CKPT:-artifacts/boxoban_2/boxoban_2_finetuned_epoch_193.pt}"
MODE="${MODE:-speed}"
TIME_LIMIT="${TIME_LIMIT:-600.0}"
NODE_LIMIT="${NODE_LIMIT:-2000000}"
JOBS="${JOBS:-1}"
NO_DL="${NO_DL:-0}"

SHARD_IDX="${SHARD_IDX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
EXP_TAG="${EXP_TAG:-final_boxoban2_10k}"

if [ ! -f "$LIST" ]; then
  echo "ERROR: list not found: $LIST"
  exit 1
fi
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

if ! [[ "$SHARD_IDX" =~ ^[0-9]+$ ]] || ! [[ "$SHARD_COUNT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SHARD_IDX and SHARD_COUNT must be non-negative integers"
  exit 1
fi
if [ "$SHARD_COUNT" -le 0 ] || [ "$SHARD_IDX" -ge "$SHARD_COUNT" ]; then
  echo "ERROR: invalid shard config SHARD_IDX=$SHARD_IDX SHARD_COUNT=$SHARD_COUNT"
  exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_BASE="results/eval/${EXP_TAG}"
mkdir -p "$RUN_BASE"
RUN_ID="${EXP_TAG}_sh${SHARD_IDX}of${SHARD_COUNT}_j${JOBS}_c${SLURM_CPUS_PER_TASK}_${SLURM_JOB_ID}"
OUT_DIR="${RUN_BASE}/${RUN_ID}"
mkdir -p "$OUT_DIR"

SHARD_LIST="${OUT_DIR}/levels_shard.txt"
python3 - "$LIST" "$SHARD_LIST" "$SHARD_IDX" "$SHARD_COUNT" <<'PY'
import sys

list_path, out_path, idx_s, cnt_s = sys.argv[1:5]
idx = int(idx_s)
cnt = int(cnt_s)
levels = []
with open(list_path, "r", encoding="utf-8") as f:
    for ln in f:
        s = ln.strip()
        if s and not s.startswith("#"):
            levels.append(s)
shard = [lid for i, lid in enumerate(levels) if (i % cnt) == idx]
with open(out_path, "w", encoding="utf-8") as f:
    for lid in shard:
        f.write(lid + "\n")
print(f"[OK] shard size: {len(shard)} / total: {len(levels)} (idx={idx}, cnt={cnt})")
PY

OUT_CSV="${OUT_DIR}/eval.csv"
WALL_CSV="${OUT_DIR}/wall.csv"

{
  echo "timestamp_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-na}"
  echo "slurm_job_nodelist=${SLURM_JOB_NODELIST:-na}"
  echo "slurm_partition=${SLURM_JOB_PARTITION:-na}"
  echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-na}"
  echo "list=$LIST"
  echo "shard_list=$SHARD_LIST"
  echo "ckpt=$CKPT"
  echo "mode=$MODE"
  echo "jobs=$JOBS"
  echo "time_limit=$TIME_LIMIT"
  echo "node_limit=$NODE_LIMIT"
  echo "no_dl=$NO_DL"
  echo "shard_idx=$SHARD_IDX"
  echo "shard_count=$SHARD_COUNT"
} | tee "${OUT_DIR}/run_meta.env"

lscpu > "${OUT_DIR}/hardware.txt" || true

cmd=(python3 -m scripts.solve.run_batch_gnn
  --list "$SHARD_LIST"
  --ckpt "$CKPT"
  --mode "$MODE"
  --out "$OUT_CSV"
  --time_limit "$TIME_LIMIT"
  --node_limit "$NODE_LIMIT"
  --jobs "$JOBS"
)
if [ "$NO_DL" = "1" ]; then
  cmd+=(--no_dl)
fi

echo "${cmd[*]}" | tee -a "${OUT_DIR}/run.log"
/usr/bin/time -f "wall_s=%e,user_s=%U,sys_s=%S,max_rss_kb=%M,cpu_pct=%P" -o "$WALL_CSV" \
  "${cmd[@]}" | tee -a "${OUT_DIR}/run.log"

echo "[DONE] shard output: ${OUT_CSV}"
