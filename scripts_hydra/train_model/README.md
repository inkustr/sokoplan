# Train model on Hydra (single GPU)

The current training entrypoint in this folder is:
- `scripts_hydra/train_model/run_single.sbatch`

Run from repository root:

```bash
source .venv/bin/activate
```

## Submit pattern

```bash
VAR1=value1 VAR2=value2 sbatch scripts_hydra/train_model/run_single.sbatch
```

## Key environment variables

Data / initialization:
- `TRAIN_DATA` (required path to training `.jsonl`)
- `VAL_DATA` (optional; if missing, train script will split train set)
- `INIT_MODEL` (optional warm start checkpoint)

Outputs:
- `OUTPUT_MODEL`
- `CHECKPOINT`
- `TEMP_FOLDER`

Training hyperparameters:
- `EPOCHS`, `BATCH`, `HIDDEN`, `LAYERS`, `LR`, `DROPOUT`
- `WORKERS`, `PREFETCH`, `TORCH_THREADS`, `AMP`

## Recommended submission (stable paths, resumable)

Use fixed output/checkpoint names instead of job-id-based defaults to make resume predictable:

```bash
RUN_NAME="boxoban_2_baseline"
OUT_DIR="artifacts/boxoban_2"
mkdir -p "$OUT_DIR"

TRAIN_DATA="data/boxoban_2/train.jsonl" \
VAL_DATA="data/boxoban_2/val.jsonl" \
OUTPUT_MODEL="$OUT_DIR/${RUN_NAME}_best.pt" \
CHECKPOINT="$OUT_DIR/${RUN_NAME}_checkpoint.pt" \
TEMP_FOLDER="$OUT_DIR/${RUN_NAME}_tmp" \
EPOCHS=30 \
BATCH=2048 \
AMP=1 \
sbatch scripts_hydra/train_model/run_single.sbatch
```

## Resume after time limit

To resume, submit again with the same `CHECKPOINT` path:

```bash
CHECKPOINT="artifacts/boxoban_2/boxoban_2_baseline_checkpoint.pt" \
OUTPUT_MODEL="artifacts/boxoban_2/boxoban_2_baseline_best.pt" \
sbatch scripts_hydra/train_model/run_single.sbatch
```

The script runs `train_gnn` with `--resume`, so it continues from `CHECKPOINT`
if the file exists.

