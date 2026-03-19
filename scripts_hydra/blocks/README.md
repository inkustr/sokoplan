# Pack-based blocks pipeline (Hydra, parameterized)

Run all commands from the repository root.

## 0) One-time setup

```bash
source .venv/bin/activate
```

For any `.sbatch` script, prefer passing paths/params via environment variables:

```bash
VAR1=value1 VAR2=value2 sbatch scripts_hydra/blocks/<script>.sbatch
```

If you change shard count, keep `NUM_SHARDS` and SLURM array in sync:

```bash
NUM_SHARDS=80 sbatch --array=0-79 scripts_hydra/blocks/<script>.sbatch
```

## 1) Step 0: filter packs

```bash
python -m scripts.blocks.utils.filter_packs \
  --config "$DATA_CONFIG" \
  --pack_subdir "$PACK_SUBDIR" \
  --out_dir "$PACKS_DIR"
```

## 2) Step 1: generate labels per pack

```bash
LABEL_SHARDS=50
PACKS_DIR="$PACKS_DIR" OUT_DIR="$LABELS_DIR" NUM_SHARDS="$LABEL_SHARDS" \
sbatch --array=0-$((LABEL_SHARDS - 1)) scripts_hydra/blocks/run_generate_labels_packs_array.sbatch
```

### 2.1 Keep only solved levels

```bash
python -m scripts.blocks.utils.filter_solved_by_labels \
  --packs_dir "$PACKS_DIR" \
  --labels_dir "$LABELS_DIR" \
  --out_dir "$PACKS_SOLVED_DIR"
```

### 2.2 Remap labels to new level IDs

Filtering changes `level_id`, so label files must be remapped to the new metadata.

```bash
python -m scripts.labels.remap_labels_to_solved \
  --meta_dir "$PACKS_SOLVED_DIR/meta" \
  --labels_in "$OFFPOLICY_LABELS_IN" \
  --labels_out "$OFFPOLICY_LABELS_SOLVED"
```

## 3) Step 2: train one GNN per pack

```bash
TRAIN_SHARDS=35
LABELS_DIR="$LABELS_DIR" OUT_DIR="$MODELS_DIR" NUM_SHARDS="$TRAIN_SHARDS" \
sbatch --array=0-$((TRAIN_SHARDS - 1)) scripts_hydra/blocks/run_train_packs_array.sbatch
```

Output files:
- `"$MODELS_DIR/<pack>_best.pt"`

## 4) Step 3.1: split packs by self-eval

1) Run self-eval:

```bash
SELF_EVAL_SHARDS=50
LABELS_DIR="$LABELS_DIR" MODELS_DIR="$MODELS_DIR" OUT_DIR="$SELF_EVAL_DIR" NUM_SHARDS="$SELF_EVAL_SHARDS" \
sbatch --array=0-$((SELF_EVAL_SHARDS - 1)) scripts_hydra/blocks/run_self_eval_packs_array.sbatch
```

2) Create split lists by per-level MAE:

```bash
python -m scripts.blocks.split.plan_splits \
  --eval_dir "$SELF_EVAL_DIR" \
  --out_lists_dir "$SPLIT_LISTS_DIR" \
  --method auto_gap \
  --auto_gap_balance_power 0.5 \
  --min_hard_levels 50
```

## 5) Step 3.2: merge packs

1) Build candidate pairs:

```bash
python -m scripts.blocks.merge.build_merge_pairs \
  --labels_dir "$LABELS_DIR" \
  --out "$MERGE_PAIRS_CSV" \
  --k 75 \
  --sample_records 2000
```

2) Run cross-eval:

```bash
CROSS_SHARDS=50
PAIRS="$MERGE_PAIRS_CSV" LABELS_DIR="$LABELS_DIR" MODELS_DIR="$MODELS_DIR" OUT_DIR="$CROSS_EVAL_DIR" NUM_SHARDS="$CROSS_SHARDS" \
sbatch --array=0-$((CROSS_SHARDS - 1)) scripts_hydra/blocks/run_cross_eval_pairs_array.sbatch
```

3) Merge shard CSVs and plan merges:

```bash
cat "$CROSS_EVAL_DIR"/cross_eval_shard_*.csv | awk 'NR==1 || $0 !~ /^model_pack,/' > "$CROSS_EVAL_CSV"

python -m scripts.blocks.merge.plan_merges \
  --cross_eval "$CROSS_EVAL_CSV" \
  --self_eval_dir "$SELF_EVAL_DIR" \
  --ratio_quantile 0.10 \
  --offset 2.0 \
  --max_degree 4 \
  --packs_lists_dir "$PACKS_DIR/lists" \
  --out_lists_dir "$MERGED_LISTS_DIR"
```


