# Pack-based blocks pipeline

## Step 0
Filter Packs

```bash
source .venv/bin/activate
python -m scripts.blocks.utils.filter_packs --config configs/data.yaml --pack_subdir packs --out_dir sokoban_core/levels/packs_filtered
```

## Step 1 — Generate labels per pack

```bash
sbatch scripts_hydra/blocks/run_generate_labels_packs_array.sbatch
```

### 1.1 Filter packs by solved labels
Keep only levels that were successfully solved.
```bash
python -m scripts.blocks.utils.filter_solved_by_labels \
    --packs_dir sokoban_core/levels/packs_filtered \
    --labels_dir data/packs_labels_festival \
    --out_dir sokoban_core/levels/packs_solved
```

### 1.2 Remap labels to new IDs
Since filtering changed level indices, we must remap the `.jsonl` files to use the new `level_id`s from `packs_solved`.
```bash
python -m scripts.labels.remap_labels_to_solved \
    --meta_dir sokoban_core/levels/packs_solved/meta \
    --labels_in data/packs_offpolicy_labels_festival \
    --labels_out data/packs_offpolicy_labels_festival_solved
```

## Step 2 — Train one small GNN per pack

Submit:

```bash
sbatch scripts_hydra/blocks/run_train_packs_array.sbatch
```

Outputs:
- `artifacts/packs_models/<pack>_best.pt`

## Step 3.1 — Split packs

1) Run self-eval:

```bash
sbatch scripts_hydra/blocks/run_self_eval_packs_array.sbatch
```

2) Create new list files based on per-level MAE (using auto_gap for balanced splits):

```bash
source .venv/bin/activate
python -m scripts.blocks.split.plan_splits \
    --eval_dir results/packs_self_eval \
    --out_lists_dir sokoban_core/levels/pack_blocks/lists \
    --method auto_gap \
    --auto_gap_balance_power 0.5 \
    --min_hard_levels 50
```

## Step 3.2 — Merge packs

1) Build candidate pairs locally:

```bash
source .venv/bin/activate
python -m scripts.blocks.merge.build_merge_pairs --labels_dir data/packs_labels_festival --out results/merge_pairs.csv --k 75 --sample_records 2000
```

2) Run cross-eval:

```bash
sbatch scripts_hydra/blocks/run_cross_eval_pairs_array.sbatch
```

3) Merge shard CSVs into one, then plan merges:

```bash
cat results/cross_eval/cross_eval_shard_*.csv | awk 'NR==1 || $0 !~ /^model_pack,/' > results/cross_eval.csv
python -m scripts.blocks.merge.plan_merges \
  --cross_eval results/cross_eval.csv \
  --self_eval_dir results/packs_self_eval \
  --ratio_quantile 0.10 --offset 2.0 --max_degree 4 \
  --packs_lists_dir sokoban_core/levels/packs_filtered/lists \
  --out_lists_dir sokoban_core/levels/pack_groups/lists
```


