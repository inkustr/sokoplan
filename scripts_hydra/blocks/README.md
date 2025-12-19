# Pack-based blocks pipeline

## Step 0
Filter Packs

```bash
source .venv/bin/activate
python -m scripts.blocks.utils.filter_packs --config configs/data.yaml --windows_subdir windows --out_dir sokoban_core/levels/packs_filtered
```

## Step 1 — Generate labels per pack

```bash
sbatch scripts_hydra/blocks/run_generate_labels_packs_array.sbatch
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

2) Create new list files based on per-level MAE:

```bash
source .venv/bin/activate
python -m scripts.blocks.split.plan_splits --eval_dir results/packs_self_eval --out_lists_dir sokoban_core/levels/pack_blocks_2/lists --method top_pct --pct 0.2
```

## Step 3.2 — Merge packs

1) Build candidate pairs locally:

```bash
source .venv/bin/activate
python -m scripts.blocks.build_merge_pairs --labels_dir data/packs_labels_festival --out results/merge_pairs.csv --k 75 --sample_records 2000
```

2) Run cross-eval:

```bash
sbatch scripts_hydra/blocks/run_cross_eval_pairs_array.sbatch
```

3) Merge shard CSVs into one, then plan merges:

```bash
cat results/cross_eval/cross_eval_shard_*.csv | awk 'NR==1 || $0 !~ /^model_pack,/' > results/cross_eval.csv
python -m scripts.blocks.plan_merges \
  --cross_eval results/cross_eval.csv \
  --self_eval_dir results/packs_self_eval \
  --strategy auto --ratio_quantile 0.10 --offset 2.0 --max_degree 4 \
  --packs_lists_dir sokoban_core/levels/packs_filtered/lists \
  --out_lists_dir sokoban_core/levels/pack_groups/lists
```


