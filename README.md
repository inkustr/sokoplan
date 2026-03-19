# Sokoplan

Sokoplan is Sokoban research/engineering project focused on **A\*** planning with classical
heuristics and learned **GNN-based heuristics**, including pipelines for
dataset preparation, label generation, training, evaluation, and
transferability-driven level grouping.

## What this project contains

- **Core Sokoban engine** (`sokoban_core/`): state representation, parsing,
  moves, deadlocks, level I/O.
- **Search stack** (`search/`, `heuristics/`): A\* with transposition table,
  classical heuristics, Festival heuristic, learned GNN heuristic.
- **GNN stack** (`gnn/`): graph construction from states, PyG model, dataset and
  training loop.
- **Experiment scripts** (`scripts/`): data fetch, split creation, label
  generation, training, solving/evaluation, clustering/grouping.
- **Hydra SLURM scripts** (`scripts_hydra/`): cluster-ready pipelines for labels,
  training, eval, and block workflow.

## Main ideas

- A level is solved by A\* over push-based Sokoban states.
- Supervised labels are generated from solved trajectories:
  each training sample is a state with target `y = pushes_to_goal`.
- A GIN/GINE model predicts `y` and is used as heuristic:
  - `speed`: use GNN output directly.
  - `optimal_mix`: `min(gnn, hungarian)` to keep conservative behavior.
- Levels can be grouped and trained/evaluated per group to study transferability.

## Repository map

```text
sokoplan/
├── sokoban_core/              # parser/state/moves/deadlocks/levels
├── search/                    # A* + priority queue + transposition table
├── heuristics/                # zero/hungarian/festival/gnn heuristics
├── gnn/                       # graph features, model, dataset, train loop
├── scripts/                   # local pipelines and utilities
│   ├── fetcher/               # download datasets (letslogic/boxoban)
│   ├── labels/                # generate supervised/off-policy labels
│   ├── train/                 # model training entrypoint
│   ├── solve/                 # batch eval for classic and GNN heuristics
│   └── blocks/                # split/merge/group pipelines
├── scripts_hydra/             # SLURM jobs for Hydra cluster
│   ├── generate_labels/
│   ├── train_model/
│   ├── eval/
│   └── blocks/
├── configs/data.yaml          # level roots, sources, filters, split paths
├── festival/                  # Festival C++ solver (binary used by scripts)
└── tests/                     # pytest suite
```

## Environment setup

### Requirements

- Python **3.11** (see `pyproject.toml`)
- Linux/macOS (Hydra scripts target SLURM)
- Optional but recommended: CUDA-capable GPU for training/eval speed

### Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Build Festival binary (required for Festival-based scripts, configured for Linux)

Many scripts expect binary at `festival/festival`.

```bash
make -C festival
```

## Data conventions

### Level IDs

Across scripts, level references use:

```text
path/to/pack.txt#idx
```

Where `idx` is the 0-based level index inside a multi-level pack file.

### Label format (`.jsonl`)

Each line stores one state sample, typically:

- board fields: `width`, `height`, `walls`, `goals`, `boxes`, `player`,
  `board_mask`
- target: `y` (remaining pushes/cost-to-go)
- provenance: `level_id`

## Local workflows

Run commands from repo root and prefer env variables over fixed paths.

### 1) Fetch levels

```bash
source .venv/bin/activate

# Windows packs from sourcecode.se -> sokoban_core/levels/windows
python -m scripts.fetcher.fetch_windows

# Boxoban packs -> sokoban_core/levels/boxoban
python -m scripts.fetcher.fetch_boxoban

# LetsLogic packs (requires API key) -> sokoban_core/levels/letslogic/raw
python -m scripts.fetcher.fetch_letslogic "$LETSLOGIC_API_KEY"
```

### 2) Build train/val/test splits

```bash
CONFIG_PATH="configs/data.yaml"
SEED=42
TRAIN_N=2000
VAL_N=200
TEST_N=200

python -m scripts.utils.make_splits \
  --config "$CONFIG_PATH" \
  --seed "$SEED" \
  --train "$TRAIN_N" \
  --val "$VAL_N" \
  --test "$TEST_N"
```

### 3) Generate supervised labels

#### Baseline (Hungarian + deadlocks)

```bash
LIST_PATH="sokoban_core/levels/splits/train.txt"
LABELS_OUT="data/generated_labels/labels.jsonl"

python -m scripts.labels.generate_labels \
  --list "$LIST_PATH" \
  --out "$LABELS_OUT" \
  --h hungarian \
  --use_dl \
  --time_limit 300 \
  --node_limit 3000000 \
  --jobs 8
```

#### Festival-based labels

```bash
LIST_PATH="sokoban_core/levels/splits/train.txt"
LABELS_OUT="data/generated_labels/labels_festival.jsonl"

python -m scripts.labels.generate_labels_festival \
  --list "$LIST_PATH" \
  --out "$LABELS_OUT" \
  --time_limit 300 \
  --jobs 8 \
  --sample_every 2
```

### 4) Train GNN heuristic

```bash
TRAIN_JSONL="data/generated_labels/labels.jsonl"
VAL_JSONL="data/generated_labels/labels_val.jsonl"   # optional
MODEL_OUT="artifacts/gnn_best.pt"
CKPT_OUT="artifacts/gnn_checkpoint.pt"

python -m scripts.train.train_gnn \
  --train "$TRAIN_JSONL" \
  --val "$VAL_JSONL" \
  --epochs 30 \
  --batch 512 \
  --hidden 128 \
  --layers 4 \
  --lr 1e-3 \
  --dropout 0.05 \
  --workers 4 \
  --prefetch_factor 4 \
  --pin_memory \
  --persistent_workers \
  --amp \
  --out "$MODEL_OUT" \
  --checkpoint "$CKPT_OUT" \
  --resume
```

### 5) Evaluate heuristics in batch

#### Classic baseline

```bash
LIST_PATH="sokoban_core/levels/splits/test.txt"
OUT_CSV="results/evaluate/hungarian.csv"

python -m scripts.solve.run_batch \
  --list "$LIST_PATH" \
  --h hungarian \
  --use_dl \
  --out "$OUT_CSV" \
  --time_limit 600 \
  --node_limit 2000000 \
  --jobs 8
```

#### GNN heuristic

```bash
LIST_PATH="sokoban_core/levels/splits/test.txt"
CKPT_PATH="artifacts/gnn_best.pt"
OUT_CSV="results/evaluate/gnn.csv"

python -m scripts.solve.run_batch_gnn \
  --list "$LIST_PATH" \
  --ckpt "$CKPT_PATH" \
  --mode speed \
  --out "$OUT_CSV" \
  --time_limit 600 \
  --node_limit 2000000 \
  --jobs 8
```

### 6) Evaluate many checkpoints + summary

```bash
EVAL_LIST="sokoban_core/levels/splits/test.txt"
MODELS_DIR="artifacts"
EVAL_DIR="results/evaluate/models"
HUN_CSV="results/evaluate/hungarian.csv"

python -m scripts.solve.eval_gnn_models run \
  --list "$EVAL_LIST" \
  --models_dir "$MODELS_DIR" \
  --out_dir "$EVAL_DIR" \
  --ensure_hungarian \
  --hungarian_csv "$HUN_CSV"

python -m scripts.solve.eval_gnn_models summarize \
  --eval_dir "$EVAL_DIR" \
  --hungarian_csv "$HUN_CSV" \
  --out "$EVAL_DIR/models_summary.csv"
```

## Hydra pipelines (SLURM)

Detailed Hydra instructions are maintained in:

- [`scripts_hydra/generate_labels/README.md`](scripts_hydra/generate_labels/README.md)
- [`scripts_hydra/train_model/README.md`](scripts_hydra/train_model/README.md)
- [`scripts_hydra/blocks/README.md`](scripts_hydra/blocks/README.md)

Recommended run pattern:

```bash
VAR1=value1 VAR2=value2 sbatch path/to/script.sbatch
```

For array jobs, keep `NUM_SHARDS` aligned with `--array`.

## Block/Grouping pipelines

The repository supports multiple level-grouping directions:

- **Transferability-driven split/merge pipeline**
  (self-eval + cross-eval driven grouping)
- **Static-feature clustering pipeline**
  (`scripts.blocks.static.*` for feature extraction and KMeans grouping)

Use the blocks Hydra README as the primary orchestration guide:
[`scripts_hydra/blocks/README.md`](scripts_hydra/blocks/README.md).

### Static-feature clustering pipeline

This pipeline groups levels using only static/topological map features
(`scripts.blocks.static.*`), without transferability cross-eval.

#### 1) Build static-feature groups (`group_XXX.list`)

```bash
IN_LISTS_DIR="sokoban_core/levels/letslogic/filtered/lists"
STATIC_LISTS_DIR="sokoban_core/levels/letslogic/clusters_static_features/lists"
STATIC_REPORT="results/static_groups/report.json"

python -m scripts.blocks.static.plan_groups_by_static_features \
  --in_lists_dir "$IN_LISTS_DIR" \
  --out_lists_dir "$STATIC_LISTS_DIR" \
  --report_json "$STATIC_REPORT" \
  --k 0 \
  --k_min 12 \
  --k_max 64 \
  --auto_k_sample_size 8000 \
  --min_cluster_size 120 \
  --seed 42
```

Notes:
- `--k 0` enables auto-selection of `K` via silhouette score.
- `--min_cluster_size` merges tiny clusters into nearest larger ones.

#### 2) Summarize and inspect the resulting groups

```bash
python -m scripts.blocks.static.summarize_static_groups \
  --report_json "$STATIC_REPORT" \
  --stat median \
  --top_features 10 \
  --out_csv "results/static_groups/summary.csv"
```

#### 3) (Optional) Convert grouped `.list` files back to pack `.txt`

```bash
STATIC_PACKS_DIR="sokoban_core/levels/letslogic/clusters_static_features"

python -m scripts.blocks.utils.list_to_pack \
  --lists_dir "$STATIC_LISTS_DIR" \
  --out_dir "$STATIC_PACKS_DIR" \
  --name_prefix_from_list_stem
```

#### 4) (Optional) Repack existing labels into static groups

```bash
PACK_LABELS_DIR="data/packs_labels_festival"
GROUP_LABELS_DIR="data/group_labels_festival_static"

python -m scripts.blocks.utils.repack_labels_to_groups \
  --labels_dir "$PACK_LABELS_DIR" \
  --groups_lists_dir "$STATIC_LISTS_DIR" \
  --out_dir "$GROUP_LABELS_DIR"
```

This creates `group_XXX.jsonl` labels that can be used to train one model per
static group.

#### 5) (Optional, Hydra) Evaluate static groups cross-pack

If you already have:
- test lists per static group (e.g. `.../splits/letslogic/static_features_clusters/test/group_XXX.list`)
- model checkpoints per static group (`group_XXX.pt`)

you can run:

```bash
TEST_LISTS_DIR="sokoban_core/levels/splits/letslogic/static_features_clusters/test"
MODELS_DIR="artifacts/letslogic/static_features"
OUT_DIR="results/evaluate/static_crosspack/raw"

TEST_LISTS_DIR="$TEST_LISTS_DIR" MODELS_DIR="$MODELS_DIR" OUT_DIR="$OUT_DIR" \
sbatch --array=0-11 scripts_hydra/eval/run_eval_static_crosspack_array.sbatch

python -m scripts.utils.build_static_crosspack_matrix \
  --raw_dir "$OUT_DIR" \
  --out_dir "results/evaluate/static_crosspack/matrix"
```

Adjust `--array=0-N` to match your number of static groups.


## Notes

- `artifacts/`, `results/`, and large `data/` subfolders contain experiment
  outputs and may vary between runs/machines.
- Most scripts expose CLI args for paths and limits; prefer passing parameters
  via env/flags instead of editing scripts.
