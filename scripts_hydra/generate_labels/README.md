# Label generation on Hydra

All scripts in this folder are runnable from repo root and accept runtime
configuration via environment variables.

## General pattern

```bash
VAR1=value1 VAR2=value2 sbatch scripts_hydra/generate_labels/<script>.sbatch
```

When using job arrays, keep `NUM_SHARDS` and SLURM array range aligned:

```bash
NUM_SHARDS=50 sbatch --array=0-49 scripts_hydra/generate_labels/<script>.sbatch
```

## Script overview

- `run.sbatch` - baseline label generation via `scripts.labels.generate_labels`
- `run_festival.sbatch` - Festival-based label generation
- `run_collect_offpolicy_labels.sbatch` - off-policy labels from one list
- `run_collect_offpolicy_packs_array.sbatch` - off-policy labels per pack list (with weighted sharding)

---

## 1) Baseline labels (`run.sbatch`)

Key env vars:
- `LIST` (input `.txt` list)
- `OUT_DIR`
- `OUTPUT_BASENAME` (default `labels_part`)
- `NUM_SHARDS`
- `HEURISTIC`, `USE_DL`, `TIME_LIMIT`, `NODE_LIMIT`, `JOBS`

Example:

```bash
NUM_SHARDS=50 \
LIST="sokoban_core/levels/splits/train.txt" \
OUT_DIR="data/generated_labels" \
OUTPUT_BASENAME="labels_part" \
sbatch --array=0-49 scripts_hydra/generate_labels/run.sbatch
```

Output pattern:
- `"$OUT_DIR/${OUTPUT_BASENAME}_<shard>.jsonl"`

---

## 2) Festival labels (`run_festival.sbatch`)

Requires Festival binary at `./festival/festival`.

Key env vars:
- `LIST`
- `OUT_DIR`
- `OUTPUT_BASENAME` (default `labels_festival_part`)
- `NUM_SHARDS`
- `TIME_LIMIT`, `SAMPLE_EVERY`, `JOBS`

Example:

```bash
NUM_SHARDS=50 \
LIST="sokoban_core/levels/splits/train.txt" \
OUT_DIR="data/generated_labels" \
OUTPUT_BASENAME="labels_festival_part" \
TIME_LIMIT=28800 \
SAMPLE_EVERY=2 \
sbatch --array=0-49 scripts_hydra/generate_labels/run_festival.sbatch
```

Output pattern:
- `"$OUT_DIR/${OUTPUT_BASENAME}_<shard>.jsonl"`

---

## 3) Off-policy labels from one list (`run_collect_offpolicy_labels.sbatch`)

Requires Festival binary at `./festival/festival`.

Key env vars:
- `LIST`
- `OUT_DIR`
- `NUM_SHARDS`
- `POLICY`, `SAMPLE_PER_LEVEL`, `FRONTIER_PER_LEVEL`, `MIN_G`
- `SEARCH_NODE_LIMIT`, `SEARCH_TIME_LIMIT`, `FESTIVAL_TIMEOUT`
- `SAMPLE_EVERY`, `START_SAMPLE_EVERY`, `JOBS`

Example:

```bash
NUM_SHARDS=50 \
LIST="sokoban_core/levels/splits/train.txt" \
OUT_DIR="data/boxoban_labels_ab" \
POLICY="hungarian" \
sbatch --array=0-49 scripts_hydra/generate_labels/run_collect_offpolicy_labels.sbatch
```

Output pattern:
- `"$OUT_DIR/boxoban_ab_shard_<shard>.jsonl"`

---

## 4) Off-policy labels per pack (`run_collect_offpolicy_packs_array.sbatch`)

Requires Festival binary at `./festival/festival`.

Key env vars:
- `PACKS_DIR` (must contain `lists/`)
- `LISTS_DIR` (optional override, defaults to `"$PACKS_DIR/lists"`)
- `OUT_DIR`
- `NUM_SHARDS`
- `SHARD_OFFSET` (useful when splitting one logical array into multiple submissions)
- `POLICY`, `SAMPLE_PER_LEVEL`, `FRONTIER_PER_LEVEL`, `MIN_G`
- `SEARCH_NODE_LIMIT`, `SEARCH_TIME_LIMIT`, `FESTIVAL_TIMEOUT`
- `SAMPLE_EVERY`, `START_SAMPLE_EVERY`, `JOBS`

Example (single submission):

```bash
NUM_SHARDS=45 \
PACKS_DIR="sokoban_core/levels/letslogic/filtered" \
OUT_DIR="data/packs_offpolicy_labels_festival" \
sbatch --array=0-44 scripts_hydra/generate_labels/run_collect_offpolicy_packs_array.sbatch
```

Example (split 90 logical shards into 2 submissions of 45):

```bash
NUM_SHARDS=90 SHARD_OFFSET=0  sbatch --array=0-44 scripts_hydra/generate_labels/run_collect_offpolicy_packs_array.sbatch
NUM_SHARDS=90 SHARD_OFFSET=45 sbatch --array=0-44 scripts_hydra/generate_labels/run_collect_offpolicy_packs_array.sbatch
```

Output pattern:
- `"$OUT_DIR/<pack>.jsonl"`

---

## 5) Merge shards (for shard-based outputs)

```bash
OUT_DIR="data/generated_labels"
OUTPUT_BASENAME="labels_part"
MERGED_FILE="data/labels.jsonl"

cat "$OUT_DIR"/"${OUTPUT_BASENAME}"_*.jsonl > "$MERGED_FILE"
```

