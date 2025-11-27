The `run.sbatch` script is configured to:
- Run a **Job Array** of 50 tasks (`--array=0-49`).
- Each task processes a slice of `sokoban_core/levels/splits/train.txt`.
- Requests 8 CPUs per task to parallelize the search within the slice.
- Runs for max 2 hours (`cpu-2h` partition).

### Submit
```bash
sbatch scripts_hydra/run.sbatch
```

### Monitor
```bash
squeue -u $USER
```

### Output
Logs will be in `logs/`.
Data will be generated in `data/generated_labels/labels_part_*.jsonl`.

## Merge Results
After all jobs finish, you can combine the parts into a single file:

```bash
cat data/generated_labels/labels_part_*.jsonl > data/labels.jsonl
```

## Customization

- **Adjust Array Size**: Edit `scripts_hydra/run.sbatch`. Change `#SBATCH --array=0-N` and `NUM_SHARDS=N+1` to match. More shards = faster total time (if sufficient nodes available).
- **Adjust Resources**: Change `--cpus-per-task` if you want more/less parallelism per shard.

