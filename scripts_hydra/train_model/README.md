### Submit Training Job

**Option A: Single GPU (simplest)**
```bash
sbatch scripts_hydra/run_single.sbatch
```
- 1 GPU

**Option B: Multi-GPU - Single/Multiple nodes (setup in script)**
```bash
sbatch scripts_hydra/run_distributed.sbatch
```
- **4 GPUs on n nodes**

## Checkpoint & Resume
If job hits the time limit:
1. It will save progress to `artifacts/gnn_checkpoint.pt`
2. Simply **resubmit the same job** - it will resume from where it stopped
3. The best model is always saved to `artifacts/gnn_best.pt`

