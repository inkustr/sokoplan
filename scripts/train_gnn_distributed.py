from __future__ import annotations
import argparse, os, json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from gnn.dataset import JsonlSokobanDataset
from gnn.model import GINHeuristic
from gnn.train_loop import train_once, eval_once


def setup_distributed():
    """Initialize distributed training environment."""
    # SLURM environment variables
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    
    # Initialize process group
    dist.init_process_group(
        backend="nccl",  # Use NCCL for GPU
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="path to JSONL with (state,y)")
    p.add_argument("--val", required=False, help="path to JSONL for val")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--out", default="artifacts/gnn_best.pt")
    p.add_argument("--checkpoint", default="artifacts/gnn_checkpoint.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--distributed", action="store_true", help="use distributed training")
    args = p.parse_args()

    # Setup distributed training if requested
    if args.distributed:
        rank, world_size, local_rank = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
        is_main = rank == 0
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    if is_main:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        print(f"[INFO] Training on {world_size} GPU(s)")
        print(f"[INFO] Device: {device}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(device)}")
            print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

    # Load datasets
    ds_train = JsonlSokobanDataset(args.train)
    if is_main:
        print(f"[INFO] Training dataset size: {len(ds_train)} samples")
        print(f"[INFO] Samples per GPU: {len(ds_train) // world_size}")
    if args.val:
        ds_val = JsonlSokobanDataset(args.val)
    else:
        n = len(ds_train)
        n_tr = int(0.9 * n)
        n_va = n - n_tr
        ds_train, ds_val = torch.utils.data.random_split(ds_train, [n_tr, n_va])

    # Create data loaders with distributed sampler if needed
    if args.distributed:
        train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True)
        dl_train = DataLoader(ds_train, batch_size=args.batch, sampler=train_sampler)
        # Validation only on main process
        if is_main:
            dl_val = DataLoader(ds_val, batch_size=args.batch)
    else:
        dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=args.batch)

    # Create model
    model = GINHeuristic(in_dim=4, hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    
    # Wrap model with DDP if distributed
    if args.distributed:
        model = DDP(model, device_ids=[local_rank])
        model_to_save = model.module  # Unwrap for saving
    else:
        model_to_save = model

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Resume from checkpoint
    start_epoch = 1
    best = float('inf')
    if args.resume and os.path.exists(args.checkpoint) and is_main:
        if is_main:
            print(f"Resuming from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model_to_save.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best = ckpt.get('best_val_loss', float('inf'))
        if is_main:
            print(f"Resuming from epoch {start_epoch}, best val loss: {best:.4f}")

    # Broadcast start_epoch and best to all processes
    if args.distributed:
        start_epoch_tensor = torch.tensor(start_epoch, device=device)
        best_tensor = torch.tensor(best, device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        dist.broadcast(best_tensor, src=0)
        start_epoch = start_epoch_tensor.item()
        best = best_tensor.item()

    # Training loop
    for epoch in tqdm(range(start_epoch, args.epochs+1), desc="Training", unit="epoch", disable=not is_main):
        if args.distributed:
            train_sampler.set_epoch(epoch)  # Important for proper shuffling
        
        tr = train_once(model, dl_train, opt, device)
        
        # Validation only on main process
        if is_main:
            va = eval_once(model, dl_val, device)
            tqdm.write(f"epoch {epoch:03d} | train {tr:.4f} | val {va:.4f}")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'config': vars(args),
                'train_loss': tr,
                'val_loss': va,
                'best_val_loss': best,
            }, args.checkpoint)
            
            # Save best model
            if va < best:
                best = va
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'config': vars(args),
                    'val_loss': va,
                    'epoch': epoch,
                }, args.out)
        
        # Synchronize all processes
        if args.distributed:
            dist.barrier()

    if is_main:
        print(f"saved best → {args.out} (val={best:.4f})")

    if args.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()


