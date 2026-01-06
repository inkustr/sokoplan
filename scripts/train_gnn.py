from __future__ import annotations
import argparse, os, json
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from gnn.dataset import JsonlSokobanDataset
from gnn.model import GINHeuristic
from gnn.train_loop import train_once, eval_once


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
    p.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA (faster on A100).")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (0=main process).")
    p.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch factor (workers>0 only).")
    p.add_argument("--pin_memory", action="store_true", help="Pin CPU memory for faster H2D transfers.")
    p.add_argument("--persistent_workers", action="store_true", help="Keep DataLoader workers alive between epochs.")
    p.add_argument("--torch_threads", type=int, default=1, help="torch.set_num_threads for CPU ops.")
    p.add_argument("--out", default="artifacts/gnn_best.pt")
    p.add_argument("--checkpoint", default="artifacts/gnn_checkpoint.pt", help="checkpoint file for resuming")
    p.add_argument("--resume", action="store_true", help="resume from checkpoint if exists")
    p.add_argument("--temp_folder", required=False, help="folder to save epoch models")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.temp_folder:
        os.makedirs(args.temp_folder, exist_ok=True)
        print(f"[INFO] Epoch models will be saved to: {args.temp_folder}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    torch.set_num_threads(int(args.torch_threads))

    ds_train = JsonlSokobanDataset(args.train)
    print(f"[INFO] Training dataset size: {len(ds_train)} samples")
    if args.val:
        ds_val = JsonlSokobanDataset(args.val)
    else:
        # split train into 90/10
        n = len(ds_train)
        n_tr = int(0.9 * n)
        n_va = n - n_tr
        ds_train, ds_val = torch.utils.data.random_split(ds_train, [n_tr, n_va])

    workers = int(args.workers)
    pin_memory = bool(args.pin_memory) and device.type == "cuda"
    dl_kwargs = dict(
        batch_size=int(args.batch),
        num_workers=workers,
        pin_memory=pin_memory,
    )
    if workers > 0:
        dl_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        dl_kwargs["persistent_workers"] = bool(args.persistent_workers)

    dl_train = DataLoader(ds_train, shuffle=True, **dl_kwargs)
    dl_val = DataLoader(ds_val, shuffle=False, **dl_kwargs)

    model = GINHeuristic(in_dim=4, hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.amp and device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except Exception:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Resume from checkpoint if requested
    start_epoch = 1
    best = float('inf')
    if args.resume and os.path.exists(args.checkpoint):
        print(f"Resuming from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt['epoch'] + 1
        best = ckpt.get('best_val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}, best val loss: {best:.4f}")

    for epoch in tqdm(range(start_epoch, args.epochs+1), desc="Training", unit="epoch"):
        tr = train_once(model, dl_train, opt, device, amp=bool(args.amp), scaler=scaler)
        va = eval_once(model, dl_val, device, amp=bool(args.amp))
        tqdm.write(f"epoch {epoch:03d} | train {tr:.4f} | val {va:.4f}")
        
        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'config': vars(args),
            'train_loss': tr,
            'val_loss': va,
            'best_val_loss': best,
        }, args.checkpoint)
        
        # Save best model
        if va < best:
            best = va
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': vars(args),
                'val_loss': va,
                'epoch': epoch,
            }, args.out)
        
        if args.temp_folder:
            epoch_path = os.path.join(args.temp_folder, f"epoch_{epoch:03d}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': vars(args),
                'train_loss': tr,
                'val_loss': va,
                'epoch': epoch,
            }, epoch_path)
    
    print(f"saved best → {args.out} (val={best:.4f})")


if __name__ == "__main__":
    main()
