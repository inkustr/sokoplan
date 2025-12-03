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

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch)

    model = GINHeuristic(in_dim=4, hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Resume from checkpoint if requested
    start_epoch = 1
    best = float('inf')
    if args.resume and os.path.exists(args.checkpoint):
        print(f"Resuming from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best = ckpt.get('best_val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}, best val loss: {best:.4f}")

    for epoch in tqdm(range(start_epoch, args.epochs+1), desc="Training", unit="epoch"):
        tr = train_once(model, dl_train, opt, device)
        va = eval_once(model, dl_val, device)
        tqdm.write(f"epoch {epoch:03d} | train {tr:.4f} | val {va:.4f}")
        
        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
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
