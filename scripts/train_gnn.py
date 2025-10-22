from __future__ import annotations
import argparse, os, json
import torch
from torch_geometric.loader import DataLoader
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
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = JsonlSokobanDataset(args.train)
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

    best = float('inf')
    for epoch in range(1, args.epochs+1):
        tr = train_once(model, dl_train, opt, device)
        va = eval_once(model, dl_val, device)
        print(f"epoch {epoch:03d} | train {tr:.4f} | val {va:.4f}")
        if va < best:
            best = va
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': vars(args),
                'val_loss': va,
            }, args.out)
    print(f"saved best → {args.out} (val={best:.4f})")


if __name__ == "__main__":
    main()
