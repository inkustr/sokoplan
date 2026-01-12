from __future__ import annotations
import os, math
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from gnn.dataset import JsonlSokobanDataset
from gnn.model import GINHeuristic
from gnn.loss import huber_loss


def train_once(model, loader, opt, device, *, amp: bool = False, scaler=None):
    model.train()
    total = 0.0
    n = 0
    use_amp = bool(amp and device.type == "cuda")
    use_bf16 = bool(use_amp and torch.cuda.is_bf16_supported())
    use_fp16 = bool(use_amp and not use_bf16)
    if use_fp16 and scaler is None:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except Exception:
            scaler = torch.cuda.amp.GradScaler()

    max_grad_norm = 1.0
    for batch in tqdm(loader, desc="Train", leave=False):
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        if use_amp:
            try:
                autocast_ctx = torch.amp.autocast("cuda", dtype=(torch.bfloat16 if use_bf16 else torch.float16))
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast()
            with autocast_ctx:
                pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                loss = huber_loss(pred, batch.y.view(-1))
            if not torch.isfinite(loss):
                print("[WARN] non-finite train loss; skipping batch")
                continue
            if use_fp16:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
        else:
            pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = huber_loss(pred, batch.y.view(-1))
            if not torch.isfinite(loss):
                print("[WARN] non-finite train loss; skipping batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
        total += float(loss.item()) * batch.num_graphs
        n += batch.num_graphs
    return total / max(1, n)


def eval_once(model, loader, device, *, amp: bool = False):
    model.eval()
    total = 0.0
    n = 0
    use_amp = bool(amp and device.type == "cuda")
    use_bf16 = bool(use_amp and torch.cuda.is_bf16_supported())
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            batch = batch.to(device)
            if use_amp:
                try:
                    autocast_ctx = torch.amp.autocast("cuda", dtype=(torch.bfloat16 if use_bf16 else torch.float16))
                except Exception:
                    autocast_ctx = torch.cuda.amp.autocast()
                with autocast_ctx:
                    pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                    loss = huber_loss(pred, batch.y.view(-1))
            else:
                pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                loss = huber_loss(pred, batch.y.view(-1))
            total += float(loss.item()) * batch.num_graphs
            n += batch.num_graphs
    return total / max(1, n)