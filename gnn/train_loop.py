from __future__ import annotations
import os, math
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from gnn.dataset import JsonlSokobanDataset
from gnn.model import GINHeuristic
from gnn.loss import huber_loss


def train_once(model, loader, opt, device):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = huber_loss(pred, batch.y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total += float(loss.item()) * batch.num_graphs
        n += batch.num_graphs
    return total / max(1, n)


def eval_once(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = huber_loss(pred, batch.y.view(-1))
            total += float(loss.item()) * batch.num_graphs
            n += batch.num_graphs
    return total / max(1, n)