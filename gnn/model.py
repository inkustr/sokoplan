from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class GINHeuristic(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 128, layers: int = 4, dropout: float = 0.0):
        super().__init__()
        convs = []
        dims = [in_dim] + [hidden] * layers
        for i in range(layers):
            mlp = MLP(dims[i], hidden, hidden, dropout)
            convs.append(GINConv(mlp))
        self.convs = nn.ModuleList(convs)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
        hg = global_mean_pool(h, batch)
        out = self.head(hg)
        return out.view(-1)