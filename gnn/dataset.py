from __future__ import annotations
from typing import Dict, Optional, List
import json
import torch
from torch_geometric.data import Dataset, Data

from sokoban_core.parser import parse_level_str
from sokoban_core.state import State
from gnn.graphs import grid_to_graph


def dict_to_state(rec: Dict) -> State:
    return State(
        width=int(rec["width"]),
        height=int(rec["height"]),
        walls=int(rec["walls"]),
        goals=int(rec["goals"]),
        boxes=int(rec["boxes"]),
        player=int(rec["player"]),
        board_mask=int(rec["board_mask"]),
    )


class JsonlSokobanDataset(Dataset):
    """Lazy JSONL reader that converts each (state,y) record to a PyG Data graph.

    Each line is a dict with fields saved by scripts/generate_labels.py
    """
    def __init__(self, path: str, transform=None, pre_transform=None):
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self._index = [f.tell()]
            while f.readline():
                self._index.append(f.tell())
        # last entry is EOF position; dataset length is len-1
        self._index = self._index[:-1]
        super().__init__(None, transform, pre_transform)

    def len(self) -> int:
        return len(self._index)

    def get(self, idx: int) -> Data:
        with open(self.path, "r", encoding="utf-8") as f:
            f.seek(self._index[idx])
            rec = json.loads(f.readline())
        s = dict_to_state(rec)
        g, _ = grid_to_graph(s)
        y = torch.tensor([float(rec["y"])], dtype=torch.float)
        g.y = y
        return g
