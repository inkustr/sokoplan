from __future__ import annotations
from typing import Dict
import json
import os
from array import array
from pathlib import Path
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
        self.path = str(path)
        self._index = self._load_or_build_index(self.path)
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

    @staticmethod
    def _idx_path(jsonl_path: str) -> Path:
        return Path(jsonl_path + ".idx_u64")

    @classmethod
    def _load_or_build_index(cls, jsonl_path: str) -> array:
        """
        Build or load a compact uint64 index of byte offsets for each JSONL line.

        This avoids storing millions of Python ints in a list.
        Index file format: raw uint64 little-endian offsets, one per record line.
        """
        p = Path(jsonl_path)
        if not p.exists():
            raise FileNotFoundError(jsonl_path)

        idx_p = cls._idx_path(jsonl_path)
        try:
            if idx_p.exists() and idx_p.stat().st_mtime >= p.stat().st_mtime:
                a = array("Q")
                with idx_p.open("rb") as f:
                    a.fromfile(f, idx_p.stat().st_size // 8)
                return a
        except Exception:
            pass

        # Build index (offset for each line start)
        a = array("Q")
        with p.open("rb") as f:
            # We read in binary to make offsets consistent across platforms/encodings.
            offset = f.tell()
            line = f.readline()
            while line:
                a.append(offset)
                offset = f.tell()
                line = f.readline()

        # Save for reuse
        try:
            tmp = idx_p.with_suffix(idx_p.suffix + ".tmp")
            with tmp.open("wb") as out:
                a.tofile(out)
            os.replace(tmp, idx_p)
        except Exception:
            pass

        return a
