from __future__ import annotations
from typing import Dict, List, Tuple
import torch
from torch import Tensor
from torch_geometric.data import Data

from sokoban_core.state import State, has_bit


def grid_to_graph(state: State) -> Tuple[Data, Dict[int, int]]:
    """Build a PyG graph from Sokoban State.

    Nodes: all *floor* cells (inside board & not walls). Walls excluded.
    Edges: 4-neighborhood between floor cells.
    Node features (x): [is_goal, has_box, is_player, walls_around/4].

    Returns (Data, idx2nid) where idx2nid maps cell index -> node id (only for floor nodes).
    """
    W, H = state.width, state.height
    size = W * H

    # Select floor cells (inside & not wall)
    floor_mask = []
    idx2nid: Dict[int, int] = {}
    nid = 0
    for idx in range(size):
        inside = state.is_inside(idx)
        wall = state.is_wall(idx)
        if inside and not wall:
            idx2nid[idx] = nid
            floor_mask.append(idx)
            nid += 1
    n = len(floor_mask)
    if n == 0:
        raise ValueError("Empty graph: no floor cells")

    # Build edges (4-neighborhood)
    src: List[int] = []
    dst: List[int] = []
    for idx in floor_mask:
        r, c = divmod(idx, W)
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr, cc = r+dr, c+dc
            if 0 <= rr < H and 0 <= cc < W:
                j = rr * W + cc
                if j in idx2nid:  # neighbor is also floor
                    src.append(idx2nid[idx])
                    dst.append(idx2nid[j])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Node features
    feats: List[List[float]] = []
    for idx in floor_mask:
        is_goal = 1.0 if state.is_goal_cell(idx) else 0.0
        has_box = 1.0 if state.has_box(idx) else 0.0
        is_player = 1.0 if idx == state.player else 0.0
        # walls around (treat outside as wall)
        walls_around = 0
        r, c = divmod(idx, W)
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr, cc = r+dr, c+dc
            if not (0 <= rr < H and 0 <= cc < W):
                walls_around += 1
            else:
                j = rr * W + cc
                if state.is_wall(j):
                    walls_around += 1
        feats.append([is_goal, has_box, is_player, walls_around/4.0])

    x = torch.tensor(feats, dtype=torch.float)

    # Global target placeholder (filled by dataset): y: Tensor([target])
    data = Data(x=x, edge_index=edge_index)
    return data, idx2nid