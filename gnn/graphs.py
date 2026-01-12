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
    Node features (x):
      [is_goal, has_box, is_player, wall_up, wall_down, wall_left, wall_right]
    (directional walls instead of a single walls_around scalar).

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

    # Build edges (4-neighborhood) + simple edge features: direction one-hot [up, down, left, right]
    src: List[int] = []
    dst: List[int] = []
    edge_attr: List[List[float]] = []
    for idx in floor_mask:
        r, c = divmod(idx, W)
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr, cc = r+dr, c+dc
            if 0 <= rr < H and 0 <= cc < W:
                j = rr * W + cc
                if j in idx2nid:  # neighbor is also floor
                    src.append(idx2nid[idx])
                    dst.append(idx2nid[j])
                    if dr == -1 and dc == 0:
                        edge_attr.append([1.0, 0.0, 0.0, 0.0])  # up
                    elif dr == 1 and dc == 0:
                        edge_attr.append([0.0, 1.0, 0.0, 0.0])  # down
                    elif dr == 0 and dc == -1:
                        edge_attr.append([0.0, 0.0, 1.0, 0.0])  # left
                    else:
                        edge_attr.append([0.0, 0.0, 0.0, 1.0])  # right
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)

    # Node features
    feats: List[List[float]] = []
    for idx in floor_mask:
        is_goal = 1.0 if state.is_goal_cell(idx) else 0.0
        has_box = 1.0 if state.has_box(idx) else 0.0
        is_player = 1.0 if idx == state.player else 0.0
        # Directional walls (treat outside as wall)
        r, c = divmod(idx, W)
        up = 1.0 if (r == 0 or state.is_wall((r - 1) * W + c)) else 0.0
        down = 1.0 if (r == H - 1 or state.is_wall((r + 1) * W + c)) else 0.0
        left = 1.0 if (c == 0 or state.is_wall(r * W + (c - 1))) else 0.0
        right = 1.0 if (c == W - 1 or state.is_wall(r * W + (c + 1))) else 0.0
        feats.append([is_goal, has_box, is_player, up, down, left, right])

    x = torch.tensor(feats, dtype=torch.float)

    # Global target placeholder (filled by dataset): y: Tensor([target])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t)
    return data, idx2nid