# --- file: heuristics/learned.py
from __future__ import annotations
from typing import Optional, Callable, Dict
import torch

from sokoban_core.zobrist import Zobrist
from sokoban_core.state import State
from gnn.graphs import grid_to_graph
from gnn.model import GINHeuristic
from heuristics.classic import h_manhattan_hungarian
from sokoban_core.deadlocks import has_deadlock, INF


class GNNHeuristic:
    """Wrap a trained GIN model to use as h(s).

    Modes:
      - mode="speed":         return gnn(s)
      - mode="optimal_mix":   return min(gnn(s), h_manh(s))  (keeps admissibility if gnn overestimates)

    Also applies deadlock filtering (INF on known deadlocks) if use_deadlocks=True.
    Caches values by Zobrist hash.
    """
    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        mode: str = "optimal_mix",
        use_deadlocks: bool = True,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = GINHeuristic(in_dim=4)
        obj = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(obj['model_state_dict'])
        self.model.eval().to(self.device)
        self.mode = mode
        self.use_deadlocks = use_deadlocks
        self._cache: Dict[int, int] = {}
        self._zobrist: Optional[Zobrist] = None

    def __call__(self, s: State) -> int:
        # lazy init zobrist (depends on board shape)
        if self._zobrist is None:
            self._zobrist = Zobrist(s.width, s.height, s.board_mask)
        key = self._zobrist.hash(s)
        if key in self._cache:
            return self._cache[key]

        if self.use_deadlocks and has_deadlock(s):
            self._cache[key] = INF
            return INF

        # build graph and run model
        data, _ = grid_to_graph(s)
        data = data.to(self.device)
        with torch.no_grad():
            y_hat = self.model(data.x, data.edge_index, torch.zeros(data.num_nodes, dtype=torch.long, device=self.device))
            # model returns shape [1]; take scalar
            gnn_val = float(y_hat.view(-1)[0].item())
        # non-negative and integerize for A*
        gnn_est = max(0, int(round(gnn_val)))

        if self.mode == "optimal_mix":
            h_val = min(gnn_est, h_manhattan_hungarian(s))
        elif self.mode == "speed":
            h_val = gnn_est
        else:
            h_val = gnn_est

        self._cache[key] = h_val
        return h_val