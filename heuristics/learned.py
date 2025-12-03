# --- file: heuristics/learned.py
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import torch

from sokoban_core.zobrist import Zobrist
from sokoban_core.state import State
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

        try:
            dummy_x = torch.randn(10, 4, device=self.device)
            dummy_edge = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long, device=self.device)
            dummy_batch = torch.zeros(10, dtype=torch.long, device=self.device)
            self.model = torch.jit.trace(self.model, (dummy_x, dummy_edge, dummy_batch))
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not JIT compile model: {e}")
        
        self.mode = mode
        self.use_deadlocks = use_deadlocks
        self._cache: Dict[int, int] = {}
        self._zobrist: Optional[Zobrist] = None
        
        # Cache static graph structure (reused across all states in a level)
        self._edge_index: Optional[torch.Tensor] = None
        self._floor_mask: Optional[List[int]] = None
        self._static_features: Optional[torch.Tensor] = None  # [is_goal, walls_around]
        self._board_signature: Optional[Tuple[int, int, int]] = None  # (width, height, board_mask)
        
        # Buffers reused across calls
        self._feature_buffer: Optional[torch.Tensor] = None  # Full [n, 4] feature matrix
        self._batch_buffer: Optional[torch.Tensor] = None

    def _build_static_structure(self, s: State) -> None:
        """Build and cache static graph structure (edges, floor cells, static features)."""
        W, H = s.width, s.height
        size = W * H
        
        # Select floor cells (inside & not wall)
        floor_mask = []
        idx2nid: Dict[int, int] = {}
        nid = 0
        for idx in range(size):
            inside = s.is_inside(idx)
            wall = s.is_wall(idx)
            if inside and not wall:
                idx2nid[idx] = nid
                floor_mask.append(idx)
                nid += 1
        
        # Build edges (4-neighborhood)
        src: List[int] = []
        dst: List[int] = []
        for idx in floor_mask:
            r, c = divmod(idx, W)
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                rr, cc = r+dr, c+dc
                if 0 <= rr < H and 0 <= cc < W:
                    j = rr * W + cc
                    if j in idx2nid:
                        src.append(idx2nid[idx])
                        dst.append(idx2nid[j])
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        
        # Build static features (goal, walls_around)
        static_feats: List[List[float]] = []
        for idx in floor_mask:
            is_goal = 1.0 if s.is_goal_cell(idx) else 0.0
            # walls around
            walls_around = 0
            r, c = divmod(idx, W)
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                rr, cc = r+dr, c+dc
                if not (0 <= rr < H and 0 <= cc < W):
                    walls_around += 1
                else:
                    j = rr * W + cc
                    if s.is_wall(j):
                        walls_around += 1
            static_feats.append([is_goal, walls_around/4.0])
        
        self._edge_index = edge_index
        self._floor_mask = floor_mask
        self._static_features = torch.tensor(static_feats, dtype=torch.float, device=self.device)
        self._board_signature = (s.width, s.height, int(s.board_mask))
    
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

        # Check if we need to rebuild static structure
        board_sig = (s.width, s.height, int(s.board_mask))
        if self._board_signature != board_sig:
            self._build_static_structure(s)
            self._feature_buffer = None
            self._batch_buffer = None

        n = len(self._floor_mask)
        if self._feature_buffer is None or self._feature_buffer.shape[0] != n:
            self._feature_buffer = torch.zeros((n, 4), dtype=torch.float, device=self.device)
            self._feature_buffer[:, 0] = self._static_features[:, 0]
            self._feature_buffer[:, 3] = self._static_features[:, 1]
            self._batch_buffer = torch.zeros(n, dtype=torch.long, device=self.device)
        
        self._feature_buffer[:, 1].zero_()
        self._feature_buffer[:, 2].zero_()
        for i, idx in enumerate(self._floor_mask):
            if s.has_box(idx):
                self._feature_buffer[i, 1] = 1.0
            if idx == s.player:
                self._feature_buffer[i, 2] = 1.0
        
        with torch.no_grad():
            y_hat = self.model(self._feature_buffer, self._edge_index, self._batch_buffer)
            gnn_val = float(y_hat.view(-1)[0].item())
        
        gnn_est = max(0, int(round(gnn_val)))

        if self.mode == "optimal_mix":
            h_val = min(gnn_est, h_manhattan_hungarian(s))
        elif self.mode == "speed":
            h_val = gnn_est
        else:
            h_val = gnn_est

        self._cache[key] = h_val
        return h_val