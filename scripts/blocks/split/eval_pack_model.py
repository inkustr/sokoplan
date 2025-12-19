from __future__ import annotations

"""
Evaluate a trained pack model on a pack label JSONL file.

Outputs:
  - per-record predictions (optional)
  - per-level aggregated error metrics
  - overall metrics

Used for:
  - splitting packs: identify "hard" levels inside a pack via per-level MAE
  - cross-pack merge decisions (when run on other pack datasets)

Example usage:
  source .venv/bin/activate
  python -m scripts.blocks.split.eval_pack_model --ckpt artifacts/packs_models/<pack>_best.pt --labels data/packs_labels/<pack>.jsonl --out results/packs_eval/<pack>.json
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch

from gnn.graphs import grid_to_graph
from gnn.model import GINHeuristic
from sokoban_core.state import State


@dataclass
class Agg:
    n: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_y: float = 0.0
    sum_y2: float = 0.0

    def add(self, y: float, yhat: float) -> None:
        e = yhat - y
        self.n += 1
        self.sum_abs += abs(e)
        self.sum_sq += e * e
        self.sum_y += y
        self.sum_y2 += y * y

    def metrics(self) -> Dict[str, float]:
        if self.n == 0:
            return {"n": 0.0, "mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "r2": float("nan")}
        mae = self.sum_abs / self.n
        mse = self.sum_sq / self.n
        rmse = mse**0.5

        mean_y = self.sum_y / self.n
        ss_tot = self.sum_y2 - self.n * (mean_y * mean_y)
        ss_res = self.sum_sq
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
        return {"n": float(self.n), "mae": float(mae), "mse": float(mse), "rmse": float(rmse), "r2": float(r2)}


def _load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    obj = torch.load(ckpt_path, map_location=device)
    cfg = obj.get("config", {})
    hidden = int(cfg.get("hidden", 128))
    layers = int(cfg.get("layers", 4))
    dropout = float(cfg.get("dropout", 0.0))
    model = GINHeuristic(in_dim=4, hidden=hidden, layers=layers, dropout=dropout).to(device)
    model.load_state_dict(obj["model_state_dict"])
    model.eval()
    return model


def _iter_jsonl(path: str, limit: Optional[int] = None):
    with open(path, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if limit is not None and i >= limit:
                return
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True, help="output JSON")
    p.add_argument("--limit_records", type=int, default=0, help="0 = no limit")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.ckpt, device)

    limit = None if args.limit_records == 0 else args.limit_records
    per_level: Dict[str, Agg] = defaultdict(Agg)
    overall = Agg()

    graph_cache: Dict[Tuple[int, int, int, int, int], Tuple[torch.Tensor, Dict[int, int], List[int], torch.Tensor]] = {}

    try:
        for rec in _iter_jsonl(args.labels, limit=limit):
            y = float(int(rec["y"]))
            level_id = str(rec["level_id"])
            s = State(
                width=int(rec["width"]),
                height=int(rec["height"]),
                walls=int(rec["walls"]),
                goals=int(rec["goals"]),
                boxes=int(rec["boxes"]),
                player=int(rec["player"]),
                board_mask=int(rec["board_mask"]),
            )

            key = (s.width, s.height, s.walls, s.goals, s.board_mask)
            if key not in graph_cache:
                data, idx2nid = grid_to_graph(s)
                floor_idxs = [0] * len(idx2nid)
                for idx_cell, nid in idx2nid.items():
                    floor_idxs[nid] = idx_cell
                static_x = data.x.clone()
                static_x[:, 1] = 0.0
                static_x[:, 2] = 0.0
                edge_index = data.edge_index.to(device)
                graph_cache[key] = (edge_index, idx2nid, floor_idxs, static_x)

            edge_index, idx2nid, floor_idxs, static_x = graph_cache[key]
            x = static_x.clone()
            for nid, idx_cell in enumerate(floor_idxs):
                if s.has_box(idx_cell):
                    x[nid, 1] = 1.0
            nid_player = idx2nid.get(s.player)
            if nid_player is not None:
                x[nid_player, 2] = 1.0

            x = x.to(device)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
            with torch.no_grad():
                y_hat = model(x, edge_index, batch).view(-1)[0].item()
            yhat = float(max(0.0, y_hat))

            overall.add(y, yhat)
            per_level[level_id].add(y, yhat)

    out_obj = {
        "ckpt": args.ckpt,
        "labels": args.labels,
        "overall": overall.metrics(),
        "per_level": {k: v.metrics() for k, v in per_level.items()},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f)


if __name__ == "__main__":
    main()


