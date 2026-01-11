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
from multiprocessing import Pool, cpu_count, current_process
from typing import Dict, List, Tuple, Optional, Iterable

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

    def merge_(self, other: "Agg") -> None:
        self.n += other.n
        self.sum_abs += other.sum_abs
        self.sum_sq += other.sum_sq
        self.sum_y += other.sum_y
        self.sum_y2 += other.sum_y2


def _load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    obj = torch.load(ckpt_path, map_location=device)
    cfg = obj.get("config", {})
    hidden = int(cfg.get("hidden", 128))
    layers = int(cfg.get("layers", 4))
    dropout = float(cfg.get("dropout", 0.0))
    sd = obj.get("model_state_dict", obj)
    if not isinstance(sd, dict):
        raise RuntimeError(f"Invalid checkpoint format at {ckpt_path}: expected dict-like state_dict.")
    w0 = sd.get("convs.0.nn.net.0.weight")
    in_dim = int(w0.shape[1]) if isinstance(w0, torch.Tensor) and w0.ndim == 2 else int(cfg.get("in_dim", 7))
    use_gine = any(str(k).startswith("convs.0.lin.") for k in sd.keys())
    model = GINHeuristic(
        in_dim=in_dim,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        conv=("gine" if use_gine else "gin"),
    ).to(device)
    model.load_state_dict(sd)
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


_WORKER_MODEL: Optional[torch.nn.Module] = None
_WORKER_DEVICE: Optional[torch.device] = None
_WORKER_GRAPH_CACHE: Optional[
    Dict[
        Tuple[int, int, int, int, int],
        Tuple[torch.Tensor, torch.Tensor, Dict[int, int], List[int], torch.Tensor],
    ]
] = None


def _worker_init(ckpt_path: str, torch_threads: int) -> None:
    """
    Pool initializer: load model once per worker (CPU) and set torch threads.
    """
    global _WORKER_MODEL, _WORKER_DEVICE, _WORKER_GRAPH_CACHE
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    _WORKER_DEVICE = torch.device("cpu")
    _WORKER_MODEL = _load_model(ckpt_path, _WORKER_DEVICE)
    _WORKER_GRAPH_CACHE = {}


def _eval_record(rec: dict) -> Tuple[str, float, float]:
    """
    Evaluate one JSONL record, returning (level_id, y, yhat).
    Uses per-process cached graphs and a per-process CPU model.
    """
    global _WORKER_MODEL, _WORKER_DEVICE, _WORKER_GRAPH_CACHE
    assert _WORKER_MODEL is not None
    assert _WORKER_DEVICE is not None
    assert _WORKER_GRAPH_CACHE is not None

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
    if key not in _WORKER_GRAPH_CACHE:
        data, idx2nid = grid_to_graph(s)
        floor_idxs = [0] * len(idx2nid)
        for idx_cell, nid in idx2nid.items():
            floor_idxs[nid] = idx_cell
        static_x = data.x.clone()
        static_x[:, 1] = 0.0
        static_x[:, 2] = 0.0
        edge_index = data.edge_index.to(_WORKER_DEVICE)
        edge_attr = data.edge_attr.to(_WORKER_DEVICE)
        _WORKER_GRAPH_CACHE[key] = (edge_index, edge_attr, idx2nid, floor_idxs, static_x)

    edge_index, edge_attr, idx2nid, floor_idxs, static_x = _WORKER_GRAPH_CACHE[key]
    x = static_x.clone()
    for nid, idx_cell in enumerate(floor_idxs):
        if s.has_box(idx_cell):
            x[nid, 1] = 1.0
    nid_player = idx2nid.get(s.player)
    if nid_player is not None:
        x[nid_player, 2] = 1.0

    x = x.to(_WORKER_DEVICE)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=_WORKER_DEVICE)
    with torch.no_grad():
        y_hat = _WORKER_MODEL(x, edge_index, batch, edge_attr).view(-1)[0].item()
    yhat = float(max(0.0, y_hat))
    return level_id, y, yhat


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True, help="output JSON")
    p.add_argument("--limit_records", type=int, default=0, help="0 = no limit")
    p.add_argument("--jobs", type=int, default=1, help="CPU processes for evaluation (1=single process).")
    args = p.parse_args()

    jobs = int(args.jobs)
    if jobs < 1:
        jobs = 1
    if jobs > cpu_count():
        jobs = cpu_count()

    # Multi-process evaluation is intended for CPU. (GPU + multiprocessing would thrash / OOM.)
    device = torch.device("cpu")
    if jobs > 1 and device.type != "cpu":
        raise SystemExit("eval_pack_model.py: --jobs>1 requires CPU execution.")

    limit = None if args.limit_records == 0 else args.limit_records
    per_level: Dict[str, Agg] = defaultdict(Agg)
    overall = Agg()

    pool = Pool(processes=jobs, initializer=_worker_init, initargs=(args.ckpt, 1))
    try:
        it: Iterable[Tuple[str, float, float]] = pool.imap_unordered(_eval_record, _iter_jsonl(args.labels, limit=limit), chunksize=256)
        for level_id, y, yhat in it:
            overall.add(y, yhat)
            per_level[level_id].add(y, yhat)
    finally:
        pool.close()
        pool.join()

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


