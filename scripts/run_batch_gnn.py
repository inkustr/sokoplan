# --- file: scripts/run_batch_gnn.py
from __future__ import annotations
import argparse, csv, os, time
from typing import List, Dict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from sokoban_core.levels.resolve import load_level_by_id
from sokoban_core.goal_check import is_goal
from sokoban_core.zobrist import Zobrist
from search.astar import astar
from search.transposition import Transposition
from heuristics.learned import GNNHeuristic


def _run_one(args_tuple) -> Dict[str, object]:
    level_id, ckpt, mode, use_deadlocks, time_limit, node_limit = args_tuple
    try:
        s = load_level_by_id(level_id)
        zob = Zobrist(s.width, s.height, s.board_mask)
        trans = Transposition(zob)
        h = GNNHeuristic(ckpt, mode=mode, use_deadlocks=use_deadlocks)
        res = astar(s, h, is_goal, trans=trans, time_limit_s=time_limit, node_limit=node_limit)
        return {
            "level_id": level_id,
            "heuristic": f"gnn[{mode}{'+dl' if use_deadlocks else ''}]",
            "success": bool(res.get("success", False)),
            "nodes": int(res.get("nodes", 0)),
            "runtime": float(res.get("runtime", 0.0)),
            "solution_len": int(res.get("solution_len", -1)),
        }
    except Exception as e:
        print(f"ERROR on {level_id}: {e}")
        import traceback
        traceback.print_exc()
        return {"level_id": level_id, "heuristic": f"gnn[{mode}]", "success": False, "nodes": 0, "runtime": 0.0, "solution_len": -1}


def main():
    p = argparse.ArgumentParser(description="Batch A* runs using a trained GNN heuristic")
    p.add_argument("--list", required=True)
    p.add_argument("--ckpt", required=True, help="path to artifacts/gnn_best.pt")
    p.add_argument("--mode", default="optimal_mix", choices=["optimal_mix", "speed"])
    p.add_argument("--no_dl", action="store_true", help="disable deadlock filter")
    p.add_argument("--out", default="results/batch_gnn.csv")
    p.add_argument("--time_limit", type=float, default=200.0)
    p.add_argument("--node_limit", type=int, default=2000000)
    p.add_argument("--jobs", type=int, default=0, help="processes (0→cpu_count)")
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"ERROR: Checkpoint file not found: {args.ckpt}")
        exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.list, "r", encoding="utf-8") as f:
        level_ids = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    jobs = args.jobs or cpu_count()
    payload = [(lid, args.ckpt, args.mode, (not args.no_dl), args.time_limit, args.node_limit) for lid in level_ids]

    started = time.time()
    if jobs == 1:
        rows = [_run_one(t) for t in tqdm(payload, desc="Running A*", unit="level")]
    else:
        with Pool(processes=jobs) as pool:
            rows = list(tqdm(pool.imap_unordered(_run_one, payload), total=len(payload), desc="Running A*", unit="level"))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["level_id", "heuristic", "success", "nodes", "runtime", "solution_len"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"done: {len(rows)} levels → {args.out}; total_time={time.time()-started:.2f}s; jobs={jobs}")


if __name__ == "__main__":
    main()