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
from heuristics.selector import get_heuristic


def _run_one(args_tuple) -> Dict[str, object]:
    level_id, heur_name, use_dl, time_limit, node_limit = args_tuple
    try:
        s = load_level_by_id(level_id)
        zob = Zobrist(s.width, s.height, s.board_mask)
        trans = Transposition(zob)
        h = get_heuristic(heur_name, use_dl)
        res = astar(s, h, is_goal, trans=trans, time_limit_s=time_limit, node_limit=node_limit)
        return {
            "level_id": level_id,
            "heuristic": f"{heur_name}{'+dl' if use_dl else ''}",
            "success": bool(res.get("success", False)),
            "nodes": int(res.get("nodes", 0)),
            "runtime": float(res.get("runtime", 0.0)),
            "solution_len": int(res.get("solution_len", -1)),
        }
    except Exception:
        return {"level_id": level_id, "heuristic": heur_name, "success": False, "nodes": 0, "runtime": 0.0, "solution_len": -1}


def main():
    p = argparse.ArgumentParser(description="Batch A* runs → CSV (flags, parallel)")
    p.add_argument("--list", required=True)
    p.add_argument("--h", default="hungarian", choices=["zero", "hungarian"]) 
    p.add_argument("--use_dl", action="store_true", help="wrap heuristic with deadlock filter")
    p.add_argument("--out", default="results/batch.csv")
    p.add_argument("--time_limit", type=float, default=10.0)
    p.add_argument("--node_limit", type=int, default=200000)
    p.add_argument("--jobs", type=int, default=0, help="processes (0→cpu_count)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.list, "r", encoding="utf-8") as f:
        level_ids = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    jobs = args.jobs or cpu_count()
    payload = [(lid, args.h, args.use_dl, args.time_limit, args.node_limit) for lid in level_ids]

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