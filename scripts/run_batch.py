from __future__ import annotations
import argparse, csv, os, time
from typing import Callable, List

from sokoban_core.levels.resolve import load_level_by_id
from sokoban_core.goal_check import is_goal
from sokoban_core.zobrist import Zobrist
from search.astar import astar
from search.transposition import Transposition
from heuristics.classic import h_zero, h_manhattan_hungarian, h_with_deadlocks


def get_heuristic(name: str) -> Callable:
    name = name.lower()
    if name == "zero":
        return h_zero
    if name == "hungarian":
        return h_manhattan_hungarian
    if name in ("hungarian+dl", "hungarian_dl", "hungarian-deadlocks"):
        return lambda s: h_with_deadlocks(s, h_manhattan_hungarian)
    raise ValueError(f"unknown heuristic: {name}")


def run_one(level_id: str, heur_name: str, time_limit: float, node_limit: int):
    s = load_level_by_id(level_id)
    zob = Zobrist(s.width, s.height, s.board_mask)
    trans = Transposition(zob)
    h = get_heuristic(heur_name)
    res = astar(s, h, is_goal, trans=trans, time_limit_s=time_limit, node_limit=node_limit)
    out = {
        "level_id": level_id,
        "heuristic": heur_name,
        "success": bool(res.get("success", False)),
        "nodes": int(res.get("nodes", 0)),
        "runtime": float(res.get("runtime", 0.0)),
        "solution_len": int(res.get("solution_len", -1)),
    }
    return out


def main():
    p = argparse.ArgumentParser(description="Batch A* runs over a split list → CSV (fixed loader)")
    p.add_argument("--list", required=True, help="path to split .txt (lines: path#idx)")
    p.add_argument("--h", default="hungarian", help="heuristic: zero|hungarian|hungarian+dl")
    p.add_argument("--out", default="results/batch.csv", help="output CSV path")
    p.add_argument("--time_limit", type=float, default=10.0)
    p.add_argument("--node_limit", type=int, default=200000)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.list, "r", encoding="utf-8") as f:
        level_ids = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    started = time.time()
    rows: List[dict] = []
    for i, lid in enumerate(level_ids, 1):
        try:
            r = run_one(lid, args.h, args.time_limit, args.node_limit)
        except Exception as e:
            r = {"level_id": lid, "heuristic": args.h, "success": False, "nodes": 0, "runtime": 0.0, "solution_len": -1}
        rows.append(r)
        if i % 20 == 0:
            print(f".. {i}/{len(level_ids)} processed")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["level_id", "heuristic", "success", "nodes", "runtime", "solution_len"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"done: {len(rows)} levels → {args.out}; total_time={time.time()-started:.2f}s")


if __name__ == "__main__":
    main()
