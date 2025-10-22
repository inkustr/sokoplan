"""Generate supervised labels (cost-to-go) from optimal paths on small levels.

For each level in --list:
  - run A* with chosen heuristic (default: hungarian + deadlocks)
  - if solved, take the reconstructed path [s0..sT]; emit pairs (state, y=T-t)
  - write to JSONL where each line is a dict with integer masks for state and y
"""
from __future__ import annotations
import argparse, json, os
from typing import Dict, List
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from sokoban_core.levels.resolve import load_level_by_id
from sokoban_core.goal_check import is_goal
from sokoban_core.zobrist import Zobrist
from search.astar import astar
from search.transposition import Transposition
from heuristics.selector import get_heuristic


def state_to_dict(s) -> Dict[str, int]:
    return {
        "width": s.width,
        "height": s.height,
        "walls": int(s.walls),
        "goals": int(s.goals),
        "boxes": int(s.boxes),
        "player": int(s.player),
        "board_mask": int(s.board_mask),
    }


def _process_one_level(args_tuple) -> List[Dict]:
    """Process one level and return list of (state, y) records."""
    level_id, heur_name, use_dl, time_limit, node_limit = args_tuple
    try:
        s0 = load_level_by_id(level_id)
        zob = Zobrist(s0.width, s0.height, s0.board_mask)
        trans = Transposition(zob)
        h = get_heuristic(heur_name, use_dl)
        res = astar(s0, h, is_goal, trans=trans, time_limit_s=time_limit, node_limit=node_limit)
        
        if not res.get("success"):
            return []
        
        path: List = res["path"]  # type: ignore
        T = len(path) - 1
        records = []
        for t, st in enumerate(path):
            y = T - t
            rec = state_to_dict(st)
            rec["y"] = int(y)
            rec["level_id"] = level_id
            records.append(rec)
        return records
    except Exception:
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True, help="split file: lines path#idx")
    p.add_argument("--out", required=True, help="output JSONL file")
    p.add_argument("--h", default="hungarian", choices=["zero", "hungarian"]) 
    p.add_argument("--use_dl", action="store_true", help="deadlock wrapper")
    p.add_argument("--time_limit", type=float, default=15.0)
    p.add_argument("--node_limit", type=int, default=300000)
    p.add_argument("--jobs", type=int, default=0, help="processes (0→cpu_count)")
    args = p.parse_args()

    with open(args.list, "r", encoding="utf-8") as f:
        level_ids = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    jobs = args.jobs or cpu_count()
    payload = [(lid, args.h, args.use_dl, args.time_limit, args.node_limit) for lid in level_ids]

    # Process levels in parallel
    if jobs == 1:
        all_records = [_process_one_level(t) for t in tqdm(payload, desc="Processing levels", unit="level")]
    else:
        with Pool(processes=jobs) as pool:
            all_records = list(tqdm(pool.imap_unordered(_process_one_level, payload), total=len(payload), desc="Processing levels", unit="level"))

    # Write all records to output file
    count_pairs = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for records in all_records:
            for rec in records:
                out.write(json.dumps(rec) + "\n")
                count_pairs += 1
    
    print(f"wrote {count_pairs} (state, y) pairs → {args.out}; jobs={jobs}")


if __name__ == "__main__":
    main()
