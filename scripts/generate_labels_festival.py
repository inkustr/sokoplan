from __future__ import annotations
import argparse, json, os
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from datetime import datetime

from sokoban_core.levels.resolve import load_level_by_id
from sokoban_core.goal_check import is_goal
from heuristics.festival import festival_heuristic, festival_to_states


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


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _level_to_string(s) -> str:
    player_mask = 1 << s.player
    
    lines = []
    for y in range(s.height):
        line = []
        for x in range(s.width):
            pos = 1 << (y * s.width + x)
            
            if s.walls & pos:
                line.append('#')
            elif s.boxes & pos and s.goals & pos:
                line.append('*')
            elif s.boxes & pos:
                line.append('$')
            elif s.goals & pos and player_mask == pos:
                line.append('+')
            elif s.goals & pos:
                line.append('.')
            elif player_mask == pos:
                line.append('@')
            else:
                line.append(' ')
        lines.append(''.join(line).rstrip())
    
    while lines and not lines[-1]:
        lines.pop()
    
    return '\n'.join(lines)


def _process_one_level(args_tuple) -> Tuple[str, List[Dict], bool]:
    level_id, time_limit, sample_every = args_tuple
    
    print(f"[{_timestamp()}] [START] {level_id}", flush=True)
    
    try:
        s0 = load_level_by_id(level_id)
        
        if is_goal(s0):
            print(f"[{_timestamp()}] [SKIPPED] {level_id} (already solved)", flush=True)
            return (level_id, [], False)
        
        level_str = _level_to_string(s0)
        
        num_boxes = bin(s0.boxes).count('1')
        if num_boxes > 100:
            print(f"[{_timestamp()}] [SKIPPED] {level_id} (too many boxes: {num_boxes} > 100)", flush=True)
            return (level_id, [], False)
        
        result = festival_heuristic(level_str, timeout=time_limit)
        
        if result is None:
            print(f"[{_timestamp()}] [FAILED] {level_id} (Festival returned None, {num_boxes} boxes)", flush=True)
            return (level_id, [], False)
        
        moves, nodes, runtime_ms, error_msg = result
        
        if error_msg:
            print(f"[{_timestamp()}] [FAILED] {level_id} ({error_msg}, {num_boxes} boxes)", flush=True)
            return (level_id, [], False)
        
        path = festival_to_states(level_str, moves, pushes_only=True)
        
        if len(path) < 2:
            print(f"[{_timestamp()}] [FAILED] {level_id} (invalid solution path)", flush=True)
            return (level_id, [], False)
        
        T = len(path) - 1
        records = []
        for t, st in enumerate(path):
            if t % sample_every == 0 or t == 0 or t == T:
                y = T - t
                rec = state_to_dict(st)
                rec["y"] = int(y)
                rec["level_id"] = level_id
                records.append(rec)
        
        print(f"[{_timestamp()}] [SOLVED] {level_id} ({len(records)} states, pushes={T}, total_moves={len(moves)}, nodes={nodes}, time={runtime_ms}ms)", flush=True)
        return (level_id, records, True)
        
    except Exception as e:
        print(f"[{_timestamp()}] [ERROR] {level_id}: {e}", flush=True)
        return (level_id, [], False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True, help="split file: lines path#idx")
    p.add_argument("--out", required=True, help="output JSONL file")
    p.add_argument("--time_limit", type=int, default=300, help="timeout per level in seconds")
    p.add_argument("--jobs", type=int, default=0, help="processes (0→cpu_count)")
    p.add_argument("--shard_idx", type=int, default=0, help="index of this shard (0..num_shards-1)")
    p.add_argument("--num_shards", type=int, default=1, help="total number of shards")
    p.add_argument("--sample_every", type=int, default=1, help="sample every Nth state from path (1=all, 2=every 2nd, 3=every 3rd)")
    args = p.parse_args()

    with open(args.list, "r", encoding="utf-8") as f:
        level_ids = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    if args.num_shards > 1:
        level_ids = [lid for i, lid in enumerate(level_ids) if i % args.num_shards == args.shard_idx]
        print(f"[Shard {args.shard_idx}/{args.num_shards}] Processing {len(level_ids)} levels")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    jobs = args.jobs or cpu_count()
    payload = [(lid, args.time_limit, args.sample_every) for lid in level_ids]

    count_pairs = 0
    count_solved = 0
    count_failed = 0
    
    with open(args.out, "w", encoding="utf-8") as out:
        if jobs == 1:
            for t in tqdm(payload, desc="Processing levels", unit="level"):
                level_id, records, success = _process_one_level(t)
                if success and records:
                    count_solved += 1
                    for rec in records:
                        out.write(json.dumps(rec) + "\n")
                        count_pairs += 1
                    out.flush()
                else:
                    count_failed += 1
        else:
            with Pool(processes=jobs) as pool:
                for level_id, records, success in tqdm(
                    pool.imap_unordered(_process_one_level, payload),
                    total=len(payload),
                    desc="Processing levels",
                    unit="level"
                ):
                    if success and records:
                        count_solved += 1
                        for rec in records:
                            out.write(json.dumps(rec) + "\n")
                            count_pairs += 1
                        out.flush()
                    else:
                        count_failed += 1
    
    total_levels = count_solved + count_failed
    print(f"\nWrote {count_pairs} (state, y) pairs to {args.out}; jobs={jobs}")
    print(f"Solved: {count_solved}/{total_levels} levels ({100*count_solved/total_levels:.1f}%)")
    print(f"Failed: {count_failed}/{total_levels} levels ({100*count_failed/total_levels:.1f}%)")


if __name__ == "__main__":
    main()


