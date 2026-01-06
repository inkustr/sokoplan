from __future__ import annotations

"""
Collect off-policy (state, y) labels.

High-level idea:
  1) Run a *bounded* best-first search (A*-like) on each level using some heuristic (Hungarian or GNN).
  2) Sample a small number of visited states (expanded states) per level.
  3) For each sampled state, call Festival to solve *from that state* and derive y = remaining pushes.
  4) Write JSONL records.

Example:
  source .venv/bin/activate
  python -m scripts.labels.collect_offpolicy_labels \
    --list sokoban_core/levels/splits/test.txt \
    --out data/offpolicy/offpolicy_labels.jsonl \
    --policy hungarian \
    --sample_per_level 8 \
    --frontier_per_level 0 \
    --min_g 3 \
    --search_node_limit 20000 \
    --search_time_limit 10.0 \
    --festival_timeout 150 \
    --include_start_solution \
    --sample_every 2 \
    --start_sample_every 1 \
    --jobs 8
"""

import argparse
import json
import os
import random
import socket
import time
import hashlib
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count, current_process
from typing import Dict, Iterable, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

from sokoban_core.levels.resolve import load_level_by_id
from sokoban_core.goal_check import is_goal
from sokoban_core.state import State
from sokoban_core.moves import successors_pushes
from sokoban_core.zobrist import Zobrist
from search.priority_queue import PriorityQueue
from heuristics.selector import get_heuristic
from heuristics.learned import GNNHeuristic
from heuristics.festival import find_festival_binary, festival_heuristic, festival_to_states

def state_to_dict(s: State) -> Dict[str, int]:
    return {
        "width": int(s.width),
        "height": int(s.height),
        "walls": int(s.walls),
        "goals": int(s.goals),
        "boxes": int(s.boxes),
        "player": int(s.player),
        "board_mask": int(s.board_mask),
    }


def _level_to_string(s: State) -> str:
    player_mask = 1 << s.player
    lines: List[str] = []
    for y in range(s.height):
        line: List[str] = []
        for x in range(s.width):
            pos = 1 << (y * s.width + x)
            if not (s.board_mask & pos):
                line.append("#")
                continue
            if s.walls & pos:
                line.append("#")
            elif s.boxes & pos and s.goals & pos:
                line.append("*")
            elif s.boxes & pos:
                line.append("$")
            elif s.goals & pos and player_mask == pos:
                line.append("+")
            elif s.goals & pos:
                line.append(".")
            elif player_mask == pos:
                line.append("@")
            else:
                line.append(" ")
        lines.append("".join(line).rstrip())
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _state_key(s: State) -> Tuple[int, int, int, int, int]:
    return (int(s.board_mask), int(s.walls), int(s.goals), int(s.boxes), int(s.player))


@dataclass(frozen=True)
class SampledState:
    state: State
    g: int
    h: int


_SAMPLING_H_POLICY = None


def _sampling_pool_init(policy: str, gnn_ckpt: str, gnn_mode: str, gnn_no_dl: bool) -> None:
    """
    Initializer for Phase-1 sampling pool.
    Creates the heuristic once per worker process (important for performance).
    """
    global _SAMPLING_H_POLICY
    if policy == "hungarian":
        _SAMPLING_H_POLICY = get_heuristic("hungarian", use_dl=True)
    else:
        _SAMPLING_H_POLICY = GNNHeuristic(
            gnn_ckpt,
            mode=gnn_mode,
            use_deadlocks=True,
        )


def _sample_one_level(args_tuple) -> Tuple[str, List[Dict[str, int]], int]:
    """
    Phase-1 worker: sample states for one level id and return state dicts.
    Returns: (level_id, [state_dict...], skipped_goal_flag)
    """
    (lid, sample_per_level, node_limit, time_limit_s, min_g, frontier_per_level, seed) = args_tuple
    s0 = load_level_by_id(lid)
    if is_goal(s0):
        return (lid, [], 1)

    h_policy = _SAMPLING_H_POLICY

    samples = _bounded_best_first_samples(
        s0,
        h_policy,
        sample_per_level=sample_per_level,
        node_limit=node_limit,
        time_limit_s=time_limit_s,
        min_g=min_g,
        frontier_per_level=frontier_per_level,
        seed=seed,
    )
    return (lid, [state_to_dict(smp.state) for smp in samples], 0)


def _bounded_best_first_samples(
    start: State,
    h_fn,
    sample_per_level: int,
    node_limit: int,
    time_limit_s: float,
    min_g: int,
    frontier_per_level: int,
    seed: int,
) -> List[SampledState]:
    """
    Bounded best-first expansion to collect candidate states.
    """
    if sample_per_level <= 0:
        return []

    rnd = random.Random(seed)
    zob = Zobrist(start.width, start.height, start.board_mask)

    g_best: Dict[int, int] = {}
    openq = PriorityQueue()

    h0 = int(h_fn(start))
    openq.push(h0, (0, start))
    g_best[zob.hash(start)] = 0

    expanded = 0
    t0 = time.time()
    reservoir: List[SampledState] = []

    while len(openq) > 0:
        if time.time() - t0 > time_limit_s:
            break
        if expanded >= node_limit:
            break

        gs, s = openq.pop()
        hs_key = zob.hash(s)
        if gs != g_best.get(hs_key, gs):
            continue
        hs = int(h_fn(s))
        if hs >= 10**9:
            continue

        expanded += 1
        if gs >= min_g and not is_goal(s):
            item = SampledState(state=s, g=gs, h=hs)
            if len(reservoir) < sample_per_level:
                reservoir.append(item)
            else:
                j = rnd.randrange(expanded)
                if j < sample_per_level:
                    reservoir[j] = item

        for ns in successors_pushes(s):
            ng = gs + 1
            k = zob.hash(ns)
            old = g_best.get(k)
            if old is not None and ng >= old:
                continue
            g_best[k] = ng
            hn = int(h_fn(ns))
            if hn < 10**9:
                openq.push(ng + hn, (ng, ns))

    if frontier_per_level > 0 and len(openq) > 0:
        heap = getattr(openq, "_h", None)
        if isinstance(heap, list) and heap:
            candidates: List[SampledState] = []
            for prio, _tb, item in heap:
                try:
                    gs2, s2 = item
                except Exception:
                    continue
                hs_key2 = zob.hash(s2)
                if gs2 != g_best.get(hs_key2, gs2):
                    continue
                hs2 = int(h_fn(s2))
                if hs2 >= 10**9:
                    continue
                if gs2 < min_g or is_goal(s2):
                    continue
                candidates.append(SampledState(state=s2, g=int(gs2), h=int(hs2)))

            if candidates:
                k = min(frontier_per_level, len(candidates))
                rnd.shuffle(candidates)
                reservoir.extend(candidates[:k])

    return reservoir


def _label_one_state(
    args_tuple,
) -> Tuple[str, Optional[Dict], str]:
    """
    Returns: (state_key_json, record_or_none, status)
    """
    (
        source_level_id,
        state_dict,
        festival_timeout,
        sample_every,
        is_offpolicy,
    ) = args_tuple
    try:
        s = State(
            width=int(state_dict["width"]),
            height=int(state_dict["height"]),
            walls=int(state_dict["walls"]),
            goals=int(state_dict["goals"]),
            boxes=int(state_dict["boxes"]),
            player=int(state_dict["player"]),
            board_mask=int(state_dict["board_mask"]),
        )
        level_str = _level_to_string(s)
        res = festival_heuristic(level_str, timeout=festival_timeout)
        if res is None:
            return (json.dumps(state_dict, sort_keys=True), None, "festival_none")
        moves, nodes, runtime_ms, error_msg = res
        if error_msg:
            return (json.dumps(state_dict, sort_keys=True), None, f"festival_{error_msg}")

        path = festival_to_states(level_str, moves, pushes_only=True)
        if len(path) < 2:
            return (json.dumps(state_dict, sort_keys=True), None, "festival_invalid_path")

        T = len(path) - 1
        out_records: List[Dict] = []
        for t, st in enumerate(path):
            if t % sample_every == 0 or t == 0 or t == T:
                rec = state_to_dict(st)
                rec["y"] = int(T - t)
                rec["level_id"] = source_level_id
                rec["_offpolicy"] = bool(is_offpolicy)
                rec["_source_level_id"] = source_level_id
                out_records.append(rec)

        wrapped = {"_records": out_records, "_festival_nodes": int(nodes), "_festival_runtime_ms": int(runtime_ms)}
        return (json.dumps(state_dict, sort_keys=True), wrapped, "ok_many")
    except Exception as e:
        return (json.dumps(state_dict, sort_keys=True), None, f"error_{type(e).__name__}")


def _read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True, help="Level-id list file (lines: path#idx).")
    p.add_argument("--out", required=True, help="Output JSONL file to write off-policy labels.")

    p.add_argument(
        "--policy",
        default="hungarian",
        choices=["hungarian", "gnn"],
        help="Which policy heuristic to use for sampling states.",
    )
    p.add_argument("--gnn_ckpt", default="", help="Required if --policy=gnn (path to model checkpoint).")
    p.add_argument("--gnn_mode", default="speed", choices=["speed", "optimal_mix"])

    p.add_argument("--sample_per_level", type=int, default=8)
    p.add_argument(
        "--frontier_per_level",
        type=int,
        default=0,
        help="Also sample this many states from the open queue after bounded search (0=disabled).",
    )
    p.add_argument("--min_g", type=int, default=2, help="Only sample expanded states with g>=min_g.")
    p.add_argument("--search_node_limit", type=int, default=20000)
    p.add_argument("--search_time_limit", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--festival_timeout", type=int, default=60)
    p.add_argument(
        "--include_start_solution",
        action="store_true",
        help=(
            "Also label the original start state for each level (Festival solution path), "
            "emitting records with _offpolicy=false into the same output JSONL."
        ),
    )
    p.add_argument(
        "--start_sample_every",
        type=int,
        default=0,
        help="If set with --include_start_solution and not --start_keep_only_start: keep every Nth state along the solution path (0 -> use --sample_every).",
    )
    p.add_argument(
        "--sample_every",
        type=int,
        default=3,
        help="If not keep_only_start: keep every Nth state along Festival path.",
    )

    p.add_argument("--jobs", type=int, default=0, help="Processes for Festival labeling (0→cpu_count).")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)

    args = p.parse_args()

    try:
        festival_bin = find_festival_binary()
        if not festival_bin:
            raise RuntimeError("Festival binary not found at ./festival/festival")
    except Exception as e:
        raise SystemExit(
            "Festival is required for off-policy labeling but is not available.\n"
            f"Reason: {e}\n"
            "Fix: compile Festival (`make -C festival`) or run this script on Hydra where it exists."
        )

    if args.policy == "gnn" and not args.gnn_ckpt:
        raise SystemExit("policy=gnn requires --gnn_ckpt")

    level_ids = _read_list(args.list)
    if args.num_shards > 1:
        level_ids = [lid for i, lid in enumerate(level_ids) if i % args.num_shards == args.shard_idx]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.jobs in (0, 1):
        if args.policy == "hungarian":
            h_policy = get_heuristic("hungarian", use_dl=True)
        else:
            h_policy = GNNHeuristic(
                args.gnn_ckpt,
                mode=args.gnn_mode,
                use_deadlocks=True,
            )
    else:
        h_policy = None

    sampled: List[Tuple[str, Dict[str, int], bool]] = []  # (level_id, state_dict, is_offpolicy)
    seen_state_keys: set[str] = set()
    skipped_goal = 0
    sampling_jobs = args.jobs or cpu_count()
    if sampling_jobs <= 1:
        for lid in tqdm(level_ids, desc="Sampling states", unit="level"):
            s0 = load_level_by_id(lid)
            if is_goal(s0):
                skipped_goal += 1
                continue
            if args.include_start_solution:
                sd0 = state_to_dict(s0)
                k0 = lid + "|" + json.dumps(sd0, sort_keys=True)
                if k0 not in seen_state_keys:
                    seen_state_keys.add(k0)
                    sampled.append((lid, sd0, False))
            samples = _bounded_best_first_samples(
                s0,
                h_policy,
                sample_per_level=args.sample_per_level,
                node_limit=args.search_node_limit,
                time_limit_s=args.search_time_limit,
                min_g=args.min_g,
                frontier_per_level=args.frontier_per_level,
                seed=(args.seed ^ hash(lid)) & 0xFFFFFFFF,
            )
            for smp in samples:
                sd = state_to_dict(smp.state)
                k = lid + "|" + json.dumps(sd, sort_keys=True)
                if k in seen_state_keys:
                    continue
                seen_state_keys.add(k)
                sampled.append((lid, sd, True))
    else:
        payload1 = [
            (
                lid,
                int(args.sample_per_level),
                int(args.search_node_limit),
                float(args.search_time_limit),
                int(args.min_g),
                int(args.frontier_per_level),
                (args.seed ^ hash(lid)) & 0xFFFFFFFF,
            )
            for lid in level_ids
        ]
        pool1 = Pool(
            processes=sampling_jobs,
            initializer=_sampling_pool_init,
            initargs=(args.policy, args.gnn_ckpt, args.gnn_mode, False),
        )
        try:
            it1 = pool1.imap_unordered(_sample_one_level, payload1)
            for lid, state_dicts, skipped in tqdm(it1, total=len(payload1), desc="Sampling states", unit="level"):
                skipped_goal += int(skipped)
                if args.include_start_solution and not int(skipped):
                    s0 = load_level_by_id(lid)
                    sd0 = state_to_dict(s0)
                    k0 = lid + "|" + json.dumps(sd0, sort_keys=True)
                    if k0 not in seen_state_keys:
                        seen_state_keys.add(k0)
                        sampled.append((lid, sd0, False))
                for sd in state_dicts:
                    k = lid + "|" + json.dumps(sd, sort_keys=True)
                    if k in seen_state_keys:
                        continue
                    seen_state_keys.add(k)
                    sampled.append((lid, sd, True))
        finally:
            pool1.close()
            pool1.join()

    jobs = args.jobs or cpu_count()
    start_sample_every = int(args.start_sample_every) if int(args.start_sample_every) > 0 else int(args.sample_every)
    payload = []
    for (lid, sd, is_offpolicy) in sampled:
        if (not is_offpolicy) and args.include_start_solution:
            payload.append(
                (lid, sd, int(args.festival_timeout), int(start_sample_every), False)
            )
        else:
            payload.append((lid, sd, int(args.festival_timeout), int(args.sample_every), True))

    ok = 0
    failed = 0
    wrote = 0
    with open(args.out, "w", encoding="utf-8") as out:
        if jobs == 1:
            it: Iterable = map(_label_one_state, payload)
        else:
            pool = Pool(processes=jobs)
            it = pool.imap_unordered(_label_one_state, payload)
        try:
            for _, rec, status in tqdm(it, total=len(payload), desc="Festival labeling", unit="state"):
                if rec is None or not status.startswith("ok"):
                    failed += 1
                    continue
                ok += 1
                if isinstance(rec, dict) and "_records" in rec:
                    for r in rec["_records"]:
                        out.write(json.dumps(r) + "\n")
                        wrote += 1
                else:
                    out.write(json.dumps(rec) + "\n")
                    wrote += 1
                out.flush()
        finally:
            if jobs != 1:
                pool.close()
                pool.join()

    print(f"levels_in_shard: {len(level_ids)} (skipped_goal={skipped_goal})")
    print(f"sampled_unique_states: {len(sampled)} (sample_per_level={args.sample_per_level})")
    print(f"festival_ok: {ok} festival_failed: {failed}")
    print(f"wrote_records: {wrote} → {args.out}")


if __name__ == "__main__":
    main()


