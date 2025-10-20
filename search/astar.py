from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple, List
import time

from sokoban_core.state import State
from sokoban_core.moves import successors_pushes
from .priority_queue import PriorityQueue
from .transposition import Transposition

Result = Dict[str, object]


def reconstruct(parent: Dict[State, Optional[State]], goal: State) -> List[State]:
    path = [goal]
    cur = goal
    while parent[cur] is not None:
        cur = parent[cur]  # type: ignore
        path.append(cur)
    path.reverse()
    return path


def astar(
    start: State,
    h_fn: Callable[[State], int],
    is_goal_fn: Callable[[State], bool],
    succ_fn: Callable[[State], List[State]] = successors_pushes,
    trans: Optional[Transposition] = None,
    time_limit_s: Optional[float] = None,
    node_limit: Optional[int] = None,
) -> Result:
    t0 = time.time()
    openq = PriorityQueue()
    g: Dict[State, int] = {start: 0}
    parent: Dict[State, Optional[State]] = {start: None}
    h0 = h_fn(start)
    if h0 >= 10 ** 9:
        # start is already in deadlock
        runtime = time.time() - t0
        return {"success": False, "nodes": 0, "runtime": runtime}
    openq.push(h0, start)

    expanded = 0
    found: Optional[State] = None

    while len(openq) > 0:
        if time_limit_s is not None and (time.time() - t0) > time_limit_s:
            break
        s: State = openq.pop()
        if is_goal_fn(s):
            found = s
            break
        expanded += 1
        if node_limit is not None and expanded >= node_limit:
            break

        gs = g[s]
        for ns in succ_fn(s):
            ng = gs + 1
            # transposition pruning
            if trans is not None and trans.seen_better(ns, ng):
                continue
            if ns not in g or ng < g[ns]:
                g[ns] = ng
                parent[ns] = s
                hn = h_fn(ns)
                if hn < 10 ** 9:
                    openq.push(ng + hn, ns)

    runtime = time.time() - t0
    if found is None:
        return {"success": False, "nodes": expanded, "runtime": runtime}
    path = reconstruct(parent, found)
    return {
        "success": True,
        "nodes": expanded,
        "runtime": runtime,
        "solution_len": len(path) - 1,
        "path": path,
    }