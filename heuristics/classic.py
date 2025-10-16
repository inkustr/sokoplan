from __future__ import annotations
from typing import List, Tuple, Callable
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from sokoban_core.state import State
from sokoban_core.moves import iter_bits, has_simple_deadlock


# ---- helpers

def idx_to_rc(idx: int, width: int) -> Tuple[int, int]:
    return (idx // width, idx % width)


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _positions(mask: int, width: int) -> List[Tuple[int, int]]:
    pos: List[Tuple[int, int]] = []
    i = 0
    m = mask
    while m:
        if m & 1:
            pos.append(idx_to_rc(i, width))
        m >>= 1
        i += 1
    return pos


# ---- classical heuristics

def h_zero(state: State) -> int:
    return 0


def h_manhattan_hungarian(state: State) -> int:
    """Cost = optimal matching of boxes → goals by Manhattan distance.
    Walls are not considered (this is a valid lower bound)."""
    boxes = _positions(state.boxes, state.width)
    goals = _positions(state.goals, state.width)
    if not boxes:
        return 0
    n = len(goals)
    boxes = boxes[:n]
    goals = goals[:n]
    if n == 0:
        return 0

    C = np.empty((n, n), dtype=np.int32)
    for i, b in enumerate(boxes):
        for j, g in enumerate(goals):
            C[i, j] = manhattan(b, g)
    r, c = linear_sum_assignment(C)
    return int(C[r, c].sum())


def h_with_deadlocks(state: State, base_h: Callable[[State], int] = h_manhattan_hungarian) -> int:
    if has_simple_deadlock(state):
        return 10 ** 9  # huge penalty → A* won't go here
    return base_h(state)
