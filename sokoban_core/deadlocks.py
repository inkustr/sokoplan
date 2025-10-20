from __future__ import annotations
from typing import Iterable, Tuple, Dict, Tuple as Tup
from collections import OrderedDict

from .state import State, has_bit
from .moves import iter_bits

INF = 10 ** 9

# Simple LRU cache for deadlock checks keyed by (boxes, walls, goals, w, h)
_DEADLOCK_CACHE: "OrderedDict[Tup[int, int, int, int, int], bool]" = OrderedDict()
_DEADLOCK_CACHE_MAX = 200000

# --- low-level helpers -------------------------------------------------------

def _is_wall_like(state: State, idx: int) -> bool:
    """Treat outside the level as a wall for deadlock checks."""
    return (not state.is_inside(idx)) or state.is_wall(idx)


def _rc(idx: int, w: int) -> Tuple[int, int]:
    return idx // w, idx % w


def _in_row_same(r: int, idx: int, w: int) -> bool:
    return (idx // w) == r

# --- individual deadlock rules ----------------------------------------------

def is_corner_deadlock(state: State, box_idx: int) -> bool:
    """Box (not on goal) in a corner of two walls/outside the level."""
    if state.is_goal_cell(box_idx):
        return False
    w = state.width
    h = state.height
    size = w * h
    up = box_idx - w if box_idx >= w else -1
    down = box_idx + w if box_idx + w < size else -1
    left = box_idx - 1 if box_idx % w != 0 else -1
    right = box_idx + 1 if box_idx % w != w - 1 else -1

    def wall(i: int) -> bool:
        return i == -1 or _is_wall_like(state, i)

    return (wall(up) and wall(left)) or (wall(up) and wall(right)) \
        or (wall(down) and wall(left)) or (wall(down) and wall(right))


def is_corridor_line_deadlock(state: State, box_idx: int) -> bool:
    """Box is squeezed in a vertical or horizontal corridor without goals along the line.

    Idea: if there is a wall/outside the level above and below, it's a vertical corridor,
    scan horizontally to the nearest walls and check if there is a goal on this segment.
    Similarly for horizontal corridor (left/right walls) and scan vertically.
    If there are no goal cells on the line → deadlock.
    """
    if state.is_goal_cell(box_idx):
        return False
    w = state.width
    h = state.height
    size = w * h

    up = box_idx - w if box_idx >= w else -1
    down = box_idx + w if box_idx + w < size else -1
    left = box_idx - 1 if box_idx % w != 0 else -1
    right = box_idx + 1 if box_idx % w != w - 1 else -1

    vert_blocked = _is_wall_like(state, up) and _is_wall_like(state, down)
    if vert_blocked:
        # scan horizontally from this row between nearest walls
        # left bound
        l = box_idx
        while l % w != 0 and not _is_wall_like(state, l - 1):
            l -= 1
        # right bound
        r = box_idx
        while (r % w) != (w - 1) and not _is_wall_like(state, r + 1):
            r += 1
        row = box_idx // w
        corridor_ok = True
        c = l
        while _in_row_same(row, c, w) and c <= r:
            cu = c - w if c >= w else -1
            cd = c + w if c + w < size else -1
            if not (_is_wall_like(state, cu) and _is_wall_like(state, cd)):
                corridor_ok = False
                break
            c += 1
        if corridor_ok:
            # check goals on [l..r]
            has_goal_on_line = False
            c = l
            while _in_row_same(row, c, w) and c <= r:
                if state.is_goal_cell(c):
                    has_goal_on_line = True
                    break
                c += 1
            if not has_goal_on_line:
                return True

    horiz_blocked = _is_wall_like(state, left) and _is_wall_like(state, right)
    if horiz_blocked:
        # scan vertically between nearest walls
        # up bound
        u = box_idx
        while u >= w and not _is_wall_like(state, u - w):
            u -= w
        # down bound
        d = box_idx
        while d + w < size and not _is_wall_like(state, d + w):
            d += w
        # ensure corridor property holds across entire [u..d]: left and right are walls
        corridor_ok = True
        c = u
        while c <= d:
            cl = c - 1 if (c % w) != 0 else -1
            cr = c + 1 if (c % w) != (w - 1) else -1
            if not (_is_wall_like(state, cl) and _is_wall_like(state, cr)):
                corridor_ok = False
                break
            c += w
        if corridor_ok:
            # check goals on [u..d] by step w
            has_goal_on_line = False
            c = u
            while c <= d:
                if state.is_goal_cell(c):
                    has_goal_on_line = True
                    break
                c += w
            if not has_goal_on_line:
                return True

    return False


def is_2x2_deadlock(state: State, box_idx: int) -> bool:
    """If the box participates in a 2x2 block where all cells are walls/boxes and
    at least one box in this block is not on the goal, then it's a deadlock.
    (Classic rule: 2x2 without goal is unsolvable).
    """
    w = state.width
    h = state.height
    r, c = _rc(box_idx, w)

    # four possible 2x2 blocks around the cell
    candidates = []
    if r > 0 and c > 0:
        candidates.append((box_idx - w - 1, box_idx - w, box_idx - 1, box_idx))
    if r > 0 and c + 1 < w:
        candidates.append((box_idx - w, box_idx - w + 1, box_idx, box_idx + 1))
    if r + 1 < h and c > 0:
        candidates.append((box_idx - 1, box_idx, box_idx + w - 1, box_idx + w))
    if r + 1 < h and c + 1 < w:
        candidates.append((box_idx, box_idx + 1, box_idx + w, box_idx + w + 1))

    for a, b, c2, d in candidates:
        cells = (a, b, c2, d)
        if any(not state.is_inside(x) for x in cells):
            continue
        # are all cells occupied by walls/boxes?
        solid = 0
        boxes = 0
        goal_on_any_box = False
        for x in cells:
            if state.is_wall(x) or state.has_box(x):
                solid += 1
            if state.has_box(x):
                boxes += 1
                if state.is_goal_cell(x):
                    goal_on_any_box = True
        if solid == 4 and boxes >= 1 and not goal_on_any_box:
            return True
    return False


# --- combined API ------------------------------------------------------------

def has_deadlock(state: State) -> bool:
    """Combines several simple deadlock rules."""
    key = (state.boxes, state.walls, state.goals, state.width, state.height)
    cached = _DEADLOCK_CACHE.get(key)
    if cached is not None:
        _DEADLOCK_CACHE.move_to_end(key)
        return cached

    deadlocked = False
    for b in iter_bits(state.boxes):
        if state.is_goal_cell(b):
            continue
        if is_corner_deadlock(state, b) or \
           is_corridor_line_deadlock(state, b) or \
           is_2x2_deadlock(state, b):
            deadlocked = True
            break

    _DEADLOCK_CACHE[key] = deadlocked
    if len(_DEADLOCK_CACHE) > _DEADLOCK_CACHE_MAX:
        _DEADLOCK_CACHE.popitem(last=False)
    return deadlocked


# --- score/penalty -------------------------------------------------

def deadlock_penalty(state: State, base_h: int) -> int:
    """Returns INF if there is a deadlock, otherwise base_h — convenient for wrapping the heuristic."""
    return INF if has_deadlock(state) else base_h

