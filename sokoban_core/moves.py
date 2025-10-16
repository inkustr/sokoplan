from collections import deque
from typing import Iterable, List
from .state import State, set_bit, has_bit, clear_bit


def iter_bits(mask: int) -> Iterable[int]:
    """Iterates over the indices of set bits."""
    idx = 0
    m = mask
    while m:
        if m & 1:
            yield idx
        m >>= 1
        idx += 1


def player_reachable(state: State) -> int:
    """Returns the bitmask of cells reachable by the player without pushing boxes."""
    start = state.player
    visited = 0
    q = deque([start])
    visited = set_bit(visited, start)

    while q:
        cur = q.popleft()
        for nb in state.neighbors(cur):
            if not state.is_inside(nb):
                continue
            if state.is_wall(nb) or state.has_box(nb):
                continue
            if not has_bit(visited, nb):
                visited = set_bit(visited, nb)
                q.append(nb)
    return visited


def successors_pushes(state: State) -> List[State]:
    """Generates only states after *pushing boxes* (cost of step = 1 push).

    Algorithm:
      1) find cells reachable by the player without pushing boxes (BFS),
      2) for each box check 4 directions: if the cell behind the box is reachable,
         and the cell in front of the box is free → push is possible.
      3) after pushing the player is on the position of the box before pushing.
    """
    reachable = player_reachable(state)
    succs: List[State] = []

    for b in iter_bits(state.boxes):
        for nb in state.neighbors(b):
            back = 2 * b - nb  # cell behind the push direction
            if back < 0 or back >= state.width * state.height:
                continue
            # the cell in front of the box (nb) must be free; the cell behind (back) must be able to stand the player
            if state.is_free_floor(nb) and has_bit(reachable, back):
                new_boxes = clear_bit(state.boxes, b)
                new_boxes = set_bit(new_boxes, nb)
                succs.append(State(
                    width=state.width,
                    height=state.height,
                    walls=state.walls,
                    goals=state.goals,
                    boxes=new_boxes,
                    player=b,            # the player is on the old position of the box
                    board_mask=state.board_mask,
                ))
    return succs


# ---- simple deadlocks

def _wall_like(state: State, idx: int) -> bool:
    """Wall/outside the level is considered a wall for checking corners."""
    return not state.is_inside(idx) or state.is_wall(idx)


def _is_corner_cell(state: State, idx: int) -> bool:
    w = state.width
    size = state.width * state.height
    up = idx - w if idx >= w else None
    down = idx + w if idx + w < size else None
    left = idx - 1 if idx % w != 0 else None
    right = idx + 1 if idx % w != w - 1 else None

    def wall(i):
        return _wall_like(state, i) if i is not None else True

    return (wall(up) and wall(left)) or (wall(up) and wall(right)) \
        or (wall(down) and wall(left)) or (wall(down) and wall(right))


def has_simple_deadlock(state: State) -> bool:
    """True if any box (not on goal) is in a corner of two walls."""
    for b in iter_bits(state.boxes):
        if not state.is_goal_cell(b) and _is_corner_cell(state, b):
            return True
    return False