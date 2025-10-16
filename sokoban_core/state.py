from dataclasses import dataclass
from typing import Iterable, Tuple

# Bit helpers
__all__ = [
    "State",
    "bit",
    "has_bit",
    "set_bit",
    "clear_bit",
]

def bit(idx: int) -> int:
    return 1 << idx

def has_bit(mask: int, idx: int) -> bool:
    return (mask >> idx) & 1 == 1

def set_bit(mask: int, idx: int) -> int:
    return mask | bit(idx)

def clear_bit(mask: int, idx: int) -> int:
    return mask & ~bit(idx)


@dataclass(frozen=True, slots=True)
class State:
    """
    Immutable representation of Sokoban's state.

    Stores the field as bitmaps: walls, goals, boxes, player position.
    Cell indexing: idx = r*width + c.
    board_mask: bits for all cells inside the level (not “void”).
    """

    width: int
    height: int
    walls: int # bitset
    goals: int # bitset
    boxes: int # bitset
    player: int # index (r*W + c)
    board_mask: int # bitset: which cells belong to the level (not “void”)


    # ---- state properties
    def is_goal(self) -> bool:
        """All boxes are on goals: boxes ⊆ goals."""
        return (self.boxes & ~self.goals) == 0


    # ---- convenient checks/conversions
    def idx_to_rc(self, idx: int) -> Tuple[int, int]:
        return (idx // self.width, idx % self.width)


    def rc_to_idx(self, r: int, c: int) -> int:
        return r * self.width + c


    def is_wall(self, idx: int) -> bool:
        return has_bit(self.walls, idx)


    def is_goal_cell(self, idx: int) -> bool:
        return has_bit(self.goals, idx)


    def has_box(self, idx: int) -> bool:
        return has_bit(self.boxes, idx)


    def is_inside(self, idx: int) -> bool:
        return has_bit(self.board_mask, idx)


    def is_free_floor(self, idx: int) -> bool:
        """Cell inside the level, not a wall, not a box."""
        return self.is_inside(idx) and not self.is_wall(idx) and not self.has_box(idx)


    def neighbors(self, idx: int) -> Iterable[int]:
        """4-neighborhood without diagonals."""
        w = self.width
        h = self.height
        r, c = self.idx_to_rc(idx)
        if r > 0: yield idx - w
        if r + 1 < h: yield idx + w
        if c > 0: yield idx - 1
        if c + 1 < w: yield idx + 1