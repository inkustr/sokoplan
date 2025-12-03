import random
from typing import Tuple
from .state import State

class Zobrist:
    """Zobrist hash for Sokoban state.

    For a fixed field size, we generate:
      - key for the player position on each cell inside board_mask
      - key for the presence of a box on each cell inside board_mask
    Walls/goals are not included in the hash (they are fixed for the level).
    """
    def __init__(self, width: int, height: int, board_mask: int, seed: int = 12345) -> None:
        rng = random.Random(seed)
        self.width = width
        self.height = height
        size = width * height
        self.player_keys = [0] * size
        self.box_keys = [0] * size
        for idx in range(size):
            if (board_mask >> idx) & 1:
                self.player_keys[idx] = rng.getrandbits(64)
                self.box_keys[idx] = rng.getrandbits(64)

    def hash(self, s: State) -> int:
        h = 0
        boxes = s.boxes
        idx = 0
        while boxes:
            if boxes & 1:
                h ^= self.box_keys[idx]
            boxes >>= 1
            idx += 1
        h ^= self.player_keys[s.player]
        return h

