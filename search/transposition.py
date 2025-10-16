from __future__ import annotations
from typing import Dict, Optional
from sokoban_core.state import State
from sokoban_core.zobrist import Zobrist

class Transposition:
    """Store the best known g(s) by Zobrist hash."""
    def __init__(self, zobrist: Zobrist) -> None:
        self.z = zobrist
        self.best_g: Dict[int, int] = {}

    def seen_better(self, s: State, g: int) -> bool:
        key = self.z.hash(s)
        old = self.best_g.get(key)
        if old is None or g < old:
            self.best_g[key] = g
            return False
        return True