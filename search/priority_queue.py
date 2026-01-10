from __future__ import annotations
import heapq
from typing import Any, List, Tuple

class PriorityQueue:
    def __init__(self) -> None:
        self._h: List[Tuple[float, int, Any]] = []
        self._tiebreak = 0

    def push(self, priority: float, item: Any) -> None:
        self._tiebreak += 1
        heapq.heappush(self._h, (priority, self._tiebreak, item))

    def pop(self) -> Any:
        return heapq.heappop(self._h)[2]

    def __len__(self) -> int:
        return len(self._h)
