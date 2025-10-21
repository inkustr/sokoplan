from __future__ import annotations
from typing import Callable

from heuristics.classic import h_zero, h_manhattan_hungarian, h_with_deadlocks


def get_heuristic(name: str, use_dl: bool = False) -> Callable:
    name = name.lower()
    if name == "zero":
        base = h_zero
    elif name == "hungarian":
        base = h_manhattan_hungarian
    else:
        raise ValueError(f"unknown heuristic: {name}")
    if use_dl:
        return lambda s: h_with_deadlocks(s, base)
    return base