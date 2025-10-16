from __future__ import annotations
import argparse

from sokoban_core.parser import parse_level_file, parse_level_str
from sokoban_core.render import render_ascii
from sokoban_core.zobrist import Zobrist
from search.astar import astar
from search.transposition import Transposition
from heuristics.classic import h_zero, h_manhattan_hungarian, h_with_deadlocks
from sokoban_core.goal_check import is_goal

LVL = """
#####
#.@ #
# $ #
# . #
#####
"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=str, default="inline", help="path to .txt level or 'inline'")
    p.add_argument("--h", type=str, default="hungarian", choices=["zero", "hungarian", "hungarian+dl"], help="heuristic")
    args = p.parse_args()

    if args.level == "inline":
        s = parse_level_str(LVL)
    else:
        s = parse_level_file(args.level)

    zob = Zobrist(s.width, s.height, s.board_mask)
    trans = Transposition(zob)

    if args.h == "zero":
        h = h_zero
    elif args.h == "hungarian":
        h = h_manhattan_hungarian
    else:
        h = lambda st: h_with_deadlocks(st, h_manhattan_hungarian)

    res = astar(s, h, is_goal, trans=trans, time_limit_s=5.0, node_limit=100000)
    print("Result:", {k: v for k, v in res.items() if k != "path"})
    if res.get("success"):
        from typing import List
        path = res["path"]  # type: ignore
        for i, st in enumerate(path):
            print(f"\n-- step {i} --\n{render_ascii(st)}")

if __name__ == "__main__":
    main()