from sokoban_core.parser import parse_level_str
from sokoban_core.goal_check import is_goal
from heuristics.classic import h_zero, h_manhattan_hungarian
from search.astar import astar
from sokoban_core.zobrist import Zobrist
from search.transposition import Transposition

LVL = """
#####
#.@ #
# $ #
# . #
#####
"""

def test_astar_solve_simple():
    s = parse_level_str(LVL)
    trans = Transposition(Zobrist(s.width, s.height, s.board_mask))
    res = astar(s, h_manhattan_hungarian, is_goal, trans=trans)
    assert res["success"] is True
    assert res["solution_len"] == 1


def test_astar_zero_heuristic_works():
    s = parse_level_str(LVL)
    trans = Transposition(Zobrist(s.width, s.height, s.board_mask))
    res = astar(s, h_zero, is_goal, trans=trans)
    assert res["success"] is True