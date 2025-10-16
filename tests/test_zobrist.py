from sokoban_core.parser import parse_level_str
from sokoban_core.zobrist import Zobrist
from sokoban_core.moves import successors_pushes

LVL = """
#####
#.@ #
# $ #
# . #
#####
"""

def test_zobrist_changes_on_push():
    s = parse_level_str(LVL)
    zob = Zobrist(s.width, s.height, s.board_mask)
    h0 = zob.hash(s)
    ns = successors_pushes(s)[0]
    h1 = zob.hash(ns)
    assert h0 != h1