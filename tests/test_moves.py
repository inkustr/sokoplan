from sokoban_core.parser import parse_level_str
from sokoban_core.moves import successors_pushes, has_simple_deadlock, player_reachable

LVL = """
#####
#.@ #
# $ #
# . #
#####
"""

def test_successors_single_push():
    s = parse_level_str(LVL)
    succs = successors_pushes(s)
    assert len(succs) == 1
    ns = succs[0]
    # the box should be on the goal
    assert ns.is_goal()


def test_reachability_non_empty():
    s = parse_level_str(LVL)
    reach = player_reachable(s)
    assert reach != 0


def test_deadlock_basic_false():
    s = parse_level_str(LVL)
    assert has_simple_deadlock(s) is False