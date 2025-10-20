from sokoban_core.parser import parse_level_str
from sokoban_core.deadlocks import (
    is_corner_deadlock,
    is_corridor_line_deadlock,
    is_2x2_deadlock,
    has_deadlock,
)


def test_corner_deadlock():
    lvl = """
#####
# $##
# @ #
#  .#
#####
"""
    s = parse_level_str(lvl)
    # box in the left upper corner (not goal) → deadlock
    # find the index of the box
    # check the general flag
    assert has_deadlock(s)


def test_corridor_line_deadlock():
    # Vertical corridor without goals on the line of the box
    lvl = """
######
# ####
# $  #
# ####
#  @ #
######
"""
    s = parse_level_str(lvl)
    assert has_deadlock(s)


def test_2x2_deadlock():
    lvl = """
#####
#$$ #
#@  #
# ..#
#####
"""
    s = parse_level_str(lvl)
    assert has_deadlock(s)