import pytest
from sokoban_core.parser import parse_level_str
from sokoban_core.state import State
from sokoban_core.render import render_ascii

LVL = """
#####
#.@ #
# $ #
# . #
#####
"""

def test_parse_and_render_basic():
    s = parse_level_str(LVL)
    txt = render_ascii(s)
    assert txt.splitlines()[0] == "#####"
    assert s.width == 5 and s.height == 5
    assert s.player >= 0
    # the goal is present
    assert any(c == '.' for c in txt)