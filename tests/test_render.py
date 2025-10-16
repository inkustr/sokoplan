"""Tests for the render module."""

from sokoban_core.parser import parse_level_str
from sokoban_core.render import render_ascii
from sokoban_core.moves import successors_pushes, has_simple_deadlock


def test_render_basic_level():
    """Test rendering a basic Sokoban level."""
    lvl = """
#####
#.@ #
# $ #
# . #
#####
"""
    s = parse_level_str(lvl)
    rendered = render_ascii(s)
    
    # Basic sanity check - should contain expected characters
    assert '@' in rendered  # player
    assert '$' in rendered  # box
    assert '.' in rendered  # goal
    assert '#' in rendered  # wall
    
    print("Rendered level:")
    print(rendered)


def test_successors_and_deadlock():
    """Test successor generation and deadlock detection."""
    lvl = """
#####
#.@ #
# $ #
# . #
#####
"""
    s = parse_level_str(lvl)
    
    # Test successor generation
    succs = successors_pushes(s)
    print(f"Number of successors: {len(succs)}")
    
    for i, ns in enumerate(succs):
        print(f"\nSuccessor {i + 1}:")
        print(render_ascii(ns))
    
    # Test deadlock detection
    has_deadlock = has_simple_deadlock(s)
    print(f"Has deadlock: {has_deadlock}")
    
    # Basic assertions
    assert isinstance(succs, list)
    assert isinstance(has_deadlock, bool)


if __name__ == "__main__":
    test_render_basic_level()
    test_successors_and_deadlock()
