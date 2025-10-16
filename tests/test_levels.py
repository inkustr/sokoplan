from sokoban_core.levels.io import iterate_level_strings


def test_examples_iterate():
    pairs = list(iterate_level_strings("sokoban_core/levels", ["examples"]))
    # Two files, one level each
    assert len(pairs) >= 2
    (ref0, s0) = pairs[0]
    assert "@" in s0 and "$" in s0
