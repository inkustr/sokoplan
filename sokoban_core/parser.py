from typing import List
from .state import State, set_bit

TOK_WALL = "#"
TOK_GOAL = "."
TOK_BOX = "$"
TOK_BOX_ON_GOAL = "*"
TOK_PLAYER = "@"
TOK_PLAYER_ON_GOAL = "+"
TOK_VOID = " "


def parse_level_str(level_str: str) -> State:
    """Parses ASCII level into State.

    Supported characters:
      '#': wall
      '.': goal
      '$': box
      '*': box on goal
      '@': player
      '+': player on goal
      ' ' (space): void/outside the level
    Other characters are treated as floor (nothing).
    """
    lines = [line.rstrip("\n") for line in level_str.splitlines() if line.strip() != ""]
    if not lines:
        raise ValueError("Empty level")
    height = len(lines)
    width = max(len(line) for line in lines)
    # align lines with space on the right
    lines = [line.ljust(width, TOK_VOID) for line in lines]

    walls = goals = boxes = 0
    player_idx = -1
    board_mask = 0

    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            idx = r * width + c
            board_mask = set_bit(board_mask, idx)
            
            if ch == TOK_WALL:
                walls = set_bit(walls, idx)
            elif ch == TOK_GOAL:
                goals = set_bit(goals, idx)
            elif ch == TOK_BOX:
                boxes = set_bit(boxes, idx)
            elif ch == TOK_BOX_ON_GOAL:
                boxes = set_bit(boxes, idx)
                goals = set_bit(goals, idx)
            elif ch == TOK_PLAYER:
                player_idx = idx
            elif ch == TOK_PLAYER_ON_GOAL:
                player_idx = idx
                goals = set_bit(goals, idx)

    if player_idx == -1:
        raise ValueError("No player '@' or '+' found in level")

    return State(width=width, height=height, walls=walls, goals=goals,
                 boxes=boxes, player=player_idx, board_mask=board_mask)


def parse_level_file(path: str) -> State:
    with open(path, "r", encoding="utf-8") as f:
        return parse_level_str(f.read())