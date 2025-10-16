from .state import State, has_bit


def render_ascii(state: State) -> str:
    """ASCII visualization of the state."""
    out_lines = []
    for r in range(state.height):
        row_chars = []
        for c in range(state.width):
            idx = r * state.width + c
            if not state.is_inside(idx):
                row_chars.append(' ')
                continue
            if state.is_wall(idx):
                row_chars.append('#')
                continue
            has_goal = state.is_goal_cell(idx)
            has_box = state.has_box(idx)
            if idx == state.player:
                row_chars.append('+' if has_goal else '@')
            elif has_box:
                row_chars.append('*' if has_goal else '$')
            else:
                row_chars.append('.' if has_goal else ' ')
        out_lines.append(''.join(row_chars))
    return "\n".join(out_lines)

