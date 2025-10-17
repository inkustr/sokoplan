# --- file: sokoban_core/levels/resolve.py
from __future__ import annotations
from typing import Tuple

from ..parser import parse_level_str


def parse_level_id(level_id: str) -> Tuple[str, int]:
    """Parses a string of the form "path/to/file.txt#3" into (path, index)."""
    if "#" not in level_id:
        return level_id, 0
    path, idx = level_id.rsplit("#", 1)
    try:
        k = int(idx)
    except ValueError:
        k = 0
    return path, k


def _split_on_blank_lines(text: str):
    blocks = []
    cur = []
    for line in text.splitlines():
        if line.strip() == "":
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line.rstrip("\n"))
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def load_level_by_id(level_id: str):
    """Loads a SPECIFIC level file#idx even if the file contains dozens of levels.

    Does not depend on directory traversal: open the file directly and take the desired block by index.
    """
    path, wanted = parse_level_id(level_id)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = _split_on_blank_lines(content)
    if not blocks:
        raise ValueError(f"No levels found in {path}")
    if wanted < 0 or wanted >= len(blocks):
        raise IndexError(f"Index {wanted} out of range for {path} (total {len(blocks)})")
    return parse_level_str(blocks[wanted])
