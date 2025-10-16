from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List, Tuple, Optional
import os

from sokoban_core.parser import parse_level_str

@dataclass
class LevelRef:
    path: str
    index: int  # index of the level inside the file (if there are multiple levels)


def _split_on_blank_lines(text: str) -> List[str]:
    blocks: List[str] = []
    cur: List[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def iterate_level_strings(root_dir: str, rel_dirs: List[str]) -> Iterator[Tuple[LevelRef, str]]:
    """Iterate over all .txt in the given subfolders and return (level reference, level string)."""
    for rel in rel_dirs:
        abs_dir = os.path.join(root_dir, rel)
        if not os.path.isdir(abs_dir):
            continue
        for fname in sorted(os.listdir(abs_dir)):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(abs_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            blocks = _split_on_blank_lines(content)
            if not blocks:
                continue
            for i, block in enumerate(blocks):
                yield LevelRef(path=fpath, index=i), block


def count_boxes(level_str: str) -> int:
    return sum(1 for ch in level_str if ch == '$' or ch == '*')


def dims(level_str: str) -> Tuple[int, int]:
    lines = [ln for ln in level_str.splitlines() if ln.strip() != ""]
    h = len(lines)
    w = max((len(ln) for ln in lines), default=0)
    return w, h


def filter_level(level_str: str, *, max_w: Optional[int], max_h: Optional[int], min_b: Optional[int], max_b: Optional[int]) -> bool:
    w, h = dims(level_str)
    b = count_boxes(level_str)
    if max_w is not None and w > max_w: return False
    if max_h is not None and h > max_h: return False
    if min_b is not None and b < min_b: return False
    if max_b is not None and b > max_b: return False
    # check if it parses
    try:
        _ = parse_level_str(level_str)
    except Exception:
        return False
    return True