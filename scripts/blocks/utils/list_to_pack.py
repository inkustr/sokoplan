from __future__ import annotations

"""
Create a Sokoban .txt level pack from a .list of level_ids (path#idx).

Example:
  source .venv/bin/activate
  python -m scripts.blocks.utils.list_to_pack \
    --list sokoban_core/levels/pack_groups/lists/group_001.list \
    --out sokoban_core/levels/pack_groups/group_001.txt \
    --name_prefix group_001
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _split_on_blank_lines(text: str) -> List[str]:
    blocks: List[str] = []
    cur: List[str] = []
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


def _strip_leading_comment_lines(block: str) -> str:
    lines = [ln.rstrip("\n") for ln in block.splitlines()]
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith(";"):
        i += 1
    lines = lines[i:]
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines)


def _parse_level_id(level_id: str) -> Tuple[str, int]:
    if "#" not in level_id:
        return level_id, 0
    path, idx_s = level_id.rsplit("#", 1)
    try:
        idx = int(idx_s)
    except ValueError:
        idx = 0
    return path, idx


@dataclass(frozen=True)
class LevelRef:
    path: str
    idx: int


def _read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def _default_level_name(ref: LevelRef) -> str:
    # Use file stem + 1-based index to match common pack naming
    stem = os.path.splitext(os.path.basename(ref.path))[0]
    return f"{stem}_{ref.idx + 1}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name_prefix", default="", help="optional prefix for '; <name>' lines")
    args = p.parse_args()

    level_ids = _read_list(args.list)
    if not level_ids:
        raise SystemExit(f"Empty list: {args.list}")

    cache: Dict[str, List[str]] = {}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        for lid in level_ids:
            path, idx = _parse_level_id(lid)
            ref = LevelRef(path=path, idx=idx)

            if path not in cache:
                with open(path, "r", encoding="utf-8") as f:
                    cache[path] = _split_on_blank_lines(f.read())

            blocks = cache[path]
            if idx < 0 or idx >= len(blocks):
                raise SystemExit(f"Index {idx} out of range for {path} (total {len(blocks)})")

            body = _strip_leading_comment_lines(blocks[idx])
            if not body.strip():
                continue

            base_name = _default_level_name(ref)
            name = f"{args.name_prefix}_{base_name}" if args.name_prefix else base_name

            out_f.write(f"; {name}\n")
            out_f.write(body.rstrip("\n"))
            out_f.write("\n\n")
            written += 1

    print(f"wrote {written} levels → {args.out}")


if __name__ == "__main__":
    main()


