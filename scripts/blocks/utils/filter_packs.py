from __future__ import annotations

"""
Filter Sokoban packs into per-pack files with valid levels only.

Filters:
  - constraints from configs/data.yaml (filters section):
      max_width, max_height, min_boxes, max_boxes
  - #boxes == #goals

Output:
  - out_dir/<pack>.txt              levels separated by blank lines, each preceded by "; <name>"
  - out_dir/lists/<pack>.list       level ids: out_dir/<pack>.txt#idx

Example usage:
  source .venv/bin/activate
  python -m scripts.blocks.utils.filter_packs \
    --config configs/data.yaml \
    --pack_subdir windows \
    --out_dir sokoban_core/levels/packs_filtered \
    --min_levels 20
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import yaml

from sokoban_core.levels.io import iterate_level_strings, filter_level
from sokoban_core.parser import parse_level_str


def _strip_comment_lines(level_str: str) -> str:
    lines: List[str] = []
    for ln in level_str.splitlines():
        if ln.lstrip().startswith(";"):
            continue
        if ln.strip() == "":
            continue
        lines.append(ln.rstrip("\n"))
    return "\n".join(lines)


def _boxes_goals_match(level_str: str) -> bool:
    s = parse_level_str(level_str)
    return int(s.boxes.bit_count()), int(s.goals.bit_count())


def _read_filters(config_path: str) -> Tuple[str, Dict[str, Optional[int]]]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root_dir = str(cfg["levels"]["root_dir"])
    flt = cfg.get("filters", {})
    return root_dir, {
        "max_w": flt.get("max_width"),
        "max_h": flt.get("max_height"),
        "min_b": flt.get("min_boxes"),
        "max_b": flt.get("max_boxes"),
    }


def _write_pack(out_path: str, pack_name: str, levels: List[str]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, lvl in enumerate(levels):
            f.write(f"; {pack_name}_{i+1}\n")
            f.write(lvl.rstrip("\n"))
            f.write("\n\n")


def _write_list(list_path: str, pack_txt_path: str, n: int) -> None:
    os.makedirs(os.path.dirname(list_path), exist_ok=True)
    with open(list_path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(f"{pack_txt_path}#{i}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--pack_subdir", default="windows")
    p.add_argument("--out_dir", default="sokoban_core/levels/packs_filtered")
    p.add_argument(
        "--min_levels",
        type=int,
        default=20,
        help="Skip writing packs that end up with fewer than this many valid levels after filtering.",
    )
    args = p.parse_args()

    root_dir, flt = _read_filters(args.config)

    pack_root = os.path.join(root_dir, args.pack_subdir)
    if not os.path.isdir(pack_root):
        raise SystemExit(f"pack dir not found: {pack_root}")

    pack_dirs = [
        os.path.join(args.pack_subdir, d)
        for d in sorted(os.listdir(pack_root))
        if os.path.isdir(os.path.join(pack_root, d))
    ]
    if not pack_dirs:
        raise SystemExit(f"No pack directories under {pack_root}")

    out_lists_dir = os.path.join(args.out_dir, "lists")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(out_lists_dir, exist_ok=True)

    total_in = 0
    total_out = 0
    packs_written = 0
    packs_skipped_small = 0

    for rel_pack_dir in pack_dirs:
        pack_name = os.path.basename(rel_pack_dir)
        valid_levels: List[str] = []
        seen_files: set[str] = set()

        for ref, raw in iterate_level_strings(root_dir, [rel_pack_dir]):
            seen_files.add(ref.path)
            total_in += 1

            lvl = _strip_comment_lines(raw)
            if not filter_level(
                lvl,
                max_w=flt["max_w"],
                max_h=flt["max_h"],
                min_b=flt["min_b"],
                max_b=flt["max_b"],
            ):
                continue

            if not _boxes_goals_match(lvl):
                continue

            valid_levels.append(lvl)

        if not valid_levels:
            continue
        if len(valid_levels) < int(args.min_levels):
            packs_skipped_small += 1
            continue

        out_pack_path = os.path.join(args.out_dir, f"{pack_name}.txt")
        _write_pack(out_pack_path, pack_name, valid_levels)
        _write_list(os.path.join(out_lists_dir, f"{pack_name}.list"), out_pack_path, len(valid_levels))

        packs_written += 1
        total_out += len(valid_levels)
        print(f"[pack] {pack_name}: kept {len(valid_levels)} levels → {out_pack_path}")

    print("\nDONE")
    print(f"packs_written: {packs_written}")
    print(f"packs_skipped_small: {packs_skipped_small}")
    print(f"levels_in: {total_in}")
    print(f"levels_out: {total_out}")


if __name__ == "__main__":
    main()


