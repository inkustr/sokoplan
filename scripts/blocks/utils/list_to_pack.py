from __future__ import annotations

"""
Create a Sokoban .txt level pack from a .list of level_ids (path#idx).

Example:
  source .venv/bin/activate
  python -m scripts.blocks.utils.list_to_pack \
    --list sokoban_core/levels/pack_groups/lists/group_001.list \
    --out sokoban_core/levels/pack_groups/group_001.txt \
    --name_prefix group_001

Batch example (convert all lists in a folder):
  source .venv/bin/activate
  python -m scripts.blocks.utils.list_to_pack \
    --lists_dir sokoban_core/levels/pack_groups/lists \
    --out_dir sokoban_core/levels/pack_groups \
    --name_prefix_from_list_stem
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


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


@dataclass
class PackTextCache:
    blocks: List[str]
    # If the pack has '; name' headers ending with an integer, map that integer -> block index.
    # This helps recover from packs whose internal numbering has gaps (e.g. missing level 2).
    header_num_to_block_idx: Dict[int, int]


def _read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def _default_level_name(ref: LevelRef) -> str:
    # Use file stem + 1-based index to match common pack naming
    stem = os.path.splitext(os.path.basename(ref.path))[0]
    return f"{stem}_{ref.idx + 1}"


_TRAILING_INT_RE = re.compile(r"(\d+)\s*$")


def _build_header_index(blocks: List[str]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for i, b in enumerate(blocks):
        lines = b.splitlines()
        if not lines:
            continue
        first = lines[0].strip()
        if not first.startswith(";"):
            continue
        m = _TRAILING_INT_RE.search(first)
        if not m:
            continue
        try:
            num = int(m.group(1))
        except Exception:
            continue
        out[num] = i
    return out


def _split_pack_text(text: str) -> PackTextCache:
    blocks = _split_on_blank_lines(text)
    return PackTextCache(blocks=blocks, header_num_to_block_idx=_build_header_index(blocks))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--list", default="", help="Single .list file (path with level_ids).")
    p.add_argument("--out", default="", help="Output .txt pack path for --list mode.")
    p.add_argument("--lists_dir", default="", help="Folder containing many *.list files (batch mode).")
    p.add_argument("--out_dir", default="", help="Output folder for batch mode (one .txt per input .list).")
    p.add_argument("--name_prefix", default="", help="Optional prefix for '; <name>' lines (single mode).")
    p.add_argument(
        "--name_prefix_from_list_stem",
        action="store_true",
        help="Batch mode: use each list filename stem as the name prefix (e.g., group_001.list -> '; group_001_<level>').",
    )
    args = p.parse_args()

    single_mode = bool(args.list or args.out)
    batch_mode = bool(args.lists_dir or args.out_dir)
    if single_mode and batch_mode:
        raise SystemExit("Use either --list/--out (single) OR --lists_dir/--out_dir (batch), not both.")
    if not single_mode and not batch_mode:
        raise SystemExit("Provide --list/--out (single) OR --lists_dir/--out_dir (batch).")

    if single_mode:
        if not args.list or not args.out:
            raise SystemExit("Single mode requires both --list and --out.")
        jobs = [(args.list, args.out, args.name_prefix)]
    else:
        if not args.lists_dir or not args.out_dir:
            raise SystemExit("Batch mode requires both --lists_dir and --out_dir.")
        if not os.path.isdir(args.lists_dir):
            raise SystemExit(f"lists_dir not found: {args.lists_dir}")
        os.makedirs(args.out_dir, exist_ok=True)
        list_files = sorted(
            [
                os.path.join(args.lists_dir, fn)
                for fn in os.listdir(args.lists_dir)
                if fn.endswith(".list") and os.path.isfile(os.path.join(args.lists_dir, fn))
            ]
        )
        if not list_files:
            raise SystemExit(f"No .list files found in: {args.lists_dir}")
        jobs = []
        for lp in list_files:
            stem = os.path.splitext(os.path.basename(lp))[0]
            outp = os.path.join(args.out_dir, f"{stem}.txt")
            prefix = stem if args.name_prefix_from_list_stem else ""
            jobs.append((lp, outp, prefix))

    cache: Dict[str, PackTextCache] = {}

    total_written = 0
    packs_written = 0

    for list_path, out_path, name_prefix in jobs:
        level_ids = _read_list(list_path)
        if not level_ids:
            raise SystemExit(f"Empty list: {list_path}")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        written = 0

        with open(out_path, "w", encoding="utf-8") as out_f:
            for lid in level_ids:
                path, idx = _parse_level_id(lid)
                ref = LevelRef(path=path, idx=idx)

                if path not in cache:
                    with open(path, "r", encoding="utf-8") as f:
                        cache[path] = _split_pack_text(f.read())

                pack_cache = cache[path]
                blocks = pack_cache.blocks
                if idx < 0 or idx >= len(blocks):
                    # Recovery: some packs have header numbering gaps (e.g. levels 1,3,4,...),
                    # and some list files may reference the header number instead of the 0-based block index.
                    recovered: Optional[int] = None
                    # Common case: list uses 0-based "idx", header uses 1-based number.
                    recovered = pack_cache.header_num_to_block_idx.get(idx + 1)
                    # Fallback: list might already be 1-based.
                    if recovered is None:
                        recovered = pack_cache.header_num_to_block_idx.get(idx)
                    if recovered is None:
                        raise SystemExit(f"Index {idx} out of range for {path} (total {len(blocks)})")
                    idx = recovered
                    ref = LevelRef(path=path, idx=idx)

                body = _strip_leading_comment_lines(blocks[idx])
                if not body.strip():
                    continue

                base_name = _default_level_name(ref)
                name = f"{name_prefix}_{base_name}" if name_prefix else base_name

                out_f.write(f"; {name}\n")
                out_f.write(body.rstrip("\n"))
                out_f.write("\n\n")
                written += 1

        print(f"{os.path.basename(list_path)}: wrote {written} levels → {out_path}", flush=True)
        total_written += written
        packs_written += 1

    if packs_written > 1:
        print(f"TOTAL: packs={packs_written} levels={total_written} → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()


