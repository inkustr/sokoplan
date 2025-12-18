from __future__ import annotations

"""
Create a level-id list (.list) from a labels JSONL file.

Run:
  source .venv/bin/activate
  python -m scripts.blocks.utils.labels_to_list \
    --labels data/group_labels_festival/group_000.jsonl \
    --out sokoban_core/levels/pack_groups/lists/group_000.from_labels.list
"""

import argparse
import json
import os
from typing import Iterable, List, Set


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, help="Input JSONL file with records containing 'level_id'.")
    p.add_argument("--out", required=True, help="Output .list file (one level_id per line).")
    args = p.parse_args()

    if not os.path.exists(args.labels):
        raise SystemExit(f"labels file not found: {args.labels}")

    seen: Set[str] = set()
    ids: List[str] = []
    total = 0
    missing_level_id = 0

    for rec in _iter_jsonl(args.labels):
        total += 1
        lid = rec.get("level_id")
        if not lid:
            missing_level_id += 1
            continue
        lid = str(lid)
        if lid in seen:
            continue
        seen.add(lid)
        ids.append(lid)

    ids = sorted(ids)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for lid in ids:
            f.write(lid + "\n")

    print(f"records_read: {total}")
    print(f"unique_level_ids: {len(ids)}")
    print(f"records_missing_level_id: {missing_level_id}")


if __name__ == "__main__":
    main()


