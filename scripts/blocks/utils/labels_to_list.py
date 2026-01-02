from __future__ import annotations

"""
Create a level-id list (.list) from a labels JSONL file.

Run:
  source .venv/bin/activate
  python -m scripts.blocks.utils.labels_to_list \
    --labels data/group_labels_festival/group_000.jsonl \
    --out sokoban_core/levels/pack_groups/lists/group_000.from_labels.list

Batch mode (convert all *.jsonl in a folder):
  source .venv/bin/activate
  python -m scripts.blocks.utils.labels_to_list \
    --labels_dir data/group_labels_festival \
    --out_dir sokoban_core/levels/pack_groups/lists
"""

import argparse
import json
import os
from typing import Iterable, List, Set, Tuple


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


def _labels_file_to_list(labels_path: str, out_path: str) -> Tuple[int, int, int]:
    """
    Returns: (records_read, unique_level_ids, records_missing_level_id)
    """
    if not os.path.exists(labels_path):
        raise SystemExit(f"labels file not found: {labels_path}")

    seen: Set[str] = set()
    ids: List[str] = []
    total = 0
    missing_level_id = 0

    for rec in _iter_jsonl(labels_path):
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

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for lid in ids:
            f.write(lid + "\n")

    return total, len(ids), missing_level_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="", help="Input JSONL file with records containing 'level_id' (single mode).")
    p.add_argument("--out", default="", help="Output .list file (single mode).")
    p.add_argument("--labels_dir", default="", help="Folder containing many *.jsonl label files (batch mode).")
    p.add_argument("--out_dir", default="", help="Output folder for batch mode (one .list per input .jsonl).")
    args = p.parse_args()

    single_mode = bool(args.labels or args.out)
    batch_mode = bool(args.labels_dir or args.out_dir)
    if single_mode and batch_mode:
        raise SystemExit("Use either --labels/--out (single) OR --labels_dir/--out_dir (batch), not both.")
    if not single_mode and not batch_mode:
        raise SystemExit("Provide --labels/--out (single) OR --labels_dir/--out_dir (batch).")

    if single_mode:
        if not args.labels or not args.out:
            raise SystemExit("Single mode requires both --labels and --out.")
        jobs = [(args.labels, args.out)]
    else:
        if not args.labels_dir or not args.out_dir:
            raise SystemExit("Batch mode requires both --labels_dir and --out_dir.")
        if not os.path.isdir(args.labels_dir):
            raise SystemExit(f"labels_dir not found: {args.labels_dir}")
        os.makedirs(args.out_dir, exist_ok=True)
        in_files = sorted(
            [
                os.path.join(args.labels_dir, fn)
                for fn in os.listdir(args.labels_dir)
                if fn.endswith(".jsonl") and os.path.isfile(os.path.join(args.labels_dir, fn))
            ]
        )
        if not in_files:
            raise SystemExit(f"No .jsonl files found in: {args.labels_dir}")
        jobs = [(p, os.path.join(args.out_dir, os.path.splitext(os.path.basename(p))[0] + ".list")) for p in in_files]

    total_all = 0
    unique_all = 0
    missing_all = 0
    for labels_path, out_path in jobs:
        total, unique, missing = _labels_file_to_list(labels_path, out_path)
        total_all += total
        unique_all += unique
        missing_all += missing
        print(f"{os.path.basename(labels_path)} -> {out_path}: records={total} unique_level_ids={unique} missing_level_id={missing}", flush=True)

    if len(jobs) > 1:
        print(f"TOTAL: files={len(jobs)} records={total_all} unique_level_ids(sum)={unique_all} missing_level_id(sum)={missing_all}", flush=True)


if __name__ == "__main__":
    main()


