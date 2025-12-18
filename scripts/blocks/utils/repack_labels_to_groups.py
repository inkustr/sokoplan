from __future__ import annotations

"""
Repack existing Festival labels into per-group label files.

Run:
  source .venv/bin/activate
  python -m scripts.blocks.utils.repack_labels_to_groups \
    --labels_dir data/packs_labels_festival \
    --groups_lists_dir sokoban_core/levels/pack_groups/lists \
    --out_dir data/group_labels_festival
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


def _read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels_dir", default="data/packs_labels_festival", help="Directory with existing *.jsonl label files.")
    p.add_argument("--groups_lists_dir", default="sokoban_core/levels/pack_groups/lists", help="Directory with group_*.list files.")
    p.add_argument("--out_dir", default="data/group_labels_festival", help="Output directory for group_XXX.jsonl files.")
    p.add_argument("--labels_glob_suffix", default=".jsonl", help="Only read label files ending with this suffix.")
    args = p.parse_args()

    if not os.path.isdir(args.labels_dir):
        raise SystemExit(f"labels_dir not found: {args.labels_dir}")
    if not os.path.isdir(args.groups_lists_dir):
        raise SystemExit(f"groups_lists_dir not found: {args.groups_lists_dir}")

    _ensure_dir(args.out_dir)

    group_list_files = sorted([f for f in os.listdir(args.groups_lists_dir) if f.startswith("group_") and f.endswith(".list")])
    if not group_list_files:
        raise SystemExit(f"No group_*.list found under: {args.groups_lists_dir}")

    level_to_group: Dict[str, str] = {}
    expected_by_group: Dict[str, Set[str]] = {}
    dup_in_groups: List[Tuple[str, str, str]] = []
    for fn in group_list_files:
        group = os.path.splitext(fn)[0]  # group_000
        ids = _read_list(os.path.join(args.groups_lists_dir, fn))
        sids: Set[str] = set(ids)
        expected_by_group[group] = sids
        for lid in sids:
            prev = level_to_group.get(lid)
            if prev is not None and prev != group:
                dup_in_groups.append((lid, prev, group))
            level_to_group[lid] = group

    if dup_in_groups:
        sample = "\n".join([f"- {lid} in {g1} and {g2}" for (lid, g1, g2) in dup_in_groups[:20]])
        raise SystemExit(f"ERROR: level_id appears in multiple groups (showing first 20):\n{sample}")

    expected_total_levels = len(level_to_group)

    # Writers for groups
    writers: Dict[str, object] = {}

    def _get_writer(group: str):
        w = writers.get(group)
        if w is not None:
            return w
        path = os.path.join(args.out_dir, f"{group}.jsonl")
        f = open(path, "a", encoding="utf-8")
        writers[group] = f
        return f

    label_files = sorted([f for f in os.listdir(args.labels_dir) if f.endswith(args.labels_glob_suffix)])
    if not label_files:
        raise SystemExit(f"No label files (*{args.labels_glob_suffix}) found under: {args.labels_dir}")

    written_records_by_group: Dict[str, int] = {os.path.splitext(f)[0]: 0 for f in group_list_files}
    seen_expected_levels: Set[str] = set()

    unassigned_level_ids: Set[str] = set()
    unassigned_records = 0
    total_records = 0

    first_source_for_level: Dict[str, str] = {}
    duplicate_level_ids: List[Tuple[str, str, str]] = []

    for lf in label_files:
        src_path = os.path.join(args.labels_dir, lf)
        for rec in _iter_jsonl(src_path):
            total_records += 1
            lid = str(rec.get("level_id", ""))
            if not lid:
                continue

            prev_src = first_source_for_level.get(lid)
            if prev_src is None:
                first_source_for_level[lid] = lf
            elif prev_src != lf:
                if len(duplicate_level_ids) < 2000:
                    duplicate_level_ids.append((lid, prev_src, lf))

            grp = level_to_group.get(lid)
            if grp is None:
                unassigned_level_ids.add(lid)
                unassigned_records += 1
                continue

            w = _get_writer(grp)
            w.write(json.dumps(rec) + "\n")
            written_records_by_group[grp] = written_records_by_group.get(grp, 0) + 1
            seen_expected_levels.add(lid)

    for f in writers.values():
        try:
            f.close()
        except Exception:
            pass

    missing_level_ids = sorted(set(level_to_group.keys()) - seen_expected_levels)
    unassigned_level_ids_sorted = sorted(unassigned_level_ids)

    missing_path = os.path.join(args.out_dir, "_missing_level_ids.txt")
    with open(missing_path, "w", encoding="utf-8") as f:
        for lid in missing_level_ids:
            f.write(lid + "\n")

    unassigned_path = os.path.join(args.out_dir, "_unassigned_level_ids.txt")
    with open(unassigned_path, "w", encoding="utf-8") as f:
        for lid in unassigned_level_ids_sorted:
            f.write(lid + "\n")

    dup_path = os.path.join(args.out_dir, "_duplicate_level_ids.txt")
    with open(dup_path, "w", encoding="utf-8") as f:
        for lid, a, b in duplicate_level_ids:
            f.write(f"{lid}\t{a}\t{b}\n")

    report = {
        "labels_dir": args.labels_dir,
        "groups_lists_dir": args.groups_lists_dir,
        "out_dir": args.out_dir,
        "groups": len(group_list_files),
        "expected_total_levels": expected_total_levels,
        "seen_expected_levels": len(seen_expected_levels),
        "missing_expected_levels": len(missing_level_ids),
        "unassigned_level_ids": len(unassigned_level_ids_sorted),
        "total_records_read": total_records,
        "unassigned_records": unassigned_records,
        "groups_records": {k: int(v) for k, v in sorted(written_records_by_group.items())},
        "notes": {
            "missing_level_ids_file": os.path.basename(missing_path),
            "unassigned_level_ids_file": os.path.basename(unassigned_path),
            "duplicate_level_ids_file": os.path.basename(dup_path),
            "duplicates_sampled": len(duplicate_level_ids),
        },
    }

    report_path = os.path.join(args.out_dir, "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"groups: {report['groups']}")
    print(f"expected_total_levels: {report['expected_total_levels']}")
    print(f"seen_expected_levels: {report['seen_expected_levels']}")
    print(f"missing_expected_levels: {report['missing_expected_levels']} -> {missing_path}")
    print(f"unassigned_level_ids: {report['unassigned_level_ids']} -> {unassigned_path}")
    print(f"total_records_read: {report['total_records_read']}")


if __name__ == "__main__":
    main()


