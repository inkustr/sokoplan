from __future__ import annotations

"""
Filter packs to keep only *solved* levels according to labels.

Inputs:
  - packs_dir: directory containing <pack>.txt and packs_dir/lists/<pack>.list
  - labels_dir: directory containing <pack>.jsonl

Outputs:
  - out_dir/<pack>.txt
  - out_dir/lists/<pack>.list
  - out_dir/meta/<pack>.json (counts + mapping)

Example:
  source .venv/bin/activate
  python -m scripts.blocks.utils.filter_solved_by_labels \
    --packs_dir sokoban_core/levels/packs_filtered \
    --labels_dir data/packs_labels_festival \
    --out_dir sokoban_core/levels/packs_solved
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        raise SystemExit(0)


def _read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


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


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue


def _labels_level_ids(labels_path: str) -> Set[str]:
    out: Set[str] = set()
    for rec in _iter_jsonl(labels_path):
        lid = rec.get("level_id")
        if isinstance(lid, str) and lid:
            out.add(lid)
    return out


@dataclass(frozen=True)
class PackResult:
    pack: str
    in_levels: int
    solved_levels: int
    dropped_levels: int
    labels_records: int
    labels_unique_level_ids: int


def _count_jsonl_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                n += 1
    return n


def _write_pack(out_path: str, pack_name: str, levels: List[Tuple[int, str]]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for new_i, (orig_idx, body) in enumerate(levels):
            f.write(f"; {pack_name}_{orig_idx + 1}\n")
            f.write(body.rstrip("\n"))
            f.write("\n\n")


def _write_list(list_path: str, pack_txt_path: str, n: int) -> None:
    os.makedirs(os.path.dirname(list_path), exist_ok=True)
    with open(list_path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(f"{pack_txt_path}#{i}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--packs_dir", default="sokoban_core/levels/packs_filtered")
    p.add_argument("--labels_dir", default="data/packs_labels_festival")
    p.add_argument("--out_dir", default="sokoban_core/levels/packs_solved")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    lists_dir = os.path.join(args.packs_dir, "lists")
    if not os.path.isdir(lists_dir):
        raise SystemExit(f"packs lists dir not found: {lists_dir}")
    if not os.path.isdir(args.labels_dir):
        raise SystemExit(f"labels_dir not found: {args.labels_dir}")

    out_lists_dir = os.path.join(args.out_dir, "lists")
    out_meta_dir = os.path.join(args.out_dir, "meta")

    list_files = sorted([f for f in os.listdir(lists_dir) if f.endswith(".list")])
    if not list_files:
        raise SystemExit(f"No *.list under {lists_dir}")

    total_in = 0
    total_solved = 0
    total_dropped = 0
    packs_written = 0
    packs_skipped = 0

    for fn in list_files:
        pack = os.path.splitext(fn)[0]
        list_path = os.path.join(lists_dir, fn)
        level_ids = _read_list(list_path)
        if not level_ids:
            continue

        labels_path = os.path.join(args.labels_dir, f"{pack}.jsonl")
        if not os.path.exists(labels_path):
            packs_skipped += 1
            _safe_print(f"[skip] {pack}: missing labels jsonl: {labels_path}")
            continue

        labeled_ids = _labels_level_ids(labels_path)
        kept = [lid for lid in level_ids if lid in labeled_ids]

        total_in += len(level_ids)
        total_solved += len(kept)
        total_dropped += (len(level_ids) - len(kept))

        if not kept:
            packs_skipped += 1
            _safe_print(f"[skip] {pack}: 0/{len(level_ids)} levels have labels")
            continue

        # All ids in a pack list should point to the same pack txt path.
        pack_txt_path, _ = _parse_level_id(kept[0])
        if not os.path.exists(pack_txt_path):
            raise SystemExit(f"Referenced pack txt not found: {pack_txt_path} (from {list_path})")

        blocks = _split_on_blank_lines(open(pack_txt_path, "r", encoding="utf-8").read())
        picked: List[Tuple[int, str]] = []
        bad_idx = 0
        for lid in kept:
            src_path, idx = _parse_level_id(lid)
            if src_path != pack_txt_path:
                continue
            if idx < 0 or idx >= len(blocks):
                bad_idx += 1
                continue
            body = _strip_leading_comment_lines(blocks[idx])
            if not body.strip():
                continue
            picked.append((idx, body))

        if bad_idx:
            _safe_print(f"[warn] {pack}: {bad_idx} kept ids had out-of-range indices for {pack_txt_path}")

        if not picked:
            packs_skipped += 1
            _safe_print(f"[skip] {pack}: all kept ids became empty after parsing")
            continue

        out_pack_path = os.path.join(args.out_dir, f"{pack}.txt")
        out_list_path = os.path.join(out_lists_dir, f"{pack}.list")
        meta_path = os.path.join(out_meta_dir, f"{pack}.json")

        report = {
            "pack": pack,
            "source_list": list_path,
            "source_pack_txt": pack_txt_path,
            "labels": labels_path,
            "levels_in_list": len(level_ids),
            "levels_with_labels": len(kept),
            "levels_written": len(picked),
            "levels_dropped_no_labels": len(level_ids) - len(kept),
            "kept_original_indices": [idx for idx, _ in picked],
        }

        if args.dry_run:
            packs_written += 1
            _safe_print(f"[dry] {pack}: keep {len(picked)}/{len(level_ids)} (labels={len(kept)})")
            continue

        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(out_lists_dir, exist_ok=True)
        os.makedirs(out_meta_dir, exist_ok=True)

        _write_pack(out_pack_path, pack, picked)
        _write_list(out_list_path, out_pack_path, len(picked))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)

        packs_written += 1
        labels_records = _count_jsonl_lines(labels_path)
        _safe_print(
            f"[pack] {pack}: kept {len(picked)}/{len(level_ids)} levels "
            f"(labels unique_ids={len(labeled_ids)}, records={labels_records}) → {out_pack_path}"
        )

    _safe_print("\nDONE")
    _safe_print(f"packs_written: {packs_written}")
    _safe_print(f"packs_skipped: {packs_skipped}")
    _safe_print(f"levels_in: {total_in}")
    _safe_print(f"levels_out: {total_solved}")
    _safe_print(f"levels_dropped_no_labels: {total_dropped}")


if __name__ == "__main__":
    main()


