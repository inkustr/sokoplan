from __future__ import annotations

"""
Delete parent files that have been split into more specific child files.

Rule:
  A stem is considered NON-LEAF if any other stem starts with "<stem>_".
  Example:
    - 696.jsonl is a parent of 696_easy.jsonl / 696_hard.jsonl  -> delete 696.jsonl
    - 696_easy.list is a parent of 696_easy_easy.list / 696_easy_hard.list -> delete 696_easy.list

Safety:
  - Default is dry-run
  - Use --yes to actually delete

Run:
  python -m scripts.blocks.utils.cleanup_split_files \
    --label_dirs data/packs_labels_festival data/pack_blocks_labels_festival \
    --list_dirs sokoban_core/levels/pack_blocks/lists sokoban_core/levels/pack_blocks_2/lists \
    --yes
"""

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple


@dataclass(frozen=True)
class FileRef:
    kind: str
    dir: str
    path: str
    stem: str


def _list_files(d: str, *, kind: str, ext: str) -> List[FileRef]:
    if not os.path.isdir(d):
        return []
    out: List[FileRef] = []
    for fn in os.listdir(d):
        if not fn.endswith(ext):
            continue
        stem = fn[: -len(ext)]
        out.append(FileRef(kind=kind, dir=d, path=os.path.join(d, fn), stem=stem))
    return out


def _compute_non_leaf_stems(stems: Set[str]) -> Set[str]:
    non_leaf: Set[str] = set()
    stems_set = set(stems)
    for s in stems_set:
        prefix = s + "_"
        for t in stems_set:
            if t.startswith(prefix):
                non_leaf.add(s)
                break
    return non_leaf


def _cleanup_kind(files: List[FileRef], yes: bool) -> Tuple[int, int]:
    if not files:
        return (0, 0)

    stems = {f.stem for f in files}
    non_leaf = _compute_non_leaf_stems(stems)
    to_delete = [f for f in files if f.stem in non_leaf]

    print(f"{files[0].kind}_files_found: {len(files)}")
    print(f"{files[0].kind}_non_leaf_stems: {len(non_leaf)}")
    print(f"{files[0].kind}_files_to_delete: {len(to_delete)}")

    if not to_delete:
        return (0, 0)

    for f in sorted(to_delete, key=lambda x: x.path)[:50]:
        print(f"- {f.path}")
    if len(to_delete) > 50:
        print(f"... and {len(to_delete) - 50} more")

    if not yes:
        return (len(to_delete), 0)

    deleted = 0
    for f in to_delete:
        try:
            os.remove(f.path)
            deleted += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[WARN] failed to delete {f.path}: {e}")

    return (len(to_delete), deleted)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--label_dirs",
        nargs="*",
        default=[],
        help="Directories containing *.jsonl label files",
    )
    p.add_argument(
        "--list_dirs",
        nargs="*",
        default=[],
        help="Directories containing *.list files",
    )
    p.add_argument("--yes", action="store_true", help="Actually delete files (default: dry-run)")
    args = p.parse_args()

    if not args.label_dirs and not args.list_dirs:
        raise SystemExit("Provide at least one of: --label_dirs or --list_dirs")

    files_labels: List[FileRef] = []
    for d in args.label_dirs:
        files_labels.extend(_list_files(d, kind="labels", ext=".jsonl"))

    files_lists: List[FileRef] = []
    for d in args.list_dirs:
        files_lists.extend(_list_files(d, kind="lists", ext=".list"))

    if not files_labels and not files_lists:
        print("No matching files found in provided dirs.")
        return

    print(f"mode: {'DELETE' if args.yes else 'DRY-RUN'}")
    if args.label_dirs:
        print(f"label_dirs: {len(args.label_dirs)}")
    if args.list_dirs:
        print(f"list_dirs: {len(args.list_dirs)}")

    if files_labels:
        _cleanup_kind(files_labels, args.yes)
    if files_lists:
        _cleanup_kind(files_lists, args.yes)

    if not args.yes:
        print("Dry-run only. Re-run with --yes to delete.")


if __name__ == "__main__":
    main()


