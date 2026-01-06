from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

"""
Combine per-pack JSONL labels from two directories into one.

Run:
  source .venv/bin/activate
  python -m scripts.blocks.utils.combine_labels \
    --general_dir data/packs_labels_festival \
    --offpolicy_dir data/packs_labels_festival_offpolicy \
    --out_dir data/packs_labels_festival_combined
"""



@dataclass(frozen=True)
class DedupConfig:
    mode: str  # "none" | "state" | "state_y"


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"Failed to parse JSONL: {path}:{i}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object in {path}:{i}, got {type(obj)}")
            yield obj


def _dedup_key(rec: dict[str, Any], cfg: DedupConfig) -> tuple[Any, ...] | None:
    if cfg.mode == "none":
        return None

    try:
        base = (
            rec.get("level_id"),
            rec.get("walls"),
            rec.get("goals"),
            rec.get("boxes"),
            rec.get("player"),
        )
    except Exception:  # noqa: BLE001
        base = (rec.get("level_id"),)

    if cfg.mode == "state":
        return base
    if cfg.mode == "state_y":
        return base + (rec.get("y"),)

    raise ValueError(f"Unknown dedup mode: {cfg.mode!r}")


def _collect_pack_files(d: Path) -> dict[str, Path]:
    if not d.exists():
        return {}
    if not d.is_dir():
        raise ValueError(f"Not a directory: {d}")
    return {p.name: p for p in sorted(d.glob("*.jsonl")) if p.is_file()}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def combine_one(
    *,
    general_path: Path | None,
    offpolicy_path: Path | None,
    out_path: Path,
    dedup: DedupConfig,
) -> tuple[int, int, int]:
    """
    Returns: (n_general, n_offpolicy, n_written)
    """
    _ensure_dir(out_path.parent)

    seen: set[tuple[Any, ...]] = set()
    n_gen = 0
    n_off = 0
    n_out = 0

    def write_recs(recs: Iterable[dict[str, Any]]) -> None:
        nonlocal n_out
        for r in recs:
            key = _dedup_key(r, dedup)
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            out_f.write(json.dumps(r, separators=(",", ":")) + "\n")
            n_out += 1

    with out_path.open("w", encoding="utf-8") as out_f:
        if general_path is not None and general_path.exists():
            for r in _iter_jsonl(general_path):
                n_gen += 1
                write_recs([r])

        if offpolicy_path is not None and offpolicy_path.exists():
            for r in _iter_jsonl(offpolicy_path):
                n_off += 1
                write_recs([r])

    return n_gen, n_off, n_out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Combine per-pack JSONL labels from a 'general' directory and an 'offpolicy' directory "
            "into one per-pack JSONL output directory."
        )
    )
    ap.add_argument(
        "--general_dir",
        type=Path,
        required=True,
        help="Directory with general per-pack labels (*.jsonl).",
    )
    ap.add_argument(
        "--offpolicy_dir",
        type=Path,
        required=True,
        help="Directory with off-policy per-pack labels (*.jsonl).",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for combined per-pack labels (*.jsonl).",
    )
    ap.add_argument(
        "--dedup",
        choices=["none", "state", "state_y"],
        default="state",
        help=(
            "Deduplicate records across inputs. "
            "'state' removes exact duplicate states (level_id,walls,goals,boxes,player). "
            "'state_y' also includes y in the key."
        ),
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, error when a pack file exists in one dir but not the other.",
    )
    args = ap.parse_args()

    general_dir: Path = args.general_dir
    offpolicy_dir: Path = args.offpolicy_dir
    out_dir: Path = args.out_dir
    dedup = DedupConfig(mode=args.dedup)

    gen_files = _collect_pack_files(general_dir)
    off_files = _collect_pack_files(offpolicy_dir)

    all_names = sorted(set(gen_files.keys()) | set(off_files.keys()))
    if not all_names:
        raise SystemExit(
            f"No .jsonl files found in either {general_dir} or {offpolicy_dir}. "
            "Check the paths."
        )

    if args.strict:
        only_gen = sorted(set(gen_files.keys()) - set(off_files.keys()))
        only_off = sorted(set(off_files.keys()) - set(gen_files.keys()))
        if only_gen or only_off:
            raise SystemExit(
                "Mismatched pack files between directories.\n"
                f"Only in general_dir ({general_dir}): {only_gen[:10]}{' ...' if len(only_gen) > 10 else ''}\n"
                f"Only in offpolicy_dir ({offpolicy_dir}): {only_off[:10]}{' ...' if len(only_off) > 10 else ''}"
            )

    _ensure_dir(out_dir)

    total_gen = 0
    total_off = 0
    total_out = 0
    for name in all_names:
        gp = gen_files.get(name)
        op = off_files.get(name)
        out_path = out_dir / name
        n_gen, n_off, n_written = combine_one(
            general_path=gp,
            offpolicy_path=op,
            out_path=out_path,
            dedup=dedup,
        )
        total_gen += n_gen
        total_off += n_off
        total_out += n_written

        print(
            f"{name}: general={n_gen} offpolicy={n_off} -> out={n_written} ({out_path})",
            flush=True,
        )

    print(
        f"TOTAL: general={total_gen} offpolicy={total_off} -> out={total_out}  (dedup={dedup.mode})",
        flush=True,
    )


if __name__ == "__main__":
    main()


