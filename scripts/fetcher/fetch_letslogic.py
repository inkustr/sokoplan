#!/usr/bin/env python3
"""
Download ALL Sokoban packs from letslogic.com and write them as standard Sokoban ASCII.

Usage:
  python scripts/fetcher/fetch_letslogic.py YOUR_API_KEY

LetsLogic digit encoding -> standard Sokoban ASCII:
  1 -> #  (wall)
  0,7 -> space (empty floor)
  2 -> @  (player)
  3 -> $  (box)
  4 -> .  (goal)
  5 -> *  (box on goal)
  6 -> +  (player on goal)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except Exception:
    requests = None


BASE_URL = "https://letslogic.com/api/v1"
OUT_ROOT = Path("sokoban_core/levels/letslogic")


TILE_MAP: Dict[str, str] = {
    "1": "#",
    "0": " ",
    "7": " ",
    "2": "@",
    "3": "$",
    "4": ".",
    "5": "*",
    "6": "+",
}


def _slugify(s: str, max_len: int = 80) -> str:
    s = s.strip()
    s = s.replace(os.sep, "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "collection"
    return s[:max_len]


def _request_json(
    session: "requests.Session",
    url: str,
    params: Dict[str, Any],
    timeout_s: float = 60.0,
    max_retries: int = 3,
    retry_sleep_s: float = 1.0,
) -> Any:
    last_err: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout_s)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep_s * (2**attempt))
    raise RuntimeError(f"Failed GET {url}: {last_err}")


def _decode_digit_map(digit_map: str, width: int, height: int) -> List[str]:
    if len(digit_map) != width * height:
        raise ValueError(
            f"Invalid map length: got {len(digit_map)} but expected {width*height} (w={width}, h={height})"
        )
    unknown = sorted(set(digit_map) - set(TILE_MAP.keys()))
    if unknown:
        raise ValueError(f"Unknown tile codes in map: {unknown}")

    rows: List[str] = []
    for r in range(height):
        segment = digit_map[r * width : (r + 1) * width]
        rows.append("".join(TILE_MAP[ch] for ch in segment))
    return rows


def _write_pack_txt(
    out_file: Path,
    pack_name: str,
    levels: List[Dict[str, Any]],
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for idx, lvl in enumerate(levels, 1):
            w = int(lvl["width"])
            h = int(lvl["height"])
            rows = _decode_digit_map(str(lvl["map"]), w, h)
            f.write(f"; {pack_name}_{idx}\n")
            for row in rows:
                if len(row) != w:
                    row = row.ljust(w)
                f.write(row + "\n")
            f.write("\n")


def fetch_all_collections(key: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    collections = _request_json(
        session,
        f"{BASE_URL}/collections",
        params={"key": key},
    )
    if not isinstance(collections, list):
        raise RuntimeError(f"Unexpected /collections response type: {type(collections)}")

    print(f"[+] LetsLogic collections: {len(collections)}")
    for i, c in enumerate(collections, 1):
        cid = int(c["id"])
        title = str(c.get("title") or "")

        pack_name = f"{cid:04d}_{(_slugify(title) or 'collection').lower()}"
        pack_dir = OUT_ROOT / pack_name
        pack_txt_path = pack_dir / f"{pack_name}.txt"

        if pack_txt_path.exists():
            print(f"[{i}/{len(collections)}] skip {pack_name}")
            continue

        print(f"[{i}/{len(collections)}] fetch {pack_name}")
        levels = _request_json(
            session,
            f"{BASE_URL}/collection/{cid}",
            params={"key": key},
        )
        if not isinstance(levels, list):
            raise RuntimeError(f"Unexpected /collection/{cid} response type: {type(levels)}")

        _write_pack_txt(pack_txt_path, pack_name, levels)

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Fetch LetsLogic packs.")
    p.add_argument("key", type=str, help="LetsLogic API key")
    args = p.parse_args(argv)

    fetch_all_collections(args.key)
    print(f"[=] Done.")


if __name__ == "__main__":
    main()
