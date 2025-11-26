#!/usr/bin/env python3
"""
Download "All levels in one ZIP" from sourcecode.se
and parse them to sokoban_core/levels/windows/
"""

import os
import re
import zipfile
import shutil
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

ZIP_URL = "https://www.sourcecode.se/sokoban/download/Levels.zip"

MAP_LINE_RE = re.compile(r"^[#@\$\.\*\+ ]+$")
MAP_CHARS = set("#@$.*+ ")

def ensure_ascii(lines: List[str]) -> List[str]:
    lines = [ln.rstrip("\n") for ln in lines]
    if not lines:
        return []
    if not all(MAP_LINE_RE.match(ln) for ln in lines):
        return []
    w = max(len(ln) for ln in lines)
    return [ln.ljust(w) for ln in lines]

def extract_maps_from_text(text: str) -> List[List[str]]:
    """Parsing ASCII maps from a file."""
    maps = []
    cur = []
    def flush():
        nonlocal cur, maps
        grid = ensure_ascii(cur)
        if grid:
            maps.append(grid)
        cur = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if MAP_LINE_RE.match(line):
            cur.append(line)
        else:
            if cur:
                flush()
    if cur:
        flush()
    return maps


def extract_maps_from_xml(text: str) -> List[List[str]]:
    """Parsing levels from XML (.slc)."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    maps = []
    for level_node in root.findall(".//Level"):
        lines = []
        for l_node in level_node.findall("L"):
            lines.append(l_node.text if l_node.text else "")
        
        grid = ensure_ascii(lines)
        if grid:
            maps.append(grid)
            
    return maps


def process_slc(text: str) -> List[List[str]]:
    """
    SLC — Sokoban Levels Collection format.
    Can be XML or plain text.
    """
    stripped = text.lstrip()
    if stripped.startswith("<?xml") or "<SokobanLevels" in text:
        maps = extract_maps_from_xml(text)
        if maps:
            return maps

    return extract_maps_from_text(text)


def convert_pack(out_dir: Path, package: str, maps: List[List[str]]):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{package}.txt"
    with out_file.open("w", encoding="utf-8") as f:
        for i, grid in enumerate(maps, 1):
            f.write(f"; {package}_{i}\n")
            for ln in grid:
                f.write(ln + "\n")
            f.write("\n")
    print(f"[+] Saved {len(maps)} levels → {out_file}")


def main():
    root = Path("sokoban_core/levels/windows")
    root.mkdir(parents=True, exist_ok=True)

    tmp = Path("tmp_sourcecode_zip")
    
    need_download = True
    if tmp.exists():
        if any(tmp.iterdir()):
             print("[i] tmp_sourcecode_zip exists and is not empty. Skipping download.")
             need_download = False
    else:
        tmp.mkdir()

    if need_download:
        print(f"[+] Download ZIP: {ZIP_URL}")
        zip_path = tmp / "levels.zip"
        with requests.get(ZIP_URL, stream=True) as r:
            r.raise_for_status()
            with zip_path.open("wb") as f:
                shutil.copyfileobj(r.raw, f)

        print("[+] Extracting ZIP…")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

    print("[+] Parsing extracted files…")

    count = 0
    for file in tmp.rglob("*"):
        if file.suffix.lower() not in {".slc", ".txt", ".xsb"}:
            continue

        package = file.stem.lower().replace(" ", "_")
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except:
             text = file.read_text(encoding="latin-1", errors="ignore")

        if file.suffix.lower() == ".slc":
            maps = process_slc(text)
        else:
            maps = extract_maps_from_text(text)

        if maps:
            convert_pack(root / package, package, maps)
            count += 1

    print("[=] Done.")
    if count == 0:
        print("[!] Warning: No levels were converted. Check parser.")
    else:
        print(f"[i] Converted {count} packs to sokoban_core/levels/windows/")

    if tmp.exists():
        shutil.rmtree(tmp)
        print(f"[-] Removed temp directory: {tmp}")


if __name__ == "__main__":
    main()
