from __future__ import annotations
import argparse, yaml, os
from sokoban_core.levels.io import iterate_level_strings, filter_level


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/data.yaml")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = cfg["levels"]["root_dir"]
    rels = cfg["levels"]["sources"]
    flt  = cfg.get("filters", {})

    ok = 0
    bad = 0
    for ref, s in iterate_level_strings(root, rels):
        if filter_level(s,
                        max_w=flt.get("max_width"),
                        max_h=flt.get("max_height"),
                        min_b=flt.get("min_boxes"),
                        max_b=flt.get("max_boxes")):
            ok += 1
        else:
            bad += 1
            print(f"[skip] {ref.path}#{ref.index}")
    print(f"valid: {ok}, skipped: {bad}")

if __name__ == "__main__":
    main()