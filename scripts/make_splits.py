from __future__ import annotations
import argparse, yaml, os, random
from typing import List
from sokoban_core.levels.io import iterate_level_strings, filter_level


"""
Make splits from the levels in the config.

Usage:
  python -m scripts.make_splits --config configs/data.yaml --seed 42 --train 2000 --val 200 --test 200
"""

def write_list(path: str, items: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/data.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=int, default=2000, help="how many levels in train")
    p.add_argument("--val", type=int, default=200, help="how many levels in val")
    p.add_argument("--test", type=int, default=200, help="how many levels in test")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = cfg["levels"]["root_dir"]
    rels = cfg["levels"]["sources"]
    flt  = cfg.get("filters", {})

    all_ids: List[str] = []
    for ref, s in iterate_level_strings(root, rels):
        if filter_level(s,
                        max_w=flt.get("max_width"),
                        max_h=flt.get("max_height"),
                        min_b=flt.get("min_boxes"),
                        max_b=flt.get("max_boxes")):
            all_ids.append(f"{ref.path}#{ref.index}")

    rng = random.Random(args.seed)
    rng.shuffle(all_ids)

    n_tr, n_v, n_te = args.train, args.val, args.test
    tr = all_ids[:n_tr]
    va = all_ids[n_tr:n_tr+n_v]
    te = all_ids[n_tr+n_v:n_tr+n_v+n_te]

    write_list(cfg["splits"]["train"], tr)
    write_list(cfg["splits"]["val"], va)
    write_list(cfg["splits"]["test"], te)

    print("written:")
    print(" train:", cfg["splits"]["train"], len(tr))
    print(" val:",   cfg["splits"]["val"], len(va))
    print(" test:",  cfg["splits"]["test"], len(te))

if __name__ == "__main__":
    main()