from __future__ import annotations

"""
Plan per-pack splits from eval JSON produced by scripts.blocks.split.eval_pack_model.

Given per-level MAE inside a pack, produce 2 list files:
  - <pack>_easy.list: levels with MAE <= threshold
  - <pack>_hard.list: levels with MAE > threshold

These lists can be used as new "blocks" for next iteration label generation/training.

Run:
  source .venv/bin/activate
  python -m scripts.blocks.split.plan_splits \
    --eval_dir results/packs_self_eval \
    --out_lists_dir sokoban_core/levels/pack_blocks/lists \
    --method top_pct --pct 0.2
"""

import argparse
import json
import os
from typing import Dict, List, Tuple


def _write_list(path: str, ids: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in ids:
            f.write(it + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default="results/packs_self_eval", help="dir with <pack>.json from eval_pack_model")
    p.add_argument("--out_lists_dir", default="sokoban_core/levels/pack_blocks/lists")
    p.add_argument("--method", choices=["mae", "top_pct"], default="top_pct")
    p.add_argument("--mae_threshold", type=float, default=3.0)
    p.add_argument("--pct", type=float, default=0.2, help="fraction of worst levels to mark as hard")
    p.add_argument("--min_levels_to_split", type=int, default=50)
    args = p.parse_args()

    if not os.path.isdir(args.eval_dir):
        raise SystemExit(f"eval_dir not found: {args.eval_dir}")

    files = sorted([f for f in os.listdir(args.eval_dir) if f.endswith(".json")])
    if not files:
        raise SystemExit(f"No eval json files in {args.eval_dir}")

    os.makedirs(args.out_lists_dir, exist_ok=True)

    splits = 0
    for fn in files:
        pack = os.path.splitext(fn)[0]
        path = os.path.join(args.eval_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        per_level: Dict[str, Dict[str, float]] = obj.get("per_level", {})
        items: List[Tuple[str, float]] = []
        for level_id, m in per_level.items():
            mae = float(m.get("mae", 0.0))
            items.append((level_id, mae))

        if len(items) < args.min_levels_to_split:
            continue

        items.sort(key=lambda x: x[1], reverse=True)
        if args.method == "mae":
            hard = [lid for lid, mae in items if mae > args.mae_threshold]
        else:
            k = max(1, int(round(args.pct * len(items))))
            hard = [lid for lid, _ in items[:k]]

        hard_set = set(hard)
        easy = [lid for lid, _ in items if lid not in hard_set]

        if len(hard) < 30 or len(easy) < 30:
            continue

        _write_list(os.path.join(args.out_lists_dir, f"{pack}_easy.list"), sorted(easy))
        _write_list(os.path.join(args.out_lists_dir, f"{pack}_hard.list"), sorted(hard))
        splits += 1

    print(f"wrote splits for {splits} packs → {args.out_lists_dir}")


if __name__ == "__main__":
    main()


