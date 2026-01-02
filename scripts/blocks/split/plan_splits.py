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

Or with guards:
python -m scripts.blocks.split.plan_splits \
  --eval_dir results/packs_self_eval \
  --out_lists_dir sokoban_core/levels/pack_blocks_2/lists \
  --method top_pct --pct 0.2 \
  --min_hard_mae 60 \
  --min_hard_easy_gap 30
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
    p.add_argument(
        "--min_hard_mae",
        type=float,
        default=0.0,
        help=(
            "Extra guardrail: only split if the average MAE of the selected hard set is >= this value. "
            "Useful with --method top_pct to avoid splitting 'good' packs just because they are large."
        ),
    )
    p.add_argument(
        "--min_hard_easy_gap",
        type=float,
        default=0.0,
        help=(
            "Extra guardrail: only split if mean(hard_mae) - mean(easy_mae) is >= this value. "
            "Useful with --method top_pct to ensure there is a real hard tail."
        ),
    )
    p.add_argument(
        "--min_hard_levels",
        type=int,
        default=30,
        help="Minimum number of levels in hard split (only used when splitting).",
    )
    p.add_argument(
        "--min_easy_levels",
        type=int,
        default=30,
        help="Minimum number of levels in easy split (only used when splitting).",
    )
    args = p.parse_args()

    if not os.path.isdir(args.eval_dir):
        raise SystemExit(f"eval_dir not found: {args.eval_dir}")

    files = sorted([f for f in os.listdir(args.eval_dir) if f.endswith(".json")])
    if not files:
        raise SystemExit(f"No eval json files in {args.eval_dir}")

    os.makedirs(args.out_lists_dir, exist_ok=True)

    splits = 0
    kept = 0
    skipped_by_guards = 0
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

        if not items:
            # Nothing to write.
            continue

        items.sort(key=lambda x: x[1], reverse=True)
        should_consider_split = len(items) >= args.min_levels_to_split
        hard: List[str] = []
        if should_consider_split:
            if args.method == "mae":
                hard = [lid for lid, mae in items if mae > args.mae_threshold]
            else:
                k = max(1, int(round(args.pct * len(items))))
                hard = [lid for lid, _ in items[:k]]

        hard_set = set(hard)
        easy = [lid for lid, _ in items if lid not in hard_set]

        can_split = should_consider_split and (len(hard) >= args.min_hard_levels and len(easy) >= args.min_easy_levels)
        if can_split and (args.min_hard_mae > 0.0 or args.min_hard_easy_gap > 0.0):
            hard_maes = [mae for _, mae in items if _ in hard_set]
            easy_maes = [mae for _, mae in items if _ not in hard_set]
            mean_hard = sum(hard_maes) / max(1, len(hard_maes))
            mean_easy = sum(easy_maes) / max(1, len(easy_maes))
            gap = mean_hard - mean_easy
            if (args.min_hard_mae > 0.0 and mean_hard < args.min_hard_mae) or (
                args.min_hard_easy_gap > 0.0 and gap < args.min_hard_easy_gap
            ):
                can_split = False
                skipped_by_guards += 1
        if not can_split:
            # Keep the pack as-is for the next iteration.
            _write_list(os.path.join(args.out_lists_dir, f"{pack}.list"), sorted([lid for lid, _ in items]))
            kept += 1
            continue

        _write_list(os.path.join(args.out_lists_dir, f"{pack}_easy.list"), sorted(easy))
        _write_list(os.path.join(args.out_lists_dir, f"{pack}_hard.list"), sorted(hard))
        splits += 1

    if args.method == "top_pct" and args.min_hard_mae <= 0.0 and args.min_hard_easy_gap <= 0.0:
        print(
            "NOTE: --method top_pct will split most sufficiently-large packs unless you add guardrails like "
            "--min_hard_mae and/or --min_hard_easy_gap.",
            flush=True,
        )
    print(f"wrote: splits={splits} kept={kept} skipped_by_guards={skipped_by_guards} → {args.out_lists_dir}")


if __name__ == "__main__":
    main()


