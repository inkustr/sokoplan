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
    --method auto_gap

Or with guards:
python -m scripts.blocks.split.plan_splits \
  --eval_dir results/packs_self_eval \
  --out_lists_dir sokoban_core/levels/pack_blocks/lists \
  --method auto_gap \
  --auto_gap_balance_power 1.0 \
  --auto_gap_min_hard_frac 0.05 \
  --auto_gap_max_hard_frac 0.50
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple


def _write_list(path: str, ids: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in ids:
            f.write(it + "\n")

def _auto_gap_split(
    items_desc: List[Tuple[str, float]],
    *,
    min_hard: int,
    min_easy: int,
    min_hard_frac: float,
    max_hard_frac: float,
    balance_power: float,
) -> List[str]:
    """
    Auto-select a hard split (top-k) by finding the largest MAE "gap" in the
    descending-sorted list, while respecting minimum split sizes.

    Returns the hard level_ids (top-k). If no valid split point exists, returns [].
    """
    n = len(items_desc)
    if n <= 0:
        return []

    # k is the size of hard set; it must allow at least min_easy remaining.
    k_min = max(1, int(min_hard))
    k_max = n - int(min_easy)
    if k_min > k_max:
        return []

    # items_desc are already sorted by mae desc: maes[i] >= maes[i+1]
    best_k: Optional[int] = None
    best_score: Optional[float] = None
    for k in range(k_min, k_max + 1):
        # split between k-1 and k (0-index): hard = [0..k-1], easy = [k..]
        if k >= n:
            continue
        frac = k / max(1, n)
        if min_hard_frac > 0.0 and frac < min_hard_frac:
            continue
        if max_hard_frac < 1.0 and frac > max_hard_frac:
            continue
        gap = items_desc[k - 1][1] - items_desc[k][1]
        # Penalize tiny splits by preferring boundaries that separate sizeable groups.
        # This keeps auto_gap from selecting an early huge jump caused by a handful of outliers.
        balance = float(min(k, n - k))
        score = float(gap) * (balance ** float(balance_power))
        if best_score is None or score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        return []
    return [lid for lid, _ in items_desc[:best_k]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default="results/packs_self_eval", help="dir with <pack>.json from eval_pack_model")
    p.add_argument("--out_lists_dir", default="sokoban_core/levels/pack_blocks/lists")
    p.add_argument("--method", choices=["mae", "top_pct", "auto_gap"], default="top_pct")
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
    p.add_argument(
        "--auto_gap_min_abs_gap",
        type=float,
        default=0.0,
        help=(
            "Only used with --method auto_gap. Require the chosen MAE jump at the split boundary "
            "(mae[k-1] - mae[k]) to be >= this value, otherwise keep pack as-is."
        ),
    )
    p.add_argument(
        "--auto_gap_min_rel_gap",
        type=float,
        default=0.0,
        help=(
            "Only used with --method auto_gap. Require relative jump (mae[k-1]-mae[k]) / max(1e-9, mae[k]) "
            "to be >= this value, otherwise keep pack as-is."
        ),
    )
    p.add_argument(
        "--auto_gap_min_hard_frac",
        type=float,
        default=0.0,
        help="Only used with --method auto_gap. Minimum fraction of levels to put in 'hard'.",
    )
    p.add_argument(
        "--auto_gap_max_hard_frac",
        type=float,
        default=1.0,
        help="Only used with --method auto_gap. Maximum fraction of levels to put in 'hard'.",
    )
    p.add_argument(
        "--auto_gap_balance_power",
        type=float,
        default=1.0,
        help=(
            "Only used with --method auto_gap. Objective is gap * balance^power, where balance=min(k,n-k). "
            "Higher values penalize tiny hard tails more strongly."
        ),
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
            elif args.method == "top_pct":
                k = max(1, int(round(args.pct * len(items))))
                hard = [lid for lid, _ in items[:k]]
            else:
                hard = _auto_gap_split(
                    items,
                    min_hard=args.min_hard_levels,
                    min_easy=args.min_easy_levels,
                    min_hard_frac=float(args.auto_gap_min_hard_frac),
                    max_hard_frac=float(args.auto_gap_max_hard_frac),
                    balance_power=float(args.auto_gap_balance_power),
                )

                # Optional extra "gap" requirements to avoid splitting on tiny/noisy elbows.
                if hard:
                    k = len(hard)
                    # boundary is between k-1 and k (descending)
                    if k < len(items):
                        mae_prev = float(items[k - 1][1])
                        mae_next = float(items[k][1])
                        gap = mae_prev - mae_next
                        rel_gap = gap / max(1e-9, abs(mae_next))
                        if (args.auto_gap_min_abs_gap > 0.0 and gap < args.auto_gap_min_abs_gap) or (
                            args.auto_gap_min_rel_gap > 0.0 and rel_gap < args.auto_gap_min_rel_gap
                        ):
                            hard = []

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
    if args.method == "auto_gap" and args.auto_gap_min_abs_gap <= 0.0 and args.auto_gap_min_rel_gap <= 0.0:
        print(
            "NOTE: --method auto_gap chooses a split based on the largest MAE jump, but it may split even when the "
            "jump is small. Consider adding --auto_gap_min_abs_gap and/or --auto_gap_min_rel_gap for stability.",
            flush=True,
        )
    print(f"wrote: splits={splits} kept={kept} skipped_by_guards={skipped_by_guards} → {args.out_lists_dir}")


if __name__ == "__main__":
    main()


