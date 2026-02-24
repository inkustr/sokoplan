from __future__ import annotations

"""
Plan multi-way per-pack splits from self-eval JSON files.

New default behavior:
  - sort levels by per-level MAE (desc)
  - recursively cut at statistically strong MAE gaps
  - emit 1..N segments (not only easy/hard)

This creates behavior-driven blocks without static map features.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Item:
    level_id: str
    mae: float


@dataclass(frozen=True)
class SplitCandidate:
    cut_idx: int
    score: float
    abs_gap: float
    rel_gap: float


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if q <= 0.0:
        return float(ys[0])
    if q >= 1.0:
        return float(ys[-1])
    k = int(round((len(ys) - 1) * q))
    return float(ys[max(0, min(len(ys) - 1, k))])


def _write_list(path: str, ids: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in ids:
            f.write(it + "\n")


def _best_cut_for_segment(
    items_desc: List[Item],
    lo: int,
    hi: int,
    *,
    min_segment_size: int,
    min_abs_gap: float,
    min_rel_gap: float,
    balance_power: float,
) -> Optional[SplitCandidate]:
    n = hi - lo
    if n < 2 * min_segment_size:
        return None

    best: Optional[SplitCandidate] = None
    for k in range(lo + min_segment_size, hi - min_segment_size + 1):
        left_mae = float(items_desc[k - 1].mae)
        right_mae = float(items_desc[k].mae)
        abs_gap = left_mae - right_mae
        rel_gap = abs_gap / max(1e-9, abs(right_mae))
        if min_abs_gap > 0.0 and abs_gap < min_abs_gap:
            continue
        if min_rel_gap > 0.0 and rel_gap < min_rel_gap:
            continue

        left_n = k - lo
        right_n = hi - k
        balance = float(min(left_n, right_n))
        score = abs_gap * (balance**float(balance_power))
        cand = SplitCandidate(cut_idx=k, score=float(score), abs_gap=float(abs_gap), rel_gap=float(rel_gap))
        if best is None or cand.score > best.score:
            best = cand
    return best


def _recursive_multi_gap_split(
    items_desc: List[Item],
    *,
    min_segment_size: int,
    min_abs_gap: float,
    min_rel_gap: float,
    balance_power: float,
    max_segments: int,
) -> List[Tuple[int, int]]:
    """
    Return segment boundaries over [0, len(items_desc)).
    Greedy global recursion: at each step, split the segment with strongest valid gap.
    """
    n = len(items_desc)
    segments: List[Tuple[int, int]] = [(0, n)]
    if n < 2 * min_segment_size:
        return segments

    while True:
        if max_segments > 0 and len(segments) >= max_segments:
            break

        best_global: Optional[Tuple[int, SplitCandidate]] = None
        for i, (lo, hi) in enumerate(segments):
            cand = _best_cut_for_segment(
                items_desc,
                lo,
                hi,
                min_segment_size=min_segment_size,
                min_abs_gap=min_abs_gap,
                min_rel_gap=min_rel_gap,
                balance_power=balance_power,
            )
            if cand is None:
                continue
            if best_global is None or cand.score > best_global[1].score:
                best_global = (i, cand)

        if best_global is None:
            break

        seg_idx, cand = best_global
        lo, hi = segments[seg_idx]
        left = (lo, cand.cut_idx)
        right = (cand.cut_idx, hi)
        segments = segments[:seg_idx] + [left, right] + segments[seg_idx + 1 :]

    segments.sort(key=lambda x: x[0])
    return segments


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default="results/packs_self_eval", help="Directory with <pack>.json from eval_pack_model.")
    p.add_argument("--out_lists_dir", default="sokoban_core/levels/pack_blocks/lists")
    p.add_argument(
        "--method",
        choices=["multi_gap", "auto_gap"],
        default="multi_gap",
        help="multi_gap (new default) produces 1..N segments; auto_gap keeps the strongest single cut (legacy-like).",
    )
    p.add_argument("--min_levels_to_split", type=int, default=80)
    p.add_argument("--min_segment_size", type=int, default=30, help="Minimum levels in each emitted segment.")
    p.add_argument("--max_segments", type=int, default=0, help="0 = unlimited; otherwise cap segments per pack.")
    p.add_argument(
        "--gap_min_abs",
        type=float,
        default=-1.0,
        help="Minimum MAE jump to accept a cut. Use <0 for auto-calibration from current eval set.",
    )
    p.add_argument(
        "--gap_min_rel",
        type=float,
        default=-1.0,
        help="Minimum relative MAE jump to accept a cut. Use <0 for auto-calibration from current eval set.",
    )
    p.add_argument(
        "--auto_abs_quantile",
        type=float,
        default=0.20,
        help="If --gap_min_abs<0, use this quantile of best-cut absolute gaps across eligible packs.",
    )
    p.add_argument(
        "--auto_rel_quantile",
        type=float,
        default=0.50,
        help="If --gap_min_rel<0, use this quantile of best-cut relative gaps across eligible packs.",
    )
    p.add_argument(
        "--gap_balance_power",
        type=float,
        default=1.0,
        help="Cut objective is gap * balance^power; larger values penalize tiny tail splits.",
    )
    args = p.parse_args()

    if not os.path.isdir(args.eval_dir):
        raise SystemExit(f"eval_dir not found: {args.eval_dir}")

    files = sorted([f for f in os.listdir(args.eval_dir) if f.endswith(".json")])
    if not files:
        raise SystemExit(f"No eval json files in {args.eval_dir}")

    os.makedirs(args.out_lists_dir, exist_ok=True)

    pack_items: List[Tuple[str, List[Item]]] = []
    for fn in files:
        pack = os.path.splitext(fn)[0]
        path = os.path.join(args.eval_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        per_level: Dict[str, Dict[str, float]] = obj.get("per_level", {})
        items = [Item(level_id=lid, mae=float(m.get("mae", 0.0))) for lid, m in per_level.items()]
        if not items:
            continue
        items.sort(key=lambda x: x.mae, reverse=True)
        pack_items.append((pack, items))

    # Auto-calibrate thresholds from best available split in each eligible pack.
    auto_abs = float(args.gap_min_abs)
    auto_rel = float(args.gap_min_rel)
    if auto_abs < 0.0 or auto_rel < 0.0:
        abs_samples: List[float] = []
        rel_samples: List[float] = []
        for _, items in pack_items:
            if len(items) < int(args.min_levels_to_split):
                continue
            cand = _best_cut_for_segment(
                items,
                0,
                len(items),
                min_segment_size=int(args.min_segment_size),
                min_abs_gap=0.0,
                min_rel_gap=0.0,
                balance_power=float(args.gap_balance_power),
            )
            if cand is None:
                continue
            abs_samples.append(float(cand.abs_gap))
            rel_samples.append(float(cand.rel_gap))
        if auto_abs < 0.0:
            auto_abs = _quantile(abs_samples, float(args.auto_abs_quantile))
        if auto_rel < 0.0:
            auto_rel = _quantile(rel_samples, float(args.auto_rel_quantile))
        print(
            f"auto thresholds: gap_min_abs={auto_abs:.6f} (q={args.auto_abs_quantile}), "
            f"gap_min_rel={auto_rel:.6f} (q={args.auto_rel_quantile}), samples={len(abs_samples)}",
            flush=True,
        )

    packs_kept = 0
    packs_split = 0
    total_segments = 0
    for pack, items in pack_items:
        if len(items) < int(args.min_levels_to_split):
            _write_list(os.path.join(args.out_lists_dir, f"{pack}.list"), sorted([x.level_id for x in items]))
            packs_kept += 1
            total_segments += 1
            continue

        max_segments = int(args.max_segments)
        if args.method == "auto_gap":
            max_segments = 2

        segs = _recursive_multi_gap_split(
            items,
            min_segment_size=int(args.min_segment_size),
            min_abs_gap=float(auto_abs),
            min_rel_gap=float(auto_rel),
            balance_power=float(args.gap_balance_power),
            max_segments=max_segments,
        )

        if len(segs) <= 1:
            _write_list(os.path.join(args.out_lists_dir, f"{pack}.list"), sorted([x.level_id for x in items]))
            packs_kept += 1
            total_segments += 1
            continue

        packs_split += 1
        total_segments += len(segs)
        for i, (lo, hi) in enumerate(segs):
            seg_ids = sorted([x.level_id for x in items[lo:hi]])
            out_name = f"{pack}_seg{i:02d}.list"
            _write_list(os.path.join(args.out_lists_dir, out_name), seg_ids)

    print(
        f"wrote: packs_split={packs_split} packs_kept={packs_kept} total_output_segments={total_segments} "
        f"-> {args.out_lists_dir}"
    )


if __name__ == "__main__":
    main()
