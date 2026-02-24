from __future__ import annotations

"""
Plan merges from cross-evaluation results using graph communities.

Input:
  cross_eval.csv with columns: model_pack,data_pack,mae,mse,...

Creates weighted undirected graph:
  - score(A,B) = max( mae(A on B)/self_mae[A], mae(B on A)/self_mae[B] )  (lower is better)
  - keep strong edges by global quantile + mutual-kNN filtering
  - convert score -> affinity and run weighted label propagation

Run:
    python -m scripts.blocks.merge.plan_merges \
    --cross_eval results/cross_eval.csv \
    --self_eval_dir results/packs_self_eval \
    -- knn 0 \
    --affinity_temperature 0.5 \
    --community_max_iter 50 \
    --seed 42 \
    --min_group_size 1 \
    --allow_missing_lists
"""

import argparse
import csv
import os
import json
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set


def _read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def _write_list(path: str, items: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it + "\n")


def _load_self_mae(self_eval_dir: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not self_eval_dir or not os.path.isdir(self_eval_dir):
        return out
    for fn in os.listdir(self_eval_dir):
        if not fn.endswith(".json"):
            continue
        pack = os.path.splitext(fn)[0]
        path = os.path.join(self_eval_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            mae = float(obj.get("overall", {}).get("mae", float("nan")))
            if mae == mae:  # not NaN
                out[pack] = mae
        except Exception:
            continue
    return out


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return float("inf")
    xs_sorted = sorted(xs)
    if q <= 0.0:
        return xs_sorted[0]
    if q >= 1.0:
        return xs_sorted[-1]
    k = int(round((len(xs_sorted) - 1) * q))
    return xs_sorted[max(0, min(len(xs_sorted) - 1, k))]


def _build_mutual_knn_edges(
    candidates: List[Tuple[float, str, str]],
    packs: Set[str],
    *,
    knn: int,
) -> List[Tuple[float, str, str]]:
    """
    Keep only edges that are in top-k neighbors for both endpoints.
    candidates: (score, a, b), lower score = better.
    """
    if knn <= 0:
        return sorted(candidates, key=lambda t: t[0])

    nbrs: Dict[str, List[Tuple[float, str]]] = {p: [] for p in packs}
    for score, a, b in candidates:
        nbrs[a].append((score, b))
        nbrs[b].append((score, a))

    topk: Dict[str, Set[str]] = {}
    for p, vs in nbrs.items():
        vs.sort(key=lambda x: x[0])
        topk[p] = set([q for _, q in vs[:knn]])

    picked: List[Tuple[float, str, str]] = []
    for score, a, b in candidates:
        if b in topk.get(a, set()) and a in topk.get(b, set()):
            picked.append((score, a, b))
    picked.sort(key=lambda t: t[0])
    return picked


def _score_to_affinity(score: float, temperature: float) -> float:
    # score around 1.0 means "same-quality transfer as self".
    # larger score => weaker affinity.
    t = max(1e-6, float(temperature))
    return float(math.exp(-(score - 1.0) / t))


def _weighted_label_propagation(
    packs: Set[str],
    weighted_edges: List[Tuple[str, str, float]],
    *,
    max_iter: int,
    seed: int,
) -> Dict[str, str]:
    """
    Weighted label propagation community detection.
    Deterministic for fixed seed.
    """
    labels: Dict[str, str] = {p: p for p in packs}
    adj: Dict[str, List[Tuple[str, float]]] = {p: [] for p in packs}
    for a, b, w in weighted_edges:
        adj[a].append((b, w))
        adj[b].append((a, w))

    rng = random.Random(int(seed))
    nodes = sorted(packs)
    for _ in range(max_iter):
        changed = 0
        order = nodes[:]
        rng.shuffle(order)
        for p in order:
            if not adj[p]:
                continue
            acc: Dict[str, float] = defaultdict(float)
            for q, w in adj[p]:
                acc[labels[q]] += float(w)
            if not acc:
                continue
            best_label = min(acc.keys())
            best_w = acc[best_label]
            for lab, tot_w in acc.items():
                if tot_w > best_w or (tot_w == best_w and lab < best_label):
                    best_label = lab
                    best_w = tot_w
            if labels[p] != best_label:
                labels[p] = best_label
                changed += 1
        if changed == 0:
            break
    return labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cross_eval", required=True)
    p.add_argument("--packs_lists_dir", default="sokoban_core/levels/packs_filtered/lists")
    p.add_argument("--out_lists_dir", default="sokoban_core/levels/pack_groups/lists")
    p.add_argument(
        "--self_eval_dir",
        default="results/packs_self_eval",
        help="Directory with <pack>.json self-eval files.",
    )
    p.add_argument(
        "--ratio_quantile",
        type=float,
        default=0.15,
        help="Keep candidate edges with score <= quantile(scores, q). Lower = stricter.",
    )
    p.add_argument(
        "--offset",
        type=float,
        default=2.0,
        help="Deprecated (kept for CLI compatibility).",
    )
    p.add_argument(
        "--max_degree",
        type=int,
        default=4,
        help="Deprecated (kept for CLI compatibility).",
    )
    p.add_argument("--knn", type=int, default=6, help="Mutual-kNN filtering on candidate edge graph (0=disabled).")
    p.add_argument("--affinity_temperature", type=float, default=0.5, help="Score->affinity temperature.")
    p.add_argument("--community_max_iter", type=int, default=50, help="Max iterations for weighted label propagation.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for community detection.")
    p.add_argument(
        "--min_group_size",
        type=int,
        default=1,
        help="Minimum number of packs in a group to write. Use 1 to include singleton packs.",
    )
    p.add_argument(
        "--allow_missing_lists",
        action="store_true",
        help=(
            "By default, fail if any pack referenced by cross-eval is missing its .list file in --packs_lists_dir. "
            "Set this to allow missing packs to be skipped (NOT recommended, can create empty/partial groups)."
        ),
    )
    args = p.parse_args()

    mae: Dict[Tuple[str, str], float] = {}
    r2: Dict[Tuple[str, str], float] = {}
    packs: set[str] = set()
    with open(args.cross_eval, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            a = row["model_pack"]
            b = row["data_pack"]
            packs.add(a)
            packs.add(b)
            mae[(a, b)] = float(row["mae"])
            if "r2" in row and row["r2"] not in ("", None):
                try:
                    r2[(a, b)] = float(row["r2"])
                except Exception:
                    pass

    self_mae = _load_self_mae(args.self_eval_dir)
    if not self_mae:
        raise SystemExit(f"self_eval_dir is empty or missing: {args.self_eval_dir}")

    # Build undirected candidate edges with a numeric "merge score" (lower = better).
    # max( mae(A on B)/self_mae[A], mae(B on A)/self_mae[B] )
    eps = 1e-9
    cand: List[Tuple[float, str, str]] = []
    missing_self = 0
    mutual_pairs = 0

    for (a, b), mab in mae.items():
        mba = mae.get((b, a))
        if mba is None:
            continue
        mutual_pairs += 1
        aa, bb = (a, b) if a < b else (b, a)

        sa = self_mae.get(a)
        sb = self_mae.get(b)
        if sa is None or sb is None:
            missing_self += 1
            continue

        # direction-normalized ratios
        rab = mab / max(sa, eps)
        rba = mba / max(sb, eps)
        score = max(rab, rba)  # symmetric
        cand.append((score, aa, bb))

    # Global thresholding
    auto_ratio_thr: Optional[float]
    scores = [s for s, _, _ in cand]
    auto_ratio_thr = _quantile(scores, args.ratio_quantile)
    cand = [(s, a, b) for (s, a, b) in cand if s <= auto_ratio_thr]

    # Local filtering to suppress bridge chains.
    picked = _build_mutual_knn_edges(cand, packs, knn=int(args.knn))

    # Weighted label propagation communities.
    weighted_edges: List[Tuple[str, str, float]] = []
    for score, a, b in picked:
        weighted_edges.append((a, b, _score_to_affinity(score, float(args.affinity_temperature))))
    labels = _weighted_label_propagation(
        packs,
        weighted_edges,
        max_iter=int(args.community_max_iter),
        seed=int(args.seed),
    )
    comps: Dict[str, List[str]] = defaultdict(list)
    for pck, lab in labels.items():
        comps[lab].append(pck)

    groups = [sorted(v) for v in comps.values() if len(v) >= args.min_group_size]
    groups.sort(key=len, reverse=True)

    os.makedirs(args.out_lists_dir, exist_ok=True)
    wrote = 0
    missing_lists: List[str] = []
    empty_groups = 0
    for i, group in enumerate(groups):
        ids: List[str] = []
        for pack in group:
            lp = os.path.join(args.packs_lists_dir, f"{pack}.list")
            if not os.path.exists(lp):
                missing_lists.append(lp)
                continue
            ids.extend(_read_list(lp))
        if missing_lists and not args.allow_missing_lists:
            sample = "\n".join(missing_lists[:15])
            raise SystemExit(
                "Missing pack list files under --packs_lists_dir.\n"
                f"packs_lists_dir={args.packs_lists_dir}\n"
                f"missing_count={len(missing_lists)} (showing first 15):\n{sample}\n\n"
                "This usually means you passed the wrong lists directory for the packs referenced in cross_eval.csv.\n"
                "If you really want to proceed anyway, pass --allow_missing_lists (NOT recommended)."
            )
        outp = os.path.join(args.out_lists_dir, f"group_{i:03d}.list")
        if not ids:
            empty_groups += 1
        _write_list(outp, ids)
        wrote += 1

    print(f"packs={len(packs)} mutual_pairs={mutual_pairs}")
    print(f"auto_ratio_thr={auto_ratio_thr}")
    if missing_self:
        print(f"missing_self_eval_for={missing_self} mutual_pairs (skipped)")
    print(f"candidates_after_quantile={len(cand)}")
    print(f"picked_mutual_knn_edges={len(picked)}")
    print(f"groups_written={wrote}")
    if missing_lists:
        print(f"WARNING: missing_list_files={len(missing_lists)} (use correct --packs_lists_dir or omit --allow_missing_lists)")
    if empty_groups:
        print(f"WARNING: empty_group_files={empty_groups} (this indicates missing lists or empty input packs)")


if __name__ == "__main__":
    main()


