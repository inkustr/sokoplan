from __future__ import annotations

"""
Group Sokoban levels by static structural features.

Pipeline:
1) Read level_ids from input .list files.
2) Extract static/topological features from each level state.
3) Standardize features and cluster with KMeans.
4) Auto-select K via silhouette score (on a sample), if requested.
5) Merge tiny clusters into nearest larger clusters.
6) Write group_XXX.list files.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict, deque
from statistics import median

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from sokoban_core.levels.resolve import load_level_by_id
from sokoban_core.moves import iter_bits, successors_pushes

FEATURE_NAMES = [
    "width",
    "height",
    "board_cells",
    "floor_ratio",
    "wall_ratio",
    "n_boxes",
    "n_goals",
    "free_per_box",
    "mean_floor_degree",
    "dead_end_ratio",
    "tunnel_ratio",
    "n_floor_components",
    "articulation_ratio",
    "corner_ratio",
    "box_goal_min_mean",
    "box_goal_min_max",
    "box_goal_lb_approx_per_box",
    "reachable_ratio_from_player",
    "initial_pushes",
]


def _read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def _write_list(path: str, ids: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for lid in ids:
            f.write(lid + "\n")


def _idx_rc(idx: int, w: int) -> Tuple[int, int]:
    return idx // w, idx % w


def _manhattan(a: int, b: int, w: int) -> int:
    ar, ac = _idx_rc(a, w)
    br, bc = _idx_rc(b, w)
    return abs(ar - br) + abs(ac - bc)


def _is_wall_like(state, idx: int) -> bool:
    if idx < 0 or idx >= state.width * state.height:
        return True
    return (not state.is_inside(idx)) or state.is_wall(idx)


def _is_corner_cell(state, idx: int) -> bool:
    w = state.width
    size = state.width * state.height
    up = idx - w if idx >= w else -1
    down = idx + w if idx + w < size else -1
    left = idx - 1 if idx % w != 0 else -1
    right = idx + 1 if idx % w != w - 1 else -1
    return (
        (_is_wall_like(state, up) and _is_wall_like(state, left))
        or (_is_wall_like(state, up) and _is_wall_like(state, right))
        or (_is_wall_like(state, down) and _is_wall_like(state, left))
        or (_is_wall_like(state, down) and _is_wall_like(state, right))
    )


def _floor_cells(state) -> List[int]:
    return [i for i in iter_bits(state.board_mask) if not state.is_wall(i)]


def _components_count(nodes: List[int], adj: Dict[int, List[int]]) -> int:
    unseen = set(nodes)
    comps = 0
    while unseen:
        comps += 1
        s = unseen.pop()
        q: deque[int] = deque([s])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v in unseen:
                    unseen.remove(v)
                    q.append(v)
    return comps


def _articulation_count(nodes: List[int], adj: Dict[int, List[int]]) -> int:
    # Iterative Tarjan articulation points for undirected graph.
    # Using an explicit stack avoids recursion-depth failures on long corridor maps.
    time = 0
    tin: Dict[int, int] = {}
    low: Dict[int, int] = {}
    parent: Dict[int, int] = {}
    child_count: Dict[int, int] = defaultdict(int)
    visited = set()
    arts = set()

    for root in nodes:
        if root in visited:
            continue
        parent[root] = -1
        visited.add(root)
        time += 1
        tin[root] = time
        low[root] = time

        # stack frames: (node, next-neighbor-index)
        stack: List[Tuple[int, int]] = [(root, 0)]
        while stack:
            u, i = stack[-1]
            nbrs = adj.get(u, [])
            if i < len(nbrs):
                v = nbrs[i]
                stack[-1] = (u, i + 1)
                if v == parent.get(u, -1):
                    continue
                if v in visited:
                    low[u] = min(low[u], tin[v])
                    continue
                parent[v] = u
                child_count[u] += 1
                visited.add(v)
                time += 1
                tin[v] = time
                low[v] = time
                stack.append((v, 0))
                continue

            stack.pop()
            p = parent.get(u, -1)
            if p != -1:
                low[p] = min(low[p], low[u])
                if parent.get(p, -1) != -1 and low[u] >= tin[p]:
                    arts.add(p)
            else:
                if child_count[u] > 1:
                    arts.add(u)
    return len(arts)


@dataclass
class LevelFeatures:
    level_id: str
    values: List[float]


def _extract_features(level_id: str) -> LevelFeatures:
    s = load_level_by_id(level_id)

    w = float(s.width)
    h = float(s.height)
    board_cells = float(int(s.board_mask).bit_count())
    wall_cells = float(int(s.walls & s.board_mask).bit_count())
    floor_cells = _floor_cells(s)
    floor_n = float(len(floor_cells))
    boxes = list(iter_bits(s.boxes))
    goals = list(iter_bits(s.goals))
    n_boxes = float(len(boxes))
    n_goals = float(len(goals))

    area = max(1.0, w * h)
    floor_ratio = floor_n / area
    wall_ratio = wall_cells / max(1.0, board_cells)

    # Build floor graph
    floor_set = set(floor_cells)
    adj: Dict[int, List[int]] = {}
    degs: List[int] = []
    tunnel_like = 0
    for u in floor_cells:
        nbrs = [v for v in s.neighbors(u) if v in floor_set]
        adj[u] = nbrs
        d = len(nbrs)
        degs.append(d)
        if d == 2:
            a, b = nbrs[0], nbrs[1]
            ur, uc = _idx_rc(u, s.width)
            ar, ac = _idx_rc(a, s.width)
            br, bc = _idx_rc(b, s.width)
            if (ar == ur == br) or (ac == uc == bc):
                tunnel_like += 1

    mean_deg = float(sum(degs) / max(1, len(degs)))
    dead_end_ratio = float(sum(1 for d in degs if d <= 1) / max(1, len(degs)))
    tunnel_ratio = float(tunnel_like / max(1, len(degs)))
    comps = float(_components_count(floor_cells, adj))
    art_count = float(_articulation_count(floor_cells, adj))
    art_ratio = art_count / max(1.0, floor_n)

    corner_cells = sum(1 for u in floor_cells if _is_corner_cell(s, u))
    corner_ratio = float(corner_cells / max(1.0, floor_n))

    # Box-goal geometric distances.
    if boxes and goals:
        nearest_goal_per_box = [min(_manhattan(b, g, s.width) for g in goals) for b in boxes]
        box_goal_min_mean = float(sum(nearest_goal_per_box) / len(nearest_goal_per_box))
        box_goal_min_max = float(max(nearest_goal_per_box))
        # lower-bound approximation using sorted nearest distances (cheap and stable).
        box_goal_lb_approx = float(sum(sorted(nearest_goal_per_box)))
    else:
        box_goal_min_mean = 0.0
        box_goal_min_max = 0.0
        box_goal_lb_approx = 0.0

    reachable = 0.0
    if s.player in floor_set:
        q: deque[int] = deque([s.player])
        vis = {s.player}
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in vis and not s.has_box(v):
                    vis.add(v)
                    q.append(v)
        reachable = float(len(vis))
    reachable_ratio = reachable / max(1.0, floor_n)

    # Initial branching proxy for push-search.
    init_pushes = float(len(successors_pushes(s)))

    free_per_box = (floor_n - n_boxes) / max(1.0, n_boxes)

    values = [
        w,
        h,
        board_cells,
        floor_ratio,
        wall_ratio,
        n_boxes,
        n_goals,
        free_per_box,
        mean_deg,
        dead_end_ratio,
        tunnel_ratio,
        comps,
        art_ratio,
        corner_ratio,
        box_goal_min_mean,
        box_goal_min_max,
        box_goal_lb_approx / max(1.0, n_boxes),
        reachable_ratio,
        init_pushes,
    ]
    return LevelFeatures(level_id=level_id, values=values)


def _summary_stats(vals: np.ndarray) -> Dict[str, float]:
    if vals.size == 0:
        return {
            "min": 0.0,
            "p10": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    return {
        "min": float(np.min(vals)),
        "p10": float(np.quantile(vals, 0.10)),
        "median": float(np.quantile(vals, 0.50)),
        "p90": float(np.quantile(vals, 0.90)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


def _auto_choose_k(
    X: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    sample_size: int,
    seed: int,
) -> int:
    n = X.shape[0]
    k_min = max(2, min(k_min, n - 1))
    k_max = max(k_min, min(k_max, n - 1))
    if k_min == k_max:
        return k_min

    rng = np.random.default_rng(seed)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    best_k = k_min
    best_score = -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=300)
        labs = km.fit_predict(Xs)
        # Need at least 2 non-empty clusters for silhouette.
        if len(set(labs.tolist())) < 2:
            continue
        sc = float(silhouette_score(Xs, labs, metric="euclidean"))
        if sc > best_score:
            best_score = sc
            best_k = k
    return best_k


def _merge_tiny_clusters(
    labels: np.ndarray,
    X: np.ndarray,
    *,
    min_cluster_size: int,
) -> np.ndarray:
    labels = labels.copy()
    while True:
        uniq, cnts = np.unique(labels, return_counts=True)
        tiny = [int(c) for c, n in zip(uniq, cnts) if int(n) < min_cluster_size]
        if not tiny or len(uniq) <= 1:
            break

        # Recompute centroids for current clustering.
        centroids: Dict[int, np.ndarray] = {}
        for c in uniq:
            mask = labels == c
            centroids[int(c)] = X[mask].mean(axis=0)

        for tc in tiny:
            idxs = np.where(labels == tc)[0]
            if idxs.size == 0:
                continue
            other = [int(c) for c in uniq if int(c) != tc]
            if not other:
                continue
            best = min(other, key=lambda c: float(np.linalg.norm(centroids[tc] - centroids[c])))
            labels[idxs] = best

        # Renumber labels to 0..K-1 for stability.
        uniq2 = sorted(set(labels.tolist()))
        remap = {old: i for i, old in enumerate(uniq2)}
        labels = np.array([remap[int(x)] for x in labels], dtype=np.int32)
    return labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in_lists_dir",
        required=True,
        help="Input lists dir with source level ids (e.g. sokoban_core/levels/pack_blocks_after_merge_1/lists).",
    )
    p.add_argument("--out_lists_dir", required=True, help="Output dir for grouped lists (group_XXX.list).")
    p.add_argument("--report_json", default="", help="Optional report JSON with stats and chosen K.")

    p.add_argument("--k", type=int, default=0, help="Fixed K. Use 0 to auto-choose by silhouette.")
    p.add_argument("--k_min", type=int, default=12)
    p.add_argument("--k_max", type=int, default=64)
    p.add_argument("--auto_k_sample_size", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_cluster_size", type=int, default=120, help="Tiny clusters are merged to nearest.")

    args = p.parse_args()

    if not os.path.isdir(args.in_lists_dir):
        raise SystemExit(f"in_lists_dir not found: {args.in_lists_dir}")
    os.makedirs(args.out_lists_dir, exist_ok=True)

    list_files = sorted(
        [os.path.join(args.in_lists_dir, fn) for fn in os.listdir(args.in_lists_dir) if fn.endswith(".list")]
    )
    if not list_files:
        raise SystemExit(f"No .list files in {args.in_lists_dir}")

    level_ids: List[str] = []
    for lp in list_files:
        level_ids.extend(_read_list(lp))
    level_ids = sorted(set(level_ids))
    if len(level_ids) < 2:
        raise SystemExit("Need at least 2 unique level_ids.")

    feats: List[LevelFeatures] = []
    for i, lid in enumerate(level_ids, start=1):
        feats.append(_extract_features(lid))
        if i % 500 == 0:
            print(f"features: {i}/{len(level_ids)}", flush=True)

    X = np.array([f.values for f in feats], dtype=np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if int(args.k) > 0:
        k = int(args.k)
    else:
        k = _auto_choose_k(
            Xs,
            k_min=int(args.k_min),
            k_max=int(args.k_max),
            sample_size=int(args.auto_k_sample_size),
            seed=int(args.seed),
        )

    km = KMeans(n_clusters=k, random_state=int(args.seed), n_init=30, max_iter=400)
    labels = km.fit_predict(Xs)
    labels = _merge_tiny_clusters(labels, Xs, min_cluster_size=int(args.min_cluster_size))

    groups: Dict[int, List[str]] = defaultdict(list)
    for lid, lb in zip(level_ids, labels.tolist()):
        groups[int(lb)].append(lid)

    ordered = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    for i, (_, ids) in enumerate(ordered):
        _write_list(os.path.join(args.out_lists_dir, f"group_{i:03d}.list"), sorted(ids))

    sizes = [len(ids) for _, ids in ordered]
    print(
        f"levels={len(level_ids)} k_chosen={k} groups_written={len(ordered)} "
        f"size[min/med/max]={min(sizes)}/{int(median(sizes))}/{max(sizes)}"
    )

    if args.report_json:
        idx_by_level = {lid: i for i, lid in enumerate(level_ids)}

        global_feature_stats = {}
        for j, fname in enumerate(FEATURE_NAMES):
            global_feature_stats[fname] = _summary_stats(X[:, j])

        groups_report = []
        for i, (_, ids) in enumerate(ordered):
            idxs = np.array([idx_by_level[lid] for lid in ids], dtype=np.int64)
            Xg = X[idxs]
            feat_stats = {fname: _summary_stats(Xg[:, j]) for j, fname in enumerate(FEATURE_NAMES)}
            groups_report.append(
                {
                    "group_id": f"group_{i:03d}",
                    "n_levels": len(ids),
                    "feature_stats": feat_stats,
                }
            )

        # Separation score per feature:
        # ratio = var(group_means) / global_var, higher -> feature better separates groups.
        sep_scores: List[Tuple[str, float]] = []
        eps = 1e-12
        for j, fname in enumerate(FEATURE_NAMES):
            gmeans = []
            for _, ids in ordered:
                idxs = np.array([idx_by_level[lid] for lid in ids], dtype=np.int64)
                gmeans.append(float(np.mean(X[idxs, j])))
            gv = float(np.var(X[:, j]))
            bv = float(np.var(np.array(gmeans, dtype=np.float64)))
            score = bv / max(eps, gv)
            sep_scores.append((fname, score))
        sep_scores.sort(key=lambda t: t[1], reverse=True)

        report = {
            "input": {
                "in_lists_dir": args.in_lists_dir,
                "n_list_files": len(list_files),
                "n_unique_levels": len(level_ids),
            },
            "params": {
                "k": args.k,
                "k_min": args.k_min,
                "k_max": args.k_max,
                "auto_k_sample_size": args.auto_k_sample_size,
                "seed": args.seed,
                "min_cluster_size": args.min_cluster_size,
            },
            "result": {
                "k_chosen_pre_merge": int(k),
                "groups_written_post_merge": len(ordered),
                "group_sizes": [len(ids) for _, ids in ordered],
            },
            "feature_names": FEATURE_NAMES,
            "global_feature_stats": global_feature_stats,
            "top_separating_features": [{"feature": f, "score": float(s)} for f, s in sep_scores[:10]],
            "groups": groups_report,
        }
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"report_json={args.report_json}")


if __name__ == "__main__":
    main()

