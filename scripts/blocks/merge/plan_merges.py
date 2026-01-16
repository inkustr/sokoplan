from __future__ import annotations

"""
Plan merges from cross-evaluation results.

Input:
  cross_eval.csv with columns: model_pack,data_pack,mae,mse,...

Creates merge edges using following method:
  - choose the ratio threshold automatically from the distribution of mutual pairs
  - score(A,B) = max( mae(A on B)/self_mae[A], mae(B on A)/self_mae[B] )
  - keep pairs with score <= quantile(score, q)

Then compute connected components (union-find) and write group list files:
  out_lists_dir/group_XXX.list

Run:
    python -m scripts.blocks.merge.plan_merges \
    --cross_eval results/cross_eval.csv \
    --self_eval_dir results/packs_self_eval \
    --ratio_quantile 0.10 --offset 2.0 --max_degree 4 \
    --packs_lists_dir sokoban_core/levels/pack_groups/lists \
    --out_lists_dir sokoban_core/levels/pack_groups_2/lists
"""

import argparse
import csv
import os
import json
from typing import Dict, List, Tuple, Optional


class DSU:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


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


def _select_edges_degree_limited(
    candidates: List[Tuple[float, str, str]],
    max_degree: int,
) -> List[Tuple[str, str]]:
    """
    Greedy degree-limited edge selection.
    candidates: (score, a, b) where a<b (undirected)
    """
    if max_degree <= 0:
        return [(a, b) for _, a, b in sorted(candidates)]
    deg: Dict[str, int] = {}
    picked: List[Tuple[str, str]] = []
    for score, a, b in sorted(candidates, key=lambda t: t[0]):
        da = deg.get(a, 0)
        db = deg.get(b, 0)
        if da >= max_degree or db >= max_degree:
            continue
        picked.append((a, b))
        deg[a] = da + 1
        deg[b] = db + 1
    return picked


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
    p.add_argument("--ratio_quantile", type=float, default=0.10, help="Quantile for auto thresholding (lower → stricter merges).")
    p.add_argument(
        "--offset",
        type=float,
        default=2.0,
        help="Additive slack for auto strategy (helps when self_mae is tiny).",
    )
    p.add_argument(
        "--max_degree",
        type=int,
        default=4,
        help="Limit merges per pack to avoid giant 'standardized' groups (0=unlimited).",
    )
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

    dsu = DSU()
    for a in packs:
        dsu.find(a)

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
        # auto: threshold later
        cand.append((score, aa, bb))

    # auto: thresholding
    auto_ratio_thr: Optional[float] = None
    scores = [s for s, _, _ in cand]
    auto_ratio_thr = _quantile(scores, args.ratio_quantile)
    cand = [(s, a, b) for (s, a, b) in cand if s <= auto_ratio_thr]

    # degree-limited selection to prevent large mega-groups
    picked = _select_edges_degree_limited(cand, args.max_degree)
    for a, b in picked:
        dsu.union(a, b)

    # connected components
    comps: Dict[str, List[str]] = {}
    for pck in packs:
        root = dsu.find(pck)
        comps.setdefault(root, []).append(pck)

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
    print(f"picked_edges={len(picked)}")
    print(f"groups_written={wrote}")
    if missing_lists:
        print(f"WARNING: missing_list_files={len(missing_lists)} (use correct --packs_lists_dir or omit --allow_missing_lists)")
    if empty_groups:
        print(f"WARNING: empty_group_files={empty_groups} (this indicates missing lists or empty input packs)")


if __name__ == "__main__":
    main()


