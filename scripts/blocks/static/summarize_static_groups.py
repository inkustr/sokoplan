from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from scripts.blocks.static.plan_groups_by_static_features import FEATURE_NAMES, _extract_features, _summary_stats


def _parse_feature_list(raw: str) -> List[str]:
    parts = [x.strip() for x in raw.split(",")]
    return [x for x in parts if x]


def _pick_features(obj: Dict, top_n: int, explicit: List[str]) -> List[str]:
    if explicit:
        return explicit
    ranked = obj.get("top_separating_features", [])
    out: List[str] = []
    for row in ranked[: max(1, top_n)]:
        feat = row.get("feature")
        if feat:
            out.append(str(feat))
    if out:
        return out
    return list(obj.get("feature_names", []))[: max(1, top_n)]


def _group_profile_text(
    g: Dict,
    features: List[str],
    global_stats: Dict[str, Dict[str, float]],
    stat_key: str,
    top_k: int = 4,
) -> str:
    fs = g.get("feature_stats", {})
    scored: List[Tuple[float, str, str]] = []
    for feat in features:
        gs = global_stats.get(feat, {})
        gmean = float(gs.get("mean", 0.0))
        gstd = float(gs.get("std", 0.0))
        gstd = max(1e-9, gstd)
        v = float(fs.get(feat, {}).get(stat_key, 0.0))
        z = (v - gmean) / gstd
        direction = "high" if z >= 0.0 else "low"
        scored.append((abs(z), feat, direction))
    scored.sort(key=lambda t: t[0], reverse=True)
    parts = [f"{direction} {feat}" for _, feat, direction in scored[: max(1, top_k)]]
    return ", ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--report_json",
        default="",
        help="Path to report from plan_groups_by_static_features.py. Optional if --groups_lists_dir is provided.",
    )
    p.add_argument(
        "--groups_lists_dir",
        default="",
        help="Directory with group_XXX.list files. If set, profiles can be computed even without report_json.",
    )
    p.add_argument(
        "--stat",
        default="median",
        choices=["min", "p10", "median", "p90", "max", "mean", "std"],
        help="Which statistic to print per feature.",
    )
    p.add_argument("--top_features", type=int, default=8, help="How many features to print per group.")
    p.add_argument(
        "--features",
        default="",
        help="Comma-separated feature list override. If empty, uses top-separating features from report.",
    )
    p.add_argument("--out_csv", default="", help="Optional output CSV path.")
    args = p.parse_args()

    obj: Dict = {}
    groups: List[Dict] = []
    global_stats: Dict[str, Dict[str, float]] = {}

    if args.report_json:
        with open(args.report_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        groups = list(obj.get("groups", []))
        global_stats = obj.get("global_feature_stats", {})

    if (not groups) and args.groups_lists_dir:
        if not os.path.isdir(args.groups_lists_dir):
            raise SystemExit(f"groups_lists_dir not found: {args.groups_lists_dir}")
        list_files = sorted(
            [
                os.path.join(args.groups_lists_dir, fn)
                for fn in os.listdir(args.groups_lists_dir)
                if fn.endswith(".list")
            ]
        )
        if not list_files:
            raise SystemExit(f"No .list files in {args.groups_lists_dir}")

        # Load level ids per group and extract features once per unique level.
        group_to_ids: Dict[str, List[str]] = {}
        all_ids: List[str] = []
        for lp in list_files:
            gid = os.path.splitext(os.path.basename(lp))[0]
            with open(lp, "r", encoding="utf-8") as f:
                ids = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
            group_to_ids[gid] = ids
            all_ids.extend(ids)

        uniq_ids = sorted(set(all_ids))
        feat_map: Dict[str, List[float]] = {}
        for i, lid in enumerate(uniq_ids, start=1):
            feat_map[lid] = _extract_features(lid).values
            if i % 500 == 0:
                print(f"features: {i}/{len(uniq_ids)}", flush=True)

        Xall = np.array([feat_map[lid] for lid in uniq_ids], dtype=np.float64)
        global_stats = {fname: _summary_stats(Xall[:, j]) for j, fname in enumerate(FEATURE_NAMES)}

        groups = []
        for gid, ids in group_to_ids.items():
            if not ids:
                continue
            Xg = np.array([feat_map[lid] for lid in ids], dtype=np.float64)
            feat_stats = {fname: _summary_stats(Xg[:, j]) for j, fname in enumerate(FEATURE_NAMES)}
            groups.append({"group_id": gid, "n_levels": len(ids), "feature_stats": feat_stats})

        # Compute top separating features directly from computed groups.
        gmeans_by_feat: Dict[str, List[float]] = {fname: [] for fname in FEATURE_NAMES}
        for g in groups:
            fs = g.get("feature_stats", {})
            for fname in FEATURE_NAMES:
                gmeans_by_feat[fname].append(float(fs.get(fname, {}).get("mean", 0.0)))
        sep_scores: List[Tuple[str, float]] = []
        eps = 1e-12
        for j, fname in enumerate(FEATURE_NAMES):
            gv = float(np.var(Xall[:, j]))
            bv = float(np.var(np.array(gmeans_by_feat[fname], dtype=np.float64)))
            sep_scores.append((fname, bv / max(eps, gv)))
        sep_scores.sort(key=lambda t: t[1], reverse=True)
        obj["top_separating_features"] = [{"feature": f, "score": float(s)} for f, s in sep_scores]
        obj["feature_names"] = FEATURE_NAMES

    if not groups:
        raise SystemExit("No groups found. Provide --report_json or --groups_lists_dir.")

    explicit = _parse_feature_list(args.features) if args.features else []
    features = _pick_features(obj, args.top_features, explicit)
    if not features:
        raise SystemExit("No features selected to print.")

    if not global_stats:
        global_stats = obj.get("global_feature_stats", {})
    if not global_stats:
        raise SystemExit("Global feature stats are missing. Re-run with --groups_lists_dir or use richer report_json.")
    groups = sorted(groups, key=lambda g: int(g.get("n_levels", 0)), reverse=True)

    header = ["group_id", "n_levels", "profile"] + [f"{feat}:{args.stat}" for feat in features]
    print("\t".join(header))

    rows: List[List[str]] = []
    for g in groups:
        gid = str(g.get("group_id", "group_???"))
        n_levels = int(g.get("n_levels", 0))
        fs = g.get("feature_stats", {})
        profile = _group_profile_text(g, features, global_stats, args.stat)
        vals: List[str] = []
        for feat in features:
            v = float(fs.get(feat, {}).get(args.stat, 0.0))
            vals.append(f"{v:.4f}")
        row = [gid, str(n_levels), profile] + vals
        rows.append(row)
        print("\t".join(row))

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"\nSaved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()

