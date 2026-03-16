#!/usr/bin/env python3
"""Build cross-pack evaluation tables/matrices from per-run CSV outputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class RunStats:
    target_group: str
    model_group: str
    num_levels: int
    success_count: int
    success_rate: float
    avg_runtime_all: float
    avg_nodes_all: float
    avg_runtime_success: float
    avg_nodes_success: float


def parse_group_from_token(token: str, prefix: str) -> str:
    if token.startswith(prefix):
        token = token[len(prefix) :]
    token = token.replace(".csv", "")
    if token.startswith("group_"):
        return token
    raise ValueError(f"Cannot parse group id from token='{token}' with prefix='{prefix}'")


def discover_runs(raw_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    for target_dir in sorted(raw_dir.glob("target_group_*")):
        if not target_dir.is_dir():
            continue
        target_group = parse_group_from_token(target_dir.name, "target_")
        for csv_path in sorted(target_dir.glob("model_group_*.csv")):
            model_group = parse_group_from_token(csv_path.name, "model_")
            yield target_group, model_group, csv_path


def read_run_stats(target_group: str, model_group: str, csv_path: Path) -> RunStats:
    num_levels = 0
    success_count = 0
    sum_runtime_all = 0.0
    sum_nodes_all = 0.0
    sum_runtime_success = 0.0
    sum_nodes_success = 0.0

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("level_id"):
                continue
            num_levels += 1
            runtime = float(row["runtime"])
            nodes = float(row["nodes"])
            success = row["success"] == "True"

            sum_runtime_all += runtime
            sum_nodes_all += nodes
            if success:
                success_count += 1
                sum_runtime_success += runtime
                sum_nodes_success += nodes

    if num_levels == 0:
        raise ValueError(f"Empty results file: {csv_path}")

    avg_runtime_all = sum_runtime_all / num_levels
    avg_nodes_all = sum_nodes_all / num_levels
    success_rate = success_count / num_levels
    avg_runtime_success = (sum_runtime_success / success_count) if success_count else float("nan")
    avg_nodes_success = (sum_nodes_success / success_count) if success_count else float("nan")

    return RunStats(
        target_group=target_group,
        model_group=model_group,
        num_levels=num_levels,
        success_count=success_count,
        success_rate=success_rate,
        avg_runtime_all=avg_runtime_all,
        avg_nodes_all=avg_nodes_all,
        avg_runtime_success=avg_runtime_success,
        avg_nodes_success=avg_nodes_success,
    )


def write_long_table(out_path: Path, stats: List[RunStats]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "target_group",
            "model_group",
            "num_levels",
            "success_count",
            "success_rate",
            "avg_runtime_all",
            "avg_nodes_all",
            "avg_runtime_success",
            "avg_nodes_success",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in stats:
            w.writerow(
                {
                    "target_group": s.target_group,
                    "model_group": s.model_group,
                    "num_levels": s.num_levels,
                    "success_count": s.success_count,
                    "success_rate": f"{s.success_rate:.6f}",
                    "avg_runtime_all": f"{s.avg_runtime_all:.6f}",
                    "avg_nodes_all": f"{s.avg_nodes_all:.6f}",
                    "avg_runtime_success": f"{s.avg_runtime_success:.6f}",
                    "avg_nodes_success": f"{s.avg_nodes_success:.6f}",
                }
            )


def write_metric_matrix(
    out_path: Path, stats: List[RunStats], metric: str, targets: List[str], models: List[str]
) -> None:
    metric_map = {
        "success_rate": lambda s: s.success_rate,
        "avg_runtime_all": lambda s: s.avg_runtime_all,
        "avg_nodes_all": lambda s: s.avg_nodes_all,
        "avg_runtime_success": lambda s: s.avg_runtime_success,
        "avg_nodes_success": lambda s: s.avg_nodes_success,
    }
    if metric not in metric_map:
        raise ValueError(f"Unsupported metric: {metric}")

    by_pair: Dict[Tuple[str, str], RunStats] = {(s.target_group, s.model_group): s for s in stats}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target_group", *models])
        for target in targets:
            row = [target]
            for model in models:
                entry = by_pair.get((target, model))
                if entry is None:
                    row.append("")
                else:
                    row.append(f"{metric_map[metric](entry):.6f}")
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="Build matrices from static cross-pack eval CSVs")
    p.add_argument(
        "--raw_dir",
        default="results/evaluate/static_crosspack/raw",
        help="Directory with target_group_*/model_group_*.csv files",
    )
    p.add_argument(
        "--out_dir",
        default="results/evaluate/static_crosspack/matrix",
        help="Where to store aggregated tables/matrices",
    )
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    stats: List[RunStats] = []
    for target_group, model_group, csv_path in discover_runs(raw_dir):
        stats.append(read_run_stats(target_group, model_group, csv_path))

    if not stats:
        raise RuntimeError(f"No cross-pack CSV files found under: {raw_dir}")

    stats.sort(key=lambda s: (s.target_group, s.model_group))
    targets = sorted({s.target_group for s in stats})
    models = sorted({s.model_group for s in stats})

    long_table_path = out_dir / "crosspack_long.csv"
    write_long_table(long_table_path, stats)

    metrics = [
        "success_rate",
        "avg_runtime_all",
        "avg_nodes_all",
        "avg_runtime_success",
        "avg_nodes_success",
    ]
    for metric in metrics:
        write_metric_matrix(out_dir / f"crosspack_matrix_{metric}.csv", stats, metric, targets, models)

    print(f"[OK] processed runs: {len(stats)}")
    print(f"[OK] targets: {len(targets)} ; models: {len(models)}")
    print(f"[OK] long table: {long_table_path}")
    for metric in metrics:
        print(f"[OK] matrix: {out_dir / f'crosspack_matrix_{metric}.csv'}")


if __name__ == "__main__":
    main()
