from __future__ import annotations

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

"""
Evaluate many GNN models and summarize results.

Usage:

Generate CSVs for all models:
python -m scripts.eval.eval_gnn_models run --list sokoban_core/levels/splits_boxoban/temp.txt \
  --models_dir artifacts/boxoban \
  --out_dir results/evaluate/boxoban \
  --mode speed \
  --ensure_hungarian \
  --hungarian_csv results/evaluate/boxoban/hungarian.csv
  --sort total_time_s (total_nodes, node_overhead_vs_hun, name)

Summarize CSVs:
python -m scripts.eval.eval_gnn_models summarize \
  --eval_dir results/evaluate/boxoban \
  --hungarian_csv results/evaluate/boxoban/hungarian.csv \
  --out results/evaluate/boxoban/models_summary.csv \
  --sort total_time_s (total_nodes, node_overhead_vs_hun, name)
"""

@dataclass(frozen=True)
class Totals:
    total_time_s: float
    total_nodes: int
    n_levels: int
    n_success: int

    @property
    def time_per_node_ms(self) -> float:
        if self.total_nodes <= 0:
            return float("inf")
        return 1000.0 * self.total_time_s / float(self.total_nodes)


def _iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row.get("level_id"):
                continue
            yield row


def _totals_from_csv(path: Path) -> Totals:
    total_time = 0.0
    total_nodes = 0
    n = 0
    n_success = 0
    for row in _iter_csv_rows(path):
        n += 1
        try:
            total_time += float(row.get("runtime", "0") or 0.0)
        except Exception:
            pass
        try:
            total_nodes += int(float(row.get("nodes", "0") or 0))
        except Exception:
            pass
        n_success += 1 if (row.get("success") == "True") else 0
    return Totals(total_time_s=total_time, total_nodes=total_nodes, n_levels=n, n_success=n_success)


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def cmd_run(args: argparse.Namespace) -> None:
    models_dir: Path = args.models_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(models_dir.glob(args.models_glob))
    if not models:
        raise SystemExit(f"No models found in {models_dir} matching {args.models_glob!r}")

    if args.ensure_hungarian:
        if not args.hungarian_csv:
            raise SystemExit("--ensure_hungarian requires --hungarian_csv")
        hun_csv = Path(args.hungarian_csv)
        if not hun_csv.exists() or args.force:
            hun_csv.parent.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    args.python,
                    "-m",
                    "scripts.run_batch",
                    "--list",
                    args.list,
                    "--h",
                    "hungarian",
                    "--use_dl",
                    "--out",
                    str(hun_csv),
                    "--time_limit",
                    str(args.time_limit),
                    "--node_limit",
                    str(args.node_limit),
                    "--jobs",
                    "0",
                ]
            )

    for m in models:
        stem = m.stem
        out_csv = out_dir / f"{args.prefix}{stem}.csv"
        if out_csv.exists() and not args.force:
            print(f"[skip] {out_csv} exists", flush=True)
            continue
        _run(
            [
                args.python,
                "-m",
                "scripts.run_batch_gnn",
                "--list",
                args.list,
                "--ckpt",
                str(m),
                "--mode",
                args.mode,
                "--out",
                str(out_csv),
                "--time_limit",
                str(args.time_limit),
                "--node_limit",
                str(args.node_limit),
                "--jobs",
                "0",
            ]
            + (["--no_dl"] if args.no_dl else [])
        )

    print(f"Done. Wrote CSVs to {out_dir}", flush=True)


def cmd_summarize(args: argparse.Namespace) -> None:
    eval_dir: Path = args.eval_dir
    if not eval_dir.exists():
        raise SystemExit(f"eval_dir not found: {eval_dir}")

    csvs = sorted(eval_dir.glob(args.eval_glob))
    if not csvs:
        raise SystemExit(f"No eval CSVs found in {eval_dir} matching {args.eval_glob!r}")

    hun_per_node_ms = float(args.hungarian_time_per_node_ms)
    hun_totals = None
    if args.hungarian_csv:
        hun_path = Path(args.hungarian_csv)
        if hun_path.exists():
            hun_totals = _totals_from_csv(hun_path)
            hun_per_node_ms = hun_totals.time_per_node_ms

    rows = []
    for p in csvs:
        t = _totals_from_csv(p)
        overhead = t.time_per_node_ms / hun_per_node_ms if hun_per_node_ms > 0 else float("inf")
        rows.append(
            {
                "model_csv": p.name,
                "levels": str(t.n_levels),
                "success": f"{t.n_success}/{t.n_levels}",
                "total_time_s": f"{t.total_time_s:.2f}",
                "total_nodes": str(t.total_nodes),
                "time_per_node_ms": f"{t.time_per_node_ms:.4f}",
                "node_overhead_vs_hun": f"{overhead:.2f}",
            }
        )

    key = args.sort
    if key == "total_time_s":
        rows.sort(key=lambda r: float(r["total_time_s"]))
    elif key == "total_nodes":
        rows.sort(key=lambda r: int(r["total_nodes"]))
    elif key == "node_overhead_vs_hun":
        rows.sort(key=lambda r: float(r["node_overhead_vs_hun"]))
    else:
        rows.sort(key=lambda r: r["model_csv"])

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "model_csv",
                    "levels",
                    "success",
                    "total_time_s",
                    "total_nodes",
                    "time_per_node_ms",
                    "node_overhead_vs_hun",
                ],
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote summary CSV: {out_path}", flush=True)
        return

    print("model_csv, levels, success, total_time_s, total_nodes, time_per_node_ms, node_overhead_vs_hun", flush=True)
    for r in rows:
        print(
            f"{r['model_csv']}, {r['levels']}, {r['success']}, {r['total_time_s']}, {r['total_nodes']}, {r['time_per_node_ms']}, {r['node_overhead_vs_hun']}",
            flush=True,
        )
    if hun_totals is not None:
        print(
            f"\nHungarian baseline: total_time_s={hun_totals.total_time_s:.2f} total_nodes={hun_totals.total_nodes} "
            f"time_per_node_ms={hun_per_node_ms:.4f}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate many GNN models and summarize results.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Run scripts/run_batch_gnn.py for all models (jobs=1).")
    runp.add_argument("--python", default="python3", help="Python executable to use (default: python3).")
    runp.add_argument("--list", required=True, help="List file of level_ids to evaluate.")
    runp.add_argument("--models_dir", type=Path, default=Path("artifacts"), help="Directory containing model .pt files.")
    runp.add_argument("--models_glob", default="gnn_best_*.pt", help="Glob pattern under models_dir.")
    runp.add_argument("--out_dir", type=Path, default=Path("results/evaluate/models"), help="Output directory for per-model CSVs.")
    runp.add_argument("--prefix", default="batch_gnn_", help="Prefix for output CSV filenames.")
    runp.add_argument("--mode", default="speed", choices=["speed", "optimal_mix"])
    runp.add_argument("--no_dl", action="store_true", help="Disable deadlock filter for GNN eval.")
    runp.add_argument("--time_limit", type=float, default=600.0)
    runp.add_argument("--node_limit", type=int, default=2_000_000)
    runp.add_argument("--force", action="store_true", help="Recompute even if CSV exists.")
    runp.add_argument("--ensure_hungarian", action="store_true", help="Also compute Hungarian+DL baseline CSV once.")
    runp.add_argument("--hungarian_csv", default="results/evaluate/hungarian.csv")

    sump = sub.add_parser("summarize", help="Summarize per-model CSVs into one table.")
    sump.add_argument("--eval_dir", type=Path, default=Path("results/evaluate/models"))
    sump.add_argument("--eval_glob", default="batch_gnn_*.csv")
    sump.add_argument("--hungarian_csv", default="results/evaluate/hungarian.csv")
    sump.add_argument(
        "--hungarian_time_per_node_ms",
        type=float,
        default=0.21,
        help="Fallback Hungarian time-per-node in ms, if --hungarian_csv missing (default 0.21).",
    )
    sump.add_argument("--sort", default="total_time_s", choices=["total_time_s", "total_nodes", "node_overhead_vs_hun", "name"])
    sump.add_argument("--out", default="results/evaluate/models_summary.csv", help="Write summary CSV here (empty to print).")

    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        cmd_summarize(args)


if __name__ == "__main__":
    main()


