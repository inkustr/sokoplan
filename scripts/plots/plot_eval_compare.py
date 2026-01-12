from __future__ import annotations

"""
Plot GNN vs Hungarian evaluation CSV comparisons.

Run:
  python -m scripts.plots.plot_eval_compare \
    --gnn_csv results/evaluate/batch_gnn_speed.sorted.csv \
    --hun_csv results/evaluate/train.sorted.csv \
    --out_dir results/evaluate/plots
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Row:
    level_id: str
    heuristic: str
    success: bool
    nodes: int
    runtime_s: float
    solution_len: int


def _read_csv(path: str) -> Dict[str, Row]:
    out: Dict[str, Row] = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            level_id = (row.get("level_id") or "").strip()
            if not level_id:
                continue
            out[level_id] = Row(
                level_id=level_id,
                heuristic=(row.get("heuristic") or "").strip(),
                success=str(row.get("success", "")).strip() == "True",
                nodes=int(row.get("nodes") or 0),
                runtime_s=float(row.get("runtime") or 0.0),
                solution_len=int(row.get("solution_len") or -1),
            )
    return out


def _savefig(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _safe_ratio(a: float, b: float) -> float:
    if b == 0.0:
        return float("inf") if a > 0 else 1.0
    return a / b


def _category(gnn: Row, hun: Row) -> str:
    if gnn.success and hun.success:
        return "both_success"
    if gnn.success and not hun.success:
        return "gnn_only"
    if (not gnn.success) and hun.success:
        return "hun_only"
    return "both_fail"


def _common_levels(gnn: Dict[str, Row], hun: Dict[str, Row]) -> List[str]:
    return sorted(set(gnn.keys()) & set(hun.keys()))


def plot_success_breakdown(
    common: List[str],
    gnn: Dict[str, Row],
    hun: Dict[str, Row],
    out_dir: str,
) -> None:
    counts = {"both_success": 0, "gnn_only": 0, "hun_only": 0, "both_fail": 0}
    for lid in common:
        counts[_category(gnn[lid], hun[lid])] += 1

    labels = ["both_success", "gnn_only", "hun_only", "both_fail"]
    vals = [counts[k] for k in labels]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, vals, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#7f7f7f"])
    plt.title("Success breakdown (common levels)")
    plt.ylabel("levels")
    plt.xticks(rotation=15, ha="right")
    _savefig(os.path.join(out_dir, "01_success_breakdown.png"))


def plot_scatter_runtime(
    common: List[str],
    gnn: Dict[str, Row],
    hun: Dict[str, Row],
    out_dir: str,
) -> None:
    xs, ys, cs = [], [], []
    cmap = {
        "both_success": "#2ca02c",
        "gnn_only": "#1f77b4",
        "hun_only": "#ff7f0e",
        "both_fail": "#7f7f7f",
    }
    for lid in common:
        gr, hr = gnn[lid].runtime_s, hun[lid].runtime_s
        xs.append(max(gr, 1e-6))
        ys.append(max(hr, 1e-6))
        cs.append(cmap[_category(gnn[lid], hun[lid])])

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(xs, ys, s=24, c=cs, alpha=0.85, edgecolors="none")
    lim = max(max(xs), max(ys)) * 1.1
    plt.plot([1e-6, lim], [1e-6, lim], "--", color="black", linewidth=1, label="equal time")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("GNN runtime (s)")
    plt.ylabel("Hungarian runtime (s)")
    plt.title("Runtime per level (log-log)")
    plt.legend(loc="lower right", frameon=True)
    _savefig(os.path.join(out_dir, "02_runtime_scatter_loglog.png"))


def plot_scatter_nodes(
    common: List[str],
    gnn: Dict[str, Row],
    hun: Dict[str, Row],
    out_dir: str,
) -> None:
    xs, ys, cs = [], [], []
    cmap = {
        "both_success": "#2ca02c",
        "gnn_only": "#1f77b4",
        "hun_only": "#ff7f0e",
        "both_fail": "#7f7f7f",
    }
    for lid in common:
        xs.append(max(gnn[lid].nodes, 1))
        ys.append(max(hun[lid].nodes, 1))
        cs.append(cmap[_category(gnn[lid], hun[lid])])

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(xs, ys, s=24, c=cs, alpha=0.85, edgecolors="none")
    lim = max(max(xs), max(ys)) * 1.1
    plt.plot([1, lim], [1, lim], "--", color="black", linewidth=1, label="equal nodes")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("GNN expanded nodes")
    plt.ylabel("Hungarian expanded nodes")
    plt.title("Nodes per level (log-log)")
    plt.legend(loc="lower right", frameon=True)
    _savefig(os.path.join(out_dir, "03_nodes_scatter_loglog.png"))


def plot_speedup_hist_cdf(
    common: List[str],
    gnn: Dict[str, Row],
    hun: Dict[str, Row],
    out_dir: str,
) -> None:
    speedups: List[float] = []
    node_ratios: List[float] = []
    for lid in common:
        if not (gnn[lid].success and hun[lid].success):
            continue
        speedups.append(_safe_ratio(hun[lid].runtime_s, gnn[lid].runtime_s))  # >1 => GNN faster
        node_ratios.append(_safe_ratio(gnn[lid].nodes, hun[lid].nodes))  # <1 => GNN fewer nodes

    if not speedups:
        return

    sp = np.array(speedups, dtype=float)
    nr = np.array(node_ratios, dtype=float)

    # Histogram of log2(speedup) centered at 0
    plt.figure(figsize=(7, 4))
    vals = np.log2(sp)
    plt.hist(vals, bins=30, color="#1f77b4", alpha=0.9)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel("log2(Hungarian_time / GNN_time)  ( >0 means GNN faster )")
    plt.ylabel("levels")
    plt.title("Speedup distribution (both succeeded)")
    _savefig(os.path.join(out_dir, "04_speedup_hist_log2.png"))

    # CDF of speedup
    plt.figure(figsize=(7, 4))
    xs = np.sort(sp)
    ys = np.linspace(0.0, 1.0, len(xs), endpoint=True)
    plt.plot(xs, ys, color="#1f77b4")
    plt.axvline(1.0, color="black", linewidth=1, linestyle="--")
    plt.xscale("log")
    plt.xlabel("Hungarian_time / GNN_time  (log scale)")
    plt.ylabel("CDF")
    plt.title("Speedup CDF (both succeeded)")
    _savefig(os.path.join(out_dir, "05_speedup_cdf.png"))

    # CDF of node ratio
    plt.figure(figsize=(7, 4))
    xs = np.sort(nr)
    ys = np.linspace(0.0, 1.0, len(xs), endpoint=True)
    plt.plot(xs, ys, color="#2ca02c")
    plt.axvline(1.0, color="black", linewidth=1, linestyle="--")
    plt.xscale("log")
    plt.xlabel("GNN_nodes / Hungarian_nodes  (log scale)")
    plt.ylabel("CDF")
    plt.title("Node ratio CDF (both succeeded)")
    _savefig(os.path.join(out_dir, "06_node_ratio_cdf.png"))


def plot_node_ratio_vs_time_ratio(
    common: List[str],
    gnn: Dict[str, Row],
    hun: Dict[str, Row],
    out_dir: str,
) -> None:
    xs, ys = [], []
    for lid in common:
        if not (gnn[lid].success and hun[lid].success):
            continue
        xs.append(_safe_ratio(gnn[lid].nodes, hun[lid].nodes))
        ys.append(_safe_ratio(gnn[lid].runtime_s, hun[lid].runtime_s))

    if not xs:
        return

    plt.figure(figsize=(7, 6))
    plt.scatter(xs, ys, s=26, alpha=0.85, color="#9467bd", edgecolors="none")
    plt.axvline(1.0, color="black", linewidth=1, linestyle="--")
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("GNN_nodes / Hungarian_nodes  (<1 means fewer nodes)")
    plt.ylabel("GNN_time / Hungarian_time  (<1 means faster)")
    plt.title("Where GNN wins/loses: nodes vs time (both succeeded)")
    _savefig(os.path.join(out_dir, "07_nodes_vs_time_ratio.png"))


def plot_per_node_cost(
    common: List[str],
    gnn: Dict[str, Row],
    hun: Dict[str, Row],
    out_dir: str,
) -> None:
    # milliseconds per expanded node
    gnn_ms, hun_ms = [], []
    for lid in common:
        gr, hr = gnn[lid], hun[lid]
        if gr.nodes > 0:
            gnn_ms.append((gr.runtime_s * 1000.0) / gr.nodes)
        if hr.nodes > 0:
            hun_ms.append((hr.runtime_s * 1000.0) / hr.nodes)

    if not gnn_ms or not hun_ms:
        return

    plt.figure(figsize=(7, 4))
    plt.hist(np.log10(np.array(hun_ms)), bins=30, alpha=0.65, label="Hungarian", color="#ff7f0e")
    plt.hist(np.log10(np.array(gnn_ms)), bins=30, alpha=0.65, label="GNN", color="#1f77b4")
    plt.xlabel("log10(ms per expanded node)")
    plt.ylabel("levels")
    plt.title("Per-node evaluation cost distribution")
    plt.legend()
    _savefig(os.path.join(out_dir, "08_per_node_cost_hist_log10.png"))


def plot_sorted_runtime_curves(
    common: List[str],
    gnn: Dict[str, Row],
    hun: Dict[str, Row],
    out_dir: str,
) -> None:
    # Show distribution of runtimes by sorting each method independently
    g = sorted([gnn[lid].runtime_s for lid in common])
    h = sorted([hun[lid].runtime_s for lid in common])
    x = np.arange(len(common))

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, g, label="GNN", color="#1f77b4")
    plt.plot(x, h, label="Hungarian", color="#ff7f0e")
    plt.yscale("log")
    plt.xlabel("levels (sorted within each method)")
    plt.ylabel("runtime (s, log scale)")
    plt.title("Sorted runtime curves")
    plt.legend()
    _savefig(os.path.join(out_dir, "09_sorted_runtime_curves.png"))


def plot_solution_len_scatter(
    common: List[str],
    gnn: Dict[str, Row],
    hun: Dict[str, Row],
    out_dir: str,
) -> None:
    """
    Compare solution lengths on levels where both methods succeeded.
    Hungarian is treated as the reference baseline.
    """
    xs, ys = [], []
    for lid in common:
        gr, hr = gnn[lid], hun[lid]
        if not (gr.success and hr.success):
            continue
        if gr.solution_len < 0 or hr.solution_len < 0:
            continue
        xs.append(float(hr.solution_len))
        ys.append(float(gr.solution_len))

    if not xs:
        return

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(xs, ys, s=24, alpha=0.85, color="#1f77b4", edgecolors="none")
    lim = max(max(xs), max(ys)) * 1.1
    plt.plot([0, lim], [0, lim], "--", color="black", linewidth=1, label="equal length")
    plt.xlabel("Hungarian solution length (pushes)")
    plt.ylabel("GNN solution length (pushes)")
    plt.title("Solution length per level (both succeeded)")
    plt.legend(loc="upper left", frameon=True)
    _savefig(os.path.join(out_dir, "10_solution_len_scatter.png"))


def plot_solution_len_ratio_hist_cdf(
    common: List[str],
    gnn: Dict[str, Row],
    hun: Dict[str, Row],
    out_dir: str,
) -> None:
    """
    Distribution of "how much worse" GNN is in solution length vs Hungarian on levels
    where both methods succeeded:

      worse_pct = (gnn_len / hun_len - 1) * 100

    (0% => equal length; 10% => GNN solution is 10% longer)
    """
    worse_pcts: List[float] = []
    for lid in common:
        gr, hr = gnn[lid], hun[lid]
        if not (gr.success and hr.success):
            continue
        if gr.solution_len <= 0 or hr.solution_len <= 0:
            continue
        ratio = float(gr.solution_len) / float(hr.solution_len)
        worse_pcts.append((ratio - 1.0) * 100.0)

    if not worse_pcts:
        return

    w = np.array(worse_pcts, dtype=float)

    # Histogram of percent worse, centered at 0 (0 => equal)
    plt.figure(figsize=(7, 4))
    plt.hist(w, bins=30, color="#1f77b4", alpha=0.9)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel("Percent worse vs Hungarian: (GNN_len / Hungarian_len - 1) * 100%")
    plt.ylabel("levels")
    plt.title("Solution length: % worse distribution (both succeeded)")
    _savefig(os.path.join(out_dir, "11_solution_len_worse_pct_hist.png"))

    # CDF of percent worse
    plt.figure(figsize=(7, 4))
    xs = np.sort(w)
    ys = np.linspace(0.0, 1.0, len(xs), endpoint=True)
    plt.plot(xs, ys, color="#1f77b4")
    plt.axvline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Percent worse vs Hungarian")
    plt.ylabel("CDF")
    plt.title("Solution length: % worse CDF (both succeeded)")
    _savefig(os.path.join(out_dir, "12_solution_len_worse_pct_cdf.png"))


def main() -> None:
    p = argparse.ArgumentParser(description="Plot GNN vs Hungarian evaluation CSV comparisons.")
    p.add_argument("--gnn_csv", required=True)
    p.add_argument("--hun_csv", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    gnn = _read_csv(args.gnn_csv)
    hun = _read_csv(args.hun_csv)
    common = _common_levels(gnn, hun)
    if not common:
        raise SystemExit("No common levels between provided CSVs.")

    os.makedirs(args.out_dir, exist_ok=True)

    plot_success_breakdown(common, gnn, hun, args.out_dir)
    plot_scatter_runtime(common, gnn, hun, args.out_dir)
    plot_scatter_nodes(common, gnn, hun, args.out_dir)
    plot_speedup_hist_cdf(common, gnn, hun, args.out_dir)
    plot_node_ratio_vs_time_ratio(common, gnn, hun, args.out_dir)
    plot_per_node_cost(common, gnn, hun, args.out_dir)
    plot_sorted_runtime_curves(common, gnn, hun, args.out_dir)
    plot_solution_len_scatter(common, gnn, hun, args.out_dir)
    plot_solution_len_ratio_hist_cdf(common, gnn, hun, args.out_dir)

    print(f"wrote plots → {args.out_dir}")


if __name__ == "__main__":
    main()


