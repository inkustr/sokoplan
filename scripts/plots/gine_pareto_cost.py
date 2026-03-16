from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


@dataclass(frozen=True)
class ModelPoint:
    label: str  # "<hidden>_<layers>"
    hidden: int
    layers: int
    total_nodes: int
    total_time_s: float
    time_per_node_ms: float


@dataclass(frozen=True)
class BaselinePoint:
    label: str
    total_nodes: int
    total_time_s: float
    time_per_node_ms: float


def _parse_hidden_layers(model_csv: str) -> Optional[Tuple[int, int]]:
    # expected pattern examples:
    # batch_gnn_gnn_best_128_4_3872256_lr1e-4.csv
    # batch_gnn_gnn_best_64_2_3872251_lr1e-4.csv
    m = re.search(r"gnn_best_(\d+)_(\d+)_", model_csv)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def load_gine_summary(path: Path) -> List[ModelPoint]:
    out: List[ModelPoint] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            model_csv = (row.get("model_csv") or "").strip()
            hl = _parse_hidden_layers(model_csv)
            if hl is None:
                continue
            hidden, layers = hl
            try:
                total_nodes = int(float(row.get("total_nodes") or 0))
                total_time_s = float(row.get("total_time_s") or 0.0)
                time_per_node_ms = float(row.get("time_per_node_ms") or 0.0)
            except Exception:
                continue
            out.append(
                ModelPoint(
                    label=f"{hidden}_{layers}",
                    hidden=hidden,
                    layers=layers,
                    total_nodes=total_nodes,
                    total_time_s=total_time_s,
                    time_per_node_ms=time_per_node_ms,
                )
            )
    if not out:
        raise SystemExit(f"No valid model points parsed from {path}")
    return out


def load_hungarian_baseline(path: Path) -> BaselinePoint:
    total_nodes = 0
    total_time_s = 0.0
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not (row.get("level_id") or "").strip():
                continue
            try:
                total_nodes += int(float(row.get("nodes") or 0))
            except Exception:
                pass
            try:
                total_time_s += float(row.get("runtime") or 0.0)
            except Exception:
                pass
    if total_nodes <= 0:
        raise SystemExit(f"Could not compute Hungarian baseline from {path}: total_nodes={total_nodes}")
    return BaselinePoint(
        label="hungarian",
        total_nodes=total_nodes,
        total_time_s=total_time_s,
        time_per_node_ms=1000.0 * total_time_s / float(total_nodes),
    )


def _annotate(
    ax,
    x: float,
    y: float,
    text: str,
    color: str = "black",
    xytext: Tuple[float, float] = (4, 4),
    ha: str = "left",
) -> None:
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=xytext,
        ha=ha,
        fontsize=8,
        color=color,
    )


def _hidden_color_map(points: List[ModelPoint]) -> Dict[int, Tuple[float, float, float, float]]:
    hidden_vals = sorted({p.hidden for p in points})
    cmap = plt.get_cmap("tab10")
    return {h: cmap(i % 10) for i, h in enumerate(hidden_vals)}


def _best_cloud_bounds(points: List[ModelPoint]) -> Tuple[float, float, float, float]:
    # Choose "best" points by joint rank in (nodes, time), then build inset bounds.
    n = len(points)
    if n == 0:
        return (0.0, 1.0, 0.0, 1.0)
    k = max(6, n // 3)
    by_nodes = {id(p): i for i, p in enumerate(sorted(points, key=lambda x: x.total_nodes))}
    by_time = {id(p): i for i, p in enumerate(sorted(points, key=lambda x: x.total_time_s))}
    ranked = sorted(points, key=lambda p: by_nodes[id(p)] + by_time[id(p)])
    best = ranked[:k]

    xs = [p.total_nodes for p in best]
    ys = [p.total_time_s for p in best]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add margins.
    dx = max(1.0, 0.15 * (x_max - x_min))
    dy = max(1.0, 0.15 * (y_max - y_min))
    return (x_min - dx, x_max + dx, y_min - dy, y_max + dy)


def plot_pareto(points: List[ModelPoint], base: BaselinePoint, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    color_map = _hidden_color_map(points)

    for h in sorted(color_map.keys()):
        hp = [p for p in points if p.hidden == h]
        ax.scatter(
            [p.total_nodes for p in hp],
            [p.total_time_s for p in hp],
            s=54,
            marker="o",
            alpha=0.9,
            label=f"GNN hidden={h}",
            color=color_map[h],
        )
        for p in hp:
            _annotate(ax, p.total_nodes, p.total_time_s, p.label, color=color_map[h])

    ax.scatter(
        [base.total_nodes],
        [base.total_time_s],
        s=120,
        marker="*",
        color="#d62728",
        label="Hungarian baseline",
        zorder=5,
    )
    _annotate(
        ax,
        base.total_nodes,
        base.total_time_s,
        base.label,
        color="#d62728",
        xytext=(-8, 4),
        ha="right",
    )

    # Inset around the best cloud.
    x1, x2, y1, y2 = _best_cloud_bounds(points)
    # Place inset under the legend area (right side).
    axins = inset_axes(
        ax,
        width="70%",
        height="38%",
        loc="upper right",
        bbox_to_anchor=(0.0, 0.0, 1.0, 0.78),
        bbox_transform=ax.transAxes,
        borderpad=0.8,
    )
    # Inset x-range: from best-cloud left bound to ~32_6 nodes (+small margin).
    p326 = next((p for p in points if p.hidden == 32 and p.layers == 6), None)
    x_left = x1
    x_right = (float(p326.total_nodes) * 1.08) if p326 is not None else x2

    for h in sorted(color_map.keys()):
        hp = [p for p in points if p.hidden == h and (x_left <= p.total_nodes <= x_right and y1 <= p.total_time_s <= y2)]
        if not hp:
            continue
        axins.scatter(
            [p.total_nodes for p in hp],
            [p.total_time_s for p in hp],
            s=38,
            marker="o",
            alpha=0.9,
            color=color_map[h],
        )
        for p in hp:
            _annotate(axins, p.total_nodes, p.total_time_s, p.label, color=color_map[h])
    axins.set_xlim(x_left, x_right)
    axins.set_ylim(y1, y2)
    axins.set_title("", fontsize=8)
    axins.grid(alpha=0.2)
    axins.tick_params(labelsize=7)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4", lw=0.9)

    ax.set_xlabel("Total expanded nodes")
    ax.set_ylabel("Total runtime (s)")
    ax.set_title("Total runtime vs total expanded nodes")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cost_vs_effort(points: List[ModelPoint], base: BaselinePoint, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    color_map = _hidden_color_map(points)

    for h in sorted(color_map.keys()):
        hp = [p for p in points if p.hidden == h]
        ax.scatter(
            [p.total_nodes for p in hp],
            [p.time_per_node_ms for p in hp],
            s=54,
            marker="o",
            alpha=0.9,
            label=f"GNN hidden={h}",
            color=color_map[h],
        )
        for p in hp:
            _annotate(ax, p.total_nodes, p.time_per_node_ms, p.label, color=color_map[h])

    ax.scatter(
        [base.total_nodes],
        [base.time_per_node_ms],
        s=120,
        marker="*",
        color="#d62728",
        label="Hungarian baseline",
        zorder=5,
    )
    _annotate(
        ax,
        base.total_nodes,
        base.time_per_node_ms,
        base.label,
        color="#d62728",
        xytext=(-8, 4),
        ha="right",
    )

    ax.set_xlabel("total_nodes")
    ax.set_ylabel("time_per_node_ms")
    ax.set_title("Cost vs Effort: nodes vs per-node cost")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Pareto and cost-vs-effort charts for GINE configs.")
    ap.add_argument(
        "--gine_summary_csv",
        default="__old_labels/evaluate/gine/models_summary_lr1e-4.csv",
        help="CSV with model summary rows.",
    )
    ap.add_argument(
        "--hungarian_csv",
        default="__old_labels/evaluate/hungarian.csv",
        help="Hungarian per-level eval CSV.",
    )
    ap.add_argument(
        "--out_dir",
        default="__old_labels/evaluate/gine/plots_lr1e-4",
        help="Directory to save plots.",
    )
    args = ap.parse_args()

    summary_path = Path(args.gine_summary_csv)
    hungarian_path = Path(args.hungarian_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = load_gine_summary(summary_path)
    base = load_hungarian_baseline(hungarian_path)

    out_pareto = out_dir / "pareto_nodes_vs_time.png"
    out_cost = out_dir / "cost_vs_effort_nodes_vs_time_per_node.png"
    plot_pareto(points, base, out_pareto)
    plot_cost_vs_effort(points, base, out_cost)

    print(f"[DONE] wrote: {out_pareto}")
    print(f"[DONE] wrote: {out_cost}")


if __name__ == "__main__":
    main()
